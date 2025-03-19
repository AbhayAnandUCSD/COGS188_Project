import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
from gymnasium import spaces
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import logging
import carla
from typing import Tuple, Dict, Any, List, Optional
import time
import os
import collections
import json
import queue
from torch.utils.data import Dataset, DataLoader

# Fix for collections.MutableMapping in Python 3.10+
if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === CARLA Environment Wrapper ===

class CarlaEnvWrapper(gym.Env):
    """Environment wrapper for CARLA simulator with fixes for synchronization and action mapping."""
    def __init__(self, host='localhost', port=2000):
        super().__init__()
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle = None
        self.lidar_sensor = None
        self.collision_sensor = None
        self.lane_sensor = None
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)  # Throttle/Brake, Steering
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)
        self.max_retries = 3
        self.last_position = None
        
        # Enable synchronous mode for consistent updates
        self._set_synchronous_mode(True)
        
        # Queue for LiDAR data to ensure latest sensor readings
        self.lidar_data_queue = queue.Queue()
        
        # Tracking variables for rewards and navigation
        self.steps_in_episode = 0
        self.total_distance = 0.0
        self.current_waypoint = None
        self.next_waypoint = None
        self.target_speed = 30.0  # km/h
        self.previous_lane_deviation = 0.0
        self.cumulative_lane_invasions = 0
        self.invaded_opposite_lane = False
        self.map = self.world.get_map()
        self.route = []
        self.current_route_index = 0
        self.debug_navigation = True
        self.previous_actions = []
        self.last_positions = []
        self.position_history_limit = 50

    def _set_synchronous_mode(self, synchronous_mode):
        """Set simulation to synchronous or asynchronous mode."""
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        settings.fixed_delta_seconds = 0.05 if synchronous_mode else None
        self.world.apply_settings(settings)
        logger.info(f"Set simulation to {'synchronous' if synchronous_mode else 'asynchronous'} mode")

    def plan_route_circle(self):
        """Plan a circular route using CARLA waypoints."""
        if not self.current_waypoint:
            logger.warning("Can't create route: current_waypoint is None")
            return []
        
        route = []
        wp = self.current_waypoint
        for _ in range(60):
            next_wps = wp.next(10.0)
            if not next_wps:
                break
            wp = next_wps[0]
            route.append(wp)
        
        last_wp = route[-1] if route else wp
        turn_wp = last_wp
        for _ in range(20):
            right_wp = turn_wp.get_right_lane()
            if right_wp and right_wp.lane_type == carla.LaneType.Driving:
                next_wps = right_wp.next(7.0)
                if next_wps:
                    turn_wp = next_wps[0]
                    route.append(turn_wp)
            else:
                next_wps = turn_wp.next(7.0)
                if next_wps:
                    turn_wp = next_wps[0]
                    route.append(turn_wp)
        
        return_points = []
        for _ in range(40):
            next_wps = turn_wp.next(10.0)
            if not next_wps:
                break
            turn_wp = next_wps[0]
            return_points.append(turn_wp)
            if turn_wp.transform.location.distance(self.current_waypoint.transform.location) < 30.0:
                break
        
        route = route + return_points
        if self.debug_navigation and route:
            self.draw_waypoints(route)
        logger.info(f"Created navigation route with {len(route)} waypoints")
        return route

    def draw_waypoints(self, waypoints, lifetime=20.0):
        """Draw waypoints in the simulator for debugging."""
        for i, waypoint in enumerate(waypoints):
            color = carla.Color(0, 0, 255) if i != self.current_route_index else carla.Color(255, 0, 0)
            self.world.debug.draw_point(waypoint.transform.location + carla.Location(z=0.5), size=0.1, color=color, life_time=lifetime)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and initialize sensors."""
        for attempt in range(self.max_retries):
            try:
                # Clean up existing actors
                if self.vehicle:
                    self.vehicle.destroy()
                if self.lidar_sensor:
                    self.lidar_sensor.destroy()
                if self.collision_sensor:
                    self.collision_sensor.destroy()
                if self.lane_sensor:
                    self.lane_sensor.destroy()

                # Spawn vehicle
                vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
                spawn_point = self.world.get_map().get_spawn_points()[0]
                self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                
                # Attach LiDAR sensor
                lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
                lidar_bp.set_attribute('range', '50')
                lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
                self.lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
                self.lidar_data_queue = queue.Queue()
                self.lidar_sensor.listen(lambda data: self.lidar_data_queue.put(data))
                
                # Attach collision sensor
                collision_bp = self.blueprint_library.find('sensor.other.collision')
                self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
                self.collision_flag = False
                self.collision_sensor.listen(lambda event: setattr(self, 'collision_flag', True))
                
                # Attach lane invasion sensor
                lane_bp = self.blueprint_library.find('sensor.other.lane_invasion')
                self.lane_sensor = self.world.spawn_actor(lane_bp, carla.Transform(), attach_to=self.vehicle)
                self.lane_invasion_flag = False
                self.lane_sensor.listen(lambda event: setattr(self, 'lane_invasion_flag', True))
                
                # Reset tracking variables
                self.steps_in_episode = 0
                self.total_distance = 0.0
                self.cumulative_lane_invasions = 0
                
                # Initialize waypoints
                vehicle_location = self.vehicle.get_location()
                self.current_waypoint = self.map.get_waypoint(vehicle_location, lane_type=carla.LaneType.Driving)
                if self.current_waypoint:
                    self.next_waypoint = self.current_waypoint.next(10.0)[0] if self.current_waypoint.next(10.0) else self.current_waypoint
                
                self.last_position = vehicle_location
                self.route = self.plan_route_circle()
                self.current_route_index = 0
                
                # Wait for sensor data and tick world
                self._wait_for_lidar_data()
                self.world.tick()
                
                state = self._get_state()
                logger.info("CARLA environment reset successfully")
                return state, {}
            except Exception as e:
                logger.error(f"Reset attempt {attempt+1}/{self.max_retries} failed: {e}")
                time.sleep(1)
        raise RuntimeError("Failed to reset CARLA environment after maximum retries")

    def _wait_for_lidar_data(self):
        """Wait for LiDAR data to be available."""
        timeout = time.time() + 5
        while self.lidar_data_queue.empty():
            if time.time() > timeout:
                logger.warning("Timeout waiting for LiDAR data")
                return
            time.sleep(0.01)

    def _get_state(self) -> np.ndarray:
        """Get current state from LiDAR data."""
        if self.lidar_data_queue.empty():
            self._wait_for_lidar_data()
        
        if not self.lidar_data_queue.empty():
            lidar_data = self.lidar_data_queue.get()
            points = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape([-1, 4])[:, :3]
        else:
            points = np.zeros((0, 3), dtype=np.float32)
        
        if len(points) == 0:
            return np.zeros(16, dtype=np.float32)
        
        angles = np.arctan2(points[:, 1], points[:, 0]) * 180 / np.pi
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        
        sector_boundaries = np.linspace(-180, 180, 9)
        state = []
        for i in range(8):
            mask = (angles >= sector_boundaries[i]) & (angles < sector_boundaries[i+1])
            sector_distances = distances[mask]
            min_dist = 50.0 if len(sector_distances) == 0 else np.min(sector_distances)
            max_dist = 50.0 if len(sector_distances) == 0 else np.max(sector_distances)
            state.extend([min_dist, max_dist])
        return np.array(state, dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        # Apply action with correct throttle/brake mapping
        throttle_action, steering = action
        control = carla.VehicleControl()
        if throttle_action > 0:
            control.throttle = float(throttle_action)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = float(-throttle_action)
        control.steer = float(steering)
        self.vehicle.apply_control(control)
        self.world.tick()
        self.steps_in_episode += 1

        # Update state and vehicle info
        state = self._get_state()
        current_position = self.vehicle.get_location()
        distance_traveled = current_position.distance(self.last_position)
        self.total_distance += distance_traveled
        self.last_position = current_position

        # Update waypoints and compute extra features if possible
        if self.current_waypoint:
            self.current_waypoint = self.map.get_waypoint(current_position, lane_type=carla.LaneType.Driving)
            next_waypoints = self.current_waypoint.next(10.0)
            self.next_waypoint = next_waypoints[0] if next_waypoints else self.current_waypoint

        extra_features = np.zeros(2, dtype=np.float32)
        if self.next_waypoint:
            next_wp_loc = self.next_waypoint.transform.location
            dist_to_wp = current_position.distance(next_wp_loc)
            # Compute vehicle heading (assuming vehicle.get_transform().rotation.yaw gives heading in degrees)
            vehicle_yaw = self.vehicle.get_transform().rotation.yaw
            # Compute desired heading based on the vector from current_position to next waypoint
            dx = next_wp_loc.x - current_position.x
            dy = next_wp_loc.y - current_position.y
            desired_yaw = np.degrees(np.arctan2(dy, dx))
            # Compute a signed angular difference (normalized to [-180, 180])
            angle_diff = (desired_yaw - vehicle_yaw + 180) % 360 - 180

            extra_features = np.array([dist_to_wp, angle_diff], dtype=np.float32)

        # Augment the LiDAR state with the extra features
        state = np.concatenate([state, extra_features])

        # Calculate reward (existing logic + bonus for aligning with waypoint)
        reward = distance_traveled - 50.0 * int(self.collision_flag) - 1.0 * int(self.lane_invasion_flag)
        done = self.collision_flag
        truncated = self.steps_in_episode >= 2000

        # Update navigation to next waypoint in route
        if self.route and len(self.route) > 0:
            if self.current_route_index < len(self.route) and current_position.distance(
                    self.route[self.current_route_index].transform.location) < 5.0:
                self.current_route_index = min(self.current_route_index + 1, len(self.route) - 1)
            if self.current_route_index < len(self.route):
                next_wp_loc = self.route[self.current_route_index].transform.location
                dist_to_waypoint = current_position.distance(next_wp_loc)
                # Add waypoint following reward component
                reward += max(0, 5.0 - (dist_to_waypoint / 10.0))
                # Bonus: reward aligning vehicle heading
                heading_bonus = max(0, 1.0 - abs(angle_diff) / 180)
                reward += heading_bonus

        # Speed reward - encourage driving at target speed and penalize excessive steering when nearly stationary
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h
        speed_reward = -0.1 * abs(speed - self.target_speed)
        if speed < 5.0:
            speed_reward -= 1.0
            if abs(action[1]) > 0.1:
                speed_reward -= 0.5 * abs(action[1])
        reward += speed_reward

        # Track last positions to detect if vehicle is stuck
        self.last_positions.append((current_position.x, current_position.y))
        if len(self.last_positions) > self.position_history_limit:
            self.last_positions.pop(0)
        if len(self.last_positions) >= 20:
            oldest_pos = self.last_positions[0]
            current_pos = self.last_positions[-1]
            if ((current_pos[0] - oldest_pos[0])**2 + (current_pos[1] - oldest_pos[1])**2) < 1.0:
                reward -= 2.0
        
        info = {
            "distance_traveled": float(self.total_distance),
            "steps": self.steps_in_episode,
            "lane_invasions": self.cumulative_lane_invasions
        }
        if self.lane_invasion_flag:
            self.cumulative_lane_invasions += 1
        self.collision_flag = False
        self.lane_invasion_flag = False

        return state, reward, done, truncated, info

    def close(self):
        """Clean up environment resources."""
        self._set_synchronous_mode(False)
        if self.vehicle:
            self.vehicle.destroy()
        if self.lidar_sensor:
            self.lidar_sensor.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
        if self.lane_sensor:
            self.lane_sensor.destroy()

# === Custom Dataset for Lightning ===
class TrajectoryDataset(Dataset):
    """Dummy dataset for reinforcement learning trajectory collection."""
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Return dummy data, actual data comes from environment interaction
        return torch.zeros(1)

# === PPO Agent Definition ===

class PPOAgent(LightningModule):
    """PPO agent implemented with PyTorch Lightning."""
    def __init__(self, env: gym.Env, learning_rate=3e-4, gamma=0.99, clip_param=0.2, max_steps=1000,
                 entropy_coef=0.01, value_loss_coef=0.5, gae_lambda=0.95, num_episodes_per_epoch=5):
        super().__init__()
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_param = clip_param
        self.max_steps = max_steps
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.gae_lambda = gae_lambda
        self.num_episodes_per_epoch = num_episodes_per_epoch
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        # Actor network with increased variance for exploration
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim * 2)
        )
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0)
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["env"])
        self.min_std = 0.5  # Higher minimum standard deviation for exploration

    def forward(self, state: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        actor_output = self.actor(state)
        mean, log_std = actor_output.chunk(2, dim=-1)
        # Bias throttle toward moving ahead using sigmoid (range: 0 to 1)
        throttle_mean = torch.sigmoid(mean[:, 0:1])
        # Keep steering in the range [-1, 1]
        steering_mean = torch.tanh(mean[:, 1:2])
        combined_mean = torch.cat([throttle_mean, steering_mean], dim=-1)
        
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        std = torch.max(std, torch.tensor(self.min_std, device=std.device))
        dist = Normal(combined_mean, std)
        value = self.critic(state)
        return dist, value

    def training_step(self, batch, batch_idx):
        """Perform one training episode and PPO update."""
        # Ignore batch as we generate data from environment
        # Collect trajectory
        states, actions, rewards, next_states, dones = [], [], [], [], []
        state, _ = self.env.reset()
        episode_reward = 0
        for step in range(self.max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                dist, _ = self(state_tensor)
                action = dist.sample().cpu().numpy()[0]
            action = np.clip(action, -1.0, 1.0)
            next_state, reward, done, truncated, info = self.env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done or truncated)
            state = next_state
            episode_reward += reward
            if done or truncated:
                break
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        # Compute advantages using GAE
        with torch.no_grad():
            values = self.critic(states).squeeze(-1)
            next_values = self.critic(next_states).squeeze(-1)
            deltas = rewards + self.gamma * next_values * (1 - dones) - values
            advantages = torch.zeros_like(deltas)
            gae = 0
            for t in reversed(range(len(deltas))):
                gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        optimizer = self.optimizers()
        for _ in range(10):
            dist, current_values = self(states)
            current_values = current_values.squeeze(-1)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().mean()
            with torch.no_grad():
                old_dist, _ = self(states)
                old_log_probs = old_dist.log_prob(actions).sum(dim=-1)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.value_loss_coef * (returns - current_values).pow(2).mean()
            loss = policy_loss + value_loss - self.entropy_coef * entropy
            optimizer.zero_grad()
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            optimizer.step()
        
        self.log('episode_reward', episode_reward, on_step=True)
        logger.info(f"Episode reward: {episode_reward}")
        return {'loss': loss}

    def configure_optimizers(self):
        return self.optimizer
    
    def train_dataloader(self):
        """Required by Lightning, creates a dataloader for training"""
        dataset = TrajectoryDataset(self.num_episodes_per_epoch)
        return DataLoader(dataset, batch_size=1)

# === Assessment and Logging Functions ===

def assess_training(agent: PPOAgent, env: CarlaEnvWrapper, episode: int):
    """Assess training progress and log metrics."""
    state, _ = env.reset()
    actions = []
    rewards = []
    for _ in range(500):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            dist, _ = agent(state_tensor)
            action = dist.sample().cpu().numpy()[0]
        actions.append(action)
        next_state, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        state = next_state
        if done or truncated:
            break
    
    actions = np.array(actions)
    action_mean = actions.mean(axis=0)
    action_std = actions.std(axis=0)
    total_reward = sum(rewards)
    
    logger.info(f"Episode {episode} Assessment:")
    logger.info(f"  Total Reward: {total_reward}")
    logger.info(f"  Action Mean: {action_mean}")
    logger.info(f"  Action Std: {action_std}")
    
    with open(os.path.join("/teamspace/studios/this_studio/logs", f"assessment_episode_{episode}.json"), 'w') as f:
        json.dump({"total_reward": total_reward, "action_mean": action_mean.tolist(), "action_std": action_std.tolist()}, f)

# === Main Execution ===

def main():
    """Main function to run training."""
    env = CarlaEnvWrapper()
    agent = PPOAgent(env=env)
    
    # Set up logging and checkpointing
    log_dir = "/teamspace/studios/this_studio/logs"
    os.makedirs(log_dir, exist_ok=True)
    from pytorch_lightning.loggers import CSVLogger
    csv_logger = CSVLogger(log_dir, name="ppo_training")
    


    trainer = pl.Trainer(
        max_epochs=100,
        logger=csv_logger,
        log_every_n_steps=1,
        callbacks = [pl.callbacks.ModelCheckpoint(dirpath=log_dir, save_top_k=1, monitor='episode_reward', save_last=True)],
        default_root_dir=log_dir,
        accelerator='auto',
        enable_checkpointing=True,
    )
    
    try:
        # Pass checkpoint_path to the fit method instead of trainer init
        trainer.fit(agent, ckpt_path='last')
        for epoch in range(100):
            assess_training(agent, env, epoch)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        #save the state of the agent
        torch.save(agent.state_dict(), os.path.join(log_dir, "ppo_agent_state.pth"))
    finally:
        env.close()

if __name__ == "__main__":
    main()