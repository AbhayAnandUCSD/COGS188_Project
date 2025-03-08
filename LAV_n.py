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
# from donkeycar.parts.simulator import DonkeySimulator
import cv2
from typing import Tuple, Dict, Any, List, Optional
import time
import importlib
import sys
import os
import collections

# Fix for collections.MutableMapping in Python 3.10
if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Environment Wrappers ===

class CarlaEnvWrapper(gym.Env):
    """Environment wrapper for CARLA simulator."""
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
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)  # Throttle, Steering
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        self.max_retries = 3
        self.last_position = None
        
        # New tracking variables for improved rewards
        self.steps_in_episode = 0
        self.total_distance = 0.0
        self.current_waypoint = None
        self.next_waypoint = None
        self.target_speed = 30.0  # in km/h, can be adjusted based on road type
        self.previous_lane_deviation = 0.0
        self.cumulative_lane_invasions = 0
        self.invaded_opposite_lane = False
        self.map = self.world.get_map()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        for attempt in range(self.max_retries):
            try:
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
                self.lidar_data = None
                self.lidar_sensor.listen(lambda data: self._process_lidar(data))
                
                # Attach collision sensor
                collision_bp = self.blueprint_library.find('sensor.other.collision')
                self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
                self.collision_flag = False
                self.collision_sensor.listen(lambda event: setattr(self, 'collision_flag', True))
                
                # Attach lane invasion sensor
                lane_bp = self.blueprint_library.find('sensor.other.lane_invasion')
                self.lane_sensor = self.world.spawn_actor(lane_bp, carla.Transform(), attach_to=self.vehicle)
                self.lane_invasion_flag = False
                self.lane_invasion_counter = 0
                self.lane_sensor.listen(self._on_lane_invasion)
                
                # Reset tracking variables
                self.steps_in_episode = 0
                self.total_distance = 0.0
                self.cumulative_lane_invasions = 0
                self.invaded_opposite_lane = False
                
                # Initialize waypoints
                vehicle_location = self.vehicle.get_location()
                self.current_waypoint = self.map.get_waypoint(vehicle_location, lane_type=carla.LaneType.Driving)
                if self.current_waypoint:
                    self.next_waypoint = self.current_waypoint.next(10.0)[0] if len(self.current_waypoint.next(10.0)) > 0 else self.current_waypoint
                    # Adjust target speed based on road type
                    if self.current_waypoint.is_junction:
                        self.target_speed = 20.0  # Slower in junctions
                    else:
                        self.target_speed = 30.0  # Normal road speed
                
                self.last_position = self.vehicle.get_location()
                self.world.tick()
                time.sleep(0.1)  # Wait for sensor data
                state = self._get_state()
                logger.info("CARLA environment reset successfully")
                return state, {}
            except Exception as e:
                logger.error(f"Failed to reset CARLA environment (attempt {attempt+1}/{self.max_retries}): {e}")
                time.sleep(1)
        raise RuntimeError("Failed to reset CARLA environment after maximum retries")

    def _on_lane_invasion(self, event):
        """Enhanced lane invasion callback to track different types of lane invasions"""
        self.lane_invasion_flag = True
        self.lane_invasion_counter += 1
        
        # Check for opposite lane invasion (more serious)
        for marking in event.crossed_lane_markings:
            if marking.type == carla.LaneMarkingType.Solid or marking.type == carla.LaneMarkingType.SolidSolid:
                self.invaded_opposite_lane = True
                break
    
    def _calculate_lane_deviation(self):
        """Calculate how far the vehicle is from the center of its lane"""
        if not self.current_waypoint:
            return 0.0
            
        vehicle_location = self.vehicle.get_location()
        vehicle_waypoint = self.map.get_waypoint(vehicle_location)
        
        if not vehicle_waypoint:
            return 0.0
            
        # Calculate distance from lane center
        lane_center = vehicle_waypoint.transform.location
        deviation = vehicle_location.distance(lane_center)
        
        # Normalize by lane width
        lane_width = vehicle_waypoint.lane_width
        normalized_deviation = deviation / (lane_width / 2.0)
        
        return normalized_deviation

    def _calculate_heading_error(self):
        """Calculate error between vehicle heading and lane direction"""
        if not self.current_waypoint:
            return 0.0
            
        vehicle_transform = self.vehicle.get_transform()
        lane_direction = self.current_waypoint.transform.rotation.get_forward_vector()
        vehicle_direction = vehicle_transform.get_forward_vector()
        
        # Dot product gives cosine of angle between vectors
        dot_product = lane_direction.x * vehicle_direction.x + lane_direction.y * vehicle_direction.y
        # Clamp to avoid floating point errors
        dot_product = max(-1.0, min(1.0, dot_product))
        angle_diff = np.arccos(dot_product)
        
        return angle_diff

    def _process_lidar(self, data):
        self.lidar_data = data

    def _get_state(self) -> np.ndarray:
        if self.lidar_data is None:
            return np.zeros(16, dtype=np.float32)
        
        # Extract points from lidar data
        points = np.frombuffer(self.lidar_data.raw_data, dtype=np.float32).reshape([-1, 4])[:, :3]
        
        # Handle empty lidar data case
        if len(points) == 0:
            return np.zeros(16, dtype=np.float32)
            
        angles = np.arctan2(points[:, 1], points[:, 0]) * 180 / np.pi
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        
        # Create 8 sectors (-180 to 180 degrees) for the 16 values (min and max per sector)
        # Explicitly create all boundaries to avoid indexing issues
        sector_boundaries = np.linspace(-180, 180, 9)  # 9 boundaries for 8 sectors
        state = []
        
        for i in range(8):  # 8 sectors
            # Use explicit indices to avoid out of bounds
            lower_bound = sector_boundaries[i]
            upper_bound = sector_boundaries[i + 1]
            
            # Filter points in this sector
            mask = (angles >= lower_bound) & (angles < upper_bound)
            sector_distances = distances[mask]
            
            # Default values if no points in sector
            min_dist = 50.0
            max_dist = 50.0
            
            if len(sector_distances) > 0:
                min_dist = np.min(sector_distances)
                max_dist = np.max(sector_distances)
                
            state.extend([min_dist, max_dist])
            
        return np.array(state, dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        throttle, steering = action
        self.vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), steer=float(steering)))
        self.world.tick()
        self.steps_in_episode += 1
        
        # Get updated state and vehicle info
        state = self._get_state()
        current_position = self.vehicle.get_location()
        current_velocity = self.vehicle.get_velocity()
        speed_kmh = 3.6 * np.sqrt(current_velocity.x**2 + current_velocity.y**2 + current_velocity.z**2)  # m/s to km/h
        
        # Calculate step distance
        distance_traveled = current_position.distance(self.last_position)
        self.total_distance += distance_traveled
        
        # Update waypoints
        self.current_waypoint = self.map.get_waypoint(current_position, lane_type=carla.LaneType.Driving)
        if self.current_waypoint:
            next_waypoints = self.current_waypoint.next(10.0)
            if len(next_waypoints) > 0:
                self.next_waypoint = next_waypoints[0]
                # Update target speed based on road conditions
                if self.current_waypoint.is_junction:
                    self.target_speed = 20.0
                else:
                    self.target_speed = 30.0
        
        # Calculate rewards - REBALANCED for better learning
        
        # 1. Base distance reward (increased to emphasize progress)
        progress_reward = 0.5 * distance_traveled
        
        # 2. Collision penalty (increased significantly)
        collision_penalty = -50.0 * int(self.collision_flag)
        
        # 3. Lane invasion penalty (increased)
        lane_invasion_penalty = -1.0 * int(self.lane_invasion_flag)  # Regular lane crossing
        if self.invaded_opposite_lane:
            lane_invasion_penalty -= 5.0  # Stronger penalty for crossing solid lines
        
        # 4. Lane centering reward
        lane_deviation = self._calculate_lane_deviation()
        lane_centering_reward = 0.2 * (1.0 - min(1.0, lane_deviation))  # Higher when centered
        
        # 5. Heading alignment reward
        heading_error = self._calculate_heading_error()
        heading_reward = 0.2 * (1.0 - min(1.0, heading_error / np.pi))  # Higher when aligned with lane
        
        # 6. Speed management reward
        speed_diff = abs(speed_kmh - self.target_speed)
        speed_reward = 0.1 * (1.0 - min(1.0, speed_diff / self.target_speed))  # Higher when at target speed
        
        # 7. Longevity reward (increased to encourage survival)
        longevity_reward = 0.05 * self.steps_in_episode
        
        # 8. Reduced steering penalty (was too dominant before)
        steering_smoothness_reward = -0.01 * abs(steering)
        
        # Combine rewards
        reward_components = {
            "progress": progress_reward,
            "collision": collision_penalty,
            "lane_invasion": lane_invasion_penalty,
            "lane_centering": lane_centering_reward,
            "heading": heading_reward,
            "speed": speed_reward,
            "longevity": longevity_reward, 
            "steering_smoothness": steering_smoothness_reward
        }
        
        total_reward = sum(reward_components.values())
        
        # Ensure reward is properly formatted for logging
        total_reward = float(total_reward)
        
        # Determine terminal conditions
        done = self.collision_flag
        truncated = False
        
        # Force episode end after too many steps to prevent never-ending episodes
        if self.steps_in_episode >= 2000:  # Set a reasonable maximum length
            truncated = True
            logger.info(f"Episode truncated after {self.steps_in_episode} steps")
        
        # Add extra info for debugging and analysis
        info = {
            "speed_kmh": float(speed_kmh),
            "distance_traveled": float(self.total_distance),
            "steps": int(self.steps_in_episode),
            "lane_invasions": int(self.cumulative_lane_invasions),
            "lane_deviation": float(lane_deviation),
            "reward_breakdown": reward_components
        }
        
        # Enhanced logging for episode end
        if (done or truncated) and self.steps_in_episode > 0:
            logger.info(f"Episode ended after {self.steps_in_episode} steps with total distance {self.total_distance:.2f}m")
            logger.info(f"Final speed: {speed_kmh:.2f} km/h, Target speed: {self.target_speed:.2f} km/h")
            logger.info(f"Final lane deviation: {lane_deviation:.4f}, Heading error: {heading_error:.4f}")
            logger.info(f"Reward breakdown: {reward_components}")
        
        # Log reward components periodically 
        if self.steps_in_episode % 50 == 0:
            logger.info(f"Step {self.steps_in_episode}: Speed {speed_kmh:.1f} km/h, Target {self.target_speed:.1f} km/h")
            logger.info(f"Lane deviation: {lane_deviation:.4f}, Heading error: {heading_error:.4f}")
            logger.info(f"Total distance so far: {self.total_distance:.2f}m")
        
        self.last_position = current_position
        self.collision_flag = False
        self.lane_invasion_flag = False
        self.invaded_opposite_lane = False
        
        return state, total_reward, done, truncated, info

    def close(self):
        if self.vehicle:
            self.vehicle.destroy()
        if self.lidar_sensor:
            self.lidar_sensor.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
        if self.lane_sensor:
            self.lane_sensor.destroy()

class DonkeyEnvWrapper(gym.Env):
    """Environment wrapper for DonkeySim simulator."""
    def __init__(self):
        super().__init__()
        self.sim = DonkeySimulator()
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)  # Throttle, Steering
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        self.max_retries = 3
        self.last_position = None
        self.image = None

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        for attempt in range(self.max_retries):
            try:
                self.sim.reset()
                self.last_position = self.sim.get_car_position()
                self.image = self.sim.get_camera_image()
                state = self._get_state()
                logger.info("DonkeySim environment reset successfully")
                return state, {}
            except Exception as e:
                logger.error(f"Failed to reset DonkeySim environment (attempt {attempt+1}/{self.max_retries}): {e}")
                time.sleep(1)
        raise RuntimeError("Failed to reset DonkeySim environment after maximum retries")

    def _get_state(self) -> np.ndarray:
        if self.image is None:
            return np.zeros(16, dtype=np.float32)
        # Process 128x128 image into 16-value state vector
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        sectors = np.linspace(0, 128, 9)[:-1]
        state = []
        for i in range(8):
            sector = gray[:, int(sectors[i]):int(sectors[i+1])]
            distances = []
            for col in range(sector.shape[1]):
                row = np.argmax(sector[:, col] > 50)  # Threshold for obstacle
                distances.append(128 - row if row > 0 else 50.0)
            state.extend([min(distances), max(distances)])
        return np.array(state, dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        throttle, steering = action
        self.sim.set_car_control(throttle, steering)
        self.image = self.sim.get_camera_image()
        state = self._get_state()
        
        current_position = self.sim.get_car_position()
        distance = np.sqrt((current_position[0] - self.last_position[0])**2 + 
                           (current_position[1] - self.last_position[1])**2)
        on_track = self.sim.is_on_track()
        reward = 0.1 * distance - 10.0 * (1 - int(on_track))
        done = not on_track
        truncated = False
        
        logger.debug(f"Action: {action}, Reward: {reward}, State: {state[:5]}")
        self.last_position = current_position
        return state, reward, done, truncated, {}

    def close(self):
        self.sim.shutdown()

# === PPO Agent with PyTorch Lightning ===

class PPOAgent(LightningModule):
    def __init__(self, env: gym.Env, learning_rate=3e-4, gamma=0.99, clip_param=0.2, max_steps=1000,
                 entropy_coef=0.01, value_loss_coef=0.5, gae_lambda=0.95):
        super().__init__()
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_param = clip_param
        self.max_steps = max_steps
        self.entropy_coef = entropy_coef  # New parameter for entropy bonus
        self.value_loss_coef = value_loss_coef  # Weight for value loss
        self.gae_lambda = gae_lambda  # Lambda parameter for GAE
        
        # Expanded Actor and Critic networks with more capacity
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        # Actor network with more layers and units
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * 2)  # Mean and log_std
        )
        
        # Separate critic network with more layers
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Improved optimizer with gradient clipping
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, eps=1e-5)
        self.automatic_optimization = False  # Manual optimization for PPO
        
        # Save hyperparameters for logging
        self.save_hyperparameters(ignore=["env"])
        
        # For creating a dummy dataloader to satisfy PyTorch Lightning's requirements
        self.dummy_data = torch.zeros((1, 1))

    def forward(self, state: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        actor_output = self.actor(state)
        mean, log_std = actor_output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        dist = Normal(mean, log_std.exp())
        value = self.critic(state)
        return dist, value

    def train_dataloader(self):
        """
        Return a dummy dataloader to satisfy PyTorch Lightning's requirements.
        For RL, we don't use traditional dataloaders as the environment provides the data.
        """
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(self.dummy_data)
        return DataLoader(dataset, batch_size=1)
    
    def training_step(self, batch, batch_idx):
        """
        Override the training step. We ignore the batch input since
        we get data directly from the environment.
        """
        # Ignore the batch from the dataloader - we get data from the environment
        states, actions, rewards, next_states, dones = [], [], [], [], []
        state, _ = self.env.reset()
        episode_reward = 0
        steps = 0

        # Collect trajectory data with more exploration at the beginning
        for _ in range(self.max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            dist, value = self(state_tensor)
            
            # Add noise to action for exploration
            action = dist.sample().cpu().numpy()[0]
            
            # More exploration early in training
            if self.global_step < 5000:  # Early in training
                action += np.random.normal(0, 0.2, size=action.shape)  # Add noise
                action = np.clip(action, -1.0, 1.0)  # Clip to valid range
            
            next_state, reward, done, truncated, _ = self.env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done or truncated)
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done or truncated or steps >= self.max_steps:
                break

        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        with torch.no_grad():
            values = self.critic(states).squeeze(-1)
            if steps < self.max_steps:
                _, last_value = self(next_states[-1].unsqueeze(0))
                last_value = last_value.squeeze(-1)
            else:
                last_value = torch.zeros(1, device=self.device)

            # GAE calculation
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = last_value
                else:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = values[t + 1]
                
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            
            returns = advantages + values

        # PPO update with improved learning
        optimizer = self.optimizers()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Multiple PPO epochs for better sample efficiency
        for _ in range(10):  # PPO epochs
            # Get current policy distribution and values
            dist, current_values = self(states)
            current_values = current_values.squeeze(-1)
            
            # Calculate log probabilities and entropy
            log_probs = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1).mean()
            
            with torch.no_grad():
                old_dist, _ = self(states)
                old_log_probs = old_dist.log_prob(actions).sum(-1)
                
            # PPO clipped objective
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss with clipping
            value_loss = 0.5 * (returns - current_values).pow(2).mean()
            
            # Combined loss with entropy bonus
            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            
            # Optimize
            optimizer.zero_grad()
            self.manual_backward(loss)
            
            # Gradient clipping to prevent extreme updates
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            optimizer.step()

        # Logging
        self.log('episode_reward', episode_reward, on_step=True, prog_bar=True)
        self.log('loss', loss.item(), on_step=True, prog_bar=True)
        self.log('policy_loss', policy_loss.item(), on_step=True)
        self.log('value_loss', value_loss.item(), on_step=True)
        self.log('entropy', entropy.item(), on_step=True)
        self.log('episode_length', steps, on_step=True)
        
        logger.info(f"Episode reward: {episode_reward}, Loss: {loss.item()}, Entropy: {entropy.item()}")
        
        # Gradually decrease exploration over time
        if self.entropy_coef > 0.001:
            self.entropy_coef *= 0.999  # Slowly decay entropy coefficient
            
        return loss

    def configure_optimizers(self):
        # Properly return the optimizer instead of None
        return [self.optimizer]

# === Sanity Check Functions ===

def sanity_check(use_carla: bool = True) -> bool:
    """
    Perform sanity checks on the environment, model, and dependencies before training.
    Returns True if all checks pass, False otherwise.
    
    Args:
        use_carla: Whether to use CARLA or DonkeySim environment
    """
    logger.info("🔍 Running sanity checks before training...")
    all_passed = True
    
    # Check dependencies
    dependency_check = _check_dependencies(["numpy", "torch", "gymnasium", "pytorch_lightning", 
                                           "carla" if use_carla else "donkeycar"])
    all_passed = all_passed and dependency_check
    
    # Initialize test environment
    env_check, test_env = _check_environment(use_carla)
    all_passed = all_passed and env_check
    
    if test_env is not None:
        # Check basic env interaction
        interaction_check = _check_environment_interaction(test_env)
        all_passed = all_passed and interaction_check
        
        # Check model
        model_check = _check_model(test_env)
        all_passed = all_passed and model_check
        
        # Close test environment
        test_env.close()
    
    # Summary
    if all_passed:
        logger.info("✅ All sanity checks passed! Ready to start training.")
    else:
        logger.warning("❌ Some sanity checks failed. Review logs above for details.")
    
    return all_passed

def _check_dependencies(packages: List[str]) -> bool:
    """Check if required packages are properly installed and accessible."""
    logger.info("Checking required dependencies...")
    all_installed = True
    
    for package in packages:
        try:
            importlib.import_module(package)
            logger.info(f"  ✓ {package} is installed")
        except ImportError:
            logger.error(f"  ✗ {package} is not installed or accessible")
            logger.error(f"    Try: pip install {package}")
            all_installed = False
    
    return all_installed

def _check_environment(use_carla: bool) -> Tuple[bool, Optional[gym.Env]]:
    """Check if environment can be initialized."""
    logger.info("Checking environment initialization...")
    try:
        if use_carla:
            env = CarlaEnvWrapper()
            logger.info("  ✓ CARLA environment created successfully")
        else:
            env = DonkeyEnvWrapper()
            logger.info("  ✓ DonkeySim environment created successfully")
        
        # Check observation and action spaces
        logger.info(f"  ✓ Observation space: {env.observation_space}")
        logger.info(f"  ✓ Action space: {env.action_space}")
        
        return True, env
    except Exception as e:
        logger.error(f"  ✗ Failed to create environment: {e}")
        logger.error(f"    Debugging tips:")
        
        if use_carla:
            logger.error("    - Is CARLA server running? Start it with: ./CarlaUE4.sh")
            logger.error("    - Check CARLA port (default: 2000) and host")
            logger.error("    - Verify CARLA installation with: python -c 'import carla; print(carla.__file__)'")
        else:
            logger.error("    - Is DonkeySim running?")
            logger.error("    - Check simulator port and settings")
        
        return False, None

def _check_environment_interaction(env: gym.Env) -> bool:
    """Test basic environment interaction."""
    logger.info("Testing environment interaction...")
    try:
        # Test reset with explicit exception handling
        try:
            state, _ = env.reset()
            logger.info(f"  ✓ Environment reset successful, state shape: {state.shape}")
        except Exception as e:
            logger.error(f"  ✗ Environment reset failed: {e}")
            logger.error(f"    Debugging tip: Check the _get_state implementation for array indexing issues")
            return False
        
        # Test random action
        try:
            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)
            logger.info(f"  ✓ Environment step successful with action {action}")
            logger.info(f"    - Reward: {reward}")
            logger.info(f"    - Done: {done}")
            logger.info(f"    - Next state shape: {next_state.shape}")
        except Exception as e:
            logger.error(f"  ✗ Environment step failed: {e}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"  ✗ Environment interaction failed: {e}")
        return False

def _check_model(env: gym.Env) -> bool:
    """Test model creation and basic operations."""
    logger.info("Testing model initialization and forward pass...")
    try:
        # Create a test agent
        agent = PPOAgent(env=env)
        logger.info("  ✓ PPOAgent created successfully")
        
        # Test forward pass with dummy state
        # Use a dummy state if reset fails
        try:
            state = env.reset()[0]
        except Exception:
            logger.info("    Using dummy state for model testing since environment reset failed")
            state = np.zeros(env.observation_space.shape, dtype=np.float32)
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        dist, value = agent(state_tensor)
        action = dist.sample().numpy()[0]
        
        logger.info(f"  ✓ Model forward pass successful")
        logger.info(f"    - Output action: {action}")
        logger.info(f"    - Value estimate: {value.item()}")
        
        # Mini rollout test - skip if env.reset() fails
        try:
            logger.info("Testing mini policy rollout (3 steps)...")
            total_reward = 0
            state, _ = env.reset()
            for i in range(3):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                dist, _ = agent(state_tensor)
                action = dist.sample().numpy()[0]
                next_state, reward, done, truncated, _ = env.step(action)
                logger.info(f"    Step {i+1}: action={action}, reward={reward}")
                state = next_state
                total_reward += reward
                if done or truncated:
                    break
            logger.info(f"  ✓ Mini rollout complete, total reward: {total_reward}")
        except Exception as e:
            logger.warning(f"    Mini rollout skipped due to environment issues: {e}")
            logger.warning("    This is not fatal for the model check - the model itself is valid")
        
        return True
    except Exception as e:
        logger.error(f"  ✗ Model check failed: {e}")
        logger.error(f"    Error details: {str(e)}")
        return False

# === Main Execution ===

def main():
    # Choose environment type
    use_carla = True  # Set to False to use DonkeySim
    
    # Run sanity checks
    checks_passed = sanity_check(use_carla)
    if not checks_passed:
        logger.warning("Sanity checks failed. Fix issues before proceeding with training.")
        return
    
    # Create the actual training environment
    env = CarlaEnvWrapper() if use_carla else DonkeyEnvWrapper()
    
    # Initialize PPO agent
    agent = PPOAgent(env=env)
    
    # Create logs directory
    log_dir = os.path.join("/teamspace/studios/this_studio", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure a CSV logger instead of TensorBoard to avoid compatibility issues
    from pytorch_lightning.loggers import CSVLogger
    csv_logger = CSVLogger(log_dir, name="ppo_training")
    
    # Trainer with more robust configuration
    trainer = pl.Trainer(
        max_epochs=1,  # Each "epoch" is already a full rollout, so set this to 1
        enable_progress_bar=True,
        logger=csv_logger,
        log_every_n_steps=1,
        enable_checkpointing=False,  # Disable checkpointing to avoid TensorBoard issues
        default_root_dir=log_dir,
        # These flags help with compatibility
        accelerator='auto',
        devices=1,
        max_steps=100  # Limit steps for safety
    )
    
    # Train
    try:
        trainer.fit(agent)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("You can try using run_training_safe.py instead, which doesn't use Lightning.")
    finally:
        env.close()

if __name__ == "__main__":
    main()