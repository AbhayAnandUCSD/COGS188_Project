import carla
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import math
import random

# --- LiDAR Processing Utilities ---

def process_lidar_data(points, num_sectors=8, max_range=50.0):
    sector_size = 360.0 / num_sectors
    min_distances = [max_range] * num_sectors
    max_distances = [0.0] * num_sectors

    print(f"Type of points: {type(points)}")  # Debug print
    print("Has raw_data:", hasattr(points, 'raw_data'))

    if isinstance(points, carla.LidarMeasurement) and hasattr(points, 'raw_data'):
        try:
            data = np.frombuffer(points.raw_data, dtype=np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32), ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))
            for point in data:
                x, y = point['x'], point['y']
                distance = math.sqrt(x**2 + y**2)
                angle = (math.atan2(y, x) * 180 / math.pi + 360) % 360
                sector = int(angle // sector_size)
                min_distances[sector] = min(min_distances[sector], distance)
                max_distances[sector] = max(max_distances[sector], distance)
        except ValueError as e:
            if "buffer size must be a multiple of element size" in str(e):
                print("Buffer size error. Truncating buffer to fit.")
                element_size = 24  # Size of one point in bytes
                buffer_size = len(points.raw_data)
                num_complete_points = buffer_size // element_size
                truncated_buffer = points.raw_data[:num_complete_points * element_size]
                data = np.frombuffer(truncated_buffer, dtype=np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32), ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))
                for point in data:
                    x, y = point['x'], point['y']
                    distance = math.sqrt(x**2 + y**2)
                    angle = (math.atan2(y, x) * 180 / math.pi + 360) % 360
                    sector = int(angle // sector_size)
                    min_distances[sector] = min(min_distances[sector], distance)
                    max_distances[sector] = max(max_distances[sector], distance)
            else:
                raise e
    else:
        print("points is not a LidarMeasurement object or missing raw_data:", type(points))
        return [1.0] * 16

    state = [d / max_range for d in min_distances + max_distances]
    return state

# --- CARLA Environment with LiDAR ---

class CarlaEnv:
    def __init__(self, use_continuous_actions=True, max_steps_per_episode=1000):
        """Initialize the CARLA environment with a vehicle and LiDAR sensor."""
        # Connect to CARLA server
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.actor_list = []
        self.use_continuous_actions = use_continuous_actions
        self.max_steps_per_episode = max_steps_per_episode

        # For continuous actions, we'll use a different action space
        # Action space will be defined when reset is called

        # For discrete actions (backward compatibility)
        self.discrete_actions = [
            (0.5, -0.5),  # Throttle + Left
            (0.5, 0.0),   # Throttle + Straight
            (0.5, 0.5),   # Throttle + Right
            (0.0, -0.5),  # No throttle + Left
            (0.0, 0.0),   # No throttle + Straight
            (0.0, 0.5),   # No throttle + Right
        ]
        
        # Initialize vehicle, sensors, etc. in reset()
        self.vehicle = None
        self.lidar = None
        self.collision_sensor = None
        self.lidar_data = None
        self.collision_history = []
        self.debug = True  # Debug flag for LiDAR data
        
    def destroy_actors(self):
        for actor in self.actor_list:
            actor.destroy()
        self.actor_list.clear()
        
    def reset(self):
        self.collision_history = []
        self.step_count = 0
        if self.actor_list:  # Only destroy if actors exist
            self.destroy_actors()

        spawn_points = self.world.get_map().get_spawn_points()
        max_retries = 100
        retry_count = 0
        
        vehicle_spawned = False
        while not vehicle_spawned and retry_count < max_retries:
            spawn_point = random.choice(spawn_points)
            try:
                # Spawn new vehicle
                vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
                self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                self.actor_list.append(self.vehicle)
                vehicle_spawned = True
                
                # Attach LiDAR sensor
                lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
                lidar_bp.set_attribute('channels', '32')
                lidar_bp.set_attribute('range', '50.0')
                lidar_bp.set_attribute('points_per_second', '100000')
                lidar_bp.set_attribute('rotation_frequency', '20')
                lidar_transform = carla.Transform(carla.Location(z=2.0))
                self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
                self.actor_list.append(self.lidar)
                self.lidar_data = None
                def process_lidar(points):
                    self.lidar_data = points
                self.lidar.listen(process_lidar)

                # Attach collision sensor
                collision_bp = self.blueprint_library.find('sensor.other.collision')
                self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
                self.actor_list.append(self.collision_sensor)
                self.collision_history = []
                def on_collision(event):
                    self.collision_history.append(event)
                self.collision_sensor.listen(on_collision)
            except RuntimeError as e:
                if "Spawn failed because of collision" in str(e):
                    retry_count += 1
                    if self.vehicle in self.actor_list:
                        self.actor_list.remove(self.vehicle)
                        self.vehicle.destroy()
                        self.vehicle = None
                    continue
                else:
                    raise e
        
        if not vehicle_spawned:
            raise ValueError(f"Failed to spawn vehicle after {max_retries} retries.")
            
        # Wait for initial LiDAR data
        wait_count = 0
        max_wait = 20  # Maximum number of simulation steps to wait
        while self.lidar_data is None and wait_count < max_wait:
            self.world.tick()
            wait_count += 1
            
        if self.lidar_data is None:
            print("Warning: LiDAR data not initialized. Using empty state.")
            # Create a default state vector (all max range)
            return [1.0] * 16  # All sectors at max range
            
        state = process_lidar_data(self.lidar_data)
        return state

    def step(self, action):
        """Execute an action and return the next state, reward, and done flag."""
        self.step_count += 1
        if self.use_continuous_actions:
            # For continuous actions, action is a numpy array or tensor with [throttle, steer]
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
                
            # Clip actions to valid range
            throttle = np.clip(action[0], 0.0, 1.0)
            steer = np.clip(action[1], -1.0, 1.0)
        else:
            # For discrete actions, use the predefined action set
            throttle, steer = self.discrete_actions[action]
            
        self.vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), steer=float(steer)))
        self.world.tick()  # Step the simulation

        # Process new LiDAR data
        wait_count = 0
        max_wait = 10
        while self.lidar_data is None and wait_count < max_wait:
            self.world.tick()
            wait_count += 1
            
        if self.lidar_data is None:
            # If still no data, use a default state
            next_state = [1.0] * 16  # All sectors at max range
        else:
            next_state = process_lidar_data(self.lidar_data)

        # Simplified reward: +1 for progress, -10 for collision
        reward = -10.0 if self.collision_history else 1.0
        done = bool(self.collision_history)
        
        # Debug LiDAR data structure
        if self.debug and self.lidar_data is not None:
            try:
                print(f"LiDAR data type: {type(self.lidar_data)}")
                if hasattr(self.lidar_data, 'raw_data'):
                    print("Has raw_data attribute")
                elif isinstance(self.lidar_data, np.ndarray):
                    print(f"NumPy array shape: {self.lidar_data.shape}")
                elif hasattr(self.lidar_data, '__iter__'):
                    sample_point = next(iter(self.lidar_data), None)
                    if sample_point:
                        print(f"Sample point type: {type(sample_point)}")
                        print(f"Sample point attributes: {dir(sample_point)}")
            except Exception as e:
                print(f"Error inspecting LiDAR data: {e}")
            self.debug = False  # Only print once
        done = bool(self.collision_history) or self.step_count >= self.max_steps_per_episode
        return next_state, reward, done, {}

    def close(self):
        """Clean up all actors."""
        for actor in self.actor_list:
            actor.destroy()

# --- Neural Network for PPO with Continuous Actions ---

class ActorCritic(nn.Module):
    def __init__(self, use_continuous_actions=True):
        """
        Actor-Critic network for PPO:
        - Input: 16-value state vector from LiDAR
        - Shared layer: 16 -> 64
        - If continuous: Actor outputs mean and log_std for throttle and steering
        - If discrete: Actor outputs probability distribution over 6 discrete actions
        - Critic head: 64 -> 1 (value)
        """
        super(ActorCritic, self).__init__()
        self.use_continuous_actions = use_continuous_actions
        self.shared = nn.Linear(16, 64)
        
        if use_continuous_actions:
            # For continuous actions, output mean and log_std for throttle and steering
            self.actor_mean = nn.Linear(64, 2)  # 2 for [throttle, steer]
            self.actor_log_std = nn.Parameter(torch.zeros(2))  # Learnable std dev
        else:
            # For discrete actions, output probabilities for 6 actions
            self.actor = nn.Linear(64, 6)
            
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.shared(x))
        value = self.critic(x)
        
        if self.use_continuous_actions:
            action_mean = self.actor_mean(x)
            # Apply activation functions to constrain outputs
            # Throttle: [0, 1], Steering: [-1, 1]
            action_mean[:, 0] = torch.sigmoid(action_mean[:, 0])  # Throttle [0, 1]
            action_mean[:, 1] = torch.tanh(action_mean[:, 1])     # Steering [-1, 1]
            
            action_std = torch.exp(self.actor_log_std)
            return action_mean, action_std, value
        else:
            action_probs = torch.softmax(self.actor(x), dim=-1)
            return action_probs, value

# --- PPO Agent with PyTorch Lightning (Continuous Actions) ---

class PPOAgent(pl.LightningModule):
    def __init__(self, env, use_continuous_actions=True, gamma=0.99, clip_epsilon=0.2, 
                 lr=3e-4, num_epochs=10, batch_size=32):
        """Initialize the PPO agent with hyperparameters."""
        super().__init__()
        self.env = env
        self.use_continuous_actions = use_continuous_actions
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.actor_critic = ActorCritic(use_continuous_actions=use_continuous_actions)
        self.buffer = []

    def compute_returns(self, rewards, done):
        """Compute discounted returns for an episode."""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G * (1 - done)
            returns.insert(0, G)
        return returns

    def collect_trajectories(self):
        for _ in range(10):
            state = self.env.reset()
            episode = []
            done = False
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    if self.use_continuous_actions:
                        action_mean, action_std, value = self.actor_critic(state_tensor)
                        normal_dist = torch.distributions.Normal(action_mean, action_std)
                        action = normal_dist.sample()
                        log_prob = normal_dist.log_prob(action).sum(dim=-1)
                        action = action.cpu().numpy()[0]
                    else:
                        action_probs, value = self.actor_critic(state_tensor)
                        action = torch.multinomial(action_probs, 1).item()
                        log_prob = torch.log(action_probs[0, action])
                next_state, reward, done, _ = self.env.step(action)
                episode.append((state, action, reward, value.item(), log_prob.item(), done))
                state = next_state
            self.buffer.append(episode)

    def process_buffer(self):
        """Flatten the buffer and compute returns/advantages."""
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        for episode in self.buffer:
            s, a, r, v, lp, d = zip(*episode)
            states.extend(s)
            actions.extend(a)
            rewards.extend(r)
            values.extend(v)
            log_probs.extend(lp)
            dones.extend(d)
        returns = self.compute_returns(rewards, dones[-1])
        advantages = [r - v for r, v in zip(returns, values)]  # Simplified advantage
        
        # Convert to tensors
        states_tensor = torch.tensor(states, dtype=torch.float32)
        
        if self.use_continuous_actions:
            # For continuous actions
            actions_tensor = torch.tensor(actions, dtype=torch.float32)
        else:
            # For discrete actions
            actions_tensor = torch.tensor(actions, dtype=torch.long)
            
        self.buffer = {
            'states': states_tensor,
            'actions': actions_tensor,
            'log_probs': torch.tensor(log_probs, dtype=torch.float32),
            'returns': torch.tensor(returns, dtype=torch.float32),
            'advantages': torch.tensor(advantages, dtype=torch.float32),
        }

    def training_step(self, batch, batch_idx):
        """Perform one training step on a batch of data."""
        states, actions, old_log_probs, returns, advantages = batch
        
        if self.use_continuous_actions:
            action_mean, action_std, values = self.actor_critic(states)
            dist = torch.distributions.Normal(action_mean, action_std)
            new_log_probs = dist.log_prob(actions).sum(dim=1)
        else:
            action_probs, values = self.actor_critic(states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
        
        # PPO clipped objective
        ratios = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(values.squeeze(), returns)
        loss = actor_loss + 0.5 * critic_loss

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def train_dataloader(self):
        """Prepare data for training."""
        self.collect_trajectories()
        self.process_buffer()
        dataset = torch.utils.data.TensorDataset(
            self.buffer['states'], self.buffer['actions'], self.buffer['log_probs'],
            self.buffer['returns'], self.buffer['advantages']
        )
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def configure_optimizers(self):
        """Set up the optimizer."""
        return torch.optim.Adam(self.actor_critic.parameters(), lr=self.lr)

# --- Sanity Check Function ---

def sanity_check():
    """Verify CARLA connection and environment functionality with better error handling."""
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        client.get_world()  # Test connection
        print("CARLA server connection successful.")
        
        # Create environment with more robust error handling
        env = CarlaEnv(use_continuous_actions=True)
        try:
            state = env.reset()  # This should handle spawn collisions with retries
            print(f"State shape: {len(state)} (expected: 16)")
            
            # Test continuous action
            action = np.array([0.5, 0.0])  # Throttle 0.5, steering 0
            next_state, reward, done, _ = env.step(action)
            print(f"Continuous action step successful - Reward: {reward}, Done: {done}")
            
            # Test discrete action for backward compatibility
            env.use_continuous_actions = False
            action = random.randint(0, 5)
            next_state, reward, done, _ = env.step(action)
            print(f"Discrete action step successful - Reward: {reward}, Done: {done}")
            
            env.close()
            return True
        except Exception as e:
            print(f"Environment test failed: {e}")
            env.close()
            return False
    except Exception as e:
        print(f"CARLA connection failed: {e}")
        return False

# --- Main Function ---

def main():
    """Orchestrate the training process."""
    if not sanity_check():
        print("Sanity check failed. Please ensure CARLA server is running.")
        return

    # Use continuous actions for deep learning models
    env = CarlaEnv(use_continuous_actions=True)
    agent = PPOAgent(env, use_continuous_actions=True)
    trainer = pl.Trainer(
        max_epochs=1,
        log_every_n_steps=1,
    )
    trainer.fit(agent)
    env.close()

if __name__ == "__main__":
    main()