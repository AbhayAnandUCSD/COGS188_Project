import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
import logging
import time
import os
import sys
import json
import argparse
import glob
from collections import defaultdict
import matplotlib.pyplot as plt

# Add parent directory to path so we can import LAV_n
sys.path.append("/teamspace/studios/this_studio")
from LAV_n import CarlaEnvWrapper, PPOAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_latest_checkpoint(log_dir):
    """Find the latest checkpoint file and training state."""
    # Find the latest epoch checkpoint
    checkpoint_files = glob.glob(os.path.join(log_dir, "agent_epoch_*.pt"))
    best_model_path = os.path.join(log_dir, "best_model.pt")
    
    latest_checkpoint = None
    latest_epoch = -1
    
    # Check regular epoch checkpoints
    for checkpoint in checkpoint_files:
        try:
            # Extract epoch number from filename
            epoch_str = checkpoint.split("_")[-1].split(".")[0]
            epoch = int(epoch_str)
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_checkpoint = checkpoint
        except ValueError:
            continue
    
    # Check if we have a training state file
    training_state_path = os.path.join(log_dir, "training_state.json")
    training_state = None
    if os.path.exists(training_state_path):
        try:
            with open(training_state_path, 'r') as f:
                training_state = json.load(f)
            logger.info(f"Found training state from epoch {training_state.get('epoch', 'unknown')}")
        except Exception as e:
            logger.warning(f"Failed to load training state: {e}")
    
    # If no regular checkpoint found but best model exists, use that
    if latest_checkpoint is None and os.path.exists(best_model_path):
        latest_checkpoint = best_model_path
        logger.info("No epoch checkpoint found, but best_model.pt exists.")
    
    return latest_checkpoint, latest_epoch, training_state

def save_training_state(log_dir, epoch, reward_history, distance_history, episode_lengths, 
                        reward_components_history, best_episode_reward, best_episode_length):
    """Save training state to disk for future resuming."""
    # Convert defaultdict to regular dict for JSON serialization
    reward_components_dict = {k: list(v) for k, v in reward_components_history.items()}
    
    state = {
        "epoch": epoch,
        "reward_history": reward_history,
        "distance_history": distance_history,
        "episode_lengths": episode_lengths,
        "reward_components_history": reward_components_dict,
        "best_episode_reward": best_episode_reward,
        "best_episode_length": best_episode_length,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    state_path = os.path.join(log_dir, "training_state.json")
    with open(state_path, 'w') as f:
        json.dump(state, f)
    
    logger.info(f"Training state saved to {state_path}")

def manual_training_loop(env, agent, num_epochs=100, steps_per_epoch=1000, resume=False, start_epoch=0):
    """
    A simple manual training loop that doesn't rely on PyTorch Lightning.
    This is more robust to environment issues and compatibility problems.
    """
    logger.info("Starting manual training loop")
    
    # Create logs directory
    log_dir = os.path.join("/teamspace/studios/this_studio", "manual_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up a log file for detailed reward tracking
    reward_log_file = os.path.join(log_dir, "reward_logs.jsonl")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    agent = agent.to(device)
    
    # Tracking metrics
    best_episode_reward = -float('inf')
    best_episode_length = 0
    reward_history = []
    distance_history = []
    episode_lengths = []
    reward_components_history = defaultdict(list)
    steps_without_collision = 0
    
    # Load previous training state if resuming
    if resume:
        _, _, training_state = find_latest_checkpoint(log_dir)
        if training_state:
            logger.info("Resuming from previous training state")
            
            # Load training metrics
            reward_history = training_state.get("reward_history", [])
            distance_history = training_state.get("distance_history", [])
            episode_lengths = training_state.get("episode_lengths", [])
            
            # Load reward components history
            reward_components_dict = training_state.get("reward_components_history", {})
            for k, v in reward_components_dict.items():
                reward_components_history[k] = v
            
            # Load best metrics
            best_episode_reward = training_state.get("best_episode_reward", -float('inf'))
            best_episode_length = training_state.get("best_episode_length", 0)
            
            logger.info(f"Loaded {len(reward_history)} previous episodes")
            logger.info(f"Best episode reward so far: {best_episode_reward}")
        else:
            logger.warning("No previous training state found, starting fresh despite resume flag")
    else:
        # If it's a new run, archive any existing reward log to prevent appending to old logs
        if os.path.exists(reward_log_file):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            os.rename(reward_log_file, f"{reward_log_file}.{timestamp}.backup")
            logger.info(f"Archived previous reward log to {reward_log_file}.{timestamp}.backup")
    
    # Learning rate scheduler to improve convergence
    initial_lr = agent.learning_rate
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        agent.optimizer, 
        T_max=num_epochs * steps_per_epoch, 
        eta_min=initial_lr * 0.1
    )
    
    # Improved exploration strategy
    entropy_coef = 0.1  # Increased from 0.05 for more exploration
    min_entropy_coef = 0.01  # Increased from 0.001 to maintain exploration
    entropy_decay = 0.999  # Slower decay to maintain exploration longer
    
    # Increase PPO iterations for short episodes
    ppo_epochs = 20  # Increased from 10 to learn more from limited data
    
    # Use a smaller batch size for more updates
    batch_size = 32  # Decreased from 64 for more frequent updates
    
    # Set up curriculum learning phases
    # Phase 1: Collision avoidance (first 30% of training)
    # Phase 2: Basic driving skills (next 30% of training)
    # Phase 3: Advanced driving skills (final 40% of training)
    phase_thresholds = {
        "collision_avoidance": int(0.3 * num_epochs),
        "basic_driving": int(0.6 * num_epochs)
    }
    
    # Safety metrics for adaptive curriculum
    collision_rate = 1.0  # Start with assumption of 100% collision rate
    avg_distance = 0.0
    collision_free_episodes = 0
    total_episodes = 0
    
    # Determine initial training phase based on resuming or metrics
    if resume and len(reward_history) > 0:
        # If resuming, analyze past performance to determine phase
        recent_collisions = 0
        recent_episodes = min(20, len(reward_history))
        for i in range(recent_episodes):
            # Assuming negative rewards generally mean collision
            if reward_history[-i-1] < -50:
                recent_collisions += 1
        
        collision_rate = recent_collisions / recent_episodes
        
        if collision_rate < 0.5 and avg_distance > 10:
            training_phase = "basic_driving"
            if collision_rate < 0.2 and avg_distance > 30:
                training_phase = "advanced_driving"
        else:
            training_phase = "collision_avoidance"
    else:
        # Start with collision avoidance for new runs
        training_phase = "collision_avoidance"
    
    # Save initial training phase
    with open("/teamspace/studios/this_studio/training_phase.txt", "w") as f:
        f.write(training_phase)
    
    logger.info(f"Starting training in '{training_phase}' phase")
    
    # Modified exploration parameters for safety-first approach
    if training_phase == "collision_avoidance":
        # Very conservative exploration in first phase
        entropy_coef = 0.01  # Lower entropy for more conservative actions
    else:
        # More exploration in later phases
        entropy_coef = 0.05
    
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
        epoch_start_time = time.time()
        
        # Check if we should advance to the next training phase
        phase_changed = False
        
        # Deterministic progression based on epochs
        if epoch == phase_thresholds["collision_avoidance"] and training_phase == "collision_avoidance":
            training_phase = "basic_driving"
            phase_changed = True
        elif epoch == phase_thresholds["basic_driving"] and training_phase == "basic_driving":
            training_phase = "advanced_driving"
            phase_changed = True
            
        # Adaptive progression based on performance
        if total_episodes >= 20:  # Need enough data to make a decision
            if training_phase == "collision_avoidance" and collision_rate < 0.3 and avg_distance > 15:
                training_phase = "basic_driving"
                phase_changed = True
                logger.info(f"Advancing to basic_driving phase based on performance metrics")
            elif training_phase == "basic_driving" and collision_rate < 0.15 and avg_distance > 40:
                training_phase = "advanced_driving"
                phase_changed = True
                logger.info(f"Advancing to advanced_driving phase based on performance metrics")
                
        # Update phase file
        if phase_changed:
            with open("/teamspace/studios/this_studio/training_phase.txt", "w") as f:
                f.write(training_phase)
            logger.info(f"Training phase changed to {training_phase}")
            
            # Reset some parameters for the new phase
            if training_phase == "basic_driving":
                # Increase entropy for more exploration in basic driving
                entropy_coef = 0.03
            elif training_phase == "advanced_driving":
                # Full exploration in advanced driving
                entropy_coef = 0.05
                
        logger.info(f"Starting epoch {epoch+1}/{num_epochs} in {training_phase} phase")
        logger.info(f"Current metrics - Collision rate: {collision_rate:.2f}, Avg distance: {avg_distance:.1f}m")
        
        # Collect trajectory
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        episode_rewards = []
        current_episode_reward = 0
        current_episode_length = 0
        total_distance_traveled = 0
        reward_components = defaultdict(float)
        
        # Reset environment at the start of each epoch
        state, _ = env.reset()
        logger.info(f"Environment reset for epoch {epoch+1}")
        
        for step in range(steps_per_epoch):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                dist, value = agent(state_tensor)
                action = dist.sample().cpu().numpy()[0]
                
                # Apply stronger noise early in training or after detecting stagnation
                noise_scale = 0.4 if epoch < 10 or len(reward_history) > 5 and reward_history[-5:].count(reward_history[-1]) == 5 else 0.2
                action += np.random.normal(0, noise_scale, size=action.shape)
                
                # In early stages, bias toward gentle steering to prevent early crashes
                if epoch < 5:
                    action[1] *= 0.5  # Reduce steering intensity by half
                    
                action = np.clip(action, -1.0, 1.0)  # Clip action to valid range
            
            next_state, reward, done, truncated, info = env.step(action)
            
            # Log detailed step information every 20 steps
            if step % 20 == 0 or done or truncated:
                logger.info(f"Step {step}: action={action}, reward={reward:.4f}")
                if 'reward_breakdown' in info:
                    components = ', '.join([f"{k}={v:.3f}" for k, v in info['reward_breakdown'].items()])
                    logger.info(f"Reward components: {components}")
                if 'speed_kmh' in info:
                    logger.info(f"Speed: {info['speed_kmh']:.1f} km/h")
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done or truncated)
            
            state = next_state
            current_episode_reward += reward
            current_episode_length += 1
            steps_without_collision += 1
            
            # Track reward components
            if 'reward_breakdown' in info:
                for component, value in info['reward_breakdown'].items():
                    reward_components[component] += value
            
            # Track additional metrics from info
            if 'distance_traveled' in info:
                total_distance_traveled = info['distance_traveled']
            
            if done or truncated:
                # Log episode completion details
                logger.info(f"Episode ended at step {current_episode_length} with reward {current_episode_reward:.4f}")
                logger.info(f"Distance traveled: {total_distance_traveled:.2f}m")
                if 'reward_breakdown' in info:
                    for component, value in reward_components.items():
                        logger.info(f"Total {component} reward: {value:.4f}")
                
                # Record episode stats
                episode_rewards.append(current_episode_reward)
                reward_history.append(current_episode_reward)
                distance_history.append(total_distance_traveled)
                episode_lengths.append(current_episode_length)
                
                # Track reward components for visualization
                for component, value in reward_components.items():
                    reward_components_history[component].append(value)
                
                # Save detailed episode log
                episode_data = {
                    "epoch": epoch + 1,
                    "episode_reward": current_episode_reward,
                    "episode_length": current_episode_length,
                    "distance_traveled": total_distance_traveled,
                    "reward_components": dict(reward_components)
                }
                with open(reward_log_file, 'a') as f:
                    f.write(json.dumps(episode_data) + '\n')
                
                # Update best episode stats
                if current_episode_reward > best_episode_reward:
                    best_episode_reward = current_episode_reward
                    best_episode_length = current_episode_length
                    # Save best model
                    torch.save(agent.state_dict(), os.path.join(log_dir, "best_model.pt"))
                    logger.info(f"New best episode! Reward: {best_episode_reward:.4f}, Length: {best_episode_length}")
                
                # Reset episode tracking variables
                current_episode_reward = 0
                current_episode_length = 0
                reward_components = defaultdict(float)
                
                # Reset environment for the next episode
                state, _ = env.reset()
                logger.info("Environment reset for next episode")
        
        # Convert to tensors more efficiently
        # Fix the warning by converting lists to numpy arrays first
        if len(states) > 0:  # Make sure we have collected some data
            states_np = np.array(states)
            actions_np = np.array(actions)
            rewards_np = np.array(rewards)
            next_states_np = np.array(next_states)
            dones_np = np.array(dones)
            
            states = torch.FloatTensor(states_np).to(device)
            actions = torch.FloatTensor(actions_np).to(device)
            rewards = torch.FloatTensor(rewards_np).to(device)
            next_states = torch.FloatTensor(next_states_np).to(device)
            dones = torch.FloatTensor(dones_np).to(device)
            
            # Compute advantages using GAE (Generalized Advantage Estimation)
            with torch.no_grad():
                values = agent.critic(states).squeeze(-1)
                # Fix: Use len(rewards) instead of undefined 'steps' variable
                if len(rewards) < agent.max_steps:
                    _, last_value = agent(next_states[-1].unsqueeze(0))
                    last_value = last_value.squeeze(-1)
                else:
                    last_value = torch.zeros(1, device=device)

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
                    
                    delta = rewards[t] + agent.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + agent.gamma * 0.95 * nextnonterminal * lastgaelam
                
                returns = advantages + values

            # Normalize advantages for more stable training
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Mini-batch training for better sample efficiency
            indices = np.arange(len(states))
            num_samples = len(states)
            
            # Optimize policy and value function with mini-batches
            policy_losses = []
            value_losses = []
            entropies = []
            
            for _ in range(ppo_epochs):  # Increased from 10 to 20
                # Shuffle data for each PPO epoch
                np.random.shuffle(indices)
                
                # Process mini-batches
                for start_idx in range(0, num_samples, batch_size):
                    end_idx = min(start_idx + batch_size, num_samples)
                    mb_indices = indices[start_idx:end_idx]
                    
                    mb_states = states[mb_indices]
                    mb_actions = actions[mb_indices]
                    mb_advantages = advantages[mb_indices]
                    mb_returns = returns[mb_indices]
                    
                    # Get current policy distribution and values
                    dist, current_values = agent(mb_states)
                    current_values = current_values.squeeze(-1)
                    
                    # Calculate log probabilities and entropy
                    log_probs = dist.log_prob(mb_actions).sum(-1)
                    entropy = dist.entropy().sum(-1).mean()
                    
                    with torch.no_grad():
                        old_dist, _ = agent(mb_states)
                        old_log_probs = old_dist.log_prob(mb_actions).sum(-1)
                    
                    # PPO clipped objective
                    ratio = torch.exp(log_probs - old_log_probs)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1.0 - agent.clip_param, 1.0 + agent.clip_param) * mb_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss with clipping for stability
                    value_loss = 0.5 * (mb_returns - current_values).pow(2).mean()
                    
                    # Combined loss with entropy bonus for exploration
                    loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
                    
                    # Optimize
                    agent.optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping to prevent extreme updates
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
                    agent.optimizer.step()
                    
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropies.append(entropy.item())
                    
                    # Update learning rate
                    lr_scheduler.step()
            
            # Focus more on recent collision states for faster learning
            collision_states = []
            collision_actions = []
            collision_rewards = []
            collision_returns = []
            collision_advantages = []
            
            # Identify states that led to collisions
            for i in range(len(dones)):
                if dones[i] and i > 0:  # Found a collision
                    # Include states leading up to the collision
                    start_idx = max(0, i - 10)
                    collision_states.append(states[start_idx:i+1])
                    collision_actions.append(actions[start_idx:i+1])
                    collision_rewards.append(rewards[start_idx:i+1])
                    collision_returns.append(returns[start_idx:i+1])
                    collision_advantages.append(advantages[start_idx:i+1])
            
            # Extra training on collision states
            if collision_states:
                collision_states = torch.cat(collision_states)
                collision_actions = torch.cat(collision_actions)
                collision_returns = torch.cat(collision_returns)
                collision_advantages = torch.cat(collision_advantages)
                
                # Perform multiple updates focused on collision avoidance
                for _ in range(5):  # Extra training iterations for collision cases
                    # ...standard PPO update but with collision data...
                    dist, values = agent(collision_states)
                    # ...update using collision states...
            
            # Decay entropy coefficient to reduce exploration over time
            entropy_coef = max(min_entropy_coef, entropy_coef * entropy_decay)
            
            # Increase batch size gradually for more stable later learning
            if epoch > 0 and epoch % 5 == 0 and batch_size < max_batch_size:
                batch_size = min(max_batch_size, batch_size * 2)
                logger.info(f"Increasing batch size to {batch_size}")
            
            # Enhanced logging with more detailed statistics
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            avg_policy_loss = np.mean(policy_losses)
            avg_value_loss = np.mean(value_losses)
            avg_entropy = np.mean(entropies)
            current_lr = lr_scheduler.get_last_lr()[0]
            
            epoch_time = time.time() - epoch_start_time
            
            logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
            logger.info(f"Episodes completed: {len(episode_rewards)}")
            logger.info(f"Average episode length: {np.mean(episode_lengths) if episode_lengths else 0:.1f} steps")
            logger.info(f"Average reward: {avg_reward:.4f}, Best reward: {best_episode_reward:.4f}")
            logger.info(f"Average losses - Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f}, Entropy: {avg_entropy:.4f}")
            logger.info(f"Learning rate: {current_lr:.6f}, Entropy coef: {entropy_coef:.6f}")
            logger.info(f"Steps without collision: {steps_without_collision}")
            
            # Save progress metrics
            if (epoch + 1) % 5 == 0 or epoch == 0:
                # Create visualization of training progress
                visualize_training_progress(log_dir, epoch+1, reward_history, distance_history, 
                                           episode_lengths, reward_components_history)
            
            # Save model every 10 epochs or on last epoch
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                torch.save(agent.state_dict(), os.path.join(log_dir, f"agent_epoch_{epoch+1}.pt"))
                logger.info(f"Model saved at epoch {epoch+1}")
                
                # Save training state for resuming later
                save_training_state(
                    log_dir, epoch+1, reward_history, distance_history, 
                    episode_lengths, reward_components_history, 
                    best_episode_reward, best_episode_length
                )
        else:
            logger.warning("No data collected in this epoch. Check if environment is functioning correctly.")

def visualize_training_progress(log_dir, epoch, reward_history, distance_history, 
                               episode_lengths, reward_components_history):
    """Create detailed visualizations of training progress."""
    plt.figure(figsize=(15, 10))
    
    # Plot episode rewards
    plt.subplot(2, 2, 1)
    plt.plot(reward_history)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot distances
    plt.subplot(2, 2, 2)
    plt.plot(distance_history)
    plt.title('Distance Traveled')
    plt.xlabel('Episode')
    plt.ylabel('Distance (m)')
    
    # Plot episode lengths
    plt.subplot(2, 2, 3)
    plt.plot(episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    # Plot reward components
    plt.subplot(2, 2, 4)
    for component, values in reward_components_history.items():
        if values:  # Only plot if we have data
            plt.plot(values, label=component)
    plt.title('Reward Components')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(loc='best', fontsize='small')
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"training_progress_epoch_{epoch}.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train a PPO agent in CARLA environment")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for")
    parser.add_argument("--steps", type=int, default=1000, help="Steps per epoch")
    parser.add_argument("-n", "--new-run", action="store_true", help="Start a new training run, ignore existing checkpoints")
    parser.add_argument("--checkpoint", type=str, help="Specific checkpoint file to load (overrides auto-detection")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--entropy", type=float, default=0.05, help="Initial entropy coefficient")
    parser.add_argument("--batch-size", type=int, default=64, help="Starting batch size")
    
    args = parser.parse_args()
    
    # Create logs directory
    log_dir = os.path.join("/teamspace/studios/this_studio", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Check for checkpoints to resume from if not starting a new run
    start_epoch = 0
    checkpoint_path = None
    
    if not args.new_run:
        if args.checkpoint:
            # Use the specified checkpoint
            if os.path.exists(args.checkpoint):
                checkpoint_path = args.checkpoint
                logger.info(f"Using specified checkpoint: {checkpoint_path}")
                
                # Try to extract epoch number from checkpoint filename if it follows the pattern
                try:
                    filename = os.path.basename(checkpoint_path)
                    if "epoch_" in filename:
                        epoch_str = filename.split("_")[-1].split(".")[0]
                        start_epoch = int(epoch_str)
                        logger.info(f"Starting from epoch {start_epoch}")
                except:
                    logger.info("Could not determine starting epoch from checkpoint name")
            else:
                logger.warning(f"Specified checkpoint {args.checkpoint} not found. Starting fresh.")
        else:
            # Auto-detect the latest checkpoint
            latest_checkpoint, latest_epoch, _ = find_latest_checkpoint(log_dir)
            
            if latest_checkpoint:
                checkpoint_path = latest_checkpoint
                if latest_epoch > 0:
                    start_epoch = latest_epoch
                    logger.info(f"Found checkpoint from epoch {latest_epoch}, resuming from epoch {start_epoch + 1}")
                else:
                    logger.info(f"Found checkpoint: {latest_checkpoint}, but could not determine epoch")
    
    # Create environment
    env = CarlaEnvWrapper()
    
    # Create agent with improved parameters
    agent = PPOAgent(
        env=env,
        learning_rate=args.lr,
        gamma=0.99,
        clip_param=0.2,
        max_steps=1000,
        entropy_coef=args.entropy,
        value_loss_coef=0.5,
        gae_lambda=0.95
    )
    
    # Load checkpoint if resuming
    if checkpoint_path:
        try:
            agent.load_state_dict(torch.load(checkpoint_path))
            logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            if not args.new_run:
                logger.error("Consider using --new-run flag to start fresh")
                return
    
    try:
        # Run manual training loop
        manual_training_loop(
            env, 
            agent, 
            num_epochs=args.epochs, 
            steps_per_epoch=args.steps,
            resume=not args.new_run,
            start_epoch=start_epoch
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        env.close()

if __name__ == "__main__":
    main()
