#!/usr/bin/env python3
"""
Analyze training logs to identify problems and recommend solutions.
"""
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

LOG_DIR = "/teamspace/studios/this_studio/manual_logs"
REWARD_LOG_FILE = os.path.join(LOG_DIR, "reward_logs.jsonl")

def load_reward_logs(log_file=REWARD_LOG_FILE):
    """Load the reward logs from the JSONL file."""
    episodes = []
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return episodes
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(episodes)} episodes from {log_file}")
    return episodes

def analyze_episodes(episodes):
    """Analyze episode data to identify performance trends and issues."""
    if not episodes:
        print("No episode data to analyze.")
        return
    
    # Extract key metrics
    rewards = [ep["episode_reward"] for ep in episodes]
    lengths = [ep["episode_length"] for ep in episodes]
    distances = [ep["distance_traveled"] for ep in episodes if "distance_traveled" in ep]
    
    # Extract reward components
    components = defaultdict(list)
    for ep in episodes:
        if "reward_components" in ep:
            for key, value in ep["reward_components"].items():
                components[key].append(value)
    
    # Count actual collisions by checking reward components
    # Instead of using reward thresholds that may be misleading due to high positive rewards
    collision_count = 0
    for ep in episodes:
        if "reward_components" in ep and "collision" in ep["reward_components"]:
            if ep["reward_components"]["collision"] < -1.0:  # Any significant negative value indicates collision
                collision_count += 1
    
    collision_rate = collision_count / len(episodes) if episodes else 0
    
    # Calculate statistics
    stats = {
        "episodes": len(episodes),
        "avg_reward": np.mean(rewards),
        "avg_length": np.mean(lengths),
        "avg_distance": np.mean(distances) if distances else 0,
        "collision_rate": collision_rate,
        "improvement_rate": calculate_improvement_rate(rewards),
        "collision_count": collision_count
    }
    
    # Print overall statistics
    print("\n=== Training Statistics ===")
    print(f"Total episodes: {stats['episodes']}")
    print(f"Average reward: {stats['avg_reward']:.2f}")
    print(f"Average episode length: {stats['avg_length']:.2f} steps")
    print(f"Average distance traveled: {stats['avg_distance']:.2f}m")
    print(f"Collision count: {stats['collision_count']} ({stats['collision_rate']*100:.1f}%)")
    print(f"Recent improvement rate: {stats['improvement_rate']*100:.1f}%")
    
    # Also check for very short episodes
    short_episodes = sum(1 for length in lengths if length < 100)
    if short_episodes > 0:
        print(f"WARNING: {short_episodes} episodes ({short_episodes/len(lengths)*100:.1f}%) ended very early (<100 steps)")
    
    # Diagnose issues
    diagnose_training_issues(stats, rewards, distances, components)

def calculate_improvement_rate(rewards, window=20):
    """Calculate if rewards are improving in the recent window."""
    if len(rewards) < window*2:
        return 0
    
    recent_rewards = rewards[-window:]
    previous_rewards = rewards[-window*2:-window]
    
    recent_avg = np.mean(recent_rewards)
    previous_avg = np.mean(previous_rewards)
    
    # Return improvement as percentage
    return (recent_avg - previous_avg) / (abs(previous_avg) + 1e-5)

def diagnose_training_issues(stats, rewards, distances, components):
    """Identify potential issues and provide recommendations."""
    print("\n=== Training Diagnosis ===")
    
    issues = []
    
    # Check for high collision rate
    if stats["collision_rate"] > 0.7:
        issues.append({
            "issue": "High collision rate",
            "severity": "HIGH",
            "description": "The agent is crashing too frequently",
            "recommendation": "Increase collision penalty further and focus on collision avoidance phase"
        })
    
    # Check for lack of improvement
    if stats["improvement_rate"] < 0.05 and len(rewards) > 50:
        issues.append({
            "issue": "Lack of improvement",
            "severity": "MEDIUM",
            "description": "Rewards aren't increasing significantly over time",
            "recommendation": "Try adjusting learning rate or network architecture"
        })
    
    # Check for short episodes
    if stats["avg_length"] < 50 and len(rewards) > 20:
        issues.append({
            "issue": "Very short episodes",
            "severity": "HIGH", 
            "description": "Episodes are ending too quickly, limiting learning opportunities",
            "recommendation": "Check for early termination conditions or environment issues"
        })
    
    # Check for reward scaling issues
    if any(r > 10000 for r in rewards):
        issues.append({
            "issue": "Extreme reward scaling",
            "severity": "MEDIUM",
            "description": "Rewards are extremely large which can destabilize learning",
            "recommendation": "Cap exponential rewards like survival bonus to prevent them from dominating"
        })
    
    # Check for short traveled distances
    if stats["avg_distance"] < 5 and len(distances) > 20:
        issues.append({
            "issue": "Limited exploration",
            "severity": "MEDIUM",
            "description": "Agent isn't traveling far enough to learn effectively",
            "recommendation": "Increase exploration or add curriculum learning"
        })
    
    # Print issues and recommendations
    if not issues:
        print("No critical issues detected.")
    else:
        for i, issue in enumerate(issues):
            print(f"\nIssue {i+1}: [{issue['severity']}] {issue['issue']}")
            print(f"Description: {issue['description']}")
            print(f"Recommendation: {issue['recommendation']}")
    
    # Check if we should change training phase
    recommend_phase_change(stats)

def recommend_phase_change(stats):
    """Recommend whether to change training phase based on metrics."""
    phase_file = "/teamspace/studios/this_studio/training_phase.txt"
    current_phase = "collision_avoidance"  # Default
    
    if os.path.exists(phase_file):
        with open(phase_file, 'r') as f:
            current_phase = f.read().strip()
    
    print("\n=== Training Phase Recommendation ===")
    print(f"Current training phase: {current_phase}")
    
    if current_phase == "collision_avoidance":
        if stats["collision_rate"] < 0.3 and stats["avg_distance"] > 15:
            print("✓ Ready to advance to 'basic_driving' phase")
            print("  Run: python set_training_phase.py basic_driving")
        else:
            print("✗ Not ready to advance phase - keep focusing on collision avoidance")
            print(f"  Current collision rate: {stats['collision_rate']*100:.1f}% (target: <30%)")
            print(f"  Current average distance: {stats['avg_distance']:.1f}m (target: >15m)")
    
    elif current_phase == "basic_driving":
        if stats["collision_rate"] < 0.15 and stats["avg_distance"] > 40:
            print("✓ Ready to advance to 'advanced_driving' phase")
            print("  Run: python set_training_phase.py advanced_driving")
        else:
            print("✗ Not ready to advance phase - keep improving basic driving skills")
            print(f"  Current collision rate: {stats['collision_rate']*100:.1f}% (target: <15%)")
            print(f"  Current average distance: {stats['avg_distance']:.1f}m (target: >40m)")
    
    else:  # advanced_driving
        if stats["collision_rate"] < 0.05 and stats["avg_distance"] > 100:
            print("✓ Training is successful! Agent has good driving skills.")
        else:
            print("✗ Continue training to improve advanced driving performance")
            print(f"  Current collision rate: {stats['collision_rate']*100:.1f}% (target: <5%)")
            print(f"  Current average distance: {stats['avg_distance']:.1f}m (target: >100m)")

def plot_training_progress(episodes, save_path=None):
    """Generate plots showing training progress."""
    if not episodes:
        return
    
    # Prepare data
    rewards = [ep["episode_reward"] for ep in episodes]
    lengths = [ep["episode_length"] for ep in episodes]
    distances = [ep["distance_traveled"] for ep in episodes if "distance_traveled" in ep]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot rewards
    axes[0,0].plot(rewards)
    axes[0,0].set_title('Episode Rewards')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Total Reward')
    
    # Plot smooth rewards (moving average)
    window = min(len(rewards) // 10 + 1, 10)
    if len(rewards) > window:
        smooth_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0,0].plot(range(window-1, len(rewards)), smooth_rewards, 'r-', lw=2)
    
    # Plot episode lengths
    axes[0,1].plot(lengths)
    axes[0,1].set_title('Episode Lengths')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Steps')
    
    # Plot distances
    if distances:
        axes[1,0].plot(distances)
        axes[1,0].set_title('Distance Traveled')
        axes[1,0].set_xlabel('Episode')
        axes[1,0].set_ylabel('Distance (m)')
    
    # Plot recent performance table
    recent = min(10, len(rewards))
    if recent > 0:
        axes[1,1].axis('off')
        table_data = []
        table_data.append(["Metric", "Last Episode", f"Last {recent} Avg"])
        table_data.append(["Reward", f"{rewards[-1]:.2f}", f"{np.mean(rewards[-recent:]):.2f}"])
        if distances:
            table_data.append(["Distance", f"{distances[-1]:.2f}m", f"{np.mean(distances[-recent:]):.2f}m"])
        table_data.append(["Length", f"{lengths[-1]} steps", f"{np.mean(lengths[-recent:]):.1f} steps"])
        
        # Create a table at the position of axes[1,1]
        table = axes[1,1].table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        axes[1,1].set_title('Recent Performance', pad=20)
    
    plt.tight_layout()
    
    # Save or show plot
    if save_path:
        plt.savefig(save_path)
        print(f"Training progress plot saved to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze training logs and provide recommendations")
    parser.add_argument("--plot", action="store_true", help="Generate plots of training progress")
    parser.add_argument("--save", type=str, help="Save the analysis plot to this file")
    args = parser.parse_args()
    
    # Load and analyze episodes
    episodes = load_reward_logs()
    if episodes:
        analyze_episodes(episodes)
        
        # Generate plots if requested
        if args.plot or args.save:
            plot_training_progress(episodes, args.save)

if __name__ == "__main__":
    main()
