"""
Testing Script for Custom PyTorch PPO Trained Model
"""

import numpy as np
import torch
import os
import time
from px4_vision_env import PX4VisionEnv
from ppo_agent import PPO

# Configuration
USE_VISION = False
MODEL_PATH = "models/px4_ppo_final.pth"  # Change to test different models
NUM_TEST_EPISODES = 3

print("=" * 60)
print("PX4 Drone RL Testing - Custom PyTorch PPO")
print("=" * 60)

# Load waypoints
if os.path.exists('waypoints.npy'):
    waypoints = np.load('waypoints.npy')
    print(f"\n✓ Loaded {len(waypoints)} waypoints")
else:
    print("\n⚠️  No waypoints file found! Using default waypoints.")
    waypoints = np.array([
        [0, 0, -5],
        [10, 0, -7],
        [10, 10, -9],
        [0, 10, -7],
        [0, 0, -5]
    ])

# Create environment
print("\n[1/3] Creating environment...")
env = PX4VisionEnv(waypoints=waypoints, max_steps=1000, use_vision=USE_VISION)
print("✓ Environment created")

# Get dimensions
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Create agent
print("\n[2/3] Loading trained model...")
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found: {MODEL_PATH}")
    print("\nAvailable models:")
    if os.path.exists("models"):
        models = [f for f in os.listdir("models") if f.endswith('.pth')]
        for m in models:
            print(f"  - models/{m}")
    else:
        print("  No models directory found!")
    print("\nPlease train a model first using train_custom_ppo.py")
    exit(1)

agent = PPO(
    obs_dim=obs_dim,
    action_dim=action_dim,
    use_vision=USE_VISION,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

agent.load(MODEL_PATH)
print(f"✓ Model loaded from {MODEL_PATH}")
print(f"  Device: {agent.device}")
print(f"  Training updates: {agent.n_updates}")

# Test the model
print("\n[3/3] Testing trained agent...")
print("=" * 60)
print("Watch Gazebo for the drone's behavior!")
print("=" * 60)
print()

test_results = []

for episode in range(NUM_TEST_EPISODES):
    print(f"\n{'='*60}")
    print(f"Test Episode {episode + 1}/{NUM_TEST_EPISODES}")
    print(f"{'='*60}")
    
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    waypoints_reached = 0
    
    done = False
    while not done and episode_length < 1000:
        # Select action (deterministic)
        action, _, value = agent.select_action(obs, deterministic=True)
        
        # Take step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        episode_length += 1
        obs = next_obs
        
        # Check waypoint progress
        if info.get('waypoint_reached', False):
            waypoints_reached += 1
            print(f"  ✓ Waypoint {waypoints_reached} reached!")
        
        # Print status every 50 steps
        if episode_length % 50 == 0:
            distance = info.get('distance', 0)
            altitude = info.get('altitude', 0)
            angular_vel = info.get('angular_velocity', 0)
            
            print(f"  Step {episode_length:4d} | "
                  f"Reward: {reward:7.2f} | "
                  f"Distance: {distance:5.2f}m | "
                  f"Alt: {altitude:5.2f}m | "
                  f"AngVel: {angular_vel:6.2f}°/s | "
                  f"Value: {value:7.2f}")
        
        time.sleep(0.05)
    
    # Episode summary
    mission_complete = info.get('mission_complete', False)
    
    print(f"\n{'='*60}")
    print(f"Episode {episode + 1} Results:")
    print(f"{'='*60}")
    print(f"  Total reward: {episode_reward:.2f}")
    print(f"  Episode length: {episode_length} steps")
    print(f"  Waypoints reached: {waypoints_reached}/{len(waypoints)}")
    
    if mission_complete:
        print(f"  🎉 Mission complete!")
    elif info.get('collision', False):
        print(f"  💥 Episode ended: Collision")
    elif episode_length >= 1000:
        print(f"  ⏰ Episode ended: Max steps reached")
    else:
        print(f"  ⚠️  Episode ended: Other reason")
    
    test_results.append({
        'reward': episode_reward,
        'length': episode_length,
        'waypoints': waypoints_reached,
        'complete': mission_complete
    })
    
    # Wait before next episode
    if episode < NUM_TEST_EPISODES - 1:
        print("\nWaiting 5 seconds before next episode...")
        time.sleep(5)

# Overall summary
print(f"\n{'='*60}")
print(f"Overall Test Summary ({NUM_TEST_EPISODES} episodes)")
print(f"{'='*60}")

avg_reward = np.mean([r['reward'] for r in test_results])
avg_length = np.mean([r['length'] for r in test_results])
avg_waypoints = np.mean([r['waypoints'] for r in test_results])
success_rate = np.mean([r['complete'] for r in test_results]) * 100

print(f"  Average reward: {avg_reward:.2f}")
print(f"  Average length: {avg_length:.1f} steps")
print(f"  Average waypoints reached: {avg_waypoints:.1f}/{len(waypoints)}")
print(f"  Success rate: {success_rate:.1f}%")

# Detailed results
print(f"\nDetailed Results:")
for i, result in enumerate(test_results):
    status = "✓" if result['complete'] else "✗"
    print(f"  Episode {i+1}: {status} Reward={result['reward']:7.2f}, "
          f"Length={result['length']:4d}, Waypoints={result['waypoints']}/{len(waypoints)}")

env.close()
print("\n✓ Testing complete")
print("=" * 60)