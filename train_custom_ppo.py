"""
Training Script for PX4 Drone RL using Custom PyTorch PPO
No Stable-Baselines3 dependency
"""

import numpy as np
import torch
import os
import time
from torch.utils.tensorboard import SummaryWriter
from px4_vision_env import PX4VisionEnv
from ppo_agent import PPO
from collections import deque

# Configuration
USE_VISION = False  # Set to True when camera is ready
TOTAL_TIMESTEPS = 100000
BUFFER_SIZE = 2048
SAVE_FREQ = 10000
LOG_FREQ = 100

# Hyperparameters
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
N_EPOCHS = 10
BATCH_SIZE = 64

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

print("=" * 60)
print("PX4 Drone RL Training - Custom PyTorch PPO")
print("=" * 60)
print(f"\nConfiguration:")
print(f"  Vision enabled: {USE_VISION}")
print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
print(f"  Buffer size: {BUFFER_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# Load waypoints
if os.path.exists('waypoints.npy'):
    print("\n✓ Loading waypoints from 'waypoints.npy'")
    waypoints = np.load('waypoints.npy')
    print(f"  Loaded {len(waypoints)} waypoints")
else:
    print("\n⚠️  No waypoints file found!")
    print("Creating default waypoints...")
    waypoints = np.array([
        [0, 0, -5],
        [10, 0, -7],
        [10, 10, -9],
        [0, 10, -7],
        [0, 0, -5]
    ])
    np.save('waypoints.npy', waypoints)
    print(f"✓ Created {len(waypoints)} default waypoints")

# Display waypoints
print("\nWaypoints (NED coordinates):")
for i, wp in enumerate(waypoints):
    print(f"  {i+1}. X={wp[0]:6.2f}, Y={wp[1]:6.2f}, Down={wp[2]:6.2f} (Alt={-wp[2]:6.2f}m)")

# Create environment
print("\n[1/3] Creating PX4 environment...")
env = PX4VisionEnv(waypoints=waypoints, max_steps=1000, use_vision=USE_VISION)
print("✓ Environment created")

# Get observation and action dimensions
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

print(f"  Observation dim: {obs_dim}")
print(f"  Action dim: {action_dim}")

# Create PPO agent
print("\n[2/3] Creating PPO agent...")
agent = PPO(
    obs_dim=obs_dim,
    action_dim=action_dim,
    use_vision=USE_VISION,
    lr=LEARNING_RATE,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
    clip_range=CLIP_RANGE,
    ent_coef=ENT_COEF,
    vf_coef=VF_COEF,
    n_epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    buffer_size=BUFFER_SIZE,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
print(f"✓ PPO agent created (device: {agent.device})")
resume_path = "models/px4_ppo_interrupted.pth" 
# You can also change this to "models/px4_ppo_best.pth" to load your best run

if os.path.exists(resume_path):
    print(f"\n🔄 Found saved model: {resume_path}")
    print("   Loading weights to RESUME training...")
    try:
        agent.load(resume_path)
        print("   ✓ Weights loaded successfully! Resuming intelligence.")
    except Exception as e:
        print(f"   ⚠️ Failed to load model: {e}")
        print("   Starting from scratch.")
else:
    print("\n🆕 No saved model found. Starting fresh training.")
# Training loop
print("\n[3/3] Starting training...")
print("=" * 60)
print("Monitor training progress:")
print("  - Watch Gazebo for drone behavior")
print("  - Check console for episode rewards")
print("  - TensorBoard: tensorboard --logdir=logs/tensorboard")
print("  - Then open: http://localhost:6006")
print("=" * 60)
print()

# Initialize TensorBoard
writer = SummaryWriter(log_dir='logs/tensorboard')
print("✓ TensorBoard initialized")

# Training state
obs, _ = env.reset()
episode_reward = 0
episode_length = 0
episode_count = 0
total_steps = 0
best_reward = -float('inf')

# Logging
episode_rewards = []
episode_lengths = []
log_file = open("logs/training_log.txt", "w")
log_file.write("episode,steps,reward,length,avg_reward_100,waypoints_reached,collisions,avg_altitude\n")

# Progress tracking
class ProgressTracker:
    def __init__(self):
        self.start_time = time.time()
        self.episode_count = 0
        self.total_steps = 0
        self.best_reward = -float('inf')
        self.rewards_window = deque(maxlen=100)
        self.lengths_window = deque(maxlen=100)
        self.waypoints_reached = 0
        self.collisions = 0
        self.altitudes = []
        self.last_print_time = time.time()
        
    def update(self, reward, length, info):
        self.episode_count += 1
        self.rewards_window.append(reward)
        self.lengths_window.append(length)
        
        if reward > self.best_reward:
            self.best_reward = reward
        
        if info.get('waypoint_reached', False):
            self.waypoints_reached += 1
        if info.get('collision', False):
            self.collisions += 1
        if 'altitude' in info:
            self.altitudes.append(info['altitude'])
    
    def print_progress(self, force=False):
        """Print progress every 30 seconds or when forced"""
        current_time = time.time()
        
        if not force and (current_time - self.last_print_time) < 30:
            return
        
        self.last_print_time = current_time
        elapsed = current_time - self.start_time
        
        # Calculate statistics
        avg_reward = np.mean(self.rewards_window) if len(self.rewards_window) > 0 else 0
        avg_length = np.mean(self.lengths_window) if len(self.lengths_window) > 0 else 0
        avg_altitude = np.mean(self.altitudes[-100:]) if len(self.altitudes) > 0 else 0
        
        # Progress bar
        progress = (self.total_steps / TOTAL_TIMESTEPS) * 100
        bar_length = 40
        filled = int(bar_length * progress / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print("\n" + "=" * 80)
        print(f"TRAINING PROGRESS [{bar}] {progress:.1f}%")
        print("=" * 80)
        print(f"⏱️  Time Elapsed: {elapsed/3600:.2f} hours")
        print(f"📊 Episodes: {self.episode_count} | Steps: {self.total_steps}/{TOTAL_TIMESTEPS}")
        print(f"🎯 Avg Reward (100): {avg_reward:+.2f} | Best: {self.best_reward:+.2f}")
        print(f"📏 Avg Length (100): {avg_length:.0f} steps")
        print(f"🎪 Waypoints Reached: {self.waypoints_reached} | Collisions: {self.collisions}")
        print(f"✈️  Avg Altitude (100): {avg_altitude:.2f}m")
        
        # Learning indicators
        if self.episode_count >= 100:
            print(f"\n📈 Learning Indicators:")
            if avg_reward > -100:
                print(f"   ✓ Making progress! (reward > -100)")
            if avg_length > 200:
                print(f"   ✓ Surviving longer! (length > 200)")
            if avg_altitude > 3.0:
                print(f"   ✓ Good altitude control! (avg > 3m)")
            if self.waypoints_reached > 0:
                print(f"   ✓ Reaching waypoints!")
        
        print("=" * 80 + "\n")

tracker = ProgressTracker()

try:
    while total_steps < TOTAL_TIMESTEPS:
        # Collect experience
        for step in range(BUFFER_SIZE):
            # Select action
            action, log_prob, value = agent.select_action(obs)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store in buffer
            agent.buffer.add(
                torch.FloatTensor(obs).to(agent.device),
                torch.FloatTensor(action).to(agent.device),
                reward,
                float(done),
                log_prob,
                value
            )
            
            # Update state
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            # Handle episode end
            if done:
                episode_count += 1
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Update progress tracker
                tracker.update(episode_reward, episode_length, info)
                tracker.total_steps = total_steps
                
                # Calculate running average
                avg_reward_100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) > 0 else 0
                
                # Log to TensorBoard
                writer.add_scalar('Episode/Reward', episode_reward, episode_count)
                writer.add_scalar('Episode/Length', episode_length, episode_count)
                writer.add_scalar('Episode/Avg_Reward_100', avg_reward_100, episode_count)
                
                # Log waypoints and collisions
                if info.get('waypoint_reached', False):
                    writer.add_scalar('Episode/Waypoint_Reached', 1, episode_count)
                if info.get('collision', False):
                    writer.add_scalar('Episode/Collision', 1, episode_count)
                if 'altitude' in info:
                    writer.add_scalar('Episode/Final_Altitude', info['altitude'], episode_count)
                
                # Log to console (every 10 episodes)
                if episode_count % 10 == 0:
                    print(f"Ep {episode_count:4d} | "
                          f"Steps: {total_steps:6d} | "
                          f"R: {episode_reward:+7.2f} | "
                          f"Len: {episode_length:4d} | "
                          f"Avg: {avg_reward_100:+7.2f} | "
                          f"Alt: {info.get('altitude', 0):.2f}m")
                
                # Log to file
                waypoints = 1 if info.get('waypoint_reached', False) else 0
                collisions = 1 if info.get('collision', False) else 0
                altitude = info.get('altitude', 0)
                log_file.write(f"{episode_count},{total_steps},{episode_reward:.2f},{episode_length},"
                             f"{avg_reward_100:.2f},{waypoints},{collisions},{altitude:.2f}\n")
                log_file.flush()
                
                # Print progress tracker
                tracker.print_progress()
                
                # Save best model
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    agent.save("models/px4_ppo_best.pth")
                    print(f"🏆 New best reward: {best_reward:.2f} - Model saved!")
                
                # Reset episode
                obs, _ = env.reset()
                episode_reward = 0
                episode_length = 0
            
            # Check if buffer is full
            if agent.buffer.full or total_steps >= TOTAL_TIMESTEPS:
                break
        
        # Update policy
        stats = agent.update()
        
        # Log update stats to TensorBoard
        writer.add_scalar('Training/Loss', stats['loss'], agent.n_updates)
        writer.add_scalar('Training/Policy_Loss', stats['policy_loss'], agent.n_updates)
        writer.add_scalar('Training/Value_Loss', stats['value_loss'], agent.n_updates)
        writer.add_scalar('Training/Entropy', stats['entropy'], agent.n_updates)
        writer.add_scalar('Training/Total_Steps', total_steps, agent.n_updates)
        
        # Log update stats
        if total_steps % (BUFFER_SIZE * 5) == 0:
            print(f"\nUpdate {agent.n_updates} | "
                  f"Loss: {stats['loss']:.4f} | "
                  f"Policy: {stats['policy_loss']:.4f} | "
                  f"Value: {stats['value_loss']:.4f} | "
                  f"Entropy: {stats['entropy']:.4f}\n")
        
        # Save checkpoint
        if total_steps % SAVE_FREQ == 0:
            agent.save(f"models/px4_ppo_checkpoint_{total_steps}.pth")
            print(f"✓ Checkpoint saved at {total_steps} steps")
    
    # Training complete
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    
    # Final progress report
    tracker.print_progress(force=True)
    
    # Save final model
    agent.save("models/px4_ppo_final.pth")
    print(f"💾 Final model saved: models/px4_ppo_final.pth")
    
    # Summary statistics
    print(f"\n📊 TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"Total episodes: {episode_count}")
    print(f"Total steps: {total_steps}")
    print(f"Training time: {(time.time() - tracker.start_time)/3600:.2f} hours")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Final avg reward (100 eps): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Total waypoints reached: {tracker.waypoints_reached}")
    print(f"Total collisions: {tracker.collisions}")
    print(f"Success rate (last 100): {(tracker.waypoints_reached/max(100, len(episode_rewards)))*100:.1f}%")
    print(f"{'='*80}\n")
    
    # Close TensorBoard
    writer.close()
    print("✅ TensorBoard logs saved")

except KeyboardInterrupt:
    print("\n" + "=" * 80)
    print("⚠️  TRAINING INTERRUPTED BY USER")
    print("=" * 80)
    
    # Show progress before exit
    tracker.print_progress(force=True)
    
    agent.save("models/px4_ppo_interrupted.pth")
    print(f"💾 Model saved: models/px4_ppo_interrupted.pth")
    print(f"📊 Completed {episode_count} episodes in {(time.time() - tracker.start_time)/3600:.2f} hours")
    writer.close()

except Exception as e:
    print(f"\n❌ TRAINING FAILED")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    
    agent.save("models/px4_ppo_emergency.pth")
    print("💾 Emergency save: models/px4_ppo_emergency.pth")
    writer.close()

finally:
    log_file.close()
    env.close()
    writer.close()
    print("\n✅ Cleanup complete")
    print("=" * 80)