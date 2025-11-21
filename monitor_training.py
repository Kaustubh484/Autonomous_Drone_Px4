#!/usr/bin/env python3
"""
Live Training Monitor
Displays real-time training statistics from log file
"""

import time
import os
import numpy as np
from collections import deque

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def draw_sparkline(values, width=20, height=5):
    """Draw a simple ASCII sparkline"""
    if len(values) < 2:
        return [''] * height
    
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val if max_val != min_val else 1
    
    # Normalize values
    normalized = [(v - min_val) / range_val for v in values]
    
    # Create sparkline
    chars = ['_', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
    line = ''
    for val in normalized[-width:]:
        idx = int(val * (len(chars) - 1))
        line += chars[idx]
    
    return [line]

def monitor_training(log_file='logs/training_log.txt', update_interval=5):
    """Monitor training progress in real-time"""
    
    print("🚁 PX4 RL Training Monitor")
    print("=" * 80)
    print(f"Monitoring: {log_file}")
    print(f"Update interval: {update_interval}s")
    print(f"Press Ctrl+C to exit\n")
    
    start_time = time.time()
    last_size = 0
    
    try:
        while True:
            # Check if file exists and has new data
            if not os.path.exists(log_file):
                print(f"Waiting for {log_file}...")
                time.sleep(update_interval)
                continue
            
            current_size = os.path.getsize(log_file)
            if current_size == last_size and last_size > 0:
                time.sleep(update_interval)
                continue
            
            last_size = current_size
            
            # Read data
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()[1:]  # Skip header
                
                if not lines:
                    time.sleep(update_interval)
                    continue
                
                # Parse data
                episodes = []
                steps = []
                rewards = []
                lengths = []
                avg_rewards = []
                waypoints = []
                collisions = []
                altitudes = []
                
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) >= 8:
                        episodes.append(int(parts[0]))
                        steps.append(int(parts[1]))
                        rewards.append(float(parts[2]))
                        lengths.append(int(parts[3]))
                        avg_rewards.append(float(parts[4]))
                        waypoints.append(int(parts[5]))
                        collisions.append(int(parts[6]))
                        altitudes.append(float(parts[7]))
                
                if not episodes:
                    time.sleep(update_interval)
                    continue
                
                # Calculate statistics
                total_episodes = episodes[-1]
                total_steps = steps[-1]
                current_reward = rewards[-1]
                current_avg = avg_rewards[-1]
                best_reward = max(rewards)
                total_waypoints = sum(waypoints)
                total_collisions = sum(collisions)
                avg_altitude = np.mean(altitudes[-100:]) if len(altitudes) > 0 else 0
                
                # Recent trends (last 50 episodes)
                recent_rewards = rewards[-50:] if len(rewards) >= 50 else rewards
                recent_lengths = lengths[-50:] if len(lengths) >= 50 else lengths
                
                elapsed = time.time() - start_time
                
                # Display
                clear_screen()
                print("=" * 80)
                print(f"🚁 PX4 RL TRAINING MONITOR - {format_time(elapsed)} elapsed")
                print("=" * 80)
                print()
                
                # Progress
                print(f"📊 PROGRESS")
                print(f"   Episodes: {total_episodes:,} | Steps: {total_steps:,}")
                print()
                
                # Rewards
                print(f"🎯 REWARDS")
                print(f"   Current: {current_reward:+8.2f} | Best: {best_reward:+8.2f}")
                print(f"   Avg (100): {current_avg:+8.2f}")
                print(f"   Trend: {draw_sparkline(recent_rewards, width=40)[0]}")
                print()
                
                # Episode Length
                print(f"📏 EPISODE LENGTH")
                avg_len = np.mean(recent_lengths) if recent_lengths else 0
                print(f"   Current: {lengths[-1]:4d} steps | Avg (50): {avg_len:.0f}")
                print(f"   Trend: {draw_sparkline(recent_lengths, width=40)[0]}")
                print()
                
                # Performance Metrics
                print(f"🎪 PERFORMANCE")
                print(f"   Waypoints: {total_waypoints:4d} | Collisions: {total_collisions:4d}")
                print(f"   Avg Altitude: {avg_altitude:.2f}m")
                
                if total_episodes >= 100:
                    collision_rate = (total_collisions / total_episodes) * 100
                    print(f"   Collision Rate: {collision_rate:.1f}%")
                print()
                
                # Learning Indicators
                print(f"📈 LEARNING INDICATORS")
                indicators = []
                if current_avg > -100:
                    indicators.append("✓ Making progress (avg > -100)")
                if avg_len > 200:
                    indicators.append("✓ Surviving longer (len > 200)")
                if avg_altitude > 3.0:
                    indicators.append("✓ Good altitude (> 3m)")
                if total_waypoints > 0:
                    indicators.append(f"✓ {total_waypoints} waypoints reached!")
                
                if indicators:
                    for ind in indicators:
                        print(f"   {ind}")
                else:
                    print(f"   ⏳ Early training - learning in progress...")
                print()
                
                # Tips
                if total_episodes < 100:
                    print(f"💡 TIP: Early training - expect lots of crashes!")
                elif total_episodes < 500:
                    print(f"💡 TIP: Training progressing - watch for reward trends")
                elif current_avg < 0:
                    print(f"💡 TIP: Still learning - be patient!")
                else:
                    print(f"💡 TIP: Good progress - keep training!")
                
                print()
                print("=" * 80)
                print(f"📊 TensorBoard: tensorboard --logdir=logs/tensorboard")
                print(f"   Then open: http://localhost:6006")
                print("=" * 80)
                
            except Exception as e:
                print(f"Error reading log: {e}")
            
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitor stopped")
        print(f"Final stats: {total_episodes} episodes, {total_steps:,} steps")

if __name__ == "__main__":
    import sys
    
    log_file = sys.argv[1] if len(sys.argv) > 1 else 'logs/training_log.txt'
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    monitor_training(log_file, interval)
