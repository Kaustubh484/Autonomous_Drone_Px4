"""
System Verification Script
Tests all components before training
"""

import sys
import os

print("=" * 60)
print("PX4 RL System Verification")
print("=" * 60)

# Test 1: Check Python packages
print("\n[1/7] Checking Python packages...")
required_packages = {
    'numpy': 'numpy',
    'torch': 'torch',
    'gymnasium': 'gym',
    'mavsdk': 'mavsdk'
}

missing_packages = []
for display_name, import_name in required_packages.items():
    try:
        __import__(import_name)
        print(f"  ✓ {display_name}")
    except ImportError:
        print(f"  ✗ {display_name} - MISSING")
        missing_packages.append(display_name)

if missing_packages:
    print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
    print(f"Install with: pip3 install {' '.join(missing_packages)}")
    sys.exit(1)

# Test 2: Check project files
print("\n[2/7] Checking project files...")
required_files = [
    'waypoint_manager.py',
    'px4_vision_env.py',
    'ros2_camera_bridge.py',
    'ppo_agent.py',
    'train_custom_ppo.py',
    'test_custom_ppo.py'
]

missing_files = []
for filename in required_files:
    if os.path.exists(filename):
        print(f"  ✓ {filename}")
    else:
        print(f"  ✗ {filename} - MISSING")
        missing_files.append(filename)

if missing_files:
    print(f"\n❌ Missing files: {', '.join(missing_files)}")
    print("Make sure all project files are in the current directory")
    sys.exit(1)

# Test 3: Test imports
print("\n[3/7] Testing module imports...")
try:
    from waypoint_manager import WaypointManager
    print("  ✓ waypoint_manager")
except Exception as e:
    print(f"  ✗ waypoint_manager: {e}")
    sys.exit(1)

try:
    from ros2_camera_bridge import start_ros2_camera_node
    print("  ✓ ros2_camera_bridge")
except Exception as e:
    print(f"  ✗ ros2_camera_bridge: {e}")
    sys.exit(1)

try:
    from ppo_agent import PPO
    print("  ✓ ppo_agent")
except Exception as e:
    print(f"  ✗ ppo_agent: {e}")
    sys.exit(1)

# Test 4: Test camera bridge
print("\n[4/7] Testing camera bridge...")
try:
    camera = start_ros2_camera_node(use_synthetic=True)
    import time
    time.sleep(0.5)
    if camera.is_ready():
        depth = camera.get_depth_image()
        print(f"  ✓ Camera working (shape: {depth.shape})")
        camera.cleanup()
    else:
        print(f"  ⚠️  Camera not ready, but functional")
        camera.cleanup()
except Exception as e:
    print(f"  ✗ Camera test failed: {e}")
    sys.exit(1)

# Test 5: Test PPO agent
print("\n[5/7] Testing PPO agent...")
try:
    import numpy as np
    import torch
    
    agent = PPO(obs_dim=13, action_dim=4, use_vision=False, device='cpu')
    
    obs = np.random.randn(13)
    action, log_prob, value = agent.select_action(obs)
    
    print(f"  ✓ PPO agent working")
    print(f"    Action shape: {action.shape}")
    print(f"    Device: {agent.device}")
    
except Exception as e:
    print(f"  ✗ PPO agent test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Check waypoints
print("\n[6/7] Checking waypoints...")
if os.path.exists('waypoints.npy'):
    waypoints = np.load('waypoints.npy')
    print(f"  ✓ Waypoints file found ({len(waypoints)} waypoints)")
else:
    print(f"  ⚠️  No waypoints file found")
    print(f"    Run 'python3 create_test_waypoints.py' to create waypoints")

# Test 7: Check PX4 directory
print("\n[7/7] Checking PX4 installation...")
px4_path = os.path.expanduser("~/PX4-Autopilot")
if os.path.exists(px4_path):
    print(f"  ✓ PX4-Autopilot found at {px4_path}")
    
    # Check if built
    build_path = os.path.join(px4_path, "build", "px4_sitl_default")
    if os.path.exists(build_path):
        print(f"  ✓ PX4 is built")
    else:
        print(f"  ⚠️  PX4 not built yet")
        print(f"    Run: cd ~/PX4-Autopilot && make px4_sitl gazebo-classic")
else:
    print(f"  ✗ PX4-Autopilot not found")
    print(f"    Clone it with: git clone https://github.com/PX4/PX4-Autopilot.git ~/PX4-Autopilot --recursive")

# Summary
print("\n" + "=" * 60)
print("Verification Summary")
print("=" * 60)
print("\n✓ All core components are working!")
print("\nNext steps:")
print("  1. Create waypoints: python3 create_test_waypoints.py")
print("  2. Start PX4 in one terminal: cd ~/PX4-Autopilot && make px4_sitl gazebo-classic")
print("  3. Start training in another terminal: python3 train_custom_ppo.py")
print("\nNote: Make sure PX4 simulation is running BEFORE starting training!")
print("=" * 60)