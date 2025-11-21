import gymnasium as gym
from gymnasium import spaces
import numpy as np
import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, AttitudeRate
import threading
import time
import subprocess
import os
from ros2_camera_bridge import start_gazebo_camera_node

class PX4VisionEnv(gym.Env):
    """PX4 Gym Environment with Gazebo camera integration"""
    
    def __init__(self, waypoints, max_steps=500, use_vision=False):
        super(PX4VisionEnv, self).__init__()
        
        self.waypoints = np.array(waypoints)
        self.current_waypoint_idx = 0
        self.max_steps = max_steps
        self.current_step = 0
        self.use_vision = use_vision
        
        self.episode_count = 0
        self.telemetry_tasks = [] 
        
        # Action space
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
        
        # Observation space
        if use_vision:
            obs_dim = 141  # 128 visual + 13 kinematic
        else:
            obs_dim = 13
    
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Action scaling
        self.max_roll_rate = 60.0
        self.max_pitch_rate = 60.0
        self.max_yaw_rate = 30.0
        self.min_thrust = 0.55
        self.max_thrust = 0.80
        
        # Reward parameters
        self.waypoint_threshold = 1.5
        self.collision_threshold = 0.3
        self.max_angular_velocity = 200.0
        self.previous_distance = None
        
        # State variables
        self.position = np.zeros(3)
        self.orientation_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        
        self.depth_image = None
        self.camera_node = None
    
        if use_vision:
            print("Initializing Gazebo depth camera...")
            try:
                self.camera_node = start_gazebo_camera_node(use_synthetic=True, use_classic=True)
                print("Waiting for camera data...")
                timeout = 5
                start_time = time.time()
                while not self.camera_node.is_ready() and (time.time() - start_time) < timeout:
                    time.sleep(0.1)
                
                if self.camera_node.is_ready():
                    print("✓ Depth camera ready!")
                    self.depth_image = self.camera_node.get_depth_image()
                else:
                    print("⚠️  Camera timeout - using synthetic")
                    self.depth_image = np.zeros((84, 84), dtype=np.float32)
            
            except Exception as e:
                print(f"Camera initialization error: {e}")
                self.camera_node = start_gazebo_camera_node(use_synthetic=True)
                self.depth_image = self.camera_node.get_depth_image()
        
        # Drone connection
        self.drone = System()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._start_event_loop, daemon=True)
        self.thread.start()
        self.is_connected = False
        
        print("Connecting to PX4...")
        asyncio.run_coroutine_threadsafe(self._connect_drone(), self.loop)
        
        timeout = 30
        start = time.time()
        while not self.is_connected and (time.time() - start) < timeout:
            time.sleep(0.1)
        
        if not self.is_connected:
            raise ConnectionError("Failed to connect to PX4")
        
        print("✓ PX4 connected successfully!")
    
    def _start_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    async def _connect_drone(self):
            # Use udpin for robust connection in SITL
            await self.drone.connect(system_address="udp://:14540")
            
            print("   Waiting for drone connection state...")
            async for state in self.drone.core.connection_state():
                if state.is_connected:
                    self.is_connected = True
                    print("   ✓ Drone connected!")
                    break
            
            print("   Waiting for Global Position lock...")
            # FIXED: Python 3.10 compatible timeout logic
            start_wait = time.time()
            gps_locked = False
            async for health in self.drone.telemetry.health():
                if health.is_global_position_ok:
                    print("   ✓ Global Position OK")
                    gps_locked = True
                    break
                
                # Manual timeout check
                if time.time() - start_wait > 10:
                    print("   ⚠️ GPS Lock timeout - proceeding anyway (risk of EKF issues)")
                    break
                    
            # START TELEMETRY NOW (Crucial: This runs even if GPS times out)
            self.telemetry_tasks.append(asyncio.create_task(self._update_position()))
            self.telemetry_tasks.append(asyncio.create_task(self._update_attitude()))
            self.telemetry_tasks.append(asyncio.create_task(self._update_velocity()))
            self.telemetry_tasks.append(asyncio.create_task(self._update_angular_velocity()))
    
    def _stop_telemetry_tasks(self):
        """Cancels background tasks to prevent 'Socket closed' errors during restart."""
        for task in self.telemetry_tasks:
            task.cancel()
        self.telemetry_tasks = []
        
    async def _update_position(self):
        try:
            async for pos_vel in self.drone.telemetry.position_velocity_ned():
                self.position = np.array([
                    pos_vel.position.north_m,
                    pos_vel.position.east_m,
                    pos_vel.position.down_m
                ])
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
    
    async def _update_attitude(self):
        try:
            async for attitude in self.drone.telemetry.attitude_quaternion():
                self.orientation_quat = np.array([
                    attitude.w, attitude.x, attitude.y, attitude.z
                ])
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
    
    async def _update_velocity(self):
        try:
            async for pos_vel in self.drone.telemetry.position_velocity_ned():
                self.linear_velocity = np.array([
                    pos_vel.velocity.north_m_s,
                    pos_vel.velocity.east_m_s,
                    pos_vel.velocity.down_m_s
                ])
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
    
    async def _update_angular_velocity(self):
        try:
            async for attitude_euler in self.drone.telemetry.attitude_euler():
                self.angular_velocity = np.array([
                    attitude_euler.roll_deg,
                    attitude_euler.pitch_deg,
                    attitude_euler.yaw_deg
                ])
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_waypoint_idx = 0
        self.previous_distance = None
        self.episode_count += 1
        
        future = asyncio.run_coroutine_threadsafe(self._reset_drone(), self.loop)
        try:
            # Increased timeout to 180s because a full restart takes ~60s
            future.result(timeout=180)
        except Exception as e:
            print(f"Reset failed ({e}), attempting recovery...")
            time.sleep(5)
            future = asyncio.run_coroutine_threadsafe(self._reset_drone(), self.loop)
            future.result(timeout=180)
        
        return self._get_observation(), {}
    
    # --- CHANGED TO ASYNC DEF ---
    async def _full_simulation_restart(self):
        print("\n" + "!"*60)
        print("⚠️  AUTO-HEAL: Killing Zombie Sim, Wiping Memory & New Waypoints...")
        print("!"*60 + "\n")
        
        # 1. Stop telemetry
        self._stop_telemetry_tasks()

        # 2. Kill processes (Synchronous is fine here as it's fast)
        subprocess.run("pkill -9 px4", shell=True, stderr=subprocess.DEVNULL)
        subprocess.run("pkill -9 gzserver", shell=True, stderr=subprocess.DEVNULL)
        subprocess.run("pkill -9 gzclient", shell=True, stderr=subprocess.DEVNULL)
        
        # Use await sleep so we don't block the event loop
        await asyncio.sleep(2)
        
        # 3. THE BRAIN TRANSPLANT
        px4_dir = os.path.expanduser("~/PX4-Autopilot")
        subprocess.run(f"rm -rf {px4_dir}/build/px4_sitl_default/instance_0", shell=True)
        subprocess.run(f"rm -rf {px4_dir}/build/px4_sitl_default/eeprom", shell=True)
        print("✓ Corrupted memory wiped")

        # 4. Start new simulation
        print("🚀 Restarting Gazebo (this takes 20s)...")
        subprocess.Popen(
            f"cd {px4_dir} && make px4_sitl gazebo", 
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # 5. Wait for startup (ASYNC SLEEP IS CRITICAL HERE)
        await asyncio.sleep(20)
        
        # 6. Reconnect (Simply await the async function)
        print("🔌 Reconnecting MavSDK...")
        self.is_connected = False
        
        # This will now run correctly because the loop isn't blocked!
        await self._connect_drone()
        
        if not self.is_connected:
            raise ConnectionError("Failed to reconnect after auto-heal")

        # 7. TRIGGER WAYPOINT SCRIPT
        if os.path.exists("create_test_waypoints.py"):
            print("⏳ Waiting 10s for Gazebo GUI to load before spawning visuals...")
            await asyncio.sleep(10)  # ASYNC sleep
            
            print("🔄 Generating new mission waypoints...")
            subprocess.run("python3 create_test_waypoints.py", shell=True)
            
            try:
                self.waypoints = np.load('waypoints.npy')
                print(f"✓ Loaded {len(self.waypoints)} new waypoints")
                self.current_waypoint_idx = 0
            except Exception as e:
                print(f"⚠️ Failed to load new waypoints: {e}")
        else:
            print("⚠️ Waypoint script not found, keeping existing mission.")
            
        print("✓ System recovered! Retrying mission...\n")

    def _reset_gazebo_model_pose(self):
        try:
            subprocess.run(
                "gz model -m iris -x 0 -y 0 -z 0.2 -R 0 -P 0 -Y 0", 
                shell=True, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"Warning: Failed to reset Gazebo model pose: {e}")

    async def _reset_drone(self):
        print("Resetting drone...")
        
        try:
            await self.drone.offboard.stop()
            await asyncio.sleep(0.5)
        except:
            pass
        
        try:
            await self.drone.action.kill()
            await asyncio.sleep(1)
        except:
            pass
        
        self._reset_gazebo_model_pose()
        await asyncio.sleep(3)
        
        print("Arming and taking off...")
        
        armed = False
        for attempt in range(10):  
            try:
                print(f"  Arming attempt {attempt + 1}/10...")
                await self.drone.action.arm()
                await asyncio.sleep(2)
                armed = True
                print("  ✓ Armed successfully")
                break
            except Exception as e:
                if attempt < 9:
                    print(f"  ⚠️  Arm failed: {e}, retrying...")
                    await asyncio.sleep(3)
                else:
                    print(f"  ❌ Arming failed after 10 attempts! Last error: {e}")
        
        if not armed:
            print("❌ Arming failed 10 times! Triggering self-healing sequence...")
            # AWAIT the restart (CRITICAL CHANGE)
            await self._full_simulation_restart()
            
            # Recursive call to reset now that sim is fresh
            await self._reset_drone()
            return
        
        try:
            await self.drone.action.takeoff()
            print("Waiting for takeoff to complete...")
            
            safe_altitude = 2.5
            start_wait = time.time()
            while True:
                current_alt = -self.position[2]
                if current_alt >= safe_altitude:
                    print(f"✓ Reached safe altitude: {current_alt:.2f}m")
                    break
                    
                if time.time() - start_wait > 20:
                    print("⚠️ Takeoff timed out, forcing offboard anyway...")
                    break
                
                await asyncio.sleep(0.5)
                
        except Exception as e:
            print(f"⚠️ Takeoff issue: {e}")
            await asyncio.sleep(5)
        
        
        print("Starting offboard mode...")
        
        # 1. Send the "Hover/Climb" command CONTINUOUSLY for 2 seconds
        # This convinces PX4 that the connection is stable
        print("   Sending setpoints to warm up...")
        for _ in range(20):  # 20 * 0.1s = 2 seconds
            await self.drone.offboard.set_attitude_rate(
                AttitudeRate(0.0, 0.0, 0.0, 0.70) # 0.70 Thrust to hold altitude
            )
            await asyncio.sleep(0.1)
        
        # 2. NOW try to switch mode
        try:
            await self.drone.offboard.start()
            print("   ✓ Offboard started")
        except Exception as e:
            print(f"   ⚠️ Offboard switch failed: {e}")
            # Retry once
            await asyncio.sleep(0.5)
            try:
                await self.drone.offboard.start()
            except:
                pass # If it fails, the step() loop will catch it
        
        print("✓ Reset complete\n")
        
    def step(self, action):
        self.current_step += 1
        
        if self.use_vision and self.camera_node is not None:
            self.depth_image = self.camera_node.get_depth_image()
        
        # --- DEBUG SECTION ---
        if self.current_step % 100 == 0:
            target_wp = self.waypoints[self.current_waypoint_idx]
            
            # FIX: Define these variables before printing them
            drone_alt = -self.position[2]
            wp_alt = -target_wp[2]
            
            print(f"\n[Step {self.current_step}] Debug:")
            print(f"  Drone: N={self.position[0]:.2f}, E={self.position[1]:.2f}, Alt={drone_alt:.2f}m")
            print(f"  Waypoint {self.current_waypoint_idx}: N={target_wp[0]:.2f}, E={target_wp[1]:.2f}, Alt={wp_alt:.2f}m")
            print(f"  Distance: {np.linalg.norm(self.position - target_wp):.2f}m")
            print(f"  Action: roll={action[0]:.2f}, pitch={action[1]:.2f}, yaw={action[2]:.2f}, thrust={action[3]:.2f}\n")
        # ---------------------

        # 1. Calculate Action
        roll_rate = action[0] * self.max_roll_rate
        pitch_rate = action[1] * self.max_pitch_rate
        yaw_rate = action[2] * self.max_yaw_rate
        
        thrust = (action[3] + 1) / 2
        thrust = np.clip(thrust, self.min_thrust, self.max_thrust)
        
        
       # --- STRAITJACKET BOOST ---
        # Completely override the agent for the first 15 steps
        if self.current_step <= 15:
            # Calculate fade-out factor (1.0 down to 0.0)
            boost_factor = (15 - self.current_step) / 15.0
            
            # 1. THRUST: Blend 0.75 (Climb) with agent's thrust
            thrust = (0.75 * boost_factor) + (thrust * (1 - boost_factor))
            
            # 2. ROTATION: FORCE LEVEL FLIGHT
            # If we are in the first 10 steps, forbid ANY tilting
            if self.current_step < 10:
                roll_rate = 0.0
                pitch_rate = 0.0
                yaw_rate = 0.0
            else:
                # Steps 10-15: Smoothly give control back
                roll_rate *= (1 - boost_factor)
                pitch_rate *= (1 - boost_factor)
                yaw_rate *= (1 - boost_factor)
        # --------------------------

        # 3. Safety Floor
        if thrust < 0.58: 
           thrust = 0.58
           
        future = asyncio.run_coroutine_threadsafe(
            self._send_attitude_rate_command(roll_rate, pitch_rate, yaw_rate, thrust),
            self.loop
        )
        try:
            future.result(timeout=0.5)
        except:
            pass
        
        time.sleep(0.05) # Control loop rate
        
        obs = self._get_observation()
        reward, info = self._calculate_reward()
        terminated = self._check_terminated()
        truncated = self.current_step >= self.max_steps
        
        return obs, reward, terminated, truncated, info
    
    
    async def _send_attitude_rate_command(self, roll_rate, pitch_rate, yaw_rate, thrust):
        await self.drone.offboard.set_attitude_rate(
            AttitudeRate(roll_rate, pitch_rate, yaw_rate, thrust)
        )
    
    def _get_observation(self):
        target_waypoint = self.waypoints[self.current_waypoint_idx]
        
        drone_altitude = -self.position[2]
        waypoint_altitude = -target_waypoint[2]
        
        relative_position = np.array([
            target_waypoint[0] - self.position[0],
            target_waypoint[1] - self.position[1],
            waypoint_altitude - drone_altitude
        ])
        
        kinematic_state = np.concatenate([
            relative_position,
            self.orientation_quat,
            self.linear_velocity,
            self.angular_velocity
        ]).astype(np.float32)
        
        if self.use_vision:
            if self.depth_image is not None:
                import cv2
                depth_small = cv2.resize(self.depth_image, (16, 8))
                depth_features = depth_small.flatten().astype(np.float32)
            else:
                depth_features = np.zeros(128, dtype=np.float32)
            observation = np.concatenate([depth_features, kinematic_state])
        else:
            observation = kinematic_state
        
        return observation
    
    def _calculate_reward(self):
        target_waypoint = self.waypoints[self.current_waypoint_idx]
        distance = np.linalg.norm(self.position - target_waypoint)
        
        reward = 0.0
        info = {}
        
        if self.previous_distance is not None:
            distance_improvement = self.previous_distance - distance
            reward += distance_improvement * 10.0
        self.previous_distance = distance
        
        if distance < self.waypoint_threshold:
            reward += 100.0
            self.current_waypoint_idx += 1
            self.previous_distance = None
            info['waypoint_reached'] = True
            print(f"✓ Waypoint {self.current_waypoint_idx} reached!")
            
            if self.current_waypoint_idx >= len(self.waypoints):
                reward += 200.0
                info['mission_complete'] = True
        
        altitude = -self.position[2]
        
        if altitude < self.collision_threshold:
            reward -= 100.0
            info['collision'] = True
        
        angular_vel_magnitude = np.linalg.norm(self.angular_velocity)
        if angular_vel_magnitude > self.max_angular_velocity:
            reward -= (angular_vel_magnitude - self.max_angular_velocity) * 0.01
        
        reward -= 0.1  # Time penalty
        
        if altitude > 20.0:
            reward -= 2.0
        elif altitude > 15.0:
            reward -= 0.5
        elif altitude < 3.0:  #altitude > self.collision_threshold:
            reward += altitude * 0.5
        elif altitude < 2.0:
            reward -= 2.0
        elif 3.0 <= altitude <= 12.0:
            reward += 1.0
        
        velocity_magnitude = np.linalg.norm(self.linear_velocity)
        if velocity_magnitude > 5.0:
            reward -= 0.5
        
        info['distance'] = distance
        info['altitude'] = altitude
        info['angular_velocity'] = angular_vel_magnitude
        
        return reward, info
    
    def _check_terminated(self):
        if self.current_waypoint_idx >= len(self.waypoints):
            print("🎉 Mission complete!")
            return True
        
        altitude = -self.position[2]
        
        if altitude < self.collision_threshold:
            target_wp = self.waypoints[self.current_waypoint_idx]
            print(f"💥 Collision! Alt: {altitude:.2f}m")
            return True
        
        if altitude > 25.0:
            print(f"⚠️ Altitude limit! altitude={altitude:.2f}m")
            return True
        
        if abs(self.angular_velocity[0]) > 180 or abs(self.angular_velocity[1]) > 180:
            print(f"🔄 Drone flipped!")
            return True
        
        return False
    
    def render(self):
        pass
    
    def close(self):
        print("Closing environment...")
        if self.camera_node is not None:
            self.camera_node.cleanup()
        
        self._stop_telemetry_tasks()
        
        future = asyncio.run_coroutine_threadsafe(self._cleanup(), self.loop)
        try:
            future.result(timeout=10)
        except:
            pass
        self.loop.stop()
    
    async def _cleanup(self):
        try:
            await self.drone.offboard.stop()
            await self.drone.action.land()
            await asyncio.sleep(5)
            await self.drone.action.disarm()
        except:
            pass