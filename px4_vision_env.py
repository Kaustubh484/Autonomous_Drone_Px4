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
    """PX4 Gym Environment - Final Robust Version with Heartbeat & Rocket Launch"""
    
    def __init__(self, waypoints, max_steps=500, use_vision=False):
        super(PX4VisionEnv, self).__init__()
        
        self.waypoints = np.array(waypoints)
        self.current_waypoint_idx = 0
        self.max_steps = max_steps
        self.current_step = 0
        self.use_vision = use_vision
        
        self.episode_count = 0
        self.telemetry_tasks = [] 
        self.hover_task = None  # Track the heartbeat task
        
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
        self.max_roll_rate = 30.0    # Reduced for stability
        self.max_pitch_rate = 30.0 
        self.max_yaw_rate = 15.0
        self.min_thrust = 0.50       
        self.max_thrust = 0.80
        
        # Reward parameters
        self.waypoint_threshold = 1.5
        self.collision_threshold = 0.4  # Higher threshold to detect ground sooner
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
                timeout = 5
                start = time.time()
                while not self.camera_node.is_ready() and (time.time() - start) < timeout:
                    time.sleep(0.1)
                
                if self.camera_node.is_ready():
                    print("✓ Depth camera ready!")
                    self.depth_image = self.camera_node.get_depth_image()
                else:
                    print("⚠️ Camera timeout - using synthetic")
                    self.depth_image = np.zeros((84, 84), dtype=np.float32)
            except Exception as e:
                print(f"Camera init error: {e}")
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
        await self.drone.connect(system_address="udp://:14540")
        
        print("   Waiting for connection...")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                self.is_connected = True
                print("   ✓ Drone connected!")
                break
        
        print("   Waiting for Global Position lock...")
        start_wait = time.time()
        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok:
                print("   ✓ Global Position OK")
                break
            if time.time() - start_wait > 10:
                print("   ⚠️ GPS Lock timeout - proceeding anyway")
                break
                
        self.telemetry_tasks.append(asyncio.create_task(self._update_position()))
        self.telemetry_tasks.append(asyncio.create_task(self._update_attitude()))
        self.telemetry_tasks.append(asyncio.create_task(self._update_velocity()))
        self.telemetry_tasks.append(asyncio.create_task(self._update_angular_velocity()))
    
    def _stop_telemetry_tasks(self):
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
        except: pass
    
    async def _update_attitude(self):
        try:
            async for attitude in self.drone.telemetry.attitude_quaternion():
                self.orientation_quat = np.array([attitude.w, attitude.x, attitude.y, attitude.z])
        except: pass
    
    async def _update_velocity(self):
        try:
            async for pos_vel in self.drone.telemetry.position_velocity_ned():
                self.linear_velocity = np.array([pos_vel.velocity.north_m_s, pos_vel.velocity.east_m_s, pos_vel.velocity.down_m_s])
        except: pass
    
    async def _update_angular_velocity(self):
        try:
            async for attitude_euler in self.drone.telemetry.attitude_euler():
                self.angular_velocity = np.array([attitude_euler.roll_deg, attitude_euler.pitch_deg, attitude_euler.yaw_deg])
        except: pass

    # --- NEW HEARTBEAT TASK ---
    async def _maintain_hover(self):
        """Background task to keep drone alive between Reset and Step 1"""
        try:
            while True:
                # Send a "Holding" command (Thrust 0.65 = light climb/hover)
                # This prevents the Failsafe from triggering during the gap
                await self.drone.offboard.set_attitude_rate(
                    AttitudeRate(0.0, 0.0, 0.0, 0.65)
                )
                await asyncio.sleep(0.05) # 20Hz heartbeat
        except asyncio.CancelledError:
            pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_waypoint_idx = 0
        self.previous_distance = None
        self.episode_count += 1
        
        future = asyncio.run_coroutine_threadsafe(self._reset_drone(), self.loop)
        try:
            future.result(timeout=180)
        except Exception as e:
            print(f"Reset failed ({e}), attempting recovery...")
            time.sleep(5)
            future = asyncio.run_coroutine_threadsafe(self._reset_drone(), self.loop)
            future.result(timeout=180)
        
        return self._get_observation(), {}
    
    async def _reset_drone(self):
        print("Resetting drone...")
        
        # 1. Stop any previous heartbeat
        if self.hover_task and not self.hover_task.done():
            self.hover_task.cancel()
        
        # 2. Cleanup
        try:
            await self.drone.offboard.stop()
            await asyncio.sleep(0.5)
        except: pass
        try:
            await self.drone.action.kill()
            await asyncio.sleep(1)
        except: pass
        
        # 3. Teleport & Arm
        self._reset_gazebo_model_pose()
        await asyncio.sleep(2)
        
        print("Arming...")
        armed = False
        for i in range(10):
            try:
                await self.drone.action.arm()
                armed = True
                print("  ✓ Armed")
                break
            except:
                await asyncio.sleep(2)
        
        if not armed:
            print("❌ Arm failed! Triggering Auto-Heal...")
            await self._full_simulation_restart()
            await self._reset_drone()
            return

        # 4. THE ROCKET LAUNCH (Manual Ascent)
        print("🚀 Rocket Launch: Force Climbing to 3.0m...")
        
        # Warmup with high thrust to prevent mode switch dip
        for _ in range(10):
            await self.drone.offboard.set_attitude_rate(AttitudeRate(0.0, 0.0, 0.0, 0.80))
        
        try:
            await self.drone.offboard.start()
        except:
            # Retry start once if it fails
            await asyncio.sleep(0.1)
            try: await self.drone.offboard.start()
            except: pass

        # Climb Loop: Blast 80% thrust until we hit 3.0m
        start_climb = time.time()
        while True:
            await self.drone.offboard.set_attitude_rate(AttitudeRate(0.0, 0.0, 0.0, 0.80))
            await asyncio.sleep(0.05)
            
            if -self.position[2] >= 3.0:
                print(f"  ✓ Reached safe altitude: {-self.position[2]:.2f}m")
                break
            
            if time.time() - start_climb > 5.0:
                print("  ⚠️ Climb timeout - proceeding anyway")
                break

        # 5. START HEARTBEAT (Crucial)
        print("❤️ Starting Heartbeat Task...")
        self.hover_task = asyncio.create_task(self._maintain_hover())
        
        print("✓ Reset complete - drone ready at safe altitude\n")

    def step(self, action):
        # 1. KILL HEARTBEAT (Agent has control now)
        if self.hover_task and not self.hover_task.done():
            self.hover_task.cancel()
             
        self.current_step += 1
        
        if self.use_vision and self.camera_node is not None:
            self.depth_image = self.camera_node.get_depth_image()
            
        if self.current_step % 100 == 0:
            drone_alt = -self.position[2]
            print(f"\n[Step {self.current_step}] Alt={drone_alt:.2f}m Action={action}")

        # Calculate Action
        roll_rate = action[0] * self.max_roll_rate
        pitch_rate = action[1] * self.max_pitch_rate
        yaw_rate = action[2] * self.max_yaw_rate
        
        thrust = (action[3] + 1) / 2
        thrust = np.clip(thrust, self.min_thrust, self.max_thrust)
        
        # Straitjacket / Boost Logic (Locks rotation for first 20 steps)
        if self.current_step <= 20:
            thrust = 0.75
            roll_rate = 0.0
            pitch_rate = 0.0
            yaw_rate = 0.0
        elif self.current_step <= 40:
            alpha = (self.current_step - 20) / 20.0
            thrust = (0.75 * (1 - alpha)) + (thrust * alpha)
            roll_rate *= alpha
            pitch_rate *= alpha
            yaw_rate *= alpha

        if thrust < 0.58: thrust = 0.58
           
        future = asyncio.run_coroutine_threadsafe(
            self._send_attitude_rate_command(roll_rate, pitch_rate, yaw_rate, thrust),
            self.loop
        )
        try: future.result(timeout=0.5)
        except: pass
        
        time.sleep(0.05)
        
        obs = self._get_observation()
        reward, info = self._calculate_reward(action)
        terminated = self._check_terminated()
        truncated = self.current_step >= self.max_steps
        
        return obs, reward, terminated, truncated, info

    async def _send_attitude_rate_command(self, roll_rate, pitch_rate, yaw_rate, thrust):
        await self.drone.offboard.set_attitude_rate(
            AttitudeRate(roll_rate, pitch_rate, yaw_rate, thrust)
        )
    
    def _get_observation(self):
        target_waypoint = self.waypoints[self.current_waypoint_idx]
        drone_alt = -self.position[2]
        wp_alt = -target_waypoint[2]
        
        rel_pos = np.array([
            target_waypoint[0] - self.position[0], 
            target_waypoint[1] - self.position[1], 
            wp_alt - drone_alt
        ])
        
        kinematic_state = np.concatenate([
            rel_pos,
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
    
    def _calculate_reward(self, action):
        reward = 0.0
        info = {}
        
        # Constants
        TARGET_ALTITUDE = 6.0      # The optimal "safe" altitude (meters)
        ALTITUDE_BAND = 4.0        # Acceptable deviation (+/- 4m)
        DIST_COEFF = 10.0          # Weight for distance progress
        STABILITY_COEFF = 0.05     # Penalty for high angular velocity
        ACTION_COEFF = 0.1         # Penalty for violent control inputs
        
        # 1. Distance Reward (Potential-Based Shaping)
        # This rewards moving closer and penalizes moving away continuously
        target_waypoint = self.waypoints[self.current_waypoint_idx]
        current_distance = np.linalg.norm(self.position - target_waypoint)
        
        if self.previous_distance is not None:
            # Clip the progress reward to avoid massive spikes if GPS glitches
            progress = np.clip(self.previous_distance - current_distance, -2.0, 2.0)
            reward += progress * DIST_COEFF
        self.previous_distance = current_distance
        
        # 2. Continuous Altitude Reward (Gaussian Curve)
        # Instead of "steps", this provides a smooth gradient towards the target altitude.
        # Peak reward is +1.0 at 6m, dropping smoothly to 0.0 at 2m or 10m.
        altitude = -self.position[2]
        alt_diff = abs(altitude - TARGET_ALTITUDE)
        # exp(-0.5 * (x / sigma)^2)
        alt_reward = np.exp(-0.5 * (alt_diff / 2.0)**2) 
        reward += alt_reward
        
        # 3. Stability Penalty
        # Penalize high angular rates (prevents jitter/oscillation)
        # self.angular_velocity is [roll_rate, pitch_rate, yaw_rate]
        ang_vel_mag = np.linalg.norm(self.angular_velocity)
        reward -= ang_vel_mag * STABILITY_COEFF
        
        # 4. Action Smoothness Penalty
        # Penalize extreme or jerky actions (helps sim-to-real transfer)
        action_mag = np.linalg.norm(action)
        reward -= action_mag * ACTION_COEFF

        # 5. Waypoint Reached (Sparse Bonus)
        if current_distance < self.waypoint_threshold:
            reward += 100.0
            self.current_waypoint_idx += 1
            self.previous_distance = None # Reset potential for new target
            info['waypoint_reached'] = True
            print(f"✓ Waypoint {self.current_waypoint_idx} reached!")
            
            if self.current_waypoint_idx >= len(self.waypoints):
                reward += 200.0 # Big bonus for finishing
                info['mission_complete'] = True

        # 6. Survival / Safety Checks
        # Severe penalty for ground collision, but scale it to avoid gradient explosion
        if altitude < self.collision_threshold:
            reward -= 100.0
            info['collision'] = True
        
        # Soft floor penalty (discourages flying too low without killing the run)
        if altitude < 1.5:
            reward -= 2.0 # Constant pressure to rise, but not a -50 "cliff"

        # 7. Existence Penalty
        # Small negative reward every step encourages speed
        reward -= 0.1

        info['distance'] = current_distance
        info['altitude'] = altitude
        return reward, info

    def _check_terminated(self):
        if self.current_waypoint_idx >= len(self.waypoints):
            print("🎉 Mission complete!")
            return True
        altitude = -self.position[2]
        if altitude < self.collision_threshold:
            print(f"💥 Collision! Alt: {altitude:.2f}m")
            return True
        if altitude > 25.0:
            print(f"⚠️ Altitude limit! {altitude:.2f}m")
            return True
        return False
    
    def _reset_gazebo_model_pose(self):
        try:
            subprocess.run("gz model -m iris -x 0 -y 0 -z 0.2 -R 0 -P 0 -Y 0", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except: pass

    async def _full_simulation_restart(self):
        print("\n" + "!"*60)
        print("⚠️  AUTO-HEAL: Killing Zombie Sim...")
        print("!"*60 + "\n")
        self._stop_telemetry_tasks()
        subprocess.run("pkill -9 px4", shell=True, stderr=subprocess.DEVNULL)
        subprocess.run("pkill -9 gzserver", shell=True, stderr=subprocess.DEVNULL)
        subprocess.run("pkill -9 gzclient", shell=True, stderr=subprocess.DEVNULL)
        await asyncio.sleep(2)
        px4_dir = os.path.expanduser("~/PX4-Autopilot")
        subprocess.run(f"rm -rf {px4_dir}/build/px4_sitl_default/instance_0", shell=True)
        subprocess.run(f"rm -rf {px4_dir}/build/px4_sitl_default/eeprom", shell=True)
        print("🚀 Restarting Gazebo...")
        subprocess.Popen(f"cd {px4_dir} && make px4_sitl gazebo", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        await asyncio.sleep(20)
        print("🔌 Reconnecting...")
        self.is_connected = False
        await self._connect_drone()
        if not self.is_connected: raise ConnectionError("Reconnect failed")
        if os.path.exists("create_test_waypoints.py"):
            await asyncio.sleep(10)
            subprocess.run("python3 create_test_waypoints.py", shell=True)
            try: self.waypoints = np.load('waypoints.npy')
            except: pass
        print("✓ Auto-Heal Complete\n")

    def render(self): pass
    
    def close(self):
        print("Closing environment...")
        if self.hover_task: self.hover_task.cancel()
        if self.camera_node: self.camera_node.cleanup()
        self._stop_telemetry_tasks()
        future = asyncio.run_coroutine_threadsafe(self._cleanup(), self.loop)
        try: future.result(timeout=10)
        except: pass
        self.loop.stop()
    
    async def _cleanup(self):
        try:
            await self.drone.offboard.stop()
            await self.drone.action.land()
            await asyncio.sleep(5)
            await self.drone.action.disarm()
        except: pass