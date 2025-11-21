import subprocess
import numpy as np
import time

class WaypointManager:
    """
    Manages waypoints for RL training:
    - Spawns visual markers in Gazebo
    - Provides waypoint coordinates to environment
    - Supports random waypoint generation for curriculum learning
    """
    
    def __init__(self):
        self.waypoints = []
        self.marker_ids = []
    
    def add_waypoint(self, x, y, z, color="red"):
        """Add a waypoint and spawn its marker"""
        waypoint = np.array([x, y, -z])  # Convert to NED (down is negative)
        self.waypoints.append(waypoint)
        
        marker_id = len(self.waypoints)
        self.spawn_marker(x, y, z, marker_id, color)
        self.marker_ids.append(marker_id)
        
        return waypoint
    
    def spawn_marker(self, x, y, z, marker_id, color="red"):
        """Spawn a visual marker in Gazebo"""
        
        color_map = {
            "red": "1 0 0 1",
            "green": "0 1 0 1",
            "blue": "0 0 1 1",
            "yellow": "1 1 0 1",
            "cyan": "0 1 1 1",
            "magenta": "1 0 1 1",
            "orange": "1 0.5 0 1",
        }
        
        color_rgba = color_map.get(color, "1 0 0 1")
        
        # Create SDF model for waypoint marker
        sdf = f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='waypoint_{marker_id}'>
    <static>true</static>
    <link name='link'>
      <visual name='visual'>
        <geometry>
          <sphere>
            <radius>0.5</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>{color_rgba}</ambient>
          <diffuse>{color_rgba}</diffuse>
          <emissive>0.5 0.5 0.5 1</emissive>
        </material>
      </visual>
      <!-- Add text label -->
      <visual name='label'>
        <pose>0 0 1.0 0 0 0</pose>
        <geometry>
          <box>
            <size>0.1 0.1 0.1</size>
          </box>
        </geometry>
        <material>
          <ambient>1 1 1 1</ambient>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""
        
        # Save to temporary file
        sdf_file = f'/tmp/waypoint_{marker_id}.sdf'
        with open(sdf_file, 'w') as f:
            f.write(sdf)
        
        # Spawn in Gazebo
        cmd = f"gz model -f {sdf_file} -m waypoint_{marker_id} -x {x} -y {y} -z {z}"
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print(f"✓ Spawned waypoint {marker_id} at ({x:.1f}, {y:.1f}, {z:.1f}) - {color}")
    
    def spawn_path_line(self, start_idx, end_idx):
        """Spawn a line connecting two waypoints"""
        if start_idx >= len(self.waypoints) or end_idx >= len(self.waypoints):
            return
        
        start = self.waypoints[start_idx]
        end = self.waypoints[end_idx]
        
        # Calculate midpoint and vector
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        mid_z = (-start[2] - end[2]) / 2  # Convert back to altitude
        
        # Simple line visualization (cylinder)
        line_length = np.linalg.norm(end - start)
        
        sdf = f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='path_line_{start_idx}_{end_idx}'>
    <static>true</static>
    <link name='link'>
      <visual name='visual'>
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>{line_length}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0 1 1 0.3</ambient>
          <diffuse>0 1 1 0.3</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""
        
        sdf_file = f'/tmp/path_line_{start_idx}_{end_idx}.sdf'
        with open(sdf_file, 'w') as f:
            f.write(sdf)
        
        cmd = f"gz model -f {sdf_file} -m path_line_{start_idx}_{end_idx} -x {mid_x} -y {mid_y} -z {mid_z}"
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    def get_waypoints(self):
        """Return waypoints as numpy array in NED coordinates"""
        return np.array(self.waypoints)
    
    def clear_waypoints(self):
        """Remove all waypoint markers from Gazebo"""
        for marker_id in self.marker_ids:
            cmd = f"gz model -m waypoint_{marker_id} -d"
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        self.waypoints = []
        self.marker_ids = []
        print("✓ All waypoints cleared")
    
    def generate_random_waypoints(self, num_waypoints=5, 
                                   x_range=(-15, 15), 
                                   y_range=(-15, 15), 
                                   z_range=(3, 12)):
        """
        Generate random waypoints for curriculum learning
        Useful for training robustness
        """
        self.clear_waypoints()
        
        # Always start at origin
        self.add_waypoint(0, 0, 5, color="green")
        
        # Generate random intermediate waypoints
        for i in range(num_waypoints - 2):
            x = np.random.uniform(*x_range)
            y = np.random.uniform(*y_range)
            z = np.random.uniform(*z_range)
            self.add_waypoint(x, y, z, color="red")
        
        # Return to start
        self.add_waypoint(0, 0, 5, color="blue")
        
        return self.get_waypoints()
    
    def generate_circular_path(self, radius=10, num_waypoints=8, altitude=8):
        """Generate waypoints in a circular pattern"""
        self.clear_waypoints()
        
        angles = np.linspace(0, 2*np.pi, num_waypoints, endpoint=False)
        
        for i, angle in enumerate(angles):
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            color = "green" if i == 0 else "red"
            self.add_waypoint(x, y, altitude, color=color)
        
        return self.get_waypoints()
    
    def generate_square_path(self, side_length=15, altitude=8):
        """Generate waypoints in a square pattern"""
        self.clear_waypoints()
        
        half = side_length / 2
        waypoints_coords = [
            (0, 0, 5, "green"),           # Start
            (half, 0, altitude, "red"),   # Right
            (half, half, altitude, "red"), # Top-right
            (0, half, altitude, "red"),    # Top
            (-half, half, altitude, "red"), # Top-left
            (-half, 0, altitude, "red"),   # Left
            (-half, -half, altitude, "red"), # Bottom-left
            (0, -half, altitude, "red"),   # Bottom
            (half, -half, altitude, "red"), # Bottom-right
            (0, 0, 5, "blue"),            # Return home
        ]
        
        for x, y, z, color in waypoints_coords:
            self.add_waypoint(x, y, z, color=color)
        
        return self.get_waypoints()
    
    def generate_zigzag_path(self, length=20, height_variation=5, num_waypoints=6):
        """Generate a zigzag climbing/descending path"""
        self.clear_waypoints()
        
        self.add_waypoint(0, 0, 5, color="green")
        
        for i in range(1, num_waypoints):
            x = (i / num_waypoints) * length
            y = (-1) ** i * 5  # Alternate left and right
            z = 5 + (i / num_waypoints) * height_variation
            self.add_waypoint(x, y, z, color="red")
        
        return self.get_waypoints()
