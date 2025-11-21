import numpy as np
import subprocess
import time

class WaypointManager:
    """Waypoint manager with visual markers in Gazebo"""
    
    def __init__(self):
        self.waypoints = []
        self.colors = {
            'red': (1, 0, 0, 1),
            'green': (0, 1, 0, 1),
            'blue': (0, 0, 1, 1),
            'yellow': (1, 1, 0, 1),
            'cyan': (0, 1, 1, 1),
            'magenta': (1, 0, 1, 1),
            'orange': (1, 0.5, 0, 1),
        }
    
    def add_waypoint(self, north, east, altitude, color='red'):
        """Add waypoint in NED coordinates and spawn visual marker
        
        Args:
            north: North position in meters
            east: East position in meters  
            altitude: Altitude in meters (positive up)
            color: Marker color ('red', 'green', 'blue', etc.)
        """
        # Store in NED format (down is negative)
        down = -altitude
        self.waypoints.append([north, east, down])
        
        # Spawn visual marker in Gazebo
        self._spawn_marker(north, east, altitude, len(self.waypoints), color)
    
    def _spawn_marker(self, x, y, z, marker_id, color='red'):
        """Spawn a visual sphere marker at waypoint location"""
        rgba = self.colors.get(color, self.colors['red'])
        
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
          <ambient>{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}</ambient>
          <diffuse>{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
        <transparency>0.3</transparency>
      </visual>
    </link>
  </model>
</sdf>"""
        
        # Save SDF file
        sdf_path = f'/tmp/waypoint_{marker_id}.sdf'
        with open(sdf_path, 'w') as f:
            f.write(sdf)
        
        # Spawn in Gazebo
        cmd = f"gz model -f {sdf_path} -m waypoint_{marker_id} -x {x} -y {y} -z {z}"
        try:
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
            print(f"  ✓ Spawned waypoint {marker_id} at ({x:.1f}, {y:.1f}, {z:.1f}m) [{color}]")
        except Exception as e:
            print(f"  ⚠️ Failed to spawn marker {marker_id}: {e}")
        
        time.sleep(0.2)  # Small delay between spawns
    
    def get_waypoints(self):
        """Return waypoints in NED format"""
        return np.array(self.waypoints)
    
    def clear_markers(self):
        """Remove all waypoint markers from Gazebo"""
        for i in range(1, len(self.waypoints) + 1):
            cmd = f"gz model -m waypoint_{i} -d"
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


print("=" * 70)
print("CREATING WAYPOINT MISSION WITH VISUAL MARKERS")
print("=" * 70)
print()

wpm = WaypointManager()

# FIXED WAYPOINT PATTERN:
# - First waypoint AWAY from origin (no direct climb)
# - Consistent altitude (easier to learn)
# - Horizontal square pattern
# - Returns to home position

print("Spawning waypoint markers...")
wpm.add_waypoint(5, 0, 7, color="green")      # WP1: Move forward, climb to 7m
wpm.add_waypoint(10, 0, 7, color="red")       # WP2: Continue forward
wpm.add_waypoint(10, 10, 7, color="red")      # WP3: Turn right (east)
wpm.add_waypoint(0, 10, 7, color="red")       # WP4: Turn left (back west)
wpm.add_waypoint(0, 0, 7, color="blue")       # WP5: Home at same altitude

waypoints_ned = wpm.get_waypoints()
np.save('waypoints.npy', waypoints_ned)

print()
print("=" * 70)
print("WAYPOINT SUMMARY")
print("=" * 70)
print(f"\nTotal waypoints: {len(waypoints_ned)}")
print("\nWaypoints (NED coordinates):")
for i, wp in enumerate(waypoints_ned):
    alt = -wp[2]
    print(f"  {i+1}. North={wp[0]:6.2f}m, East={wp[1]:6.2f}m, Down={wp[2]:6.2f}m (Alt={alt:.2f}m)")

print("\n" + "=" * 70)
print("KEY FEATURES")
print("=" * 70)
print("✓ First waypoint at (5, 0, 7m) - AWAY from spawn point")
print("✓ All waypoints at consistent 7m altitude (no vertical changes)")
print("✓ Horizontal square pattern (easier to learn)")
print("✓ Returns to home position")
print("✓ Visual markers spawned in Gazebo (semi-transparent spheres)")
print()
print("✓ Saved to waypoints.npy")
print()
print("=" * 70)
print("EXPECTED DRONE BEHAVIOR")
print("=" * 70)
print("1. Drone takes off to ~3.5m")
print("2. Climbs and moves forward to first waypoint (5, 0, 7m)")
print("3. Continues forward to (10, 0, 7m)")
print("4. Turns right to (10, 10, 7m)")
print("5. Turns left back to (0, 10, 7m)")
print("6. Returns home to (0, 0, 7m)")
print()
print("🚁 Ready to train!")
print("=" * 70)
