import numpy as np
from waypoint_manager import WaypointManager

def create_straight_line_mission():
    """
    Level 1: The Straight Line
    - Constant Altitude (6m)
    - No Turns (Just North)
    - Short Distance
    """
    wpm = WaypointManager()
    print("=" * 60)
    print("GENERATING LEVEL 1 CURRICULUM: STRAIGHT LINE")
    print("=" * 60)

    # Clear any existing markers
    try:
        wpm.clear_waypoints()
    except:
        pass

    # Waypoint 1: Takeoff & Stabilize
    # 6m is your reward function's "sweet spot"
    wpm.add_waypoint(0, 0, 6, color="green") 
    
    # Waypoint 2: Fly Forward 5m
    # Gentle start, easy to reach
    wpm.add_waypoint(5, 0, 6, color="yellow") 
    
    # Waypoint 3: Fly Forward 10m (End)
    # Requires maintaining velocity
    wpm.add_waypoint(10, 0, 6, color="red")

    # Save to file
    waypoints = wpm.get_waypoints()
    np.save('waypoints.npy', waypoints)
    
    print(f"\n✓ Generated {len(waypoints)} waypoints.")
    print("✓ Saved to 'waypoints.npy'")
    print("\nMission Profile:")
    print("  1. Hover at (0, 0, 6m)")
    print("  2. Fly North 5m at 6m altitude")
    print("  3. Fly North 10m at 6m altitude")
    print("\nRun your training script now!")

if __name__ == "__main__":
    create_straight_line_mission()