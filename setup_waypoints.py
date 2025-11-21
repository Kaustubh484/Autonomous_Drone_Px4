from waypoint_manager import WaypointManager
import time

def main():
    """Interactive waypoint setup for RL training"""
    
    wpm = WaypointManager()
    
    print("=" * 60)
    print("Waypoint Setup for RL Training")
    print("=" * 60)
    print("\nChoose a waypoint pattern:")
    print("  1. Manual entry")
    print("  2. Square path")
    print("  3. Circular path")
    print("  4. Random waypoints")
    print("  5. Zigzag path (climbing)")
    print("  6. Simple 4-waypoint path (recommended for testing)")
    print()
    
    choice = input("Enter choice (1-6): ").strip()
    
    if choice == "1":
        # Manual entry
        print("\nEnter waypoints (x, y, z) in meters.")
        print("Type 'done' when finished.")
        print("Starting waypoint: (0, 0, 5)")
        wpm.add_waypoint(0, 0, 5, color="green")
        
        while True:
            user_input = input("Waypoint (x y z): ").strip()
            if user_input.lower() == 'done':
                break
            
            try:
                x, y, z = map(float, user_input.split())
                wpm.add_waypoint(x, y, z, color="red")
            except:
                print("Invalid format. Use: x y z (e.g., 10 5 8)")
        
        # Add return home
        add_home = input("\nAdd return to home waypoint? (y/n): ").lower()
        if add_home == 'y':
            wpm.add_waypoint(0, 0, 5, color="blue")
    
    elif choice == "2":
        # Square path
        side = float(input("Enter square side length (meters, default=15): ") or "15")
        alt = float(input("Enter altitude (meters, default=8): ") or "8")
        waypoints = wpm.generate_square_path(side_length=side, altitude=alt)
    
    elif choice == "3":
        # Circular path
        radius = float(input("Enter circle radius (meters, default=10): ") or "10")
        num_pts = int(input("Number of waypoints (default=8): ") or "8")
        alt = float(input("Enter altitude (meters, default=8): ") or "8")
        waypoints = wpm.generate_circular_path(radius=radius, 
                                               num_waypoints=num_pts, 
                                               altitude=alt)
    
    elif choice == "4":
        # Random waypoints
        num = int(input("Number of waypoints (default=5): ") or "5")
        waypoints = wpm.generate_random_waypoints(num_waypoints=num)
    
    elif choice == "5":
        # Zigzag path
        length = float(input("Path length (meters, default=20): ") or "20")
        height_var = float(input("Height variation (meters, default=5): ") or "5")
        num_pts = int(input("Number of waypoints (default=6): ") or "6")
        waypoints = wpm.generate_zigzag_path(length=length, 
                                             height_variation=height_var, 
                                             num_waypoints=num_pts)
    
    elif choice == "6":
        # Simple recommended path
        print("\nCreating simple 4-waypoint test path...")
        wpm.add_waypoint(0, 0, 5, color="green")
        wpm.add_waypoint(10, 0, 7, color="red")
        wpm.add_waypoint(10, 10, 9, color="red")
        wpm.add_waypoint(0, 10, 7, color="red")
        wpm.add_waypoint(0, 0, 5, color="blue")
    
    else:
        print("Invalid choice!")
        return
    
    # Display waypoints
    print("\n" + "=" * 60)
    print("Waypoints created:")
    print("=" * 60)
    waypoints = wpm.get_waypoints()
    for i, wp in enumerate(waypoints):
        print(f"  {i+1}. ({wp[0]:6.2f}, {wp[1]:6.2f}, {-wp[2]:6.2f})  [NED: down={wp[2]:.2f}]")
    
    # Save to file
    print("\nSaving waypoints to 'waypoints.npy'...")
    np.save('waypoints.npy', waypoints)
    print("✓ Waypoints saved")
    
    # Optional: Draw path lines
    draw_lines = input("\nDraw path lines between waypoints? (y/n): ").lower()
    if draw_lines == 'y':
        for i in range(len(waypoints) - 1):
            wpm.spawn_path_line(i, i + 1)
            time.sleep(0.2)
        print("✓ Path lines drawn")
    
    print("\n" + "=" * 60)
    print("Setup complete! Launch Gazebo to see the waypoints.")
    print("Waypoints are saved in 'waypoints.npy'")
    print("=" * 60)

if __name__ == "__main__":
    import numpy as np
    main()
