#!/usr/bin/env python3
"""
Telemetry Diagnostic Script
Tests if PX4 telemetry is updating properly
"""

import asyncio
import numpy as np
from mavsdk import System
import time

async def test_telemetry():
    print("=" * 60)
    print("PX4 TELEMETRY DIAGNOSTIC")
    print("=" * 60)
    
    drone = System()
    print("\n1. Connecting to PX4...")
    await drone.connect(system_address="udp://:14540")
    
    print("   Waiting for connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("   ✓ Connected!")
            break
    
    print("\n2. Testing position telemetry...")
    positions = []
    
    async def collect_positions():
        count = 0
        async for pos_vel in drone.telemetry.position_velocity_ned():
            positions.append([
                pos_vel.position.north_m,
                pos_vel.position.east_m,
                pos_vel.position.down_m,
                time.time()
            ])
            count += 1
            if count >= 50:  # Collect 50 samples
                break
    
    # Start telemetry collection
    task = asyncio.create_task(collect_positions())
    
    # Wait for collection
    await asyncio.sleep(5)
    
    if len(positions) < 10:
        print(f"   ❌ PROBLEM: Only received {len(positions)} position updates in 5 seconds")
        print(f"   Expected: ~50 updates (10 Hz)")
        print(f"\n   This explains the takeoff issue!")
        print(f"   Solution: Restart PX4/Gazebo or check system load")
    else:
        # Calculate update rate
        if len(positions) >= 2:
            time_span = positions[-1][3] - positions[0][3]
            update_rate = len(positions) / time_span
            print(f"   ✓ Received {len(positions)} updates in {time_span:.1f}s")
            print(f"   ✓ Update rate: {update_rate:.1f} Hz")
            
            # Check for duplicate values
            unique_positions = len(set([tuple(p[:3]) for p in positions]))
            if unique_positions < len(positions) * 0.5:
                print(f"   ⚠️ WARNING: Many duplicate values detected")
                print(f"   Unique positions: {unique_positions}/{len(positions)}")
            else:
                print(f"   ✓ Position values updating normally")
            
            # Check for negative altitude (telemetry lag indicator)
            altitudes = [-p[2] for p in positions]
            if any(alt < -1.0 for alt in altitudes):
                print(f"   ⚠️ WARNING: Negative altitudes detected (telemetry lag)")
                print(f"   Min altitude: {min(altitudes):.2f}m")
        
    task.cancel()
    
    print("\n3. Testing attitude telemetry...")
    attitudes = []
    
    async def collect_attitudes():
        count = 0
        async for att in drone.telemetry.attitude_quaternion():
            attitudes.append([att.w, att.x, att.y, att.z, time.time()])
            count += 1
            if count >= 50:
                break
    
    task = asyncio.create_task(collect_attitudes())
    await asyncio.sleep(5)
    
    if len(attitudes) < 10:
        print(f"   ❌ PROBLEM: Only received {len(attitudes)} attitude updates")
    else:
        time_span = attitudes[-1][4] - attitudes[0][4]
        update_rate = len(attitudes) / time_span
        print(f"   ✓ Received {len(attitudes)} updates in {time_span:.1f}s")
        print(f"   ✓ Update rate: {update_rate:.1f} Hz")
    
    task.cancel()
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)
    
    if len(positions) >= 30 and len(attitudes) >= 30:
        print("✓ Telemetry is working properly")
        print("✓ Manual takeoff should work correctly")
    elif len(positions) < 10 or len(attitudes) < 10:
        print("❌ CRITICAL: Telemetry is too slow or not working")
        print("\nPossible causes:")
        print("  1. PX4/Gazebo simulation overloaded (high CPU usage)")
        print("  2. Network issue with MAVLink connection")
        print("  3. Gazebo/PX4 needs restart")
        print("\nRecommended actions:")
        print("  1. Check CPU usage: top or htop")
        print("  2. Restart simulation:")
        print("     pkill -9 px4 && pkill -9 gzserver && pkill -9 gzclient")
        print("     cd ~/PX4-Autopilot && make px4_sitl gazebo-classic")
        print("  3. If still slow, reduce simulation speed in Gazebo")
    else:
        print("⚠️ Telemetry is slow but might work")
        print("Manual takeoff may have issues")
    
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_telemetry())
