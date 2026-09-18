# demo_dual_mode.py
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from cleosdk import RobotHead

def main():
    # Connect to the Pi (Change "localhost" to your Pi's IP address if running on separate machines)
    head = RobotHead(ip="localhost")

    try:
        print("\n--- 1. Initialization ---")
        print("Centering Head...")
        head.center(time_ms=2000)
        time.sleep(2.5)

        print("\n--- 3. Hardware Mode (Time-Based Profile) ---")
        
        # Look Up and Right
        head.move_head(yaw=90, pitch=10, roll=10, time_ms=1500)
        time.sleep(2.0)
        
        # Look Down and Left with a slight head tilt (roll)
        head.move_head(yaw=-45, pitch=-10, roll=-10, time_ms=1500)
        time.sleep(2.0)


        print("\n--- 4. Software Mode (Ruckig OTG) ---")
        print("Ruckig generates these waypoints at 100Hz.")
        
        # Return to center using Ruckig with a HIGH jerk (Snappy, robotic movement)
        print("-> High Jerk")
        head.move_head_ruckig(
            yaw=45, pitch=10, roll=10, 
            v_max=100,  # Fast top speed
            a_max=400, # Fast acceleration
            j_max=4000  # High Jerk (Snappy start/stop)
        )


        # Look around using Ruckig with a LOW jerk (Smooth, organic/biological movement)
        print("-> Low Jerk")
        head.move_head_ruckig(
            yaw=-45, pitch=0, roll=15, 
            v_max=100,  # Medium top speed
            a_max=400,  # Gradual acceleration
            j_max=400   # LOW Jerk (Very slow, easing start/stop)
        )


        print("\n--- 4. The Ruckig Override Advantage ---")
        print("Starting a 5-second movement...")
        head.move_head_ruckig(yaw=-90, pitch=20, roll=0, v_max=200, a_max=400, j_max=400, duration=None, wait=False)
        
        time.sleep(0.5)
        
        print("Interrupting mid-movement! Ruckig blends this perfectly without violently jerking.")
        head.move_head_ruckig(yaw=0, pitch=0, roll=0, v_max=3000, a_max=8000, j_max=30000, duration=None, wait=False)

    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
        
    finally:
        print("\nCleaning up...")
        head.center(time_ms=1500)
        time.sleep(2.0)
        head.disconnect()
        print("Done.")

if __name__ == "__main__":
    main()