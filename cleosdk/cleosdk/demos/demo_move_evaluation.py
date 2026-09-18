# demo_dual_mode.py
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from cleosdk import RobotHead

def main():
    # Connect to the Pi (Change "localhost" to your Pi's IP address if running on separate machines)
    head = RobotHead(ip="localhost")

    try:
        print("\n--- Initialization ---")
        print("Centering Head...")
        head.center(time_ms=1000)
        time.sleep(1.5)
        
        # Look Up and Right
        #head.move_head(yaw=60, pitch=0, roll=0, time_ms=1500)
        #time.sleep(2.0)

        print("\n--- Workspace evaluation ---")

        print("\n--- Pitch ---")
        print("-> Ruckig")
        head.move_head_ruckig(
            yaw=0, pitch=-80, roll=0, 
            v_max=100, 
            a_max=400, 
            j_max=400,
            duration=4,
        )

        print("-> Ruckig")
        head.move_head_ruckig(
            yaw=0, pitch=80, roll=0, 
            v_max=100, 
            a_max=400, 
            j_max=400,
            duration=4,
        )

        print("-> Ruckig")
        head.move_head_ruckig(
            yaw=0, pitch=0, roll=0, 
            v_max=100, 
            a_max=400, 
            j_max=400,
            duration=4,
        )

        print("\n--- Roll ---")
        head.move_head_ruckig(
            yaw=0, pitch=0, roll=-50, 
            v_max=100, 
            a_max=400, 
            j_max=400,
            duration=4,
        )

        print("-> Ruckig")
        head.move_head_ruckig(
            yaw=0, pitch=0, roll=50, 
            v_max=100, 
            a_max=400, 
            j_max=400,
            duration=4,
        )


        print("-> Ruckig")
        head.move_head_ruckig(
            yaw=0, pitch=0, roll=0, 
            v_max=100, 
            a_max=400, 
            j_max=400,
            duration=2,
        )


    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
        
    finally:
        print("\nCleaning up...")
        time.sleep(2.0)
        head.disconnect()
        print("Done.")

if __name__ == "__main__":
    main()