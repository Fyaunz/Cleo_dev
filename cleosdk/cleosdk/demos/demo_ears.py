# demo_ears.py -- the two ear end effectors, on both motion paths.
import time
from cleosdk import RobotHead


def main():
    head = RobotHead(ip="localhost")

    try:
        print("\n--- 1. Initialization ---")
        head.center(time_ms=1500)
        head.center_ears(time_ms=1000)
        time.sleep(2.0)

        print("\n--- 2. Hardware Mode (Time-Based Profile) ---")
        # Positive is the same physical direction on both sides: the mounting is
        # mirrored and the SDK hides that.
        print("-> Both ears forward")
        head.move_ears(ear_left=40, ear_right=40, time_ms=800)
        time.sleep(1.2)

        print("-> One ear alone (the other holds its pose)")
        head.move_ears(ear_left=-40, time_ms=800)
        time.sleep(1.2)

        print("\n--- 3. Software Mode (Ruckig OTG) ---")
        print("-> Snappy flick")
        head.move_ears_ruckig(ear_left=60, ear_right=60, v_max=400, a_max=3000, j_max=20000)

        print("-> Slow, organic settle")
        head.move_ears_ruckig(ear_left=0, ear_right=0, v_max=60, a_max=200, j_max=200)

        print("\n--- 4. Independence from the head ---")
        print("Starting a slow head turn, then flicking the ears through it.")
        head.move_head_ruckig(yaw=60, pitch=0, roll=0,
                              v_max=20, a_max=100, j_max=100, wait=False)
        time.sleep(0.3)

        # Two separate Ruckig generators on the Pi, so neither retimes the other:
        # this returns when the ears land, with the neck still turning.
        head.move_ears_ruckig(ear_left=50, ear_right=-50, v_max=400, a_max=3000, j_max=20000)
        print(f"Ears done; head still moving: {head.is_moving()}")

        print("\n--- 5. One packet for head and ears ---")
        head.move_head(yaw=0, pitch=10, roll=0, ear_left=30, ear_right=30, time_ms=1000)
        time.sleep(1.5)

    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")

    finally:
        print("\nCleaning up...")
        head.center_ears(time_ms=800)
        head.center(time_ms=1500)
        time.sleep(2.0)
        head.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
