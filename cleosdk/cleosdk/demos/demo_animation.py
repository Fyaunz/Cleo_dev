# demo/demo_playback.py
from cleosdk import RobotHead
import time

def main():
    head = RobotHead(ip="localhost")
    
    try:

        head.center(time_ms=1500)
        print("\nPlaying back animation from 'animation.json'")
        time.sleep(2.0)
        for i in range(4):
            head.play_animation("animation.json")
        
    except KeyboardInterrupt:
        print("\nPlayback interrupted by user.")
    finally:
        head.center(time_ms=1500)
        head.disconnect()
        print("Done.")

if __name__ == "__main__":
    main()