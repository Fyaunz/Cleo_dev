# demo_video.py
import cv2
from cleosdk import RobotHead

def main():
    head = RobotHead(ip="localhost")

    # The daemon keeps the camera off until a client asks for it.
    head.start_video()
    print("\nPress 'q' in the video window to quit!")

    try:
        while True:
            # 1. Get the latest data
            frame = head.get_video_frame()
            telemetry = head.get_telemetry()

            # 2. Render the GUI if we have a frame
            if frame is not None:
                
                # Overlay Telemetry Text
                if telemetry:
                    pan = telemetry['servos'].get('joint_1', 'N/A')
                    cv2.putText(frame, f"Pan (Yaw): {pan} ticks", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Show the video window
                cv2.imshow("Robot Head Vision", frame)

            # 3. Check for 'q' to quit (wait 1ms)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopping viewer...")
    finally:
        head.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()