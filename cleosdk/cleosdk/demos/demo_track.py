import cv2
import time
import numpy as np
import os
from cleosdk import RobotHead

def main():
    head = RobotHead(ip="localhost")

    # The daemon keeps the camera off until a client asks for it.
    head.start_video()

    # Load the Face Detection AI
    haarcascade_path = os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(haarcascade_path)

    # Tuning Parameter: P-Gain
    # Converts "Pixel Error" into "Degrees of Movement"
    P_GAIN = 0.05 

    print("ZeroMQ Tracking Started.")
    print("Looking for faces... Press 'q' to quit.")

    try:
        while True:
            frame = head.get_video_frame()
            
            # asynchronous, might occasionally get None
            if frame is None:
                continue

            # Setup Dimensions
            height, width, _ = frame.shape
            center_x, center_y = int(width / 2), int(height / 2)

            # Process Image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

            # Center UI
            cv2.line(frame, (center_x - 15, center_y), (center_x + 15, center_y), (0, 255, 0), 2)
            cv2.line(frame, (center_x, center_y - 15), (center_x, center_y + 15), (0, 255, 0), 2)

            # Handle Detections
            if len(faces) > 0:
                # Target only the first face found
                (x, y, w, h) = faces[0]
                face_center_x = int(x + w / 2)
                face_center_y = int(y + h / 2)

                # Target UI
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.circle(frame, (face_center_x, face_center_y), 4, (0, 0, 255), -1)

                # Distance from center -> Error
                error_x = face_center_x - center_x
                error_y = center_y - face_center_y

                # Simple conversion to Degree Delta
                delta_yaw = error_x * P_GAIN
                delta_pitch = error_y * P_GAIN

                # Draw Vector Targeting Line
                cv2.line(frame, (center_x, center_y), (face_center_x, face_center_y), (0, 255, 255), 1)

                # Display Delta
                status_text = f"Yaw: {delta_yaw:+.1f} | Pitch: {delta_pitch:+.1f}"
                cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Show the Result
            cv2.imshow("Cleo SDK - ZeroMQ Face Tracking", frame)

            # Break loop on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopping Demo...")
    finally:
        cv2.destroyAllWindows()
        head.disconnect()

if __name__ == "__main__":
    main()