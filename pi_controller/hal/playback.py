# pi-daemon/hal/playback.py
import threading
import time

class PlaybackEngine:
    def __init__(self, servo_controller, controller=None):
        """
        Manages the local 100Hz playback thread to stream frames
        directly to the KinematicController.

        `controller` is the KinematicController, held only to tell it where each
        frame put the robot. It never plans playback. Optional so a head can be
        driven by animations alone.
        """
        self.servo_controller = servo_controller
        self.controller = controller
        self.active = False
        self.thread = None

    def play(self, frames):
        """Spawns the background thread to play the animation."""
        # Stop any currently running animation safely
        self.stop()
        
        # Start the new thread
        self.active = True
        self.thread = threading.Thread(
            target=self._worker, 
            args=(frames,),
            daemon=True
        )
        self.thread.start()

    def stop(self):
        """Halts the animation and waits for the thread to close."""
        self.active = False
        if self.thread and self.thread.is_alive():
            self.thread.join()

    def _sync(self, frame):
        """Hands the pose this frame just commanded to the kinematic controller."""
        if self.controller is None:
            return
        self.controller.sync_state(
            yaw=frame['yaw'], pitch=frame['pitch'], roll=frame['roll'],
            ear_left=frame.get('ear_left'), ear_right=frame.get('ear_right')
        )

    def _worker(self, frames):
        """
        The high-speed local loop running entirely in the Pi's RAM.
        """
        frame_interval = 0.010 # 100Hz clock
        print(f"Local Playback Started: {len(frames)} frames.")

        # 1. Ease to the starting position
        start_frame = frames[0]
        self.servo_controller.go_to(
            yaw=start_frame['yaw'], pitch=start_frame['pitch'], roll=start_frame['roll'],
            t_total=100, t_accel=50,
            # Ear channels are optional in an exported animation; a frame without
            # them leaves the ears where they are rather than snapping to neutral.
            ear_left=start_frame.get('ear_left'), ear_right=start_frame.get('ear_right')
        )
        self._sync(start_frame)
        time.sleep(1.0)

        # 2. The 100Hz streaming loop
        for frame in frames:
            if not self.active:
                print("Playback aborted by user.")
                break

            start_tick = time.perf_counter()

            # Blast the target to the Kinematics Thread
            self.servo_controller.go_to(
                yaw=frame['yaw'], pitch=frame['pitch'], roll=frame['roll'],
                t_total=0, t_accel=0,
                ear_left=frame.get('ear_left'), ear_right=frame.get('ear_right')
            )
            self._sync(frame)

            elapsed = time.perf_counter() - start_tick
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.active = False
        print("Local Playback Complete.")