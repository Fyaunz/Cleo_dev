# pi-daemon/main.py
import os
import signal
import threading
import time
import zmq
from hal.servo import ServoController
from hal.controller import KinematicController
from hal.playback import PlaybackEngine
from network.session import SessionManager
from network.command import CommandListener
from network.dedup import ReplyCache
from network.telemetry import TelemetryPublisher
from network.video import VideoStreamer
from network.audio import AudioStreamer
from network.router import route_command

# The USB serial adapter is named differently on every machine: /dev/cu.usbserial-* on macOS, /dev/ttyUSB* on the Pi
DEFAULT_SERIAL_DEVICE = '/dev/cu.usbserial-FTAAMM58'

def _audio_device():
    """CLEO_AUDIO_DEVICE as an index if it is one, else a name fragment, else None."""
    spec = os.environ.get('CLEO_AUDIO_DEVICE')
    if not spec:
        return None
    return int(spec) if spec.lstrip('-').isdigit() else spec

def _install_signal_handlers(stop_event):
    """
    Turn SIGTERM and SIGINT into a request to leave the main loop.
    """
    def _request_stop(signum, _frame):
        print(f"\nReceived {signal.Signals(signum).name}, shutting down...")
        stop_event.set()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

def main():
    context = zmq.Context()

    stop_event = threading.Event()
    _install_signal_handlers(stop_event)

    device_name = os.environ.get('CLEO_SERIAL', DEFAULT_SERIAL_DEVICE)
    # Order matters: [m1, m2, m3, ear_left, ear_right]
    servo_controller = ServoController(device_name=device_name, servo_ids=[2, 3, 1, 4, 5])

    session_manager = SessionManager(timeout_sec=30.0)
    reply_cache = ReplyCache()

    cmd_listener = CommandListener(context, port=5555)
    telemetry_pub = TelemetryPublisher(context, port=5556)

    kinematic_controller = KinematicController(servo_controller=servo_controller, hz=100)
    kinematic_controller.start()

    playback = PlaybackEngine(servo_controller=servo_controller,
                              controller=kinematic_controller)

    video_pub = VideoStreamer(context, port=5557, camera_index=0)
    audio_pub = AudioStreamer(context, mic_port=5558, speaker_port=5559,
                              device=_audio_device(),
                              access_check=session_manager.is_allowed)

    print(f"Cleo Daemon Running (Ruckig Enabled) on {device_name}.")
    TELEMETRY_RATE = 1.0 / 20 
    last_telemetry_time = time.time()

    try:
        while not stop_event.is_set():
            command_msg = cmd_listener.get_command()
            if command_msg:
                route_command(command_msg=command_msg,
                              cmd_listener=cmd_listener,
                              session_manager=session_manager,
                              servo_controller=servo_controller,
                              controller=kinematic_controller,
                              playback_engine=playback,
                              video_streamer=video_pub,
                              reply_cache=reply_cache,
                              audio_streamer=audio_pub)

            current_time = time.time()
            if current_time - last_telemetry_time >= TELEMETRY_RATE:
                current_state = kinematic_controller.get_telemetry()
                telemetry_pub.publish(current_state)
                last_telemetry_time = current_time

            time.sleep(0.005)

    finally:
        playback.stop()
        kinematic_controller.stop()
        servo_controller.shutdown()
        video_pub.stop()
        audio_pub.stop()
        # context.term()

if __name__ == "__main__":
    main()