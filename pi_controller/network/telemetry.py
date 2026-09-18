# pi-daemon/network/telemetry.py
import zmq

class TelemetryPublisher:
    def __init__(self, context: zmq.Context, port: int = 5556):
        self.socket = context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        print(f" Telemetry Publisher bound to port {port}")

    def publish(self, data: dict):
        """Broadcasts the robot's state to anyone listening."""
        self.socket.send_json(data)