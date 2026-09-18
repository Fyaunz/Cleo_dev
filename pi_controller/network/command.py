# pi-daemon/network/command.py
import zmq

class CommandListener:
    def __init__(self, context: zmq.Context, port: int = 5555):
        self.socket = context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        print(f" Command Listener bound to port {port}")

    def get_command(self):
        """Checks if a command arrived. Returns None if empty."""
        events = dict(self.poller.poll(0)) # 0ms timeout (non-blocking)
        if self.socket in events:
            return self.socket.recv_json()
        return None

    def send_reply(self, status: str, message="", msg_id=None):
        """Acknowledges the command (ZMQ REP sockets MUST reply).

        msg_id is echoed so the client can tell this reply apart from the reply
        to a command it has already given up on.
        """
        self.socket.send_json({"status": status, "message": message, "msg_id": msg_id})