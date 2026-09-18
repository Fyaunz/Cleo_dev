# pi-daemon/network/session.py
import time

class SessionManager:
    def __init__(self, timeout_sec=3.0):
        self.active_client_id = None
        self.last_seen = 0
        self.timeout = timeout_sec

    def check_access(self, client_id: str) -> bool:
        """Determines if a client is allowed to execute commands."""
        # If no client_id was provided by an outdated script, reject it safely
        if not client_id:
            return False

        now = time.time()

        # 1. Clear expired locks (Client crashed or lost Wi-Fi)
        if self.active_client_id and (now - self.last_seen > self.timeout):
            print(f"Client {self.active_client_id[-4:]} timed out. Lock released.")
            self.active_client_id = None

        # 2. Claim the lock if it is currently available
        if self.active_client_id is None:
            self.active_client_id = client_id
            print(f"Lock acquired by Client {client_id[-4:]}")

        # 3. Verify ownership
        if self.active_client_id == client_id:
            self.last_seen = now # Refresh the lease
            return True
            
        return False

    def is_allowed(self, client_id: str) -> bool:
        """
        Is this client the lock holder? Read-only, unlike check_access.
        """
        if not client_id:
            return False
        if self.active_client_id is None:
            return True
        if time.time() - self.last_seen > self.timeout:
            return True
        return self.active_client_id == client_id

    def release(self, client_id: str):
        """Allows a client to gracefully release the robot when finished."""
        if self.active_client_id == client_id:
            self.active_client_id = None
            print(f"Lock cleanly released by Client {client_id[-4:]}")