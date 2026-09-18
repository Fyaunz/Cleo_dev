# pi-daemon/network/dedup.py
from collections import OrderedDict
from typing import Any, Optional

# What send_reply() puts on the wire: a status plus a JSON-serialisable payload.
Reply = tuple[str, Any]


class ReplyCache:
    """Remembers the reply sent for every msg_id so a resent command is answered
    from the cache instead of being executed a second time.

    The SDK keeps one msg_id across all retries of the same command, so a hit
    means the original reply was lost on the way back to the client, not
    that the client wants the movement repeated.
    """

    def __init__(self, max_entries: int = 256):
        self.max_entries = max_entries
        self._entries: OrderedDict[str, Reply] = OrderedDict()

    def get(self, msg_id: Optional[str]) -> Optional[Reply]:
        """Returns the (status, message) already sent for msg_id, or None."""
        if msg_id is None or msg_id not in self._entries:
            return None
        self._entries.move_to_end(msg_id)
        return self._entries[msg_id]

    def put(self, msg_id: Optional[str], status: str, message: Any = "") -> None:
        if msg_id is None:
            return
        self._entries[msg_id] = (status, message)
        self._entries.move_to_end(msg_id)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
