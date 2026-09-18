# pi-daemon/network/video.py
"""JPEG frames from the head's camera, published to whoever asks for them.

Two sources, because the Pi's camera and a development laptop's webcam are not
reachable the same way:

    rpicam   `rpicam-vid --codec mjpeg` in a subprocess, with this process
             reading finished JPEGs off its stdout. The Pi path.
    cv2      cv2.VideoCapture, encoding each frame with cv2.imencode. USB
             webcams, and macOS during development.
"""
import os
import select
import shutil
import subprocess
import threading
import time
from collections import deque

import zmq

try:
    import cv2
except ImportError:
    # Optional, and only for the cv2 source
    cv2 = None

# Set low to reduce network load. Can be increased
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_RATE = 30
JPEG_QUALITY = 80

RPICAM_BINARIES = ("rpicam-vid", "libcamera-vid")

# Camera Module mounted upside down inside Cleo
MOUNT_ROTATION = 180
VALID_ROTATIONS = (0, 180)

FIRST_FRAME_TIMEOUT = 5.0

# One read syscall's worth.
READ_SIZE = 65536

#no growing the buffer infinitely
MAX_FRAME_BYTES = 4 * 1024 * 1024

_SOI = b"\xff\xd8"  # start of image
_EOI = b"\xff\xd9"  # end of image


def find_rpicam() -> str | None:
    """
    Path to the rpicam-vid binary, or None if this machine has no such thing.
    """
    override = os.environ.get("CLEO_RPICAM_BIN")
    if override:
        return override

    for name in RPICAM_BINARIES:
        path = shutil.which(name)
        if path:
            return path
    return None


def _parse_rotation(spec) -> int | None:
    """
    A rotation from a constructor argument or CLEO_CAMERA_ROTATION.
    """
    if spec is None or spec == "":
        return None

    try:
        rotation = int(spec)
    except (TypeError, ValueError):
        raise RuntimeError(f"Camera rotation {spec!r} is not a number.") from None

    if rotation not in VALID_ROTATIONS:
        raise RuntimeError(
            f"Camera rotation {rotation} is not supported, expected "
            f"{' or '.join(str(r) for r in VALID_ROTATIONS)}.")
    return rotation


class _MjpegBuffer:
    """
    Reassembles the JPEGs in an MJPEG byte stream.
    """

    def __init__(self, max_frame_bytes: int = MAX_FRAME_BYTES):
        self._buf = bytearray()
        self._max_frame_bytes = max_frame_bytes

    def feed(self, chunk: bytes) -> None:
        self._buf += chunk

    def take(self) -> bytes | None:
        """The next complete frame, or None if one has not arrived yet."""
        start = self._buf.find(_SOI)
        if start < 0:
            del self._buf[:max(0, len(self._buf) - 1)]
            return None

        end = self._buf.find(_EOI, start + 2)
        if end < 0:
            # Keep the buffer anchored at the SOI so the partial frame survives
            # until the rest of it turns up
            del self._buf[:start]
            if len(self._buf) > self._max_frame_bytes:
                print(f"Video: no JPEG end marker in {len(self._buf)} bytes, resyncing.")
                self._buf.clear()
            return None

        frame = bytes(self._buf[start:end + 2])
        del self._buf[:end + 2]
        return frame


class _RpicamSource:
    """The CSI camera, via rpicam-vid writing MJPEG to a pipe."""

    name = "rpicam"

    def __init__(self, binary: str, camera_index: int = 0, rotation: int = MOUNT_ROTATION):
        self.binary = binary
        self.camera_index = camera_index
        self.rotation = rotation
        self._proc = None
        self._buffer = _MjpegBuffer()
        self._stderr_tail = deque(maxlen=10)
        self._stderr_thread = None
        self._first_frame = None

    def _argv(self) -> list[str]:
        # Rotating here is easier.
        rotation = ["--rotation", str(self.rotation)] if self.rotation else []
        return [
            self.binary,
            "--camera", str(self.camera_index),
            *rotation,
            "--codec", "mjpeg",
            "--width", str(FRAME_WIDTH),
            "--height", str(FRAME_HEIGHT),
            "--framerate", str(FRAME_RATE),
            "--quality", str(JPEG_QUALITY),
            # 0 means run until it is killed
            "--timeout", "0",
            "--nopreview",
            # Push each frame out as it is encoded
            "--flush",
            "--output", "-",
        ]

    def open(self) -> None:
        """
        Spawns the camera process and waits for it to produce a frame.
        """
        # bufsize=0 keeps stdout a raw pipe
        self._proc = subprocess.Popen(
            self._argv(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        # much on stderr, keeps only tail for error messagesxs
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, args=(self._proc.stderr,), daemon=True)
        self._stderr_thread.start()

        deadline = time.monotonic() + FIRST_FRAME_TIMEOUT
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.close()
                raise RuntimeError(
                    f"{os.path.basename(self.binary)} produced no frame in "
                    f"{FIRST_FRAME_TIMEOUT:.0f}s{self._stderr_hint()}")

            chunk = self._read(timeout=remaining)
            if chunk is None:
                continue
            if not chunk:
                returncode = self._proc.poll()
                self.close()
                raise RuntimeError(
                    f"{os.path.basename(self.binary)} exited "
                    f"({returncode}) without sending a frame{self._stderr_hint()}")

            self._buffer.feed(chunk)
            frame = self._buffer.take()
            if frame is not None:
                self._first_frame = frame
                return

    def frames(self):
        if self._first_frame is not None:
            frame, self._first_frame = self._first_frame, None
            yield frame

        while True:
            chunk = self._read()
            if not chunk:
                # EOF: the camera process exited, or interrupt() killed it.
                return
            self._buffer.feed(chunk)
            while (frame := self._buffer.take()) is not None:
                yield frame

    def interrupt(self) -> None:
        """
        Unblocks a grabber thread parked in read().
        """
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return

        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)

        if self._stderr_thread:
            self._stderr_thread.join(timeout=1.0)

        for pipe in (proc.stdout, proc.stderr):
            if pipe:
                pipe.close()

    def _read(self, timeout: float | None = None) -> bytes | None:
        """Up to READ_SIZE bytes: b'' at EOF, None if the timeout expired first."""
        if self._proc is None:
            return b""  # close() got here first; EOF is the honest answer.

        stdout = self._proc.stdout
        if timeout is not None:
            ready, _, _ = select.select([stdout], [], [], timeout)
            if not ready:
                return None
        return stdout.read(READ_SIZE)

    def _drain_stderr(self, stderr) -> None:
        try:
            for line in stderr:
                text = line.decode(errors="replace").strip()
                if text:
                    self._stderr_tail.append(text)
        except (ValueError, OSError):
            # close() pulled the pipe out from under us. Nothing to report.
            pass

    def _stderr_hint(self) -> str:
        """
        Whatever the camera output on its way out, for the error message.
        """
        if self._stderr_thread:
            self._stderr_thread.join(timeout=1.0)
        if not self._stderr_tail:
            return ""
        return ": " + " | ".join(self._stderr_tail)


class _Cv2Source:
    """A V4L2 / AVFoundation camera, via OpenCV."""

    name = "cv2"

    def __init__(self, camera_index: int = 0, rotation: int = 0):
        self.camera_index = camera_index
        self.rotation = rotation
        self.cap = None

    def open(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to open camera index {self.camera_index}")

        # Lower resolution to ensure low-latency network streaming
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap = cap

    def frames(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                if self.rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)

                ok, buffer = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    yield buffer.tobytes()

            # Limit to ~30 FPS to prevent overwhelming the network. The rpicam
            # source needs no equivalent: --framerate paces the pipe for us.
            time.sleep(1 / FRAME_RATE)

    def interrupt(self) -> None:
        """
        Nothing to do: cap.read() returns within a frame time on its own.
        """

    def close(self) -> None:
        cap, self.cap = self.cap, None
        if cap is not None:
            cap.release()


class VideoStreamer:
    """
    Publishes JPEG frames on a PUB socket, but only while a client wants them.
    """

    def __init__(self, context: zmq.Context, port: int = 5557, camera_index: int = 0,
                 backend: str | None = None, rotation: int | None = None):
        self.socket = context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        self.camera_index = camera_index
        # "auto" resolves at start()
        self.backend = backend or os.environ.get("CLEO_CAMERA", "auto")
        self.rotation = _parse_rotation(
            rotation if rotation is not None else os.environ.get("CLEO_CAMERA_ROTATION"))
        self.running = False
        self.thread = None
        self.source = None
        self._lock = threading.Lock()
        print(f"Video Publisher bound to port {port} "
              f"(Camera: {camera_index}, backend: {self.backend}, idle)")

    @property
    def is_streaming(self) -> bool:
        return self.running

    def _make_source(self):
        backend = self.backend
        if backend == "auto":
            backend = "rpicam" if find_rpicam() else "cv2"

        if backend == "rpicam":
            binary = find_rpicam()
            if binary is None:
                raise RuntimeError(
                    "No rpicam-vid or libcamera-vid on PATH. Install rpicam-apps, "
                    "or set CLEO_CAMERA=cv2 for a USB camera.")
            return _RpicamSource(
                binary, self.camera_index,
                MOUNT_ROTATION if self.rotation is None else self.rotation)

        if backend == "cv2":
            if cv2 is None:
                raise RuntimeError("CLEO_CAMERA=cv2 but opencv-python is not installed.")
            return _Cv2Source(self.camera_index, 0 if self.rotation is None else self.rotation)

        raise RuntimeError(f"Unknown camera backend {backend!r}, expected rpicam, cv2 or auto.")

    def start(self) -> bool:
        """Opens the camera and starts the grabber thread.

        Returns `True` if this call started the stream, `False` if it was already
        running, and raises if the camera will not open
        """
        with self._lock:
            if self.running:
                return False

            source = self._make_source()
            source.open()

            self.source = source
            self.running = True
            self.thread = threading.Thread(target=self._stream_loop, args=(source,), daemon=True)
            self.thread.start()
            print(f"Video stream started ({source.name}, rotation {source.rotation}).")
            return True

    def _stream_loop(self, source):
        try:
            for frame in source.frames():
                if not self.running:
                    break
                self.socket.send(frame)
        except Exception as exc:  # noqa: BLE001 -- a dying camera must not be silent
            print(f"Video stream ended on error: {exc}")
        finally:
            # Release even if the loop dies on an exception, or the next start()
            # finds the device still busy and the camera stays dark
            source.close()
            # A source that ended by itself (camera unplugged, process killed)
            # leaves the streamer idle rather than claiming to stream, so the
            # next start_video opens a fresh one instead of returning "already
            # running" and publishing nothing.
            self.running = False

    def stop(self) -> bool:
        """Stops the grabber and releases the camera.

        Returns `True` if this call stopped a running stream, `False` if there was
        nothing to stop.
        """
        with self._lock:
            if not self.running:
                return False

            self.running = False
            source, self.source = self.source, None
            thread, self.thread = self.thread, None
            if source:
                source.interrupt()
            if thread:
                thread.join(timeout=2.0)
            print("Video stream stopped.")
            return True
