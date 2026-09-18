# pi-daemon/network/audio.py
"""Full-duplex audio between the head and a client, on two dedicated ports.

The ReSpeaker Flex is a plain USB audio device -- a mic array in, a speaker out --
so the two halves are independent streams and get a socket each:

    mic     PUB  on 5558   
    speaker PULL on 5559   

Both are self-describing: every message is a two-part [header, pcm] where the
JSON header carries rate/channels/width.
"""
import json
import threading
import time
from typing import Any

import numpy as np
import pyaudio
import zmq

try:
    import usb.core
    import usb.util
except ImportError:
    # pyusb is optional. Without it DOA is unavailable
    usb = None

# int16 is what PortAudio, Vosk and the wave module speak natively
SAMPLE_WIDTH = 2

# Tried in order when no device is named.
DEVICE_HINTS = ("respeaker", "seeed", "xvf", "xmos")


def list_devices() -> list[dict]:
    """Every audio device PortAudio can see, printed and returned.

        python -c "import sys; sys.path.insert(0, 'pi_controller'); \\
                   from network.audio import list_devices; list_devices()"
    """
    pa = pyaudio.PyAudio()
    try:
        devices = [pa.get_device_info_by_index(i) for i in range(pa.get_device_count())]
    finally:
        pa.terminate()

    for info in devices:
        print(f"[{info['index']:>2}] {info['name']}  "
              f"in={info['maxInputChannels']} out={info['maxOutputChannels']} "
              f"rate={int(info['defaultSampleRate'])}")
    return devices


class DoaReader:
    """Direction of arrival, read from the ReSpeaker over USB.
    Only exposed through a vendor usb control interface, read-only.
    """

    VENDOR_ID = 0x2886  # Seeed Studio

    # (resource id, command id, payload length) from Seeed's control map
    DOA_VALUE = (20, 18, 4)
    VERSION = (48, 0, 3)

    # The vendor tool uses 100s.
    TIMEOUT_MS = 200

    # A control transfer costs about a millisecond, and the DSP does not update faster
    MIN_INTERVAL = 0.05

    def __init__(self, vendor_id: int = VENDOR_ID):
        self.vendor_id = vendor_id
        # pyusb hands back a dynamically built Device; annotating it keeps the
        # type checker from guessing at attributes that only exist at runtime.
        self._dev: Any = None
        self._lock = threading.Lock()
        self._cached = None
        self._cached_at = 0.0
        self._failures = 0
        self._warned = False

    @property
    def available(self) -> bool:
        return self._dev is not None

    def open(self) -> bool:
        """Finds the board. False -- never raises -- if there is nothing to find.

        DOA is an extra: a missing pyusb, a head with no ReSpeaker, or a Pi
        where the user cannot reach the USB device must all leave the audio
        working and just report no direction.
        """
        if usb is None:
            print("DOA unavailable: pyusb is not installed.")
            return False

        try:
            devices = list(usb.core.find(find_all=True, idVendor=self.vendor_id) or [])
        except Exception as e:
            print(f"DOA unavailable: cannot enumerate USB ({e}).")
            return False

        if not devices:
            print(f"DOA unavailable: no device with vendor id {self.vendor_id:#06x}.")
            return False

        # Lowest product id first, matching the vendor tool: the board enumerates
        # under different ones across firmware revisions.
        devices.sort(key=lambda d: getattr(d, "idProduct", 0))
        self._dev = devices[0]

        version = self._read(*self.VERSION)
        if version is None:
            print("DOA unavailable: the device did not answer a version read.")
            self._dev = None
            return False

        vid = getattr(self._dev, "idVendor", 0)
        pid = getattr(self._dev, "idProduct", 0)
        print(f"DOA: ReSpeaker {vid:#06x}:{pid:#06x}, "
              f"firmware {'.'.join(str(b) for b in version)}")
        return True

    def close(self) -> None:
        with self._lock:
            if self._dev is not None and usb is not None:
                usb.util.dispose_resources(self._dev)
            self._dev = None

    def read(self) -> dict | None:
        """The newest direction, or None if this head cannot report one.

        Returns `{"angle": degrees, "speech": bool}`. 
        
        **The angle only moves while speech is true!**
        """
        if self._dev is None:
            return None

        now = time.monotonic()
        if self._cached is not None and now - self._cached_at < self.MIN_INTERVAL:
            return self._cached

        payload = self._read(*self.DOA_VALUE)
        if payload is None:
            return self._cached

        # [angle low, angle high, speech detected]
        self._cached = {"angle": payload[0] | (payload[1] << 8),
                        "speech": bool(payload[2])}
        self._cached_at = now
        return self._cached

    def _read(self, resid: int, command: int, length: int):
        """One control read, or None. Returns the payload without the status byte."""
        with self._lock:
            if self._dev is None or usb is None:
                return None
            try:
                response = self._dev.ctrl_transfer(
                    usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                    0, 0x80 | command, resid, length + 1, self.TIMEOUT_MS)
            except Exception as e:
                self._note_failure(e)
                return None

            if len(response) != length + 1 or response[0] != 0:
                self._note_failure(f"status {response[0] if len(response) else 'empty'}")
                return None

            self._failures = 0
            return list(response[1:])

    def _note_failure(self, reason) -> None:
        """Gives up after a run of failures rather than logging once per chunk."""
        self._failures += 1
        if self._failures >= 20:
            if not self._warned:
                self._warned = True
                print(f"DOA reads keep failing ({reason}); giving up on direction.")
            self._dev = None
        elif not self._warned and self._failures == 1:
            print(f"DOA read failed ({reason}); will keep trying.")


def _resolve_device(pa, spec, kind: str):
    """Turns None / an index / a name fragment into a concrete (index, info).

    None means "the ReSpeaker if it is plugged in, otherwise whatever PortAudio
    calls the default"
    """
    channel_key = "maxInputChannels" if kind == "input" else "maxOutputChannels"

    if isinstance(spec, int):
        return spec, pa.get_device_info_by_index(spec)

    candidates = [pa.get_device_info_by_index(i) for i in range(pa.get_device_count())]
    usable = [info for info in candidates if int(info.get(channel_key, 0)) > 0]

    hints = (spec.lower(),) if isinstance(spec, str) else DEVICE_HINTS
    for hint in hints:
        for info in usable:
            if hint in str(info.get("name", "")).lower():
                return int(info["index"]), info

    if isinstance(spec, str):
        raise RuntimeError(
            f"No {kind} device matching '{spec}'. Available: "
            + ", ".join(f"{int(i['index'])}:{i['name']}" for i in usable))

    if kind == "input":
        info = pa.get_default_input_device_info()
    else:
        info = pa.get_default_output_device_info()
    return int(info["index"]), info


class AudioStreamer:
    """
    Publishes microphone audio and plays back what a client sends.
    """

    def __init__(self, context: zmq.Context, mic_port: int = 5558, speaker_port: int = 5559,
                 device=None, mic_rate: int = 16000, mic_channels: int = 1,
                 speaker_rate=None, speaker_channels: int = 1,
                 chunk_frames: int = 1024, access_check=None, doa: bool = True):
        """
        speaker_rate=None opens the device at its own default rate which is preferable.
        """
        self.mic_socket = context.socket(zmq.PUB)
        self.mic_socket.bind(f"tcp://*:{mic_port}")

        self.speaker_socket = context.socket(zmq.PULL)
        # PUSH blocks rather than drops when this fills
        self.speaker_socket.setsockopt(zmq.RCVHWM, 2000)
        self.speaker_socket.bind(f"tcp://*:{speaker_port}")

        self._speaker_poller = zmq.Poller()
        self._speaker_poller.register(self.speaker_socket, zmq.POLLIN)

        self.device = device
        self.mic_rate = mic_rate
        self.mic_channels = mic_channels
        self.speaker_rate = speaker_rate
        self.speaker_channels = speaker_channels
        self.chunk_frames = chunk_frames
        self.access_check = access_check

        # Opened lazily on the first start()
        self.doa_reader = DoaReader() if doa else None
        self._doa_opened = False
        self._doa_lock = threading.Lock()

        self._pa = None
        self._mic_stream = None
        self._speaker_stream = None
        self._mic_capture_channels = mic_channels
        self._mic_running = False
        self._speaker_running = False
        self._mic_thread = None
        self._speaker_thread = None
        self._flush_requested = threading.Event()
        self._seq = 0
        self._last_reject_log = 0.0

        self._resample_phase = 0.0
        self._resample_tail = None
        self._resample_src = None

        self._lock = threading.Lock()
        print(f"Audio bound to ports {mic_port} (mic) and {speaker_port} (speaker), idle")

    @property
    def is_streaming(self) -> bool:
        return self._mic_running or self._speaker_running

    @property
    def mic_running(self) -> bool:
        return self._mic_running

    @property
    def speaker_running(self) -> bool:
        return self._speaker_running

    @property
    def doa_available(self) -> bool:
        return self.doa_reader is not None and self.doa_reader.available

    def get_doa(self) -> dict | None:
        """
        The direction the board last heard a voice from, or None.
        """
        if self.doa_reader is None:
            return None
        self._open_doa()
        return self.doa_reader.read()

    def _open_doa(self) -> None:
        """
        Opens the USB side once, whatever asks for it first.
        """
        if self.doa_reader is None or self._doa_opened:
            return
        with self._doa_lock:
            if self._doa_opened:
                return
            self.doa_reader.open()
            self._doa_opened = True


    def start(self, mic: bool = True, speaker: bool = True) -> bool:
        """Opens the audio device and starts the requested directions.

        Returns `True` if this call started something, `False` if everything asked
        for was already running, and raises if the device will not open.
        """
        with self._lock:
            want_mic = mic and not self._mic_running
            want_speaker = speaker and not self._speaker_running
            if not (want_mic or want_speaker):
                return False

            pa = self._pa or pyaudio.PyAudio()
            started_pa = self._pa is None
            opened = []

            try:
                if want_mic:
                    self._mic_stream, self._mic_capture_channels = self._open_input(pa)
                    opened.append(self._mic_stream)
                if want_speaker:
                    self._speaker_stream = self._open_output(pa)
                    opened.append(self._speaker_stream)
            except Exception:
                for stream in opened:
                    stream.stop_stream()
                    stream.close()
                self._mic_stream = None if want_mic else self._mic_stream
                self._speaker_stream = None if want_speaker else self._speaker_stream
                if started_pa:
                    pa.terminate()
                raise

            self._pa = pa

            if want_mic:
                self._mic_running = True
                self._mic_thread = threading.Thread(target=self._capture_loop, daemon=True)
                self._mic_thread.start()

            if want_speaker:
                # Whatever a client pushed while the speaker was closed is stale by now
                self._drain_speaker_socket()
                self._reset_resampler()
                self._speaker_running = True
                self._speaker_thread = threading.Thread(target=self._playback_loop, daemon=True)
                self._speaker_thread.start()

            print(f"Audio started (mic={self._mic_running}, speaker={self._speaker_running}).")
            return True

    def stop(self, mic: bool = True, speaker: bool = True) -> bool:
        """Stops the requested directions and frees the device.

        Returns `True` if this call stopped something.
        """
        with self._lock:
            stopping_mic = mic and (self._mic_running or self._mic_stream is not None)
            stopping_speaker = speaker and (
                self._speaker_running or self._speaker_stream is not None)
            if not (stopping_mic or stopping_speaker):
                return False

            if stopping_mic:
                self._mic_running = False
            if stopping_speaker:
                self._speaker_running = False

            for flag, attr in ((stopping_mic, "_mic_thread"),
                               (stopping_speaker, "_speaker_thread")):
                if not flag:
                    continue
                thread = getattr(self, attr)
                setattr(self, attr, None)
                if thread:
                    thread.join(timeout=2.0)

            if stopping_mic:
                self._close_stream("_mic_stream")
            if stopping_speaker:
                self._close_stream("_speaker_stream")

            if self._mic_stream is None and self._speaker_stream is None:
                if self._pa is not None:
                    self._pa.terminate()
                    self._pa = None
                # Hand the USB device back too, so nothing holds an interface
                # open on a board the next client may want to claim.
                if self.doa_reader is not None:
                    self.doa_reader.close()
                    self._doa_opened = False

            print(f"Audio stopped (mic={self._mic_running}, speaker={self._speaker_running}).")
            return True

    def flush_playback(self) -> None:
        """Drops everything queued for the speaker, mid-sentence if need be.
        """
        self._flush_requested.set()

    def _close_stream(self, attr: str) -> None:
        stream = getattr(self, attr)
        setattr(self, attr, None)
        if stream is None:
            return
        try:
            stream.stop_stream()
        finally:
            stream.close()

    # --- device -------------------------------------------------------------

    def _open_input(self, pa):
        """Opens the capture stream, returning it with the channel count it took.
        """
        index, info = _resolve_device(pa, self.device, "input")
        native = int(info.get("maxInputChannels", 0))
        if native < 1:
            raise RuntimeError(f"Audio device '{info.get('name')}' has no input channels")

        attempts = [min(self.mic_channels, native)]
        if native not in attempts:
            attempts.append(native)

        last_error = None
        for channels in attempts:
            try:
                stream = pa.open(format=pyaudio.paInt16, channels=channels,
                                 rate=self.mic_rate, input=True,
                                 input_device_index=index,
                                 frames_per_buffer=self.chunk_frames)
            except Exception as e:  # PortAudio raises OSError
                last_error = e
                continue

            print(f"Mic: '{info.get('name')}' @ {self.mic_rate}Hz, {channels}ch"
                  + (f" -> {self.mic_channels}ch published" if channels != self.mic_channels else ""))
            return stream, channels

        raise RuntimeError(
            f"Failed to open mic '{info.get('name')}' at {self.mic_rate}Hz "
            f"(tried {attempts} channels): {last_error}")

    def _open_output(self, pa):
        index, info = _resolve_device(pa, self.device, "output")
        native = int(info.get("maxOutputChannels", 0))
        if native < 1:
            raise RuntimeError(f"Audio device '{info.get('name')}' has no output channels")

        channels = min(self.speaker_channels, native)
        rate = int(self.speaker_rate or info.get("defaultSampleRate", 48000))
        try:
            stream = pa.open(format=pyaudio.paInt16, channels=channels,
                             rate=rate, output=True,
                             output_device_index=index,
                             frames_per_buffer=self.chunk_frames)
        except Exception as e:
            raise RuntimeError(
                f"Failed to open speaker '{info.get('name')}' at {rate}Hz, "
                f"{channels}ch: {e}") from e

        self.speaker_rate = rate
        self.speaker_channels = channels
        print(f"Speaker: '{info.get('name')}' @ {self.speaker_rate}Hz, {channels}ch")
        return stream

    # --- microphone ---------------------------------------------------------

    def _capture_loop(self):
        try:
            while self._mic_running:
                pcm = self._mic_stream.read(self.chunk_frames, exception_on_overflow=False)

                if self._mic_capture_channels != self.mic_channels:
                    pcm = self._reduce_capture(pcm)

                self._seq += 1
                header = {
                    "rate": self.mic_rate,
                    "channels": self.mic_channels,
                    "width": SAMPLE_WIDTH,
                    "seq": self._seq,
                    "t": time.time(),
                }

                doa = self.get_doa()
                if doa is not None:
                    header["doa"] = doa["angle"]
                    header["speech"] = doa["speech"]

                self.mic_socket.send_multipart([json.dumps(header).encode(), pcm])
        except Exception as e:
            print(f"Mic capture stopped: {e}")
            self._mic_running = False

    def _reduce_capture(self, pcm: bytes) -> bytes:
        """Narrows a multi-channel capture down to the channels we publish.

        On the ReSpeaker Flex channel 0 is processed.
        """
        frames = np.frombuffer(pcm, dtype=np.int16).reshape(-1, self._mic_capture_channels)
        return np.ascontiguousarray(frames[:, :self.mic_channels]).tobytes()


    def _playback_loop(self):
        try:
            while self._speaker_running:
                if self._flush_requested.is_set():
                    self._do_flush()
                    continue

                events = dict(self._speaker_poller.poll(100))
                if self.speaker_socket not in events:
                    continue

                self._play_message(self.speaker_socket.recv_multipart())
        except Exception as e:
            print(f"Audio playback stopped: {e}")
            self._speaker_running = False

    def _play_message(self, parts) -> bool:
        """Plays one [header, pcm] message. Returns False if it was dropped."""
        pcm, header = self._parse(parts)
        if pcm is None:
            return False

        if self.access_check is not None and not self.access_check(header.get("client_id")):
            # Rate-limited: a rejected client keeps pushing, and one line per
            # 40ms chunk would bury the journal.
            now = time.time()
            if now - self._last_reject_log > 5.0:
                self._last_reject_log = now
                print("Dropping playback audio from a client that does not hold the lock.")
            return False

        self._write(self._convert_for_output(pcm, header))
        return True

    def _parse(self, parts):
        if len(parts) != 2:
            print(f"Ignoring malformed audio message ({len(parts)} parts)")
            return None, {}
        try:
            header = json.loads(parts[0].decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            print(f"Ignoring audio message with an unreadable header: {e}")
            return None, {}

        if int(header.get("width", SAMPLE_WIDTH)) != SAMPLE_WIDTH:
            print(f"Ignoring {header.get('width')}-byte audio; only 16-bit PCM is supported")
            return None, {}
        return parts[1], header

    def _write(self, pcm: bytes):
        """
        Writes to the device in device-sized slices.
        """
        step = self.chunk_frames * self.speaker_channels * SAMPLE_WIDTH
        for offset in range(0, len(pcm), step):
            if not self._speaker_running or self._flush_requested.is_set():
                return
            self._speaker_stream.write(pcm[offset:offset + step])

    def _do_flush(self):
        """Discards queued audio"""
        self._flush_requested.clear()
        drained = self._drain_speaker_socket()
        # stop/start is the only way to drop what PortAudio has already buffered.
        self._speaker_stream.stop_stream()
        self._speaker_stream.start_stream()
        self._reset_resampler()
        print(f"Playback flushed ({drained} queued chunks dropped).")

    def _drain_speaker_socket(self) -> int:
        dropped = 0
        while dict(self._speaker_poller.poll(0)):
            self.speaker_socket.recv_multipart()
            dropped += 1
        return dropped

    # --- format conversion --------------------------------------------------

    def _reset_resampler(self):
        self._resample_phase = 0.0
        self._resample_tail = None
        self._resample_src = None

    def _convert_for_output(self, pcm: bytes, header: dict) -> bytes:
        """
        Brings a client's PCM into the format the speaker was opened with.
        """
        src_rate = int(header.get("rate", self.speaker_rate))
        src_channels = max(1, int(header.get("channels", 1)))

        if src_rate == self.speaker_rate and src_channels == self.speaker_channels:
            return pcm

        # Trim to whole frames before interpreting
        whole = (len(pcm) // (SAMPLE_WIDTH * src_channels)) * SAMPLE_WIDTH * src_channels
        if whole == 0:
            return b""
        frames = np.frombuffer(pcm[:whole], dtype=np.int16).reshape(-1, src_channels)

        frames = self._to_channels(frames, self.speaker_channels)

        if src_rate != self.speaker_rate:
            if self._resample_src != (src_rate, self.speaker_channels):
                self._reset_resampler()
                self._resample_src = (src_rate, self.speaker_channels)
            frames = self._resample(frames, src_rate)

        return np.clip(np.rint(frames), -32768, 32767).astype(np.int16).tobytes()

    @staticmethod
    def _to_channels(frames: np.ndarray, dst: int) -> np.ndarray:
        src = frames.shape[1]
        if src == dst:
            return frames
        if dst == 1:
            return frames.mean(axis=1, keepdims=True)
        if src == 1:
            return np.repeat(frames, dst, axis=1)
        if src > dst:
            return frames[:, :dst]
        return np.concatenate(
            [frames, np.repeat(frames[:, -1:], dst - src, axis=1)], axis=1)

    def _resample(self, frames: np.ndarray, src_rate: int) -> np.ndarray:
        """Linear interpolation to the speaker's rate, continuous across chunks.

        Coordinates: `ext` is the previous chunk's last sample followed by this
        chunk, so index 0 is exactly where the last call stopped reading.
        """
        n = len(frames)
        if n == 0:
            return frames

        # Calculate the step size for resampling
        step = src_rate / self.speaker_rate
        tail = self._resample_tail if self._resample_tail is not None else frames[:1]
        ext = np.concatenate((tail, frames)).astype(np.float32)

        count = max(0, int(np.ceil((n - self._resample_phase) / step)))
        positions = self._resample_phase + step * np.arange(count)

        source = np.arange(len(ext), dtype=np.float32)
        out = np.empty((count, frames.shape[1]), dtype=np.float32)
        for c in range(frames.shape[1]):
            out[:, c] = np.interp(positions, source, ext[:, c])

        # Where the next chunk should pick up, measured from its own index 0
        self._resample_phase = self._resample_phase + count * step - n
        self._resample_tail = frames[-1:]
        return out
