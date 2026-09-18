from typing import Any, NamedTuple, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R
import itertools
import uuid
import wave
import cv2
import zmq
import time
import json


class CleoError(RuntimeError):
    """Base class for every error the SDK raises."""


class CommandTimeout(CleoError):
    """Raised when the robot never acknowledged a command, after every retry.

    Also raised when it acknowledged but never reported the work finished, as a
    trajectory that stops reaching its target does.
    """


class RobotError(CleoError):
    """Raised when the robot got the command but refused it or failed to run it.

    The robot reports refusals in the reply body, so a reply arriving is not the
    same as a command having worked.
    """


# How long move_head_ruckig(wait=True) keeps polling before it gives up, on top
# of any requested duration. Generous on purpose: it exists to break a hang, not
# to time a move, and a slow trajectory must not trip it.
DEFAULT_MOVE_WAIT_S = 60.0


class Direction(NamedTuple):
    """Where the microphone array last heard a voice.

    `angle` is degrees clockwise in the array's own frame, not the head's -- 0
    is whichever way the board is physically pointing, so a head that turns
    towards a speaker needs an offset measured once for how it is mounted.

    `speech` is the part that is easy to skip and should not be: the DSP latches
    the last direction it heard a voice from and holds it through every silence,
    so an angle with speech=False is a memory, not an observation.
    """

    angle: int
    speech: bool


class AudioChunk(NamedTuple):
    """One block of microphone audio, carrying the format it was captured in.

    The Pi describes every block it sends rather than assuming both ends agreed
    on a rate beforehand, so `pcm` can go straight into a recogniser without the
    caller hardcoding what the robot's microphone happens to be doing.

    `direction` rides along when the head has an array that reports one, so
    "what was said" and "where from" arrive together, already lined up in time.
    """

    pcm: bytes
    rate: int
    channels: int
    width: int
    seq: int
    direction: Optional[Direction] = None

    @property
    def samples(self) -> np.ndarray:
        """The block as int16, shaped (frames, channels)."""
        return np.frombuffer(self.pcm, dtype=np.int16).reshape(-1, self.channels)


class RobotHead:
    def __init__(self, ip: str = "localhost", cmd_port=5555, telemetry_port=5556, video_port=5557,
                 mic_port=5558, speaker_port=5559,
                 cmd_timeout_ms: int = 2000, cmd_attempts: int = 3, verbose: bool = False):
        self.ip = ip
        self.cmd_port = cmd_port
        self.context = zmq.Context()

        self.client_id = str(uuid.uuid4())
        print(f"Initialized with Client ID: {self.client_id[-4:]}")

        self.cmd_timeout_ms = cmd_timeout_ms
        self.cmd_attempts = cmd_attempts
        self.verbose = verbose
        self._seq = itertools.count(1)

        # 1. Command Socket (REQ)
        self.cmd_socket = self.context.socket(zmq.REQ)
        self.cmd_socket.connect(f"tcp://{self.ip}:{cmd_port}")
        
        # 2. Telemetry Socket (SUB)
        self.telemetry_socket = self.context.socket(zmq.SUB)
        self.telemetry_socket.connect(f"tcp://{self.ip}:{telemetry_port}")
        self.telemetry_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # 3. Video Socket (SUB)
        self.video_socket = self.context.socket(zmq.SUB)
        self.video_socket.connect(f"tcp://{self.ip}:{video_port}")
        self.video_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # 4. Microphone Socket (SUB)
        self.mic_socket = self.context.socket(zmq.SUB)
        self.mic_socket.connect(f"tcp://{self.ip}:{mic_port}")
        self.mic_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.speaker_socket = self.context.socket(zmq.PUSH)
        # IMMEDIATE: queue only to a connection that actually completed.
        self.speaker_socket.setsockopt(zmq.IMMEDIATE, 1)
        self.speaker_socket.connect(f"tcp://{self.ip}:{speaker_port}")

        self.telemetry_poller = zmq.Poller()
        self.telemetry_poller.register(self.telemetry_socket, zmq.POLLIN)

        self.video_poller = zmq.Poller()
        self.video_poller.register(self.video_socket, zmq.POLLIN)

        self.cmd_poller = zmq.Poller()
        self.cmd_poller.register(self.cmd_socket, zmq.POLLIN)

        self.mic_poller = zmq.Poller()
        self.mic_poller.register(self.mic_socket, zmq.POLLIN)

        self.speaker_poller = zmq.Poller()
        self.speaker_poller.register(self.speaker_socket, zmq.POLLOUT)

        self._last_telemetry = None
        self._video_started = False
        self._audio_started = False
        self._audio_seq = itertools.count(1)

        print(f"Commands: {cmd_port} | Telemetry: {telemetry_port} | Video: {video_port} "
              f"| Mic: {mic_port} | Speaker: {speaker_port}")

    def _check_reply(self, command: str, reply: dict, quiet: bool = False) -> dict:
        """Turns the robot's reply into a return value or an exception.

        """
        status = reply.get("status")
        message = reply.get("message", "")

        if status != "ok":
            raise RobotError(f"'{command}' rejected by robot: {message if message else reply}")

        if self.verbose and not quiet:
            print(f"[cleo] {command}: {message}")

        return reply

    def _send_command(self, payload: dict, timeout_ms: Optional[int] = None,
                      attempts: Optional[int] = None, quiet: bool = False) -> dict:
        """Sends a command and returns the robot's reply, resending if it goes missing.

        A silent command and a silent reply look identical from the client side, so every
        attempt reuses the same msg_id: the controller recognises the resend and replays
        its original answer instead of moving the head twice.

        Raises CommandTimeout if no attempt is acknowledged, or RobotError if the
        robot answers but reports a failure.
        """
        timeout_ms = self.cmd_timeout_ms if timeout_ms is None else timeout_ms
        attempts = self.cmd_attempts if attempts is None else attempts

        payload = dict(payload)
        payload["client_id"] = self.client_id
        payload["msg_id"] = f"{self.client_id}:{next(self._seq)}"

        for attempt in range(1, attempts + 1):
            self.cmd_socket.send_json(payload)

            events = dict(self.cmd_poller.poll(timeout_ms))
            if self.cmd_socket not in events:
                # A REQ socket stays stuck waiting for the lost reply, so the
                # only way to send again is to throw the socket away.
                print(f"Network Timeout on '{payload['command']}' "
                      f"(attempt {attempt}/{attempts})")
                self._reset_socket()
                continue

            reply: dict = self.cmd_socket.recv_json()  # type: ignore

            reply_id = reply.get("msg_id")
            if reply_id is not None and reply_id != payload["msg_id"]:
                print(f"Ignoring reply meant for {reply_id}")
                continue

            return self._check_reply(payload["command"], reply, quiet)

        raise CommandTimeout(
            f"Robot did not acknowledge '{payload['command']}' after {attempts} attempts"
        )


    def _reset_socket(self):
        """Destroys the stuck socket, creates a new one, and UPDATES THE POLLER."""
        
        # Unregister the dead socket BEFORE closing it to prevent memory leaks
        self.cmd_poller.unregister(self.cmd_socket)
        
        self.cmd_socket.setsockopt(zmq.LINGER, 0)
        self.cmd_socket.close()
        
        self.cmd_socket = self.context.socket(zmq.REQ)
        self.cmd_socket.connect(f"tcp://{self.ip}:{self.cmd_port}")
        
        self.cmd_poller.register(self.cmd_socket, zmq.POLLIN)

    def _go_to(self, yaw: float, pitch: float, roll: float, time_ms: int = 1000,
               ear_left: Optional[float] = None, ear_right: Optional[float] = None):
        """
        Moves multiple servos simultaneously.
        """
        payload = {
            "command": "go_to",
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "total_time_ms": time_ms,
            "accel_time_ms": int(time_ms / 3)
        }
        # Omitted rather than sent as 0.0: an ear the caller did not mention keeps
        # the pose it is in instead of being pulled back to neutral by a head move.
        if ear_left is not None:
            payload["ear_left"] = ear_left
        if ear_right is not None:
            payload["ear_right"] = ear_right
        return self._send_command(payload)

    def center(self, time_ms: int = 2000):
        """Centers the head smoothly."""
        return self._send_command({
            "command": "center_head",
            "total_time_ms": time_ms
        })
    
    def move_from_matrix(self, matrix: list, time_ms: int = 1000):
        """
        Accepts a 3x3 rotation matrix and moves the head to that orientation.
        """
        r = R.from_matrix(matrix)
        yaw, pitch, roll = r.as_euler('zyx', degrees=True)
        
        # 2. Pass the angles to our new Kinematics function
        return self.move_head(yaw=yaw, pitch=pitch, roll=roll, time_ms=time_ms)
    
    def move_head(self, yaw: float, pitch: float, roll: float = 0.0, time_ms: int = 1000,
                  ear_left: Optional[float] = None, ear_right: Optional[float] = None):
        """
        Moves the Head in all 3 DOF, optionally taking the ears along.

        Ear angles given here travel in the same packet as the head pose, so the
        whole expression lands on one bus write instead of two.
        """
        # 4. Send the raw angles to the Raspberry Pi
        return self._go_to(yaw, pitch, roll, time_ms=time_ms,
                           ear_left=ear_left, ear_right=ear_right)

    def move_ears(self, ear_left: Optional[float] = None, ear_right: Optional[float] = None,
                  time_ms: int = 1000):
        """Moves the ears over a timed profile, leaving the head alone.

        Angles are degrees from neutral, and mirrored: the same positive value on
        both sides moves them the same physical way. A side left as None is not
        commanded at all, so one ear can move while the other holds its pose.

        This is the non-Ruckig path -- the servo's own time-based profile, the
        same one move_head uses. Use move_ears_ruckig for a trajectory that can
        be retargeted mid-flick.
        """
        if ear_left is None and ear_right is None:
            raise ValueError("move_ears needs at least one of ear_left, ear_right")

        payload: dict[str, Any] = {
            "command": "move_ears",
            "total_time_ms": time_ms,
            "accel_time_ms": int(time_ms / 3),
        }
        if ear_left is not None:
            payload["ear_left"] = ear_left
        if ear_right is not None:
            payload["ear_right"] = ear_right
        return self._send_command(payload)

    def center_ears(self, time_ms: int = 1000):
        """Returns both ears to neutral, the head untouched."""
        return self._send_command({
            "command": "center_ears",
            "total_time_ms": time_ms
        })

    def move_ears_ruckig(self, ear_left: float, ear_right: float,
                         v_max: float = 200, a_max: float = 1000, j_max: float = 4000,
                         duration=None, wait: bool = True,
                         wait_timeout: Optional[float] = None) -> dict:
        """Moves the ears through the Ruckig OTG, independently of the head.

        The Pi runs a second trajectory generator for the ears, so this can be
        called while the head is mid-move -- and retargeted mid-flick -- without
        either trajectory disturbing the other. The defaults are livelier than
        the head's: an ear carries almost no inertia.

        Raises the same way move_head_ruckig does: RobotError on a refusal,
        CommandTimeout if the robot never answers, and with wait=True once the
        ears have reported themselves moving for wait_timeout seconds.
        """
        for name, limit in (("v_max", v_max), ("a_max", a_max), ("j_max", j_max)):
            if limit <= 0:
                raise ValueError(f"{name} must be greater than 0, got {limit}")

        payload = {
            "command": "move_ears_ruckig",
            "ear_left": ear_left,
            "ear_right": ear_right,
            "v_max": v_max,
            "a_max": a_max,
            "j_max": j_max
        }

        if duration is not None:
            if duration <= 0.0:
                raise ValueError(f"duration must be greater than 0, got {duration}")
            payload["duration"] = duration

        reply = self._send_command(payload)

        if not wait:
            return reply

        budget = (DEFAULT_MOVE_WAIT_S + (duration or 0.0)
                  if wait_timeout is None else wait_timeout)
        deadline = time.monotonic() + budget

        time.sleep(0.1)  # let the Pi's 100Hz thread pick the new target up
        while self.ears_moving():
            if time.monotonic() >= deadline:
                raise CommandTimeout(
                    f"Ears still report moving {budget:.1f}s after 'move_ears_ruckig' "
                    f"(left={ear_left}, right={ear_right}); they may still be in motion")
            time.sleep(0.02)

        return reply


    def move_head_ruckig(self, yaw: float, pitch: float, roll: float,
                         v_max: float = 100, a_max: float = 400, j_max: float = 400,
                         duration=None, wait: bool = True,
                         wait_timeout: Optional[float] = None) -> dict:
        """
        Moves the 3-DOF differential head using the Ruckig OTG trajectory generator.
        v_max: Max velocity in ticks/sec
        a_max: Max acceleration in ticks/sec^2
        j_max: Max jerk in ticks/sec^3

        Raises RobotError if the robot refuses the target (a lock held by someone
        else, say) and CommandTimeout if it never answers at all. With wait=True
        it also raises CommandTimeout once the head has reported itself moving for
        wait_timeout seconds -- by default DEFAULT_MOVE_WAIT_S on top of duration.
        Nothing on the Pi stops a trajectory, so the head may still be in motion
        when that fires; it bounds the wait, it does not undo the move.
        """
        for name, limit in (("v_max", v_max), ("a_max", a_max), ("j_max", j_max)):
            if limit <= 0:
                # Ruckig with a zero limit never reaches its target, which
                # downstream is a head that reports itself moving forever.
                raise ValueError(f"{name} must be greater than 0, got {limit}")

        payload = {
            "command": "go_to_ruckig",
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "v_max": v_max,
            "a_max": a_max,
            "j_max": j_max
        }

        if duration is not None:
            if duration <= 0.0:
                raise ValueError(f"duration must be greater than 0, got {duration}")
            payload["duration"] = duration

        reply = self._send_command(payload)

        if not wait:
            return reply

        budget = (DEFAULT_MOVE_WAIT_S + (duration or 0.0)
                  if wait_timeout is None else wait_timeout)
        deadline = time.monotonic() + budget

        # 1. Give the Pi's 100Hz thread a tiny fraction of a second to register
        # the new command and change its status to 'Working'
        time.sleep(0.1)
        # 2. Poll the Pi at ~50Hz until it reports the movement is done. A refusal
        # or an unreachable robot raises out of is_moving() rather than spinning.
        while self.is_moving():
            if time.monotonic() >= deadline:
                raise CommandTimeout(
                    f"Head still reports moving {budget:.1f}s after 'go_to_ruckig' "
                    f"(yaw={yaw}, pitch={pitch}, roll={roll}); it may still be in motion")
            time.sleep(0.02)

        return reply

    def get_telemetry(self) -> Optional[dict[str, Any]]:
        """
        Fetches the absolute newest telemetry packet from the robot.
        Returns None if no data has ever been received.
        """
        events = dict(self.telemetry_poller.poll(3))
        if self.telemetry_socket in events:
            self._last_telemetry = self.telemetry_socket.recv_json()
        else:
            return None
        return self._last_telemetry # type: ignore
    
    def play_animation(self, filepath: str, initial_delay: float = 0.5, 
                       base_v: float = 1000, base_a: float = 5000, base_j: float = 20000):
        """
        Reads a JSON keyframe sequence exported from Blender and plays it back 
        """
        try:
            with open(filepath, 'r') as f:
                keyframes = json.load(f)
        except FileNotFoundError:
            print(f"Error: Animation file '{filepath}' not found.")
            return False
        except json.JSONDecodeError:
            print(f"Error: '{filepath}' is not a valid JSON file.")
            return False

        if not keyframes:
            print("Error: Empty animation file.")
            return False

        print(f"Playing animation: {filepath} ({len(keyframes)} keyframes)")

        # 1. Ease into the starting position
        start_pos = keyframes[0]
        self.move_head_ruckig(
            yaw=start_pos.get("yaw", 0.0), 
            pitch=start_pos.get("pitch", 0.0), 
            roll=start_pos.get("roll", 0.0), # Slower, smooth start
            wait=True,
        )

        current_time = start_pos["time_sec"]

        # 2. Stream the remaining keyframes
        for i in range(1, len(keyframes)):
            kf = keyframes[i]
            delta_time = kf["time_sec"] - current_time
            print(f"Moving to Keyframe {i}/{len(keyframes)-1} | Time: {kf['time_sec']}s | Δt: {delta_time:.2f}s | Yaw: {kf.get('yaw', 0.0)}°, Pitch: {kf.get('pitch', 0.0)}°, Roll: {kf.get('roll', 0.0)}°")
            
            # Prevent negative sleep times if keyframes are stacked on the exact same frame
            if delta_time > 0:
                self.move_head_ruckig(
                    yaw=kf.get("yaw", 0.0), 
                    pitch=kf.get("pitch", 0.0), 
                    roll=kf.get("roll", 0.0),
                    v_max=base_v,
                    a_max=base_a,
                    j_max=base_j,
                    duration=delta_time,
                    wait=False
                )
                time.sleep(delta_time + 0.1) # Sleep for the duration minus a small buffer to ensure we start the next keyframe on time
            current_time = kf["time_sec"]

        print("Animation finished.")
        return True
    
    def upload_and_play(self, filepath: str):
        """
        Reads a 50Hz baked JSON animation and sends it to the Pi as one payload.
        The Pi will execute it locally with zero network latency.
        """
        import json

        try:
            with open(filepath, 'r') as f:
                frames = json.load(f)
        except FileNotFoundError:
            print(f"File {filepath} not found.")
            return False

        print(f"Uploading {len(frames)} frames to Robot...")

        # We attach the entire array to the network payload!
        self._send_command({
            "command": "upload_and_play",
            "frames": frames
        })
        return True

    def stop_animation(self):
        """Remotely aborts a playing animation on the Pi."""
        self._send_command({"command": "stop_animation"})
    
    def start_video(self):
        """Switches the robot's camera on.

        The daemon boots with the stream off, so this has to be called before
        get_video_frame() returns anything. The Pi opens the camera before it
        answers -- hence the longer timeout -- so an 'ok' here means the device
        really is streaming, but the first frames still need a moment to arrive.
        """
        reply = self._send_command({"command": "start_video"}, timeout_ms=5000)
        self._video_started = True
        return reply

    def stop_video(self):
        """Switches the robot's camera off and frees it for the next client."""
        reply = self._send_command({"command": "stop_video"})
        self._video_started = False
        return reply

    def get_video_frame(self):
        """
        Fetches the newest video frame from cleo.
        Returns an OpenCV BGR image array, or None if the stream was never
        started (see start_video) or no frame has arrived yet.
        """
        latest_frame_bytes = None
        
        # Socket Draining: Pull frames until the queue is empty to eliminate lag
        while True:
            events = dict(self.video_poller.poll(0))
            if self.video_socket in events:
                latest_frame_bytes = self.video_socket.recv()
            else:
                break

        if latest_frame_bytes:
            # Convert the raw bytes back into an OpenCV image
            nparr = np.frombuffer(latest_frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
            
        return None
    
    def start_audio(self, mic: bool = True, speaker: bool = True) -> dict:
        """Switches the robot's microphone and speaker on.

        The daemon boots with both closed, so this has to be called before
        get_audio_chunk() returns anything or play_audio() is heard. The Pi opens
        the device before it answers, and reports the format it settled on, which
        is what the returned dict carries: mic_rate, mic_channels, speaker_rate,
        speaker_channels.

        The generous timeout is the point: the *first* open after the daemon
        boots costs seconds -- measured at 7.7s against a ReSpeaker, against
        0.15s for every one after it -- because the OS is setting up access to
        the device for that process for the first time (on macOS including the
        microphone permission check). Waiting longer costs nothing when the
        device is warm, since this returns as soon as the reply lands.

        The two directions are independent: pass speaker=False for a robot that
        only listens, or mic=False for one that only talks.
        """
        reply = self._send_command(
            {"command": "start_audio", "mic": mic, "speaker": speaker}, timeout_ms=30000)
        self._audio_started = True

        info = reply.get("message")
        return info if isinstance(info, dict) else {}

    def stop_audio(self, mic: bool = True, speaker: bool = True) -> dict:
        """Closes the robot's microphone and speaker, freeing the device.

        Also slower than a normal command: the Pi joins both worker threads
        (up to 2s each) before it closes the device and answers, so the 2s
        default would be tight enough to time out on a healthy robot.
        """
        reply = self._send_command({"command": "stop_audio", "mic": mic, "speaker": speaker},
                                   timeout_ms=10000)
        if mic and speaker:
            self._audio_started = False

        info = reply.get("message")
        return info if isinstance(info, dict) else {}

    def get_audio_chunk(self, timeout_ms: int = 0) -> Optional[AudioChunk]:
        """The next block of microphone audio, oldest first.

        Unlike get_video_frame this deliberately does not skip ahead to the
        newest message. For video only the latest frame is worth having, but
        speech *is* the sequence: drop blocks out of the middle and a recogniser
        hears a different sentence. Consume this in a loop that keeps up --
        the Pi's PUB socket starts dropping blocks if you fall far enough behind.

        Returns None if nothing arrived within timeout_ms (0 = don't wait).
        """
        events = dict(self.mic_poller.poll(timeout_ms))
        if self.mic_socket not in events:
            return None

        header_raw, pcm = self.mic_socket.recv_multipart()
        header = json.loads(header_raw.decode())

        direction = None
        if "doa" in header:
            direction = Direction(angle=int(header["doa"]),
                                  speech=bool(header.get("speech", False)))

        return AudioChunk(pcm=pcm,
                          rate=int(header.get("rate", 16000)),
                          channels=int(header.get("channels", 1)),
                          width=int(header.get("width", 2)),
                          seq=int(header.get("seq", 0)),
                          direction=direction)

    def play_audio(self, pcm: bytes, rate: int, channels: int = 1,
                   chunk_ms: int = 40, wait: bool = False) -> float:
        """Sends 16-bit PCM to the robot's speaker and returns its duration.

        Whatever rate the audio is in is what gets sent: the Pi resamples to
        whatever it opened the speaker at, so a TTS engine's native output can
        go straight through without the caller matching formats.

        Sent in chunk_ms slices so the Pi can start playing before the whole
        utterance has arrived, and so flush_audio() can cut in mid-sentence.
        With wait=True this blocks for roughly as long as the audio lasts --
        approximate, since the last chunk still has to drain on the Pi.
        """
        if channels < 1:
            raise ValueError(f"channels must be at least 1, got {channels}")

        frame_bytes = channels * 2
        duration = len(pcm) / (rate * frame_bytes) if rate else 0.0
        step = max(1, int(rate * chunk_ms / 1000)) * frame_bytes
        started = time.monotonic()

        for offset in range(0, len(pcm), step):
            self._push_audio({
                "rate": rate,
                "channels": channels,
                "width": 2,
                "seq": next(self._audio_seq),
                "client_id": self.client_id,
            }, pcm[offset:offset + step])

        if wait:
            time.sleep(max(0.0, duration - (time.monotonic() - started)))

        return duration

    def play_wav(self, filepath: str, chunk_ms: int = 40, wait: bool = False) -> float:
        """Plays a 16-bit WAV file through the robot's speaker.

        The natural pairing with a TTS engine that writes to a file, which most
        of them do -- pyttsx3's save_to_file, piper, espeak -o.
        """
        with wave.open(filepath, "rb") as wav:
            if wav.getsampwidth() != 2:
                raise CleoError(
                    f"'{filepath}' is {wav.getsampwidth() * 8}-bit; only 16-bit WAV is supported")
            pcm = wav.readframes(wav.getnframes())
            rate, channels = wav.getframerate(), wav.getnchannels()

        return self.play_audio(pcm, rate=rate, channels=channels, chunk_ms=chunk_ms, wait=wait)

    def get_direction(self) -> Optional[Direction]:
        """Asks where the array last heard a voice. None if it cannot tell.

        Answers with the microphone closed too -- the DSP on the board keeps
        tracking whether or not anyone is reading samples -- so a head that only
        wants to turn towards whoever is talking never has to stream audio.

        Check `.speech` before acting on `.angle`: the board holds its last
        direction through silence, so following the angle alone means chasing a
        person who stopped talking a long time ago.
        """
        # quiet: a client that turns the head to face a speaker polls this.
        reply = self._send_command({"command": "get_doa"}, quiet=True)

        doa = reply.get("message")
        if not isinstance(doa, dict):
            return None
        return Direction(angle=int(doa.get("angle", 0)), speech=bool(doa.get("speech", False)))

    def flush_audio(self):
        """Cuts off whatever the robot is still playing, mid-word if need be.

        Barge-in: the person starts talking while the robot is halfway through a
        reply. Simply stopping the sending side is not enough -- the Pi still has
        everything already queued to get through first.
        """
        return self._send_command({"command": "flush_audio"})

    def _push_audio(self, header: dict, pcm: bytes):
        """Queues one block for the speaker, refusing to block forever.

        A PUSH socket with nowhere to push blocks on send, so a daemon that was
        never started would hang the caller instead of reporting anything. The
        poll turns that into an error, while still letting a busy-but-alive Pi
        apply backpressure.
        """
        deadline = time.monotonic() + self.cmd_timeout_ms / 1000

        while True:
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms > 0 and self.speaker_socket in dict(
                    self.speaker_poller.poll(remaining_ms)):
                try:
                    self.speaker_socket.send_multipart(
                        [json.dumps(header).encode(), pcm], zmq.DONTWAIT)
                    return
                except zmq.Again:
                    continue  # lost the race to another sender; the deadline still holds

            raise CleoError(
                "Robot is not accepting audio -- is the daemon running and start_audio() called?")

    def is_moving(self) -> bool:
        """Asks the Pi if the head's Ruckig trajectory is currently in progress.

        The head only: the ears run their own trajectory and answer to
        ears_moving(), so a twitching ear never holds up a head move's wait.
        """
        # quiet: the wait loop in move_head_ruckig polls this at ~50Hz.
        reply = self._send_command({"command": "is_moving"}, quiet=True)
        return bool(reply.get("message", False))

    def ears_moving(self) -> bool:
        """Asks the Pi if the ears' Ruckig trajectory is currently in progress."""
        # quiet: the wait loop in move_ears_ruckig polls this at ~50Hz.
        reply = self._send_command({"command": "ears_moving"}, quiet=True)
        return bool(reply.get("message", False))

    def disconnect(self):
        """Closes the network connection cleanly."""

        try: 
            self._go_to(0, -15, 0, time_ms=1500,
                                       ear_left=0, ear_right=0)
        except CleoError as e:
            print(f"Could not center the head and ears ({e}); continuing.")
            
        try:
            if self._video_started:
                # Nothing on the Pi turns the camera off on its own -- the lease
                # expiring only frees the lock -- so a client that leaves without
                # this leaves the camera running until the daemon restarts.
                self.stop_video()
        except CleoError as e:
            print(f"Could not stop the video stream ({e}); continuing.")

        try:
            if self._audio_started:
                # Same reasoning as the camera: nothing on the Pi closes the
                # audio device on its own, so a client that leaves without this
                # keeps the microphone open until the daemon restarts.
                self.stop_audio()
        except CleoError as e:
            print(f"Could not stop the audio stream ({e}); continuing.")

        try:
            self._send_command({"command": "release_lock"})
        except CleoError as e:
            # The lock expires on its own, so an unreachable or unhappy robot
            # must not stop us from freeing the sockets.
            print(f"Could not release the lock cleanly ({e}); closing anyway.")
        finally:
            self.cmd_socket.close()
            self.telemetry_socket.close()
            self.video_socket.close()
            self.mic_socket.close()
            # LINGER=0: anything still queued for the speaker is stale by the time
            # a disconnecting client cares, and without it close() waits for a Pi
            # that may be exactly what went wrong.
            self.speaker_socket.setsockopt(zmq.LINGER, 0)
            self.speaker_socket.close()