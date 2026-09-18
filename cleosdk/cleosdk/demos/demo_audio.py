# demo_audio.py
"""Talk through the robot's speaker, listen through its microphone.

    python -m cleosdk.demos.demo_audio                     # greet, then show levels
    python -m cleosdk.demos.demo_audio "Guten Tag"         # say something else
    python -m cleosdk.demos.demo_audio --file bell.wav     # play a 16-bit WAV
    python -m cleosdk.demos.demo_audio --ip 192.168.1.50   # a robot on the network
    python -m cleosdk.demos.demo_audio --follow            # turn towards the speaker

The meter shows the direction the array last heard a voice from, alongside the
level. --follow turns that into movement; it is opt-in because it energises the
servos, and it needs --offset measured once for how the array is mounted.

The speech itself is rendered by whatever TTS the machine running this already
has -- macOS `say`, espeak-ng on Linux -- because the SDK deliberately takes no
position on that: play_wav() and play_audio() accept 16-bit PCM from any engine,
and the Pi resamples it to whatever it opened the ReSpeaker at.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np

from cleosdk import RobotHead

TTS_RATE = 22050


def render_tts(text: str, path: str) -> bool:
    """Renders `text` to a 16-bit WAV. False if the machine has no TTS."""
    if shutil.which("say"):  # macOS
        command = ["say", "-o", path, "--data-format=LEI16@22050",
                   "--file-format=WAVE", text]
    elif shutil.which("espeak-ng") or shutil.which("espeak"):  # Linux, the Pi
        engine = shutil.which("espeak-ng") or "espeak"
        command = [engine, "-w", path, text]
    else:
        return False

    subprocess.run(command, check=True)
    return True


def beep(seconds: float = 0.6, hz: float = 440.0) -> bytes:
    """Something audible for a machine with no TTS installed."""
    t = np.linspace(0, seconds, int(TTS_RATE * seconds), endpoint=False)
    fade = np.minimum(1.0, np.minimum(t, seconds - t) * 20)  # no click at the edges
    return (np.sin(2 * np.pi * hz * t) * fade * 8000).astype(np.int16).tobytes()


def level_bar(chunk, width: int = 30) -> str:
    """A crude VU meter, so you can see the microphone working."""
    samples = chunk.samples.astype(np.float32)
    rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
    filled = min(width, int(width * rms / 3000))
    line = f"[{'#' * filled}{'.' * (width - filled)}] {rms:6.0f} rms"

    # The angle is only an observation while speech is true; the board holds its
    # last direction through silence, so say which of the two this is.
    if chunk.direction is not None:
        state = "voice" if chunk.direction.speech else " ... "
        line += f"   {chunk.direction.angle:3}deg {state}"
    return line


def head_yaw(angle: int, offset: float, limit: float) -> float:
    """Maps an array angle onto a head yaw.

    The array reports 0-180 across the arc it can distinguish, so its centre --
    90 -- is whatever direction the board physically faces. Which way that is on
    the head is a mounting question, not something the SDK can know, hence
    `offset`: point the array's marked front at the robot's front and this is
    zero, otherwise measure it once by talking from straight ahead.
    """
    return max(-limit, min(limit, (90.0 - angle) + offset))


def main():
    # localhost like every other demo. Not ROBOT_IP: devenv sets that to the
    # Pi's address, so honouring it would send this demo across the network
    # while its siblings talk to the daemon on this machine.
    ip, wav_path, words = "localhost", None, []
    follow, offset, limit = False, 0.0, 60.0
    args = iter(sys.argv[1:])
    for arg in args:
        if arg == "--ip":
            ip = next(args)
        elif arg == "--file":
            wav_path = next(args)
        elif arg == "--follow":
            follow = True
        elif arg == "--offset":
            offset = float(next(args))
        elif arg == "--max-yaw":
            limit = float(next(args))
        else:
            words.append(arg)

    text = " ".join(words) or "Hallo, ich bin Cleo. Ich hoere zu."

    head = RobotHead(ip=ip)

    # The daemon keeps the ReSpeaker closed until a client asks for it, and
    # reports the format it settled on.
    fmt = head.start_audio()
    print(f"Mic: {fmt.get('mic_rate')}Hz x{fmt.get('mic_channels')} | "
          f"Speaker: {fmt.get('speaker_rate')}Hz x{fmt.get('speaker_channels')}")

    try:
        # --- 1. Out: the robot says something -------------------------------
        if wav_path:
            print(f"Playing {wav_path}")
            head.play_wav(wav_path, wait=True)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "speech.wav")
                if render_tts(text, path):
                    print(f"Speaking: {text}")
                    head.play_wav(path, wait=True)
                else:
                    print("No system TTS found (say / espeak-ng) -- beeping instead.")
                    head.play_audio(beep(), rate=TTS_RATE, wait=True)

        # --- 2. In: what the microphone hears, and from where ---------------
        print("\nListening. Talk to the robot, Ctrl-C to stop.")
        print("Turning towards the speaker.\n" if follow else "")

        last_move, last_yaw = 0.0, None
        while True:
            chunk = head.get_audio_chunk(timeout_ms=200)
            if chunk is None:
                continue

            print(f"\r{level_bar(chunk)}", end="", flush=True)

            if not (follow and chunk.direction and chunk.direction.speech):
                continue

            # Only chase an angle that is both new and current: a couple of
            # degrees of jitter is not worth a servo command, and the array
            # reports a direction faster than a head can turn to it.
            target = head_yaw(chunk.direction.angle, offset, limit)
            now = time.monotonic()
            if now - last_move < 0.5 or (last_yaw is not None and abs(target - last_yaw) < 8):
                continue

            last_move, last_yaw = now, target
            head.move_head_ruckig(yaw=target, pitch=0.0, roll=0.0, wait=False)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # disconnect() closes the device and releases the lock; without it the
        # microphone stays open until the daemon restarts.
        head.disconnect()


if __name__ == "__main__":
    main()
