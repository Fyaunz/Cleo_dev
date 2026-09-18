"""End-to-end checks for the audio path, over real ZMQ sockets.

pi_controller/tests/test_audio.py mocks ZMQ away to exercise the streamer's
logic; this does the opposite. The real SDK talks to the real AudioStreamer over
loopback, with only PortAudio faked, which is the only way to cover what lives
between them: the two-part [header, pcm] framing, the PUB/SUB and PUSH/PULL
wiring, and the Pi resampling whatever rate a client's TTS happened to emit.

Run directly:
    .devenv/state/venv/bin/python3 e2e/test_audio.py -v
"""
import io
import itertools
import os
import sys
import time
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pi_controller"))
sys.path.insert(0, os.path.join(ROOT, "cleosdk"))

import zmq

from network.audio import AudioStreamer
from cleosdk.cleo import CleoError, RobotHead

# Fresh ports per test, above the range test_reliability.py uses.
_PORTS = itertools.count(5800)

MIC_RATE = 16000
SPEAKER_RATE = 48000
CHUNK = 256


class FakePortAudio:
    """A device that captures a known tone and remembers what was played."""

    def __init__(self):
        self.captured = np.arange(CHUNK, dtype=np.int16).tobytes()
        self.played = bytearray()
        self.pa = MagicMock()
        self.pa.get_device_count.return_value = 1
        self.pa.get_device_info_by_index.return_value = self._info()
        self.pa.get_default_input_device_info.return_value = self._info()
        self.pa.get_default_output_device_info.return_value = self._info()
        self.pa.open.side_effect = self._open

    @staticmethod
    def _info():
        return {"index": 0, "name": "ReSpeaker Flex (fake)", "maxInputChannels": 1,
                "maxOutputChannels": 1, "defaultSampleRate": 48000.0}

    def _open(self, **kwargs):
        stream = MagicMock()
        if kwargs.get("input"):
            stream.read.side_effect = self._read
        else:
            stream.write.side_effect = self.played.extend
        return stream

    def _read(self, frames, **kwargs):
        # A real capture blocks for as long as the audio lasts. Without that the
        # thread would publish flat out and drown the subscriber.
        time.sleep(frames / MIC_RATE)
        return self.captured


class AudioE2EBase(unittest.TestCase):
    def setUp(self):
        self.device = FakePortAudio()
        patcher = patch("network.audio.pyaudio")
        pyaudio = patcher.start()
        self.addCleanup(patcher.stop)
        pyaudio.PyAudio.return_value = self.device.pa

        mic_port, speaker_port = next(_PORTS), next(_PORTS)
        self.context = zmq.Context()

        with redirect_stdout(io.StringIO()):  # hush the bind and connect banners
            # doa=False: PortAudio is faked here, and the USB side would not be.
            # These tests must pass on a machine with no ReSpeaker plugged in.
            self.streamer = AudioStreamer(self.context, mic_port=mic_port,
                                          speaker_port=speaker_port, mic_rate=MIC_RATE,
                                          speaker_rate=SPEAKER_RATE, chunk_frames=CHUNK,
                                          doa=False)
            self.robot = RobotHead(ip="127.0.0.1", cmd_port=next(_PORTS),
                                   telemetry_port=next(_PORTS), video_port=next(_PORTS),
                                   mic_port=mic_port, speaker_port=speaker_port,
                                   cmd_timeout_ms=500)

        self.addCleanup(self._teardown)
        time.sleep(0.3)  # let the SUB and PUSH connections settle before we rely on them

    def _teardown(self):
        with redirect_stdout(io.StringIO()):
            self.streamer.stop()
        for sock in (self.robot.cmd_socket, self.robot.telemetry_socket,
                     self.robot.video_socket, self.robot.mic_socket,
                     self.robot.speaker_socket):
            sock.close(linger=0)
        self.streamer.mic_socket.close(linger=0)
        self.streamer.speaker_socket.close(linger=0)
        self.context.term()

    def start(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            self.streamer.start(**kwargs)

    def wait_for(self, predicate, timeout=3.0):
        """Polls until the audio has made it across, or gives up."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.02)
        return False

    def next_chunk(self, timeout=3.0):
        """The first microphone block to arrive, or None."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            chunk = self.robot.get_audio_chunk(timeout_ms=200)
            if chunk is not None:
                return chunk
        return None


class TestMicrophoneToSdk(AudioE2EBase):
    def test_captured_audio_arrives_intact(self):
        self.start(mic=True, speaker=False)

        chunk = self.next_chunk()

        self.assertIsNotNone(chunk, "no microphone audio reached the SDK")
        self.assertEqual(chunk.pcm, self.device.captured)
        self.assertEqual(chunk.rate, MIC_RATE)
        self.assertEqual(chunk.channels, 1)
        self.assertEqual(chunk.width, 2)

    def test_chunks_are_numbered_so_a_client_can_see_drops(self):
        self.start(mic=True, speaker=False)

        seqs = []
        while len(seqs) < 3:
            chunk = self.next_chunk()
            self.assertIsNotNone(chunk, "microphone audio stopped arriving")
            seqs.append(chunk.seq)

        self.assertEqual(seqs, sorted(seqs), "chunks arrived out of order")
        self.assertEqual(len(set(seqs)), len(seqs), "sequence numbers repeated")

    def test_samples_come_back_as_a_usable_array(self):
        """The recogniser-facing half of the SDK: bytes in, int16 frames out."""
        self.start(mic=True, speaker=False)

        chunk = self.next_chunk()

        self.assertIsNotNone(chunk)
        np.testing.assert_array_equal(chunk.samples[:, 0], np.arange(CHUNK, dtype=np.int16))

    def test_nothing_is_published_before_start(self):
        """The device stays closed until a client asks, exactly like the camera."""
        self.assertIsNone(self.robot.get_audio_chunk(timeout_ms=300))


class TestDirectionOverTheWire(AudioE2EBase):
    """Direction is published in the mic header, so it crosses the same socket
    as the audio and has to survive the same JSON round trip."""

    def test_direction_arrives_with_the_chunk(self):
        self.streamer.doa_reader = MagicMock()
        self.streamer.doa_reader.read.return_value = {"angle": 243, "speech": True}
        self.start(mic=True, speaker=False)

        chunk = self.next_chunk()

        self.assertIsNotNone(chunk)
        self.assertIsNotNone(chunk.direction, "direction did not reach the SDK")
        self.assertEqual(chunk.direction.angle, 243)
        self.assertTrue(chunk.direction.speech)

    def test_a_head_without_an_array_reports_no_direction(self):
        """Not zero: 0 degrees is a real bearing, so 'cannot tell' has to be
        distinguishable from 'straight ahead'."""
        self.start(mic=True, speaker=False)

        chunk = self.next_chunk()

        self.assertIsNotNone(chunk)
        self.assertIsNone(chunk.direction)


class TestSdkToSpeaker(AudioE2EBase):
    def tone(self, frames, rate):
        t = np.linspace(0, frames / rate, frames, endpoint=False)
        return (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)

    def test_matching_audio_reaches_the_speaker_untouched(self):
        self.start(mic=False, speaker=True)
        pcm = self.tone(1024, SPEAKER_RATE).tobytes()

        self.robot.play_audio(pcm, rate=SPEAKER_RATE, channels=1)

        self.assertTrue(self.wait_for(lambda: len(self.device.played) >= len(pcm)),
                        f"only {len(self.device.played)} of {len(pcm)} bytes were played")
        self.assertEqual(bytes(self.device.played), pcm)

    def test_a_tts_rate_is_resampled_by_the_pi(self):
        """A client sends whatever its engine emitted; matching formats is the
        Pi's job, because only the Pi knows what it opened the speaker at."""
        self.start(mic=False, speaker=True)
        pcm = self.tone(2205, 22050).tobytes()
        expected = len(pcm) * SPEAKER_RATE // 22050

        self.robot.play_audio(pcm, rate=22050, channels=1)

        self.assertTrue(self.wait_for(lambda: len(self.device.played) >= expected * 0.98),
                        f"only {len(self.device.played)} of ~{expected} bytes were played")
        # Resampling 22.05k -> 48k is not an integer ratio, so allow a chunk of slack.
        self.assertLess(abs(len(self.device.played) - expected), 4 * CHUNK)

    def test_stereo_is_mixed_down_for_a_mono_speaker(self):
        self.start(mic=False, speaker=True)
        mono = self.tone(512, SPEAKER_RATE)
        stereo = np.repeat(mono, 2).tobytes()  # same signal in both channels

        self.robot.play_audio(stereo, rate=SPEAKER_RATE, channels=2)

        self.assertTrue(self.wait_for(lambda: len(self.device.played) >= len(mono) * 2))
        np.testing.assert_allclose(
            np.frombuffer(self.device.played, dtype=np.int16), mono, atol=1)

    def test_audio_sent_before_start_is_not_played_later(self):
        """Whatever was pushed while the speaker was closed is stale by the time
        it opens; playing it would be a voice from several minutes ago."""
        self.robot.play_audio(self.tone(512, SPEAKER_RATE).tobytes(), rate=SPEAKER_RATE)
        time.sleep(0.2)

        self.start(mic=False, speaker=True)
        time.sleep(0.4)

        self.assertEqual(len(self.device.played), 0)

    def test_pushing_at_a_daemon_that_is_not_there_raises(self):
        """A PUSH socket with no peer blocks forever on send, so the SDK puts a
        deadline on it -- otherwise a stopped daemon hangs the caller."""
        with redirect_stdout(io.StringIO()):
            robot = RobotHead(ip="127.0.0.1", cmd_port=next(_PORTS),
                              telemetry_port=next(_PORTS), video_port=next(_PORTS),
                              mic_port=next(_PORTS), speaker_port=next(_PORTS),
                              cmd_timeout_ms=300)
        self.addCleanup(robot.speaker_socket.close, 0)

        with self.assertRaises(CleoError):
            robot.play_audio(b"\x00\x00" * 128, rate=SPEAKER_RATE)


if __name__ == '__main__':
    unittest.main()
