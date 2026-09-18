# pi-daemon/tests/test_audio.py
import json
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import network.audio
from network.audio import AudioStreamer, DoaReader


def _device(index, name, inputs, outputs, rate=48000.0):
    return {"index": index, "name": name, "maxInputChannels": inputs,
            "maxOutputChannels": outputs, "defaultSampleRate": rate}


class AudioStreamerTestBase(unittest.TestCase):
    """Exercises the real threads with PortAudio and ZMQ mocked out."""

    devices = [_device(0, "Built-in Microphone", 1, 0),
               _device(1, "ReSpeaker Flex", 4, 2)]

    def setUp(self):
        patcher = patch("network.audio.pyaudio")
        self.pyaudio = patcher.start()
        self.addCleanup(patcher.stop)

        # A real zmq.Poller cannot poll a mocked socket, and the playback thread
        # polls before anything else. An empty result means "nothing arrived".
        zmq_patcher = patch("network.audio.zmq")
        self.zmq = zmq_patcher.start()
        self.addCleanup(zmq_patcher.stop)
        self.zmq.Poller.return_value.poll.return_value = []

        self.pa = MagicMock()
        self.pyaudio.PyAudio.return_value = self.pa
        # Read through self.devices so a test can swap the machine's hardware.
        self.pa.get_device_count.side_effect = lambda: len(self.devices)
        self.pa.get_device_info_by_index.side_effect = lambda i: self.devices[i]
        self.pa.get_default_input_device_info.side_effect = lambda: self.devices[0]
        self.pa.get_default_output_device_info.side_effect = lambda: self.devices[0]

        # Distinct streams so a test can tell the two directions apart.
        self.streams = []
        self.pa.open.side_effect = self._open

        self.context = MagicMock()
        self.sockets = []
        self.context.socket.side_effect = self._socket

        self.streamer = self._build()
        self.addCleanup(self.streamer.stop)

    def _build(self):
        return self.make(mic_port=5558, speaker_port=5559)

    def make(self, **kwargs):
        """A streamer with the USB side off.

        doa=False throughout: the direction reader has its own tests, and a real
        ReSpeaker plugged into the machine running these must not change what
        they assert.
        """
        kwargs.setdefault("doa", False)
        streamer = AudioStreamer(self.context, **kwargs)
        self.addCleanup(streamer.stop)
        return streamer

    def _open(self, **kwargs):
        stream = MagicMock()
        stream.kwargs = kwargs
        # 16 bit, whatever channel count was asked for.
        stream.read.return_value = b"\x00\x01" * (kwargs.get("channels", 1) * 8)
        self.streams.append(stream)
        return stream

    def _socket(self, kind):
        socket = MagicMock()
        socket.kind = kind
        self.sockets.append(socket)
        return socket

    @property
    def mic_socket(self):
        """Bound first, in __init__: the PUB carrying captured audio."""
        return self.sockets[0]

    @property
    def opened_input(self):
        return [s for s in self.streams if s.kwargs.get("input")]

    @property
    def opened_output(self):
        return [s for s in self.streams if s.kwargs.get("output")]


class TestExplicitStart(AudioStreamerTestBase):
    def test_construction_leaves_the_device_closed(self):
        """The daemon builds this at boot; nothing may open the ReSpeaker yet."""
        self.pyaudio.PyAudio.assert_not_called()
        self.pa.open.assert_not_called()
        self.assertFalse(self.streamer.is_streaming)

    def test_start_opens_both_directions(self):
        self.assertTrue(self.streamer.start())

        self.assertTrue(self.streamer.mic_running)
        self.assertTrue(self.streamer.speaker_running)
        self.assertEqual(len(self.opened_input), 1)
        self.assertEqual(len(self.opened_output), 1)

    def test_second_start_does_not_open_a_second_device(self):
        self.streamer.start()
        self.assertFalse(self.streamer.start(), "a redundant start claimed it started audio")
        self.assertEqual(self.pa.open.call_count, 2)

    def test_speaker_only_leaves_the_microphone_closed(self):
        """A client that only wants the robot to talk must not open the mic."""
        self.streamer.start(mic=False, speaker=True)

        self.assertFalse(self.streamer.mic_running)
        self.assertTrue(self.streamer.speaker_running)
        self.assertEqual(self.opened_input, [])

    def test_the_second_direction_can_be_added_later(self):
        self.streamer.start(mic=False, speaker=True)
        self.assertTrue(self.streamer.start(mic=True, speaker=False))

        self.assertTrue(self.streamer.mic_running)
        self.assertTrue(self.streamer.speaker_running)

    def test_unopenable_device_raises_instead_of_failing_silently(self):
        """The router turns this into an error reply; a quiet thread death would
        leave the client waiting forever for audio that never comes."""
        self.pa.open.side_effect = OSError("Invalid sample rate")

        with self.assertRaises(RuntimeError):
            self.streamer.start()

        self.assertFalse(self.streamer.is_streaming)

    def test_a_failed_speaker_does_not_leave_the_mic_open(self):
        """Half-open is worse than closed: the client hears about the failure but
        the microphone would stay busy with nothing draining it."""
        def fail_on_output(**kwargs):
            if kwargs.get("output"):
                raise OSError("Device unavailable")
            return self._open(**kwargs)

        self.pa.open.side_effect = fail_on_output

        with self.assertRaises(RuntimeError):
            self.streamer.start()

        self.assertFalse(self.streamer.is_streaming)
        self.opened_input[0].close.assert_called_once()
        self.pa.terminate.assert_called_once()


class TestDeviceSelection(AudioStreamerTestBase):
    def test_the_respeaker_is_preferred_over_the_default(self):
        """device=None means 'the ReSpeaker if it is plugged in' -- the Pi has no
        other sound card, but the machine this gets developed on does."""
        self.streamer.start()

        self.assertEqual(self.opened_input[0].kwargs["input_device_index"], 1)
        self.assertEqual(self.opened_output[0].kwargs["output_device_index"], 1)

    def test_a_name_fragment_picks_the_device(self):
        streamer = self.make(device="built-in")
        streamer.start(mic=True, speaker=False)

        self.assertEqual(self.opened_input[0].kwargs["input_device_index"], 0)

    def test_an_index_is_used_verbatim(self):
        streamer = self.make(device=1)
        streamer.start(mic=True, speaker=False)

        self.assertEqual(self.opened_input[0].kwargs["input_device_index"], 1)

    def test_a_name_that_matches_nothing_is_an_error(self):
        streamer = self.make(device="nonexistent")

        with self.assertRaises(RuntimeError):
            streamer.start()

    def test_falls_back_to_the_default_when_no_respeaker_is_present(self):
        """The daemon has to run on the development machine too."""
        self.devices = [_device(0, "Built-in Audio", 1, 2)]
        streamer = self.make()

        streamer.start()

        self.assertEqual(self.opened_input[0].kwargs["input_device_index"], 0)

    def test_the_speaker_adopts_the_devices_own_rate(self):
        """The Flex runs at 16kHz natively, so opening it at a hardcoded 48k is
        an argument with ALSA that nothing gains: clients are converted to
        whatever rate we ended up with either way."""
        self.devices = [_device(0, "ReSpeaker Flex", 2, 2, rate=16000.0)]
        streamer = self.make()

        streamer.start(mic=False, speaker=True)

        self.assertEqual(streamer.speaker_rate, 16000)
        self.assertEqual(self.opened_output[0].kwargs["rate"], 16000)

    def test_an_explicit_speaker_rate_wins(self):
        streamer = self.make(speaker_rate=44100)

        streamer.start(mic=False, speaker=True)

        self.assertEqual(self.opened_output[0].kwargs["rate"], 44100)

    def test_mono_capture_falls_back_to_the_arrays_native_width(self):
        """The Flex exposes its microphones as one fixed multi-channel stream and
        ALSA refuses anything narrower, so mono has to be produced afterwards."""
        def mono_unsupported(**kwargs):
            if kwargs.get("input") and kwargs["channels"] == 1:
                raise OSError("Invalid number of channels")
            return self._open(**kwargs)

        self.pa.open.side_effect = mono_unsupported
        self.streamer.start(mic=True, speaker=False)

        self.assertTrue(self.streamer.mic_running)
        self.assertEqual(self.opened_input[0].kwargs["channels"], 4)


class TestMicrophonePublishing(AudioStreamerTestBase):
    def test_published_blocks_describe_their_own_format(self):
        """A client should not have to hardcode the robot's capture rate."""
        self.streamer.start(mic=True, speaker=False)
        self.streamer.stop()

        self.assertTrue(self.mic_socket.send_multipart.called, "nothing was published")
        header_raw, pcm = self.mic_socket.send_multipart.call_args.args[0]
        header = json.loads(header_raw.decode())

        self.assertEqual(header["rate"], 16000)
        self.assertEqual(header["channels"], 1)
        self.assertEqual(header["width"], 2)
        self.assertGreaterEqual(header["seq"], 1)
        self.assertTrue(pcm)

    def test_a_multi_channel_capture_is_narrowed_before_publishing(self):
        """Channel 0 only: on an array the mics sit centimetres apart, so summing
        them comb-filters the result."""
        self.streamer._mic_capture_channels = 4
        frames = np.arange(16, dtype=np.int16).reshape(-1, 4)

        narrowed = np.frombuffer(self.streamer._reduce_capture(frames.tobytes()), dtype=np.int16)

        np.testing.assert_array_equal(narrowed, frames[:, 0])


class TestStop(AudioStreamerTestBase):
    def test_stop_closes_the_device(self):
        self.streamer.start()
        mic_thread = self.streamer._mic_thread

        self.assertTrue(self.streamer.stop())

        self.assertFalse(self.streamer.is_streaming)
        self.assertFalse(mic_thread.is_alive(), "capture thread outlived stop()")
        for stream in self.streams:
            stream.close.assert_called_once()
        self.pa.terminate.assert_called_once()

    def test_stopping_idle_audio_is_harmless(self):
        """main.py's finally block calls this whether or not audio ever ran."""
        self.assertFalse(self.streamer.stop())

    def test_stopping_one_direction_keeps_the_device_open(self):
        self.streamer.start()
        self.streamer.stop(mic=True, speaker=False)

        self.assertFalse(self.streamer.mic_running)
        self.assertTrue(self.streamer.speaker_running)
        self.pa.terminate.assert_not_called()

    def test_a_device_that_failed_mid_stream_is_still_closed(self):
        """The capture thread clears its own flag when the ReSpeaker is unplugged.
        If stop() trusted the flag, that stream would never be closed and the next
        start() would find the device busy."""
        self.streamer.start()
        self.streamer._mic_running = False  # as the dying capture thread leaves it
        stream = self.opened_input[0]

        self.assertTrue(self.streamer.stop())

        stream.close.assert_called_once()
        self.pa.terminate.assert_called_once()

    def test_audio_can_be_restarted(self):
        """One client stops the stream, the next one asks for it again."""
        self.streamer.start()
        self.streamer.stop()

        self.assertTrue(self.streamer.start())
        self.assertEqual(len(self.opened_input), 2)


class TestPlaybackConversion(AudioStreamerTestBase):
    """The Pi adapts to whatever a client's TTS emitted, so no client has to ask."""

    def setUp(self):
        super().setUp()
        self.streamer.start(mic=False, speaker=True)
        self.assertEqual(self.streamer.speaker_rate, 48000,
                         "the fake device's default rate should have been adopted")
        self.assertEqual(self.streamer.speaker_channels, 1)

    def pcm(self, samples):
        return np.asarray(samples, dtype=np.int16).tobytes()

    def convert(self, samples, rate, channels=1):
        out = self.streamer._convert_for_output(
            self.pcm(samples), {"rate": rate, "channels": channels})
        return np.frombuffer(out, dtype=np.int16)

    def test_matching_format_is_passed_through_untouched(self):
        pcm = self.pcm([1, 2, 3, 4])
        result = self.streamer._convert_for_output(pcm, {"rate": 48000, "channels": 1})

        self.assertIs(result, pcm, "a client that already matches should cost nothing")

    def test_upsampling_stretches_the_block(self):
        result = self.convert([0] * 100, rate=24000)
        self.assertEqual(len(result), 200)

    def test_downsampling_shrinks_the_block(self):
        result = self.convert([0] * 200, rate=96000)
        self.assertEqual(len(result), 100)

    def test_stereo_is_mixed_down_to_the_speakers_channel_count(self):
        # Interleaved L/R: a constant 100 against a constant 300 averages to 200.
        result = self.convert([100, 300] * 8, rate=48000, channels=2)

        self.assertEqual(len(result), 8)
        np.testing.assert_array_equal(result, np.full(8, 200, dtype=np.int16))

    def test_resampling_is_continuous_across_chunks(self):
        """Resampling each chunk from scratch restarts the interpolation at every
        boundary, and at 40ms chunks that discontinuity is an audible buzz."""
        ramp = np.arange(0, 400, dtype=np.int16)  # one straight line, split in two
        first = self.convert(ramp[:200], rate=24000)
        second = self.convert(ramp[200:], rate=24000)

        joined = np.concatenate([first, second]).astype(np.float64)
        steps = np.diff(joined)
        # A ramp resampled 2x should rise by a constant 0.5 everywhere, so no step
        # may exceed one quantisation unit -- least of all at the seam.
        self.assertLess(steps.max(), 1.5, "discontinuity at the chunk boundary")
        self.assertGreater(steps.min(), -0.5)

    def test_an_odd_sized_block_does_not_crash(self):
        """A truncated final chunk is a client's business, not a reason to die."""
        result = self.streamer._convert_for_output(b"\x01\x02\x03", {"rate": 24000, "channels": 2})
        self.assertEqual(result, b"")


class TestPlaybackMessages(AudioStreamerTestBase):
    """Anyone on the network can reach the speaker port, so what arrives there is
    checked before it is played."""

    def setUp(self):
        super().setUp()
        # Through the real start(), so the streamer is in the state the daemon
        # actually puts it in -- notably with a negotiated speaker rate.
        self.streamer.start(mic=False, speaker=True)
        self.stream = self.opened_output[0]

    def message(self, client_id="client-1", rate=48000, channels=1, width=2):
        header = {"rate": rate, "channels": channels, "width": width, "client_id": client_id}
        return [json.dumps(header).encode(), np.zeros(64, dtype=np.int16).tobytes()]

    def test_audio_reaches_the_speaker(self):
        self.assertTrue(self.streamer._play_message(self.message()))
        self.stream.write.assert_called()

    def test_audio_from_the_lock_holder_is_played(self):
        self.streamer.access_check = lambda cid: cid == "client-1"
        self.assertTrue(self.streamer._play_message(self.message(client_id="client-1")))

    def test_audio_from_another_client_is_dropped(self):
        """The session lock is what stops a second client talking over the first."""
        self.streamer.access_check = lambda cid: cid == "client-1"

        self.assertFalse(self.streamer._play_message(self.message(client_id="intruder")))
        self.stream.write.assert_not_called()

    def test_an_unsupported_sample_width_is_dropped_not_played_as_noise(self):
        self.assertFalse(self.streamer._play_message(self.message(width=4)))
        self.stream.write.assert_not_called()

    def test_a_malformed_message_does_not_stop_playback(self):
        self.assertFalse(self.streamer._play_message([b"only one part"]))
        self.assertFalse(self.streamer._play_message([b"not json", b"\x00\x00"]))
        self.stream.write.assert_not_called()

    def test_a_long_upload_is_written_in_device_sized_slices(self):
        """One message holding a whole sentence would otherwise block the thread
        for the length of the sentence, and flush with it."""
        self.streamer.chunk_frames = 16
        self.streamer._play_message(
            [json.dumps({"rate": 48000, "channels": 1, "width": 2}).encode(),
             np.zeros(160, dtype=np.int16).tobytes()])

        self.assertEqual(self.stream.write.call_count, 10)


class DoaTestBase(unittest.TestCase):
    """Drives DoaReader against a faked pyusb, so no board has to be plugged in."""

    def setUp(self):
        self.usb = MagicMock()
        patcher = patch.object(network.audio, "usb", self.usb)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.dev = MagicMock()
        self.dev.idVendor, self.dev.idProduct = 0x2886, 0x001A
        self.usb.core.find.return_value = [self.dev]

        # Status byte first, then the payload: version 2.0.9, then a direction.
        self.responses = {(48, 0x80): [0, 2, 0, 9]}
        self.dev.ctrl_transfer.side_effect = self._transfer

        self.reader = DoaReader()

    def _transfer(self, request_type, request, wvalue, windex, length, timeout):
        return list(self.responses.get((windex, wvalue), [0x41] + [0] * (length - 1)))

    def set_doa(self, angle, speech):
        self.responses[(20, 0x80 | 18)] = [0, angle & 0xFF, angle >> 8, int(speech), 0]


class TestDoaDecoding(DoaTestBase):
    def test_open_identifies_the_board_by_a_version_read(self):
        """Answering a known read is the only proof the control map matches;
        the device answers a wrong command with a status byte, not a stall."""
        self.assertTrue(self.reader.open())
        self.assertTrue(self.reader.available)

    def test_angle_is_a_little_endian_pair(self):
        self.reader.open()
        self.set_doa(287, speech=True)

        self.assertEqual(self.reader.read(), {"angle": 287, "speech": True})

    def test_speech_flag_is_separate_from_the_angle(self):
        """The board holds its last direction through silence, so an angle
        without this flag says nothing about whether anyone is talking now."""
        self.reader.open()
        self.set_doa(90, speech=False)

        self.assertEqual(self.reader.read(), {"angle": 90, "speech": False})

    def test_reads_are_rate_limited(self):
        """One control transfer per audio chunk is pointless: the DSP does not
        update that fast, and the capture thread pays for every one."""
        self.reader.open()
        self.set_doa(45, speech=True)
        self.dev.ctrl_transfer.reset_mock()

        for _ in range(10):
            self.reader.read()

        self.assertEqual(self.dev.ctrl_transfer.call_count, 1)

    def test_the_command_map_is_the_one_the_board_answers(self):
        """Regression: these four numbers are the whole protocol. The device
        replies to a wrong resource or command with status 0x41 rather than an
        error, so a typo here shows up as a direction that never moves."""
        self.reader.open()
        self.dev.ctrl_transfer.reset_mock()
        self.set_doa(10, speech=True)
        self.reader.read()

        _, request, wvalue, windex, length, _ = self.dev.ctrl_transfer.call_args.args
        self.assertEqual(request, 0)
        self.assertEqual(windex, 20, "DOA resource id")
        self.assertEqual(wvalue, 0x80 | 18, "DOA command id, with the read bit")
        self.assertEqual(length, 5, "4-byte payload plus the status byte")


class TestDoaFailures(DoaTestBase):
    def test_no_pyusb_is_not_an_error(self):
        """Direction is an extra. A head without it still hears and speaks."""
        with patch.object(network.audio, "usb", None):
            reader = DoaReader()
            self.assertFalse(reader.open())
            self.assertIsNone(reader.read())

    def test_no_board_is_not_an_error(self):
        self.usb.core.find.return_value = []

        self.assertFalse(self.reader.open())
        self.assertIsNone(self.reader.read())

    def test_a_board_that_fails_the_version_read_is_refused(self):
        """Some other Seeed device on the same vendor id would answer USB but
        not this control map, and would report a fabricated direction forever."""
        self.responses.clear()

        self.assertFalse(self.reader.open())

    def test_a_bad_status_byte_is_not_read_as_data(self):
        self.reader.open()
        self.responses[(20, 0x80 | 18)] = [0x41, 99, 0, 1, 0]

        self.assertIsNone(self.reader.read())

    def test_a_run_of_failures_gives_up_instead_of_logging_forever(self):
        self.reader.open()
        self.dev.ctrl_transfer.side_effect = OSError("Pipe error")

        for _ in range(25):
            self.reader._cached_at = 0.0  # defeat the rate limit
            self.reader.read()

        self.assertFalse(self.reader.available, "kept polling a device that never answers")

    def test_a_usb_failure_keeps_the_last_known_direction(self):
        """A dropped transfer is a glitch, not a reason to tell a client that a
        head with a working array suddenly has no sense of direction."""
        self.reader.open()
        self.set_doa(120, speech=True)
        self.assertEqual(self.reader.read()["angle"], 120)

        self.dev.ctrl_transfer.side_effect = OSError("Pipe error")
        self.reader._cached_at = 0.0

        self.assertEqual(self.reader.read(), {"angle": 120, "speech": True})


class TestDoaInStream(AudioStreamerTestBase):
    """The angle rides along with the audio it describes."""

    def setUp(self):
        super().setUp()
        # Give the streamer from the base class a reader, rather than building a
        # second one whose sockets are not the ones mic_socket watches.
        self.streamer.doa_reader = MagicMock()
        self.streamer.doa_reader.read.return_value = {"angle": 75, "speech": True}

    def published_header(self):
        self.streamer.start(mic=True, speaker=False)
        self.streamer.stop()
        header_raw, _ = self.mic_socket.send_multipart.call_args.args[0]
        return json.loads(header_raw.decode())

    def test_direction_travels_with_the_audio(self):
        header = self.published_header()

        self.assertEqual(header["doa"], 75)
        self.assertTrue(header["speech"])

    def test_the_usb_device_is_opened_once_for_every_caller(self):
        """Regression: the capture thread and the command loop both ask for
        direction. Checking an 'opened' flag without holding it across the open
        let the second caller read a device the first was still opening, and the
        first chunks went out with no direction on a head that has an array."""
        self.streamer._doa_opened = False
        for _ in range(5):
            self.streamer.get_doa()

        self.streamer.doa_reader.open.assert_called_once()

    def test_a_head_without_an_array_publishes_audio_unchanged(self):
        """No direction is a missing key, not a zero -- 0 degrees is a real
        bearing and a client must be able to tell the two apart."""
        self.streamer.doa_reader.read.return_value = None
        header = self.published_header()

        self.assertNotIn("doa", header)
        self.assertNotIn("speech", header)
        self.assertEqual(header["rate"], 16000, "audio itself must be unaffected")


if __name__ == '__main__':
    unittest.main()
