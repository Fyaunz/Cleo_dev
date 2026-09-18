# pi-daemon/tests/test_video.py
import unittest
import os
import stat
import sys
import tempfile
import time
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from network.video import MOUNT_ROTATION, VideoStreamer, _MjpegBuffer, find_rpicam


def wait_until(predicate, timeout=3.0):
    """Polls until predicate() is true, returning whether it ever was."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class VideoStreamerTestBase(unittest.TestCase):
    """Exercises the real thread with cv2 and ZMQ mocked out.

    backend is pinned rather than left on "auto": these tests are about the cv2
    source, and on a Pi -- where rpicam-vid is on PATH -- auto would pick the
    other one and quietly test nothing.
    """

    def setUp(self):
        patcher = patch("network.video.cv2")
        self.cv2 = patcher.start()
        self.addCleanup(patcher.stop)

        self.cap = MagicMock()
        self.cap.isOpened.return_value = True
        self.cap.read.return_value = (True, MagicMock())
        self.cv2.VideoCapture.return_value = self.cap
        self.cv2.imencode.return_value = (True, MagicMock())

        self.streamer = VideoStreamer(MagicMock(), port=5557, camera_index=0, backend="cv2")
        self.addCleanup(self.streamer.stop)


class TestExplicitStart(VideoStreamerTestBase):
    def test_construction_leaves_the_camera_closed(self):
        """The daemon builds this at boot; nothing may touch the camera yet."""
        self.cv2.VideoCapture.assert_not_called()
        self.assertFalse(self.streamer.is_streaming)

    def test_start_opens_the_camera_and_publishes(self):
        self.assertTrue(self.streamer.start())
        self.assertTrue(self.streamer.is_streaming)
        self.cv2.VideoCapture.assert_called_once_with(0)

    def test_second_start_does_not_open_a_second_camera(self):
        self.streamer.start()
        self.assertFalse(self.streamer.start(), "a redundant start claimed it started the stream")
        self.cv2.VideoCapture.assert_called_once()

    def test_unopenable_camera_raises_instead_of_failing_silently(self):
        """The router turns this into an error reply; a quiet thread death would
        leave the client waiting forever for frames that never come."""
        self.cap.isOpened.return_value = False

        with self.assertRaises(RuntimeError):
            self.streamer.start()

        self.assertFalse(self.streamer.is_streaming)
        self.cap.release.assert_called_once()


class TestStop(VideoStreamerTestBase):
    def test_stop_releases_the_camera(self):
        self.streamer.start()
        thread = self.streamer.thread

        self.assertTrue(self.streamer.stop())
        self.assertFalse(self.streamer.is_streaming)
        self.assertFalse(thread.is_alive(), "grabber thread outlived stop()")
        self.cap.release.assert_called_once()

    def test_stopping_an_idle_streamer_is_harmless(self):
        """main.py's finally block calls this whether or not video ever ran."""
        self.assertFalse(self.streamer.stop())

    def test_stream_can_be_restarted(self):
        """One client stops the stream, the next one asks for it again."""
        self.streamer.start()
        self.streamer.stop()

        self.assertTrue(self.streamer.start())
        self.assertEqual(self.cv2.VideoCapture.call_count, 2)


class TestMjpegBuffer(unittest.TestCase):
    """The framing rpicam-vid does not do for us.

    It writes finished JPEGs back to back with no lengths and no container, so
    everything downstream depends on finding the markers correctly.
    """

    def setUp(self):
        self.buffer = _MjpegBuffer()

    @staticmethod
    def frame(payload=b"body"):
        return b"\xff\xd8" + payload + b"\xff\xd9"

    def test_a_whole_frame_comes_back_whole(self):
        self.buffer.feed(self.frame())
        self.assertEqual(self.buffer.take(), self.frame())
        self.assertIsNone(self.buffer.take())

    def test_a_partial_frame_waits_for_the_rest(self):
        self.buffer.feed(b"\xff\xd8partial")
        self.assertIsNone(self.buffer.take())

        self.buffer.feed(b"-rest\xff\xd9")
        self.assertEqual(self.buffer.take(), self.frame(b"partial-rest"))

    def test_frames_split_across_reads_at_any_byte(self):
        """A read boundary can land inside a marker, so FF and D9 arrive apart."""
        stream = self.frame(b"one") + self.frame(b"two")
        for i in range(1, len(stream)):
            with self.subTest(split=i):
                buffer = _MjpegBuffer()
                buffer.feed(stream[:i])
                got = []
                while (frame := buffer.take()) is not None:
                    got.append(frame)
                buffer.feed(stream[i:])
                while (frame := buffer.take()) is not None:
                    got.append(frame)
                self.assertEqual(got, [self.frame(b"one"), self.frame(b"two")])

    def test_several_frames_in_one_read(self):
        self.buffer.feed(self.frame(b"a") + self.frame(b"b") + self.frame(b"c"))
        self.assertEqual([self.buffer.take() for _ in range(3)],
                         [self.frame(b"a"), self.frame(b"b"), self.frame(b"c")])

    def test_leading_junk_is_discarded(self):
        """Attaching mid-stream drops us into the middle of somebody's frame."""
        self.buffer.feed(b"tail of a frame we never saw the start of" + self.frame())
        self.assertEqual(self.buffer.take(), self.frame())

    def test_a_stream_that_is_not_mjpeg_does_not_grow_forever(self):
        buffer = _MjpegBuffer(max_frame_bytes=1024)
        buffer.feed(b"\xff\xd8" + b"x" * 4096)
        self.assertIsNone(buffer.take())

        # Resynced: the next real frame is found rather than appended to garbage.
        buffer.feed(self.frame())
        self.assertEqual(buffer.take(), self.frame())


class RpicamTestBase(unittest.TestCase):
    """Runs a fake rpicam-vid: a real subprocess writing real MJPEG to a pipe.

    No camera and no libcamera involved -- the point of the subprocess design is
    that this side only ever sees bytes on a pipe, so a stand-in that produces
    the same bytes exercises the same code the Pi runs.
    """

    FRAME = b"\xff\xd8" + b"fake-jpeg-payload" + b"\xff\xd9"

    def fake_camera(self, body):
        """Writes an executable Python script and points CLEO_RPICAM_BIN at it."""
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "rpicam-vid")
        with open(path, "w") as handle:
            handle.write(f"#!{sys.executable}\n{body}")
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        patcher = patch.dict(os.environ, {"CLEO_RPICAM_BIN": path})
        patcher.start()
        self.addCleanup(patcher.stop)
        return path

    def streamer(self, **kwargs):
        streamer = VideoStreamer(MagicMock(), port=5557, backend="rpicam", **kwargs)
        self.addCleanup(streamer.stop)
        return streamer


class TestRpicamSource(RpicamTestBase):
    def test_frames_off_the_pipe_are_published_untouched(self):
        """The Pi path never re-encodes: rpicam-vid already emits JPEG."""
        self.fake_camera(
            "import sys, time\n"
            f"frame = {self.FRAME!r}\n"
            "while True:\n"
            "    sys.stdout.buffer.write(frame)\n"
            "    sys.stdout.buffer.flush()\n"
            "    time.sleep(0.01)\n")
        streamer = self.streamer()

        self.assertTrue(streamer.start())
        self.assertTrue(wait_until(lambda: streamer.socket.send.call_count >= 2),
                        "no frames reached the socket")
        self.assertEqual(streamer.socket.send.call_args[0][0], self.FRAME)

    def test_a_camera_that_dies_at_startup_raises_with_its_own_message(self):
        """rpicam-vid explains itself on stderr and exits. Spawning succeeds
        either way, so only waiting for a real frame catches this -- and the
        client deserves the reason, not just 'it did not work'."""
        self.fake_camera(
            "import sys\n"
            "sys.stderr.write('ERROR: *** no cameras available ***\\n')\n"
            "sys.exit(1)\n")
        streamer = self.streamer()

        with self.assertRaises(RuntimeError) as caught:
            streamer.start()

        self.assertIn("no cameras available", str(caught.exception))
        self.assertFalse(streamer.is_streaming)

    def test_a_camera_that_never_sends_a_frame_times_out(self):
        """Spawned, alive, silent -- a plain Popen() would call this success."""
        self.fake_camera("import time\ntime.sleep(60)\n")
        streamer = self.streamer()

        with patch("network.video.FIRST_FRAME_TIMEOUT", 0.5):
            with self.assertRaises(RuntimeError) as caught:
                streamer.start()

        self.assertIn("no frame", str(caught.exception))
        self.assertFalse(streamer.is_streaming)

    def test_stop_kills_the_camera_process(self):
        """A pipe read has no timeout, so clearing the running flag is not
        enough on its own: the process has to go so the read returns EOF."""
        self.fake_camera(
            "import sys, time\n"
            f"frame = {self.FRAME!r}\n"
            "while True:\n"
            "    sys.stdout.buffer.write(frame)\n"
            "    sys.stdout.buffer.flush()\n"
            "    time.sleep(0.01)\n")
        streamer = self.streamer()
        streamer.start()
        proc = streamer.source._proc
        thread = streamer.thread

        self.assertTrue(streamer.stop())
        self.assertFalse(thread.is_alive(), "grabber thread outlived stop()")
        self.assertIsNotNone(proc.poll(), "rpicam-vid survived stop()")

    def test_a_camera_that_exits_mid_stream_leaves_the_streamer_idle(self):
        """Otherwise the next start_video answers 'already running' and the
        client subscribes to a socket nothing publishes on."""
        self.fake_camera(
            "import sys\n"
            f"sys.stdout.buffer.write({self.FRAME!r})\n"
            "sys.stdout.buffer.flush()\n")
        streamer = self.streamer()

        self.assertTrue(streamer.start())
        self.assertTrue(wait_until(lambda: not streamer.is_streaming),
                        "streamer still claims to be streaming after the camera exited")
        self.assertTrue(streamer.start(), "a dead stream could not be restarted")


class TestMountRotation(unittest.TestCase):
    """The head's camera is mounted with its ribbon connector facing up.

    That is 180 degrees from the way the sensor reads out, so the correction
    belongs here rather than in every client that ever looks at a frame.
    """

    def source(self, backend, rotation=None, env=None):
        with patch.dict(os.environ, env or {}, clear=False):
            streamer = VideoStreamer(MagicMock(), port=5557,
                                     backend=backend, rotation=rotation)
        with patch("network.video.find_rpicam", return_value="/usr/bin/rpicam-vid"):
            return streamer._make_source()

    def test_the_head_camera_is_turned_by_default(self):
        self.assertEqual(self.source("rpicam").rotation, MOUNT_ROTATION)

    def test_rpicam_asks_the_isp_to_do_it(self):
        """Not this side: rotating here would mean decode, rotate, re-encode,
        which is the work reading finished JPEGs off a pipe exists to avoid."""
        argv = self.source("rpicam")._argv()
        self.assertIn("--rotation", argv)
        self.assertEqual(argv[argv.index("--rotation") + 1], str(MOUNT_ROTATION))

    def test_a_development_webcam_is_left_alone(self):
        """The Mac's camera is the right way up; correcting it would be a bug."""
        self.assertEqual(self.source("cv2").rotation, 0)

    def test_the_environment_can_correct_a_usb_camera_in_the_head(self):
        self.assertEqual(
            self.source("cv2", env={"CLEO_CAMERA_ROTATION": "180"}).rotation, 180)

    def test_a_head_camera_mounted_the_other_way_can_be_told_so(self):
        """Remounting the module must not mean editing the source."""
        source = self.source("rpicam", env={"CLEO_CAMERA_ROTATION": "0"})
        self.assertEqual(source.rotation, 0)
        self.assertNotIn("--rotation", source._argv(),
                         "a no-op rotation is passed to binaries that may not accept it")

    def test_an_unsupported_rotation_is_refused_at_construction(self):
        """90 is not a thing rpicam-vid does, so nothing here pretends it is --
        better than a client silently getting it on one source and not the other."""
        for spec in ("90", "270", "upside-down"):
            with self.subTest(rotation=spec):
                with self.assertRaises(RuntimeError):
                    with patch.dict(os.environ, {"CLEO_CAMERA_ROTATION": spec}):
                        VideoStreamer(MagicMock(), port=5557)


class TestCv2Rotation(VideoStreamerTestBase):
    """The rotation the cv2 source has to perform itself."""

    def test_frames_are_turned_before_they_are_encoded(self):
        streamer = VideoStreamer(MagicMock(), port=5557, backend="cv2", rotation=180)
        self.addCleanup(streamer.stop)
        streamer.start()

        self.assertTrue(wait_until(lambda: self.cv2.rotate.called),
                        "frames were published without being rotated")
        self.assertEqual(self.cv2.rotate.call_args[0][1], self.cv2.ROTATE_180)
        # imencode must see the rotated frame, not the one off the camera.
        self.assertIs(self.cv2.imencode.call_args[0][1], self.cv2.rotate.return_value)

    def test_an_unrotated_source_does_no_work(self):
        self.streamer.start()
        self.assertTrue(wait_until(lambda: self.cv2.imencode.called))
        self.cv2.rotate.assert_not_called()


class TestBackendSelection(unittest.TestCase):
    """Which source 'auto' picks, and what happens when it cannot have one."""

    def test_auto_prefers_rpicam_where_it_exists(self):
        streamer = VideoStreamer(MagicMock(), port=5557)
        with patch("network.video.find_rpicam", return_value="/usr/bin/rpicam-vid"):
            self.assertEqual(streamer._make_source().name, "rpicam")

    def test_auto_falls_back_to_cv2_on_a_machine_without_it(self):
        """The Mac. No rpicam-vid anywhere, and a webcam OpenCV can open."""
        streamer = VideoStreamer(MagicMock(), port=5557)
        with patch("network.video.find_rpicam", return_value=None):
            self.assertEqual(streamer._make_source().name, "cv2")

    def test_the_environment_can_pin_the_backend(self):
        with patch.dict(os.environ, {"CLEO_CAMERA": "cv2"}):
            streamer = VideoStreamer(MagicMock(), port=5557)
        with patch("network.video.find_rpicam", return_value="/usr/bin/rpicam-vid"):
            self.assertEqual(streamer._make_source().name, "cv2")

    def test_asking_for_rpicam_without_it_says_so(self):
        streamer = VideoStreamer(MagicMock(), port=5557, backend="rpicam")
        with patch("network.video.find_rpicam", return_value=None):
            with self.assertRaises(RuntimeError) as caught:
                streamer.start()
        self.assertIn("rpicam-vid", str(caught.exception))

    def test_a_typo_in_the_backend_is_not_silently_ignored(self):
        streamer = VideoStreamer(MagicMock(), port=5557, backend="picamera")
        with self.assertRaises(RuntimeError):
            streamer.start()

    def test_find_rpicam_honours_the_environment_override(self):
        with patch.dict(os.environ, {"CLEO_RPICAM_BIN": "/opt/cam/rpicam-vid"}):
            self.assertEqual(find_rpicam(), "/opt/cam/rpicam-vid")


if __name__ == '__main__':
    unittest.main()
