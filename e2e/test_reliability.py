"""End-to-end checks for command delivery, over real ZMQ sockets.

Unlike pi_controller/tests, these drive the real SDK against a real listener and
router on a loopback port; only the hardware is mocked. That is the only way to
cover the bits that live between the two: the REQ socket reset, resends, and the
Pi's reply cache.

Run directly:
    .devenv/state/venv/bin/python3 e2e/test_reliability.py -v
"""
import io
import itertools
import os
import sys
import threading
import time
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pi_controller"))
sys.path.insert(0, os.path.join(ROOT, "cleosdk"))

import zmq

from network.command import CommandListener
from network.dedup import ReplyCache
from network.router import route_command
from cleosdk.cleo import CleoError, CommandTimeout, RobotError, RobotHead

# Every test gets fresh ports so a resend queued by one cannot leak into the next.
_PORTS = itertools.count(5700)


class FakeDaemon:
    """The real listener, router and reply cache driving mocked hardware.

    Doubles as the SessionManager so a test can open and close the lock at will.
    """

    def __init__(self, port):
        self.port = port
        self.servo = MagicMock()
        self.servo.go_to.return_value = [1, 2, 3]
        self.controller = MagicMock()
        self.controller.get_telemetry.return_value = {"is_moving": False}
        self.playback = MagicMock()
        self.video = MagicMock()

        self.allow_access = True
        self.stall_next_for = 0.0
        self.requests = 0

        self._stop = threading.Event()
        self._thread = None

    def check_access(self, client_id):
        return self.allow_access

    def release(self, client_id):
        pass

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        time.sleep(0.3)  # let the bind settle before a client connects

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        context = zmq.Context()
        listener = CommandListener(context, port=self.port)
        cache = ReplyCache()
        while not self._stop.is_set():
            msg = listener.get_command()
            if msg:
                self.requests += 1
                if self.stall_next_for:
                    # Answer so late the client has already given up: the reply
                    # goes to a closed socket and is lost for real, exactly as a
                    # dropped packet would be.
                    delay, self.stall_next_for = self.stall_next_for, 0.0
                    time.sleep(delay)
                route_command(msg, listener, self, self.servo, self.controller,
                              self.playback, self.video, reply_cache=cache)
            time.sleep(0.005)
        listener.socket.close(linger=0)
        context.term()


class E2ETestBase(unittest.TestCase):
    timeout_ms = 400
    attempts = 5

    def setUp(self):
        self.daemon = FakeDaemon(next(_PORTS))
        self.daemon.start()
        self.addCleanup(self.daemon.stop)

        with redirect_stdout(io.StringIO()):  # hush the connection banner
            # Every port comes from the loopback range, including the ones this
            # file never uses: left at their defaults they would connect to a
            # daemon actually running on this machine.
            self.robot = RobotHead(ip="127.0.0.1", cmd_port=self.daemon.port,
                                   telemetry_port=next(_PORTS), video_port=next(_PORTS),
                                   mic_port=next(_PORTS), speaker_port=next(_PORTS),
                                   cmd_timeout_ms=self.timeout_ms, cmd_attempts=self.attempts)
        # Close the sockets by hand: teardown must not depend on the network,
        # since some tests deliberately leave the robot unreachable.
        self.addCleanup(self._close_robot)

    def _close_robot(self):
        for sock in (self.robot.cmd_socket, self.robot.telemetry_socket,
                     self.robot.video_socket, self.robot.mic_socket,
                     self.robot.speaker_socket):
            sock.close(linger=0)

    def capture(self, fn):
        buf = io.StringIO()
        with redirect_stdout(buf):
            fn()
        return buf.getvalue()


class TestLostReply(E2ETestBase):
    """A lost reply is indistinguishable from a lost command, so the SDK resends."""

    def test_resend_after_lost_reply_moves_the_head_once(self):
        self.daemon.stall_next_for = 1.0  # > timeout_ms, so the client gives up

        reply = self.robot.move_head(yaw=10.0, pitch=5.0, roll=2.0)

        self.assertEqual(reply.get("status"), "ok", "SDK never got a usable reply")
        self.assertGreaterEqual(self.daemon.requests, 2, "SDK should have resent")
        self.assertEqual(self.daemon.servo.go_to.call_count, 1,
                         "head moved more than once: the reply cache did not dedupe")

    def test_next_command_works_on_the_recycled_socket(self):
        self.daemon.stall_next_for = 1.0
        self.robot.move_head(yaw=10.0, pitch=5.0, roll=2.0)

        reply = self.robot.move_head(yaw=1.0, pitch=1.0, roll=1.0)

        self.assertEqual(reply.get("status"), "ok")
        self.assertEqual(self.daemon.servo.go_to.call_count, 2)

    def test_unreachable_robot_raises_instead_of_returning_nothing(self):
        self.daemon.stop()

        with self.assertRaises(CommandTimeout):
            self.robot.move_head(yaw=2.0, pitch=0.0, roll=0.0)


class TestOkReporting(E2ETestBase):
    def test_quiet_by_default(self):
        out = self.capture(lambda: self.robot.move_head(yaw=10.0, pitch=5.0, roll=2.0))
        self.assertNotIn("[cleo]", out)

    def test_verbose_prints_the_ok_message(self):
        self.robot.verbose = True
        out = self.capture(lambda: self.robot.move_head(yaw=10.0, pitch=5.0, roll=2.0))
        self.assertIn("[cleo] go_to: Head moving to [1, 2, 3] ticks", out)

    def test_verbose_does_not_spam_the_is_moving_poll(self):
        """move_head_ruckig(wait=True) polls is_moving at ~50Hz."""
        self.robot.verbose = True
        out = self.capture(lambda: self.robot.move_head_ruckig(yaw=1, pitch=1, roll=1, wait=True))
        self.assertNotIn("is_moving", out)


class TestRuckigWait(E2ETestBase):
    def test_wait_gives_up_on_a_head_that_never_stops_moving(self):
        """A trajectory that stops converging used to park the caller for good."""
        self.daemon.controller.get_telemetry.return_value = {"is_moving": True}

        with self.assertRaises(CommandTimeout) as ctx:
            self.robot.move_head_ruckig(yaw=1, pitch=1, roll=1, wait=True, wait_timeout=0.5)
        self.assertIn("still reports moving", str(ctx.exception))

    def test_refusal_mid_wait_reaches_the_caller(self):
        self.daemon.controller.get_telemetry.return_value = {"is_moving": True}
        self.daemon.allow_access = False

        with self.assertRaises(RobotError):
            self.robot.move_head_ruckig(yaw=1, pitch=1, roll=1, wait=True)

    def test_nonsense_limits_are_refused_before_the_head_moves(self):
        for kwargs in ({"v_max": 0}, {"a_max": -1}, {"j_max": 0}, {"duration": 0.0}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    self.robot.move_head_ruckig(yaw=1, pitch=1, roll=1, **kwargs)

        self.assertEqual(self.daemon.controller.set_target.call_count, 0)


class TestEars(E2ETestBase):
    """The ear DOFs, end to end: both motion paths and their own wait flag."""

    def test_timed_ear_move_reaches_the_hardware(self):
        reply = self.robot.move_ears(ear_left=30.0, ear_right=30.0, time_ms=600)

        self.assertEqual(reply.get("status"), "ok")
        kwargs = self.daemon.servo.go_to_ears.call_args.kwargs
        self.assertEqual((kwargs["ear_left"], kwargs["ear_right"]), (30.0, 30.0))
        self.assertEqual(kwargs["t_total"], 600)

    def test_ruckig_ear_move_reaches_the_controller(self):
        self.robot.move_ears_ruckig(ear_left=45.0, ear_right=-45.0, wait=False)

        kwargs = self.daemon.controller.set_ear_target.call_args.kwargs
        self.assertEqual((kwargs["ear_left"], kwargs["ear_right"]), (45.0, -45.0))
        self.daemon.controller.set_target.assert_not_called()

    def test_ear_wait_ignores_a_head_that_is_still_moving(self):
        """The whole point of the second generator: an ear flick during a slow
        head move must return as soon as the ears land, not wait out the neck."""
        self.daemon.controller.get_telemetry.return_value = {
            "is_moving": True, "ears_moving": False}

        self.robot.move_ears_ruckig(ear_left=20.0, ear_right=20.0,
                                    wait=True, wait_timeout=2.0)

    def test_ear_wait_gives_up_on_ears_that_never_stop(self):
        self.daemon.controller.get_telemetry.return_value = {"ears_moving": True}

        with self.assertRaises(CommandTimeout) as ctx:
            self.robot.move_ears_ruckig(ear_left=20.0, ear_right=20.0,
                                        wait=True, wait_timeout=0.5)
        self.assertIn("still report moving", str(ctx.exception))

    def test_head_wait_ignores_ears_that_are_still_moving(self):
        self.daemon.controller.get_telemetry.return_value = {
            "is_moving": False, "ears_moving": True}

        self.robot.move_head_ruckig(yaw=1, pitch=1, roll=1, wait=True, wait_timeout=2.0)

    def test_move_ears_needs_at_least_one_side(self):
        with self.assertRaises(ValueError):
            self.robot.move_ears()
        self.daemon.servo.go_to_ears.assert_not_called()

    def test_head_move_can_carry_the_ears(self):
        self.robot.move_head(yaw=10.0, pitch=0.0, roll=0.0, ear_left=15.0, ear_right=15.0)

        kwargs = self.daemon.servo.go_to.call_args.kwargs
        self.assertEqual((kwargs["ear_left"], kwargs["ear_right"]), (15.0, 15.0))


class TestErrorReporting(E2ETestBase):
    def test_refusal_raises(self):
        self.daemon.allow_access = False

        with self.assertRaises(RobotError) as ctx:
            self.robot.move_head(yaw=10.0, pitch=0.0, roll=0.0)
        self.assertIn("locked", str(ctx.exception))

    def test_is_moving_raises_instead_of_reporting_a_phantom_move(self):
        """Regression: is_moving did bool(reply["message"]).

        A refusal puts an error *string* in that field, and a non-empty string is
        truthy, so a locked robot reported 'moving' forever and the wait loop in
        move_head_ruckig hung for good.
        """
        self.daemon.allow_access = False

        with self.assertRaises(RobotError):
            self.robot.is_moving()

    def test_hardware_fault_reaches_the_caller(self):
        self.daemon.servo.go_to.side_effect = ValueError("Kinematic Limit Reached!")

        with self.assertRaises(RobotError) as ctx:
            self.robot.move_head(yaw=10.0, pitch=0.0, roll=0.0)
        self.assertIn("Kinematic Limit", str(ctx.exception))

    def test_errors_share_a_catchable_base(self):
        self.daemon.allow_access = False

        with self.assertRaises(CleoError):
            self.robot.move_head(yaw=10.0, pitch=0.0, roll=0.0)

    def test_zero_pose_is_not_an_error(self):
        """Regression: `not (yaw or pitch or roll)` refused the centred pose."""
        self.robot.move_head(yaw=0.0, pitch=0.0, roll=0.0)

        self.assertEqual(self.daemon.servo.go_to.call_count, 1)


class TestVideoOnDemand(E2ETestBase):
    """The camera is off until the client asks, so the asking has to work."""

    def test_start_and_stop_reach_the_camera(self):
        self.robot.start_video()
        self.daemon.video.start.assert_called_once()

        self.robot.stop_video()
        self.daemon.video.stop.assert_called_once()

    def test_a_dead_camera_raises_instead_of_streaming_nothing(self):
        self.daemon.video.start.side_effect = RuntimeError("Failed to open camera index 0")

        with self.assertRaises(RobotError) as ctx:
            self.robot.start_video()
        self.assertIn("camera", str(ctx.exception))

    def test_disconnect_turns_the_camera_off(self):
        """Otherwise a client that leaves holds the camera until the Pi reboots."""
        self.robot.start_video()
        self.robot.disconnect()

        self.daemon.video.stop.assert_called_once()

    def test_disconnect_leaves_an_unused_camera_alone(self):
        self.robot.disconnect()

        self.daemon.video.stop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
