# pi-daemon/tests/test_router.py
import unittest
import sys
import os
from unittest.mock import MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from network.router import route_command
from network.dedup import ReplyCache


class FakeListener:
    """Captures replies instead of putting them on the wire."""

    def __init__(self):
        self.replies = []

    def send_reply(self, status, message, msg_id=None):
        self.replies.append((status, message))

    @property
    def last(self):
        return self.replies[-1] if self.replies else (None, None)

    @property
    def status(self):
        return self.last[0]


class FakeSession:
    def __init__(self, allow=True):
        self.allow = allow
        self.released = []

    def check_access(self, client_id):
        return self.allow

    def release(self, client_id):
        self.released.append(client_id)


class RouterTestBase(unittest.TestCase):
    def setUp(self):
        self.listener = FakeListener()
        self.session = FakeSession()
        self.servo = MagicMock()
        self.controller = MagicMock()
        self.playback = MagicMock()
        self.video = MagicMock()
        self.audio = MagicMock()
        self.cache = ReplyCache()

    def route(self, msg):
        msg.setdefault("client_id", "client-1")
        route_command(msg, self.listener, self.session, self.servo,
                      self.controller, self.playback, self.video,
                      reply_cache=self.cache, audio_streamer=self.audio)


class TestGoToDispatch(RouterTestBase):
    def test_reaches_hardware_with_ok_reply(self):
        """The router's blanket try/except turns any bug into a silent 'error'.

        Asserting 'ok' is what makes such failures visible instead of quiet.
        """
        self.route({"command": "go_to", "yaw": 10.0, "pitch": 5.0, "roll": 2.0})

        self.assertEqual(self.listener.status, "ok",
                         f"router swallowed a failure: {self.listener.last}")
        self.servo.go_to.assert_called_once()

    def test_angles_arrive_as_floats_not_tuples(self):
        """Regression: trailing commas once made these 1-tuples.

        A stray comma turned `yaw = msg.get(...)` into `(0.0,)`, which blew up
        inside go_to as a TypeError and got swallowed into an 'error' reply,
        so the servos silently never moved.
        """
        self.route({"command": "go_to", "yaw": 10.0, "pitch": 5.0, "roll": 2.0})

        yaw, pitch, roll = self.servo.go_to.call_args.args[:3]
        for name, value in (("yaw", yaw), ("pitch", pitch), ("roll", roll)):
            self.assertIsInstance(value, float, f"{name} must be a float, got {type(value)}")
        self.assertEqual((yaw, pitch, roll), (10.0, 5.0, 2.0))

    def test_profile_times_forwarded(self):
        self.route({"command": "go_to", "yaw": 1.0, "pitch": 0.0, "roll": 0.0,
                    "total_time_ms": 2000, "accel_time_ms": 400})
        t_total, t_accel = self.servo.go_to.call_args.args[3:5]
        self.assertEqual((t_total, t_accel), (2000, 400))

    def test_profile_times_have_defaults(self):
        self.route({"command": "go_to", "yaw": 1.0, "pitch": 0.0, "roll": 0.0})
        t_total, t_accel = self.servo.go_to.call_args.args[3:5]
        self.assertEqual((t_total, t_accel), (1000, 200))

    def test_zero_pose_is_a_valid_target(self):
        """Regression: `not (yaw or pitch or roll)` rejected the centred pose.

        0.0 is falsy but is a real angle, so move_head(0, 0, 0) came back as
        'No positions provided' and the head never moved.
        """
        self.route({"command": "go_to", "yaw": 0.0, "pitch": 0.0, "roll": 0.0})

        self.assertEqual(self.listener.status, "ok",
                         f"zero pose refused: {self.listener.last}")
        self.servo.go_to.assert_called_once()

    def test_command_without_any_angle_is_rejected(self):
        self.route({"command": "go_to"})

        self.assertEqual(self.listener.status, "error")
        self.servo.go_to.assert_not_called()

    def test_partial_pose_defaults_the_rest_to_zero(self):
        self.route({"command": "go_to", "yaw": 10.0})

        yaw, pitch, roll = self.servo.go_to.call_args.args[:3]
        self.assertEqual((yaw, pitch, roll), (10.0, 0.0, 0.0))

    def test_hardware_error_becomes_error_reply(self):
        """A kinematic limit must surface as an error, not crash the daemon."""
        self.servo.go_to.side_effect = ValueError("Kinematic Limit Reached!")
        self.route({"command": "go_to", "yaw": 1.0, "pitch": 0.0, "roll": 0.0})
        self.assertEqual(self.listener.status, "error")


class TestSessionGate(RouterTestBase):
    def test_locked_client_is_rejected(self):
        self.session.allow = False
        self.route({"command": "go_to", "yaw": 10.0, "pitch": 0.0, "roll": 0.0})

        self.assertEqual(self.listener.status, "error")
        self.servo.go_to.assert_not_called()

    def test_lock_gate_guards_every_command(self):
        """No command may touch hardware while another client holds the lock."""
        self.session.allow = False
        for cmd in ("go_to", "go_to_ruckig", "center_head", "upload_and_play",
                    "stop_animation", "start_video", "stop_video",
                    "start_audio", "stop_audio", "flush_audio"):
            self.route({"command": cmd})

        self.servo.go_to.assert_not_called()
        self.controller.set_target.assert_not_called()
        self.playback.play.assert_not_called()
        self.playback.stop.assert_not_called()
        self.video.start.assert_not_called()
        self.video.stop.assert_not_called()
        self.audio.start.assert_not_called()
        self.audio.stop.assert_not_called()
        self.audio.flush_playback.assert_not_called()

    def test_release_lock(self):
        self.route({"command": "release_lock", "client_id": "client-1"})
        self.assertEqual(self.session.released, ["client-1"])
        self.assertEqual(self.listener.status, "ok")


class TestOtherCommands(RouterTestBase):
    def test_center_head_uses_symmetric_pose(self):
        self.route({"command": "center_head", "total_time_ms": 3000})

        kwargs = self.servo.go_to.call_args.kwargs
        self.assertEqual((kwargs["yaw"], kwargs["pitch"], kwargs["roll"]), (0.0, 0.0, 0.0))
        self.assertEqual(kwargs["t_total"], 3000)
        self.assertEqual(kwargs["t_accel"], 1000, "accel time should be a third of total")

    def test_go_to_ruckig_forwards_limits(self):
        self.route({"command": "go_to_ruckig", "yaw": 5.0, "pitch": 1.0, "roll": 2.0,
                    "v_max": 50.0, "a_max": 100.0, "j_max": 200.0, "duration": 1.5})

        kwargs = self.controller.set_target.call_args.kwargs
        self.assertEqual(kwargs["v_max"], 50.0)
        self.assertEqual(kwargs["a_max"], 100.0)
        self.assertEqual(kwargs["j_max"], 200.0)
        self.assertEqual(kwargs["duration"], 1.5)
        self.assertEqual(self.listener.status, "ok")

    def test_upload_and_play_forwards_frames(self):
        frames = [{"yaw": 0.0, "pitch": 0.0, "roll": 0.0}]
        self.route({"command": "upload_and_play", "frames": frames})

        self.playback.play.assert_called_once_with(frames)
        self.assertEqual(self.listener.status, "ok")

    def test_upload_and_play_rejects_empty(self):
        self.route({"command": "upload_and_play", "frames": []})
        self.assertEqual(self.listener.status, "error")
        self.playback.play.assert_not_called()

    def test_stop_animation(self):
        self.route({"command": "stop_animation"})
        self.playback.stop.assert_called_once()
        self.assertEqual(self.listener.status, "ok")

    def test_is_moving_reports_controller_state(self):
        self.controller.get_telemetry.return_value = {"is_moving": True}
        self.route({"command": "is_moving"})
        self.assertEqual(self.listener.last, ("ok", True))

    def test_unknown_command(self):
        self.route({"command": "fly_away"})
        self.assertEqual(self.listener.status, "error")


class TestEarCommands(RouterTestBase):
    """The ears are their own DOFs: both motion paths, independent of the head."""

    def test_move_ears_uses_the_timed_profile_path(self):
        self.route({"command": "move_ears", "ear_left": 30.0, "ear_right": -30.0,
                    "total_time_ms": 800, "accel_time_ms": 200})

        kwargs = self.servo.go_to_ears.call_args.kwargs
        self.assertEqual((kwargs["ear_left"], kwargs["ear_right"]), (30.0, -30.0))
        self.assertEqual((kwargs["t_total"], kwargs["t_accel"]), (800, 200))
        self.assertEqual(self.listener.status, "ok",
                         f"router swallowed a failure: {self.listener.last}")

    def test_neutral_ears_are_a_valid_target(self):
        """0.0 is falsy but is the neutral ear pose, exactly as for the head."""
        self.route({"command": "move_ears", "ear_left": 0.0, "ear_right": 0.0})

        self.assertEqual(self.listener.status, "ok")
        self.servo.go_to_ears.assert_called_once()

    def test_one_ear_alone_is_a_valid_command(self):
        self.route({"command": "move_ears", "ear_left": 20.0})

        kwargs = self.servo.go_to_ears.call_args.kwargs
        self.assertEqual(kwargs["ear_left"], 20.0)
        self.assertIsNone(kwargs["ear_right"], "the unmentioned ear must not be commanded")

    def test_move_ears_without_any_angle_is_rejected(self):
        self.route({"command": "move_ears"})

        self.assertEqual(self.listener.status, "error")
        self.servo.go_to_ears.assert_not_called()

    def test_move_ears_ruckig_forwards_limits(self):
        self.route({"command": "move_ears_ruckig", "ear_left": 45.0, "ear_right": 45.0,
                    "v_max": 300.0, "a_max": 1200.0, "j_max": 5000.0, "duration": 0.4})

        kwargs = self.controller.set_ear_target.call_args.kwargs
        self.assertEqual((kwargs["ear_left"], kwargs["ear_right"]), (45.0, 45.0))
        self.assertEqual(kwargs["v_max"], 300.0)
        self.assertEqual(kwargs["duration"], 0.4)
        self.assertEqual(self.listener.status, "ok")

    def test_ear_ruckig_does_not_disturb_the_head_trajectory(self):
        self.route({"command": "move_ears_ruckig", "ear_left": 45.0, "ear_right": 45.0})

        self.controller.set_target.assert_not_called()

    def test_head_ruckig_does_not_disturb_the_ears(self):
        self.route({"command": "go_to_ruckig", "yaw": 10.0, "pitch": 0.0, "roll": 0.0})

        self.controller.set_ear_target.assert_not_called()

    def test_go_to_carries_ears_when_given(self):
        self.route({"command": "go_to", "yaw": 10.0, "pitch": 0.0, "roll": 0.0,
                    "ear_left": 25.0, "ear_right": 25.0})

        kwargs = self.servo.go_to.call_args.kwargs
        self.assertEqual((kwargs["ear_left"], kwargs["ear_right"]), (25.0, 25.0))

    def test_go_to_without_ears_leaves_them_uncommanded(self):
        """An omitted ear must stay put, not be dragged to neutral by a head move."""
        self.route({"command": "go_to", "yaw": 10.0, "pitch": 0.0, "roll": 0.0})

        kwargs = self.servo.go_to.call_args.kwargs
        self.assertIsNone(kwargs["ear_left"])
        self.assertIsNone(kwargs["ear_right"])

    def test_center_ears(self):
        self.route({"command": "center_ears", "total_time_ms": 900})

        kwargs = self.servo.go_to_ears.call_args.kwargs
        self.assertEqual((kwargs["ear_left"], kwargs["ear_right"]), (0.0, 0.0))
        self.assertEqual((kwargs["t_total"], kwargs["t_accel"]), (900, 300))
        self.assertEqual(self.listener.status, "ok")

    def test_ears_moving_reports_controller_state(self):
        self.controller.get_telemetry.return_value = {"is_moving": False, "ears_moving": True}
        self.route({"command": "ears_moving"})

        self.assertEqual(self.listener.last, ("ok", True))

    def test_ears_moving_is_answered_fresh_not_replayed(self):
        """move_ears_ruckig(wait=True) polls this at ~50Hz, like is_moving."""
        self.controller.get_telemetry.return_value = {"ears_moving": True}
        self.route({"command": "ears_moving", "msg_id": "client-1:1"})

        self.controller.get_telemetry.return_value = {"ears_moving": False}
        self.route({"command": "ears_moving", "msg_id": "client-1:1"})

        self.assertEqual(self.listener.last, ("ok", False), "resent query replayed a stale state")

    def test_ear_hardware_error_becomes_error_reply(self):
        self.servo.go_to_ears.side_effect = ValueError("This head has no ear servos configured")
        self.route({"command": "move_ears", "ear_left": 10.0})

        self.assertEqual(self.listener.status, "error")

    def test_lock_gate_guards_the_ear_commands(self):
        self.session.allow = False
        for cmd in ("move_ears", "move_ears_ruckig", "center_ears"):
            self.route({"command": cmd, "ear_left": 10.0, "ear_right": 10.0})

        self.servo.go_to_ears.assert_not_called()
        self.controller.set_ear_target.assert_not_called()


class TestVideoCommands(RouterTestBase):
    """The camera is off at boot, so the client owns switching it on and off."""

    def test_start_video_starts_the_stream(self):
        self.route({"command": "start_video"})

        self.video.start.assert_called_once()
        self.assertEqual(self.listener.status, "ok",
                         f"router swallowed a failure: {self.listener.last}")

    def test_stop_video_stops_the_stream(self):
        self.route({"command": "stop_video"})

        self.video.stop.assert_called_once()
        self.assertEqual(self.listener.status, "ok")

    def test_starting_an_already_running_stream_is_ok(self):
        """The client wanted video and video is what it gets: not an error."""
        self.video.start.return_value = False
        self.route({"command": "start_video"})

        self.assertEqual(self.listener.status, "ok")

    def test_stopping_an_idle_stream_is_ok(self):
        self.video.stop.return_value = False
        self.route({"command": "stop_video"})

        self.assertEqual(self.listener.status, "ok")

    def test_camera_failure_becomes_error_reply(self):
        """A missing camera must reach the client, not just the Pi's journal."""
        self.video.start.side_effect = RuntimeError("Failed to open camera index 0")
        self.route({"command": "start_video"})

        self.assertEqual(self.listener.status, "error")

    def test_daemon_without_a_streamer_answers_with_an_error(self):
        route_command({"command": "start_video", "client_id": "client-1"},
                      self.listener, self.session, self.servo, self.controller,
                      self.playback, reply_cache=self.cache)

        self.assertEqual(self.listener.status, "error")


class TestAudioCommands(RouterTestBase):
    """Same contract as the camera: closed at boot, the client switches it on."""

    def test_start_audio_starts_both_directions(self):
        self.route({"command": "start_audio"})

        self.audio.start.assert_called_once_with(mic=True, speaker=True)
        self.assertEqual(self.listener.status, "ok",
                         f"router swallowed a failure: {self.listener.last}")

    def test_a_client_can_ask_for_one_direction(self):
        """Speech output without opening the microphone is a legitimate setup."""
        self.route({"command": "start_audio", "mic": False, "speaker": True})

        self.audio.start.assert_called_once_with(mic=False, speaker=True)

    def test_stop_audio_stops_the_stream(self):
        self.route({"command": "stop_audio"})

        self.audio.stop.assert_called_once_with(mic=True, speaker=True)
        self.assertEqual(self.listener.status, "ok")

    def test_the_reply_reports_the_negotiated_format(self):
        """The client hands the Pi whatever its TTS emits, but a recogniser needs
        to know what the microphone is actually producing."""
        self.audio.mic_rate = 16000
        self.audio.mic_channels = 1
        self.audio.speaker_rate = 48000
        self.audio.speaker_channels = 2
        self.route({"command": "start_audio"})

        status, message = self.listener.last
        self.assertEqual(status, "ok")
        self.assertEqual(message["mic_rate"], 16000)
        self.assertEqual(message["speaker_rate"], 48000)
        self.assertEqual(message["speaker_channels"], 2)

    def test_starting_already_running_audio_is_ok(self):
        """The client wanted audio and audio is what it gets: not an error."""
        self.audio.start.return_value = False
        self.route({"command": "start_audio"})

        self.assertEqual(self.listener.status, "ok")

    def test_device_failure_becomes_error_reply(self):
        """A missing ReSpeaker must reach the client, not just the Pi's journal."""
        self.audio.start.side_effect = RuntimeError("Failed to open mic")
        self.route({"command": "start_audio"})

        self.assertEqual(self.listener.status, "error")

    def test_get_doa_reports_the_direction(self):
        self.audio.get_doa.return_value = {"angle": 135, "speech": True}
        self.route({"command": "get_doa"})

        self.assertEqual(self.listener.last, ("ok", {"angle": 135, "speech": True}))

    def test_a_head_without_an_array_answers_ok_with_nothing(self):
        """No array is not a failure: a client asks and finds out it cannot know."""
        self.audio.get_doa.return_value = None
        self.route({"command": "get_doa"})

        self.assertEqual(self.listener.last, ("ok", None))

    def test_get_doa_is_answered_fresh_not_replayed(self):
        """A client turning the head to face a speaker polls this, so a replayed
        answer would aim at where the room was, not where it is."""
        self.audio.get_doa.return_value = {"angle": 10, "speech": True}
        self.route({"command": "get_doa", "msg_id": "client-1:1"})

        self.audio.get_doa.return_value = {"angle": 170, "speech": True}
        self.route({"command": "get_doa", "msg_id": "client-1:1"})

        self.assertEqual(self.listener.last[1]["angle"], 170,
                         "resent query replayed a stale direction")

    def test_flush_audio_drops_queued_speech(self):
        self.route({"command": "flush_audio"})

        self.audio.flush_playback.assert_called_once()
        self.assertEqual(self.listener.status, "ok")

    def test_daemon_without_a_streamer_answers_with_an_error(self):
        for cmd in ("start_audio", "flush_audio"):
            route_command({"command": cmd, "client_id": "client-1"},
                          self.listener, self.session, self.servo, self.controller,
                          self.playback, reply_cache=self.cache)

            self.assertEqual(self.listener.status, "error")


class TestResendDeduplication(RouterTestBase):
    """A resend means the client lost our reply, not that it wants a second move."""

    def move(self, msg_id):
        self.route({"command": "go_to", "msg_id": msg_id,
                    "yaw": 10.0, "pitch": 5.0, "roll": 2.0})

    def test_resent_command_touches_hardware_once(self):
        self.move("client-1:1")
        self.move("client-1:1")

        self.servo.go_to.assert_called_once()

    def test_resent_command_still_gets_the_original_reply(self):
        """The client is waiting for an answer; a silent drop would hang it."""
        self.move("client-1:1")
        first = self.listener.last

        self.move("client-1:1")

        self.assertEqual(len(self.listener.replies), 2, "every request needs a reply")
        self.assertEqual(self.listener.last, first)

    def test_distinct_ids_are_executed_separately(self):
        self.move("client-1:1")
        self.move("client-1:2")

        self.assertEqual(self.servo.go_to.call_count, 2)

    def test_error_replies_are_replayed_too(self):
        """Replaying beats re-running: the second attempt could disagree."""
        self.servo.go_to.side_effect = ValueError("Kinematic Limit Reached!")
        self.move("client-1:1")
        self.move("client-1:1")

        self.servo.go_to.assert_called_once()
        self.assertEqual(self.listener.status, "error")

    def test_replayed_reply_ignores_a_lock_taken_since(self):
        """The answer already happened; the lock cannot retroactively veto it."""
        self.move("client-1:1")
        self.session.allow = False
        self.move("client-1:1")

        self.assertEqual(self.listener.status, "ok")

    def test_upload_and_play_is_not_replayed_as_a_second_animation(self):
        frames = [{"yaw": 0.0, "pitch": 0.0, "roll": 0.0}]
        for _ in range(2):
            self.route({"command": "upload_and_play", "msg_id": "client-1:9",
                        "frames": frames})

        self.playback.play.assert_called_once()

    def test_queries_are_answered_fresh_not_replayed(self):
        """is_moving is a pure read polled at ~50Hz: a stale replay would be a lie,
        and caching every poll would evict the moves that must not repeat."""
        self.controller.get_telemetry.return_value = {"is_moving": True}
        self.route({"command": "is_moving", "msg_id": "client-1:1"})

        self.controller.get_telemetry.return_value = {"is_moving": False}
        self.route({"command": "is_moving", "msg_id": "client-1:1"})

        self.assertEqual(self.listener.last, ("ok", False),
                         "resent query replayed a stale answer")

    def test_client_without_msg_id_is_unaffected(self):
        """Older scripts predate msg_id and must keep working, uncached."""
        self.route({"command": "go_to", "yaw": 1.0, "pitch": 0.0, "roll": 0.0})
        self.route({"command": "go_to", "yaw": 1.0, "pitch": 0.0, "roll": 0.0})

        self.assertEqual(self.servo.go_to.call_count, 2)
        self.assertEqual(self.listener.status, "ok")

    def test_reply_carries_msg_id_back_to_client(self):
        listener = MagicMock()
        route_command({"command": "go_to", "msg_id": "client-1:1", "client_id": "client-1",
                       "yaw": 1.0, "pitch": 0.0, "roll": 0.0},
                      listener, self.session, self.servo, self.controller,
                      self.playback, reply_cache=self.cache)

        self.assertEqual(listener.send_reply.call_args.kwargs["msg_id"], "client-1:1")


class TestReplyCache(unittest.TestCase):
    def test_evicts_oldest_beyond_capacity(self):
        cache = ReplyCache(max_entries=2)
        cache.put("a", "ok", "1")
        cache.put("b", "ok", "2")
        cache.put("c", "ok", "3")

        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("c"), ("ok", "3"))

    def test_hit_refreshes_entry(self):
        """A client retrying an old command must not have it evicted underneath."""
        cache = ReplyCache(max_entries=2)
        cache.put("a", "ok", "1")
        cache.put("b", "ok", "2")
        cache.get("a")
        cache.put("c", "ok", "3")

        self.assertEqual(cache.get("a"), ("ok", "1"))
        self.assertIsNone(cache.get("b"))

    def test_missing_msg_id_is_never_cached(self):
        cache = ReplyCache()
        cache.put(None, "ok", "1")
        self.assertIsNone(cache.get(None))


if __name__ == '__main__':
    unittest.main()
