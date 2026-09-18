# pi-daemon/network/router.py
from typing import Optional

from network.dedup import Reply, ReplyCache

# Pure reads: re-running one on a resend is free and answers with fresher state
# than a replay would. Keeping them out of the cache also stops is_moving, which
# the SDK polls at ~50Hz, from evicting the movement commands that must not run twice.
QUERY_COMMANDS = frozenset({"is_moving", "ears_moving", "get_doa"})


class _ReplyRecorder:
    """
    Forwards replies to the real listener while remembering what was sent.
    """

    def __init__(self, listener, msg_id):
        self._listener = listener
        self._msg_id = msg_id
        self.sent: Optional[Reply] = None

    def send_reply(self, status, message=""):
        if self.sent is not None:
            print(f"Dropping extra reply for {self._msg_id}: {status} {message}")
            return
        self.sent = (status, message)
        self._listener.send_reply(status, message, msg_id=self._msg_id)


def route_command(command_msg: dict, cmd_listener, session_manager, servo_controller,
                  controller, playback_engine, video_streamer=None,
                  reply_cache: Optional[ReplyCache] = None, audio_streamer=None):
    """
    Parses incoming network commands and executes the corresponding hardware actions.
    """
    msg_id = command_msg.get("msg_id")

    # Queries are answered fresh, so they neither read nor fill the cache.
    cache = None if command_msg.get("command") in QUERY_COMMANDS else reply_cache

    if cache is not None:
        cached = cache.get(msg_id)
        if cached is not None:
            status, message = cached
            print(f"Resend of {msg_id} ({command_msg.get('command')}); replaying reply.")
            cmd_listener.send_reply(status, message, msg_id=msg_id)
            return

    recorder = _ReplyRecorder(cmd_listener, msg_id)
    _dispatch(command_msg, recorder, session_manager, servo_controller, controller,
              playback_engine, video_streamer, audio_streamer)

    reply = recorder.sent
    if reply is None:
        # The REP socket owes the client exactly one reply; never leave it unanswered.
        reply = ("error", "Command produced no reply")
        recorder.send_reply(*reply)

    if cache is not None:
        cache.put(msg_id, *reply)


def _dispatch(command_msg: dict, cmd_listener, session_manager, servo_controller, controller,
              playback_engine, video_streamer=None, audio_streamer=None):
    client_id = command_msg.get("client_id")

    # 2. The Gatekeeper: Reject the command if another client owns the lock
    if not session_manager.check_access(client_id):
        cmd_listener.send_reply("error", "Robot is currently locked by another client.")
        return

    # 3. If access is granted, route the command normally
    command = command_msg.get("command")

    try:
        match command:
            case "go_to":
                # Test for the keys, not their truth: 0.0 is a falsy float but a
                # perfectly good angle, and centring the head is all zeroes.
                if not any(k in command_msg for k in ("yaw", "pitch", "roll")):
                    cmd_listener.send_reply("error", "No positions provided")
                else:
                    yaw = command_msg.get("yaw", 0.0)
                    pitch = command_msg.get("pitch", 0.0)
                    roll = command_msg.get("roll", 0.0)
                    t_total = command_msg.get("total_time_ms", 1000)
                    t_accel = command_msg.get("accel_time_ms", 200)
                    # Absent, not zero: an omitted ear keeps its current pose
                    # instead of being dragged back to neutral by a head move.
                    ear_left = command_msg.get("ear_left")
                    ear_right = command_msg.get("ear_right")

                    targets = servo_controller.go_to(yaw, pitch, roll, t_total, t_accel,
                                                     ear_left=ear_left, ear_right=ear_right)
                    # The servos, not Ruckig, just moved the head: hand the pose to
                    # the loop or the next go_to_ruckig starts from a stale position.
                    controller.sync_state(yaw=yaw, pitch=pitch, roll=roll,
                                          ear_left=ear_left, ear_right=ear_right)
                    cmd_listener.send_reply("ok", f"Head moving to {targets} ticks")

            case "move_ears":
                if not any(k in command_msg for k in ("ear_left", "ear_right")):
                    cmd_listener.send_reply("error", "No ear positions provided")
                else:
                    targets = servo_controller.go_to_ears(
                        ear_left=command_msg.get("ear_left"),
                        ear_right=command_msg.get("ear_right"),
                        t_total=command_msg.get("total_time_ms", 1000),
                        t_accel=command_msg.get("accel_time_ms", 200),
                    )
                    controller.sync_state(ear_left=command_msg.get("ear_left"),
                                          ear_right=command_msg.get("ear_right"))
                    cmd_listener.send_reply("ok", f"Ears moving to {targets} ticks")

            case "move_ears_ruckig":
                controller.set_ear_target(
                    ear_left=command_msg.get("ear_left", 0.0),
                    ear_right=command_msg.get("ear_right", 0.0),
                    v_max=command_msg.get("v_max", 200.0),
                    a_max=command_msg.get("a_max", 1000.0),
                    j_max=command_msg.get("j_max", 4000.0),
                    duration=command_msg.get("duration")
                )
                cmd_listener.send_reply("ok", "Ear Trajectory Updated")

            case "center_ears":
                t_total = command_msg.get("total_time_ms", 1000)
                servo_controller.go_to_ears(ear_left=0.0, ear_right=0.0,
                                            t_total=t_total, t_accel=int(t_total / 3))
                controller.sync_state(ear_left=0.0, ear_right=0.0)
                cmd_listener.send_reply("ok", "Centering ears.")

            case "ears_moving":
                # Separate from is_moving so a client waiting on a head move is
                # not held up by an ear twitch, and vice versa.
                state: dict = controller.get_telemetry()
                cmd_listener.send_reply("ok", state.get("ears_moving", False))

            case "go_to_ruckig":
                controller.set_target(
                    yaw=command_msg.get("yaw", 0.0),
                    pitch=command_msg.get("pitch", 0.0),
                    roll=command_msg.get("roll", 0.0),
                    v_max=command_msg.get("v_max", 100.0),
                    a_max=command_msg.get("a_max", 400.0),
                    j_max=command_msg.get("j_max", 400.0),
                    duration=command_msg.get("duration")
                )
                cmd_listener.send_reply("ok", "Trajectory Updated")

            case "center_head":
                t_total = command_msg.get("total_time_ms", 2000)
                servo_controller.go_to(yaw=0.0, pitch=0.0, roll=0.0, t_total=t_total, t_accel=int(t_total/3))
                controller.sync_state(yaw=0.0, pitch=0.0, roll=0.0)
                cmd_listener.send_reply("ok", "Centering head.")

            case "is_moving":
                # Fetch the current state from the thread-safe dictionary
                state: dict = controller.get_telemetry()
                is_moving = state.get("is_moving", False)
                
                # Send the boolean back to the Mac
                cmd_listener.send_reply("ok", is_moving)

            case "upload_and_play":
                frames = command_msg.get("frames", [])
                
                if not frames:
                    cmd_listener.send_reply("error", "No frames received")
                    return

                # Pass the data to the engine
                playback_engine.play(frames)
                
                # Instantly acknowledge receipt over the network
                cmd_listener.send_reply("ok", f"Received {len(frames)} frames. Playing locally.")

            case "stop_animation":
                # Safely trigger the engine's halt method
                playback_engine.stop()
                cmd_listener.send_reply("ok", "Playback stopped.")

            case "start_video" | "stop_video":
                # The camera is off until a client asks for it, so both of these
                # are no-ops when the stream is already in the requested state --
                # still an "ok", since the client's intent is satisfied either way.
                if video_streamer is None:
                    cmd_listener.send_reply("error", "This daemon has no video streamer.")
                elif command == "start_video":
                    started = video_streamer.start()
                    cmd_listener.send_reply(
                        "ok", "Video stream started." if started else "Video stream already running.")
                else:
                    stopped = video_streamer.stop()
                    cmd_listener.send_reply(
                        "ok", "Video stream stopped." if stopped else "Video stream was not running.")

            case "start_audio" | "stop_audio":
                # Same contract as the camera: the device stays closed until a
                # client asks, and asking for a state it is already in is an
                # "ok" -- the client's intent is satisfied either way. mic and
                # speaker are independent, so a client that only wants the robot
                # to talk never opens the microphone.
                if audio_streamer is None:
                    cmd_listener.send_reply("error", "This daemon has no audio streamer.")
                else:
                    mic = command_msg.get("mic", True)
                    speaker = command_msg.get("speaker", True)
                    if command == "start_audio":
                        changed = audio_streamer.start(mic=mic, speaker=speaker)
                        verb = "started" if changed else "already running"
                    else:
                        changed = audio_streamer.stop(mic=mic, speaker=speaker)
                        verb = "stopped" if changed else "was not running"
                    cmd_listener.send_reply("ok", {
                        "message": f"Audio {verb}.",
                        "mic": audio_streamer.mic_running,
                        "speaker": audio_streamer.speaker_running,
                        "mic_rate": audio_streamer.mic_rate,
                        "mic_channels": audio_streamer.mic_channels,
                        "speaker_rate": audio_streamer.speaker_rate,
                        "speaker_channels": audio_streamer.speaker_channels,
                    })

            case "get_doa":
                # A pure read, and polled like one by a client following a
                # speaker -- hence its place in QUERY_COMMANDS, where a replayed
                # answer would report a direction the room has already left.
                if audio_streamer is None:
                    cmd_listener.send_reply("error", "This daemon has no audio streamer.")
                else:
                    cmd_listener.send_reply("ok", audio_streamer.get_doa())

            case "flush_audio":
                # Barge-in: drop whatever speech is still queued, now.
                if audio_streamer is None:
                    cmd_listener.send_reply("error", "This daemon has no audio streamer.")
                else:
                    audio_streamer.flush_playback()
                    cmd_listener.send_reply("ok", "Playback flush requested.")

            case "release_lock":
                session_manager.release(client_id)
                cmd_listener.send_reply("ok", "Lock released.")

            case _:
                cmd_type = command_msg.get("command", "None")
                cmd_listener.send_reply("error", f"Unknown command: {cmd_type}")

    except Exception as e:
        cmd_listener.send_reply("error", str(e))