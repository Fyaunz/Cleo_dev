# pi-daemon/hal/controller.py
import threading
import time
from typing import Optional
from ruckig import InputParameter, OutputParameter, Ruckig, Result

class KinematicController:
    def __init__(self, servo_controller, hz=50, debug=False):
        self.servo_controller = servo_controller
        self.control_rate = 1.0 / hz
        # Reuse the solver owned by the ServoController (stateless, no need for a second instance)
        self.solver = servo_controller.solver
        self.debug = debug

        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        # Poses written by the direct paths (go_to, center_head, playback), waiting
        # for the loop to fold them into Ruckig's own state. See sync_state.
        self._resync_head: Optional[dict] = None
        self._resync_ears: Optional[dict] = None

        # --- Shared States ---
        self.target = {
            'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
            'v_max': 100.0, 'a_max': 400.0, 'j_max': 400.0, 'duration': None
        }

        # ear target and limits
        self.ear_target = {
            'ear_left': 0.0, 'ear_right': 0.0,
            'v_max': 200.0, 'a_max': 1000.0, 'j_max': 4000.0, 'duration': None
        }

        # Telemetry
        self.telemetry = {
            'yaw': 180.0, 'pitch': 180.0, 'roll': 180.0, 'timestamp': 0, 'is_moving': False,
            'ear_left': 0.0, 'ear_right': 0.0, 'ears_moving': False
        }

    def start(self):
        """Spins up the dedicated thread."""
        self.running = True
        self.thread = threading.Thread(target=self._realtime_loop, daemon=True)
        self.thread.start()
        print(f"Kinematics Thread running at {int(1/self.control_rate)}Hz")

    def set_target(self, yaw, pitch, roll, v_max, a_max, j_max, duration=None):
        """Thread-safe way for main.py to send new commands."""
        with self.lock:
            self.target['yaw'] = yaw
            self.target['pitch'] = pitch
            self.target['roll'] = roll
            self.target['v_max'] = v_max
            self.target['a_max'] = a_max
            self.target['j_max'] = j_max
            self.target['duration'] = duration
            print("Target Updated!")

    def set_ear_target(self, ear_left, ear_right, v_max, a_max, j_max, duration=None):
        """
        Thread-safe way to retarget the ears, independently of the head.
        """
        with self.lock:
            self.ear_target['ear_left'] = ear_left
            self.ear_target['ear_right'] = ear_right
            self.ear_target['v_max'] = v_max
            self.ear_target['a_max'] = a_max
            self.ear_target['j_max'] = j_max
            self.ear_target['duration'] = duration
            print("Ear Target Updated!")

    def sync_state(self,
                   yaw: Optional[float] = None,
                   pitch: Optional[float] = None,
                   roll: Optional[float] = None,
                   ear_left: Optional[float] = None,
                   ear_right: Optional[float] = None):
        """Tell the loop where the robot ended up after a move it did not plan.

        Direct writes leave Ruckig holding a stale position, so both its state
        and its target move to the new pose. Position alone would drive the
        head straight back. An axis left as None keeps whatever the loop has.
        """
        with self.lock:
            head = {k: float(v) for k, v in
                    (('yaw', yaw), ('pitch', pitch), ('roll', roll)) if v is not None}
            if head:
                self.target.update(head)
                # The duration belonged to the go_to_ruckig this move superseded;
                # leaving it set makes Ruckig hold "moving" while standing still.
                self.target['duration'] = None
                self.telemetry.update(head)
                self._resync_head = {**(self._resync_head or {}), **head}

            ears = {k: float(v) for k, v in
                    (('ear_left', ear_left), ('ear_right', ear_right)) if v is not None}
            if ears:
                self.ear_target.update(ears)
                self.ear_target['duration'] = None
                self.telemetry.update(ears)
                self._resync_ears = {**(self._resync_ears or {}), **ears}

            if head or ears:
                self.telemetry['timestamp'] = time.time()

    def set_telemetry(self,
                  yaw: Optional[float] = None,
                  pitch: Optional[float] = None,
                  roll: Optional[float] = None,
                  is_moving: Optional[bool] = None,
                  ear_left: Optional[float] = None,
                  ear_right: Optional[float] = None,
                  ears_moving: Optional[bool] = None):

        with self.lock:
            if yaw is not None:
                self.telemetry['yaw'] = yaw

            if pitch is not None:
                self.telemetry['pitch'] = pitch

            if roll is not None:
                self.telemetry['roll'] = roll

            if is_moving is not None:
                self.telemetry['is_moving'] = is_moving

            if ear_left is not None:
                self.telemetry['ear_left'] = ear_left

            if ear_right is not None:
                self.telemetry['ear_right'] = ear_right

            if ears_moving is not None:
                self.telemetry['ears_moving'] = ears_moving

            self.telemetry['timestamp'] = time.time()

    def get_telemetry(self) -> dict:
        """Thread-safe way to read current hardware state."""
        with self.lock:
            return dict(self.telemetry) # Return a copy so it can't be mutated

    def _realtime_loop(self):
        otg = Ruckig(3, self.control_rate)
        inp = InputParameter(3)
        out = OutputParameter(3)

        # Ruckig is now tracking Yaw, Pitch, Roll. It starts at (0,0,0)
        inp.current_position = [0.0, 0.0, 0.0]
        inp.current_velocity = [0.0, 0.0, 0.0]
        inp.current_acceleration = [0.0, 0.0, 0.0]

        # The ears get their own generator rather than two more DOFs on the head's:
        # Ruckig time-synchronises the DOFs within one instance, so a flicking ear
        # would stretch or compress the head trajectory it happened to ride on.
        ear_otg = Ruckig(2, self.control_rate)
        ear_inp = InputParameter(2)
        ear_out = OutputParameter(2)

        ear_inp.current_position = [0.0, 0.0]
        ear_inp.current_velocity = [0.0, 0.0]
        ear_inp.current_acceleration = [0.0, 0.0]

        while self.running:
            start_time = time.perf_counter()

            # Read Targets
            with self.lock:
                inp.target_position = [
                    self.target['yaw'],
                    self.target['pitch'],
                    self.target['roll']
                ]
                v, a, j = self.target['v_max'], self.target['a_max'], self.target['j_max']
                dur = self.target['duration']

                ear_inp.target_position = [
                    self.ear_target['ear_left'],
                    self.ear_target['ear_right']
                ]
                ev, ea, ej = (self.ear_target['v_max'], self.ear_target['a_max'],
                              self.ear_target['j_max'])
                ear_dur = self.ear_target['duration']

                resync, self._resync_head = self._resync_head, None
                ear_resync, self._resync_ears = self._resync_ears, None

            # Adopt a pose written by one of the direct paths. Velocity and
            # acceleration go to zero with it: whatever ramp Ruckig was on ended
            # when something else took the servos.
            if resync is not None:
                axes = {'yaw': 0, 'pitch': 1, 'roll': 2}
                position = list(inp.current_position)
                velocity = list(inp.current_velocity)
                acceleration = list(inp.current_acceleration)
                for axis, value in resync.items():
                    position[axes[axis]] = value
                    velocity[axes[axis]] = 0.0
                    acceleration[axes[axis]] = 0.0
                inp.current_position = position
                inp.current_velocity = velocity
                inp.current_acceleration = acceleration

            if ear_resync is not None:
                sides = {'ear_left': 0, 'ear_right': 1}
                ear_position = list(ear_inp.current_position)
                ear_velocity = list(ear_inp.current_velocity)
                ear_acceleration = list(ear_inp.current_acceleration)
                for side, value in ear_resync.items():
                    ear_position[sides[side]] = value
                    ear_velocity[sides[side]] = 0.0
                    ear_acceleration[sides[side]] = 0.0
                ear_inp.current_position = ear_position
                ear_inp.current_velocity = ear_velocity
                ear_inp.current_acceleration = ear_acceleration

            inp.max_velocity = [v, v, v]
            inp.max_acceleration = [a, a, a]
            inp.max_jerk = [j, j, j]
            if dur is not None and dur > 0.0:
                inp.minimum_duration = dur
            else:
                inp.minimum_duration = None

            ear_inp.max_velocity = [ev, ev]
            ear_inp.max_acceleration = [ea, ea]
            ear_inp.max_jerk = [ej, ej]
            if ear_dur is not None and ear_dur > 0.0:
                ear_inp.minimum_duration = ear_dur
            else:
                ear_inp.minimum_duration = None

            # Generate trajectory steps for the Head Angles and the Ears
            res = otg.update(inp, out)
            ear_res = ear_otg.update(ear_inp, ear_out)

            is_moving = (res == Result.Working)
            ears_moving = (ear_res == Result.Working)

            current_yaw = out.new_position[0]
            current_pitch = out.new_position[1]
            current_roll = out.new_position[2]

            # None leaves an ear servo out of the packet
            ear_left = ear_out.new_position[0] if ears_moving else None
            ear_right = ear_out.new_position[1] if ears_moving else None

            # One bus write per tick
            if is_moving:
                try:
                    m1_angle, m2_angle = self.servo_controller.head_motor_angles(
                        current_pitch, current_roll
                    )
                   # yaw angle extra
                    m3_angle = current_yaw + 180.0
                except ValueError as e:
                    # If Ruckig generated an angle that physically binds the robot,
                    # the solver will catch it. We can log it and safely skip the write.
                    print(f"Kinematic Limit Hit: {e}")
                    # An unreachable head pose must not freeze the ears with it.
                    if ears_moving:
                        self.servo_controller.set_ears(ear_left=ear_left, ear_right=ear_right)
                else:
                    # Write to Hardware
                    self.servo_controller.set_servos(m1=m1_angle, m2=m2_angle, m3=m3_angle,
                                                     ear_left=ear_left, ear_right=ear_right)
                    if self.debug:
                        print(f"OTG Step | Yaw: {current_yaw:.2f}°, Pitch: {current_pitch:.2f}°, Roll: {current_roll:.2f}° -> M1: {m1_angle:.2f}°, M2: {m2_angle:.2f}°, M3: {m3_angle:.2f}°")
            elif ears_moving:
                self.servo_controller.set_ears(ear_left=ear_left, ear_right=ear_right)
                if self.debug:
                    print(f"Ear OTG Step | L: {ear_left:.2f}°, R: {ear_right:.2f}°")

            if is_moving:
                out.pass_to_input(inp)
            if ears_moving:
                ear_out.pass_to_input(ear_inp)

            # Finally Update Telemetry (positions + both moving states in one lock)
            self.set_telemetry(
                yaw=current_yaw if is_moving else None,
                pitch=current_pitch if is_moving else None,
                roll=current_roll if is_moving else None,
                is_moving=is_moving,
                ear_left=ear_left,
                ear_right=ear_right,
                ears_moving=ears_moving,
            )

            # Timing Management
            elapsed = time.perf_counter() - start_time
            sleep_time = self.control_rate - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
    def stop(self):
        """
        Safely shuts down the motor thread.
        """
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                print("Warning: kinematics thread did not stop within 2s")