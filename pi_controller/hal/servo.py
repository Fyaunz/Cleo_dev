# pi-daemon/hal/servo.py
import math
from typing import Optional
from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite, GroupSyncRead
from .solver import KinematicSolver, SIDES

class ServoController:
    EAR_CENTER_DEG = 180.0
    EAR_DIRECTION = (1.0, -1.0)  # (left, right)
    EAR_RANGE_DEG = 90.0         # soft travel limit either side of centre

    def __init__(self, device_name='/dev/cu.usbserial-FTAAMM58', baudrate=2000000, servo_ids=[1, 2, 3, 4, 5]):
        self.servo_ids = servo_ids
        # Empty on a head wired without ears; the ear paths then refuse rather
        # than indexing off the end of the list.
        self.ear_ids = tuple(servo_ids[3:5])
        self.portHandler = PortHandler(device_name)
        self.packetHandler = PacketHandler(2.0)
        self.solver = KinematicSolver()

        # Addresses
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_DRIVE_MODE = 10
        self.ADDR_PROFILE_ACCEL = 108
        self.ADDR_PROFILE_VELOCITY = 112
        self.ADDR_GOAL_POSITION = 116
        
        self.ADDR_PRESENT_POSITION = 132
        self.LEN_PRESENT_POSITION = 4

        # Hardware Limits (XL330)
        self.MIN_TICK = 0
        self.MAX_TICK = 4095

        self._home_deg = {k: self.solver.HOME_COUNT[k] * 360.0 / self.solver.COUNTS_PER_REV
                          for k in SIDES}
        # Never None: the solver refuses to construct if the level pose is unreachable.
        self._alpha_home_deg = {k: math.degrees(self.solver.alpha_home[k])  # type: ignore[arg-type]
                                for k in SIDES}

        if not self.portHandler.openPort() or not self.portHandler.setBaudRate(baudrate):
            raise Exception(f"Failed to initialize hardware on {device_name}")
        
        self.sync_writer = GroupSyncWrite(self.portHandler, self.packetHandler, self.ADDR_PROFILE_ACCEL, 12)
        self.groupSyncRead = GroupSyncRead(self.portHandler, self.packetHandler, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION)

        for dxl_id in self.servo_ids:
            if not self.groupSyncRead.addParam(dxl_id):
                print(f"Warning: Failed to add Motor {dxl_id} to SyncRead Group")

        self._initialize_servos()

    def _initialize_servos(self):
        for dxl_id in self.servo_ids:
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, 0)
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, self.ADDR_DRIVE_MODE, 4)
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, 1)
            

    # Conversions
    def _convert_to_ticks(self, value: float, unit: str) -> int:
        """Converts human units to Dynamixel ticks (0-4095)."""
        if unit == "ticks":
            ticks = int(value)
        elif unit == "deg":
            ticks = int((value / 360.0) * 4096)
        elif unit == "rad":
            ticks = int((value / (2 * math.pi)) * 4096)
        else:
            raise ValueError(f"Unknown unit: {unit}")
        
        # Clamp to physical hardware limits to prevent breaking the servo
        return max(self.MIN_TICK, min(self.MAX_TICK, ticks))

    # Thread-Safe Hardware Writes
    def _write_goals(self, targets: dict, accel: int = 0, vel: int = 0):
        """
        Batch-writes Profile Accel + Profile Velocity + Goal Position for each
        servo in one GroupSyncWrite broadcast.

        `targets` maps dxl_id -> angle in degrees.
        
        `accel`/`vel` map to the
        Profile Acceleration/Velocity registers (interpreted as ms in the
        time-based profile configured in _initialize_servos).
        """
        # Clear the parameter list from the previous call
        self.sync_writer.clearParam()

        # accel/vel are constant across servos, so build their bytes once.
        accel_bytes = list(accel.to_bytes(4, byteorder='little', signed=True))
        vel_bytes = list(vel.to_bytes(4, byteorder='little', signed=True))

        for dxl_id, angle_deg in targets.items():
            target_ticks = self._convert_to_ticks(angle_deg, "deg")
            pos_bytes = list(target_ticks.to_bytes(4, byteorder='little', signed=True))

            # Concatenate: 4 (Accel) + 4 (Vel) + 4 (Pos) = 12 bytes
            self.sync_writer.addParam(dxl_id, accel_bytes + vel_bytes + pos_bytes)

        # Broadcast the packet to all servos simultaneously
        self.sync_writer.txPacket()

    def head_motor_angles(self, pitch: float, roll: float) -> tuple[float, float]:
        """
        Servo angles for m1/m2 from a head pitch/roll, both in degrees.
        """

        # This is the one place the conversion happens; the Ruckig loop calls it
        # rather than the solver so the two motion paths cannot drift apart.

        crank = self.solver.get_crank_angles(pitch_deg=pitch, roll_deg=roll)
        return tuple(  # type: ignore[return-value]
            self._home_deg[k] + self.solver.SIGMA[k] * (crank[k] - self._alpha_home_deg[k])
            for k in SIDES
        )

    def _ear_targets(self, ear_left: Optional[float] = None,
                     ear_right: Optional[float] = None) -> dict:
        """
        Maps ear angles onto motor angles for the two ear servos.
        """
        if ear_left is None and ear_right is None:
            return {}
        if len(self.ear_ids) < 2:
            raise ValueError("This head has no ear servos configured")

        targets = {}
        for angle, dxl_id, direction in ((ear_left, self.ear_ids[0], self.EAR_DIRECTION[0]),
                                         (ear_right, self.ear_ids[1], self.EAR_DIRECTION[1])):
            if angle is None:
                continue
            clamped = max(-self.EAR_RANGE_DEG, min(self.EAR_RANGE_DEG, float(angle)))
            targets[dxl_id] = self.EAR_CENTER_DEG + direction * clamped
        return targets

    def set_servos(self, m1: float, m2: float, m3: float,
                   ear_left: Optional[float] = None, ear_right: Optional[float] = None):
        """Writes exact motor angles at max speed (accel=vel=0).

        m1/m2/m3 are motor angles, but ear_left/ear_right are *ear* angles: the
        mounting offset lives in _ear_targets so the realtime loop and go_to
        cannot drift apart on it.
        """
        targets = {
            self.servo_ids[0]: m1,
            self.servo_ids[1]: m2,
            self.servo_ids[2]: m3,
        }
        targets.update(self._ear_targets(ear_left, ear_right))
        self._write_goals(targets)

    def set_ears(self, ear_left: Optional[float] = None, ear_right: Optional[float] = None):
        """Writes ear angles at max speed, leaving the head servos alone."""
        targets = self._ear_targets(ear_left, ear_right)
        if targets:
            self._write_goals(targets)
        return targets

    def go_to(self, yaw: float, pitch: float, roll: float, t_total: int, t_accel: int,
              ear_left: Optional[float] = None, ear_right: Optional[float] = None):
        """Moves the head to a pose over a timed profile (time-based).

        Ear angles are optional
        """
        m1_angle, m2_angle = self.head_motor_angles(pitch, roll)
        m3_angle = yaw + 180.0

        targets = {
            self.servo_ids[0]: m1_angle,
            self.servo_ids[1]: m2_angle,
            self.servo_ids[2]: m3_angle,
        }
        targets.update(self._ear_targets(ear_left, ear_right))
        self._write_goals(targets, accel=t_accel, vel=t_total)
        return targets

    def go_to_ears(self, ear_left: Optional[float] = None, ear_right: Optional[float] = None,
                   t_total: int = 1000, t_accel: int = 200):
        """Moves the ears alone over a timed profile, the head untouched."""
        targets = self._ear_targets(ear_left, ear_right)
        if targets:
            self._write_goals(targets, accel=t_accel, vel=t_total)
        return targets

    def get_single_position(self, dxl_id: int):
        """Helper to get just one motor's position in ticks."""
        position, result, error = self.packetHandler.read4ByteTxRx(
            self.portHandler, dxl_id, self.ADDR_PRESENT_POSITION)
        return position if result == 0 else None

    def get_telemetry(self) -> dict:
        """
        Fetches the physical position of all servos simultaneously using GroupSyncRead.
        """
        telemetry = {}

        # Fire the single broadcast request to the serial bus
        dxl_comm_result = self.groupSyncRead.txRxPacket()
        
        if dxl_comm_result != 0:
            # If the communication failed (e.g., loose wire), log it but don't crash
            print(f"SyncRead Error: {self.packetHandler.getTxRxResult(dxl_comm_result)}")
            return telemetry

        # Unpack single response packet for each motor
        for dxl_id in self.servo_ids:
            # Verify the motor actually returned its specific data
            if self.groupSyncRead.isAvailable(dxl_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION):
                pos = self.groupSyncRead.getData(dxl_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION)
                telemetry[f"joint_{dxl_id}"] = pos
            else:
                print(f"SyncRead Error: No data available for motor {dxl_id}")
                pass 

        return telemetry

    def shutdown(self):
        for dxl_id in self.servo_ids:
            self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, 0)
        self.portHandler.closePort()