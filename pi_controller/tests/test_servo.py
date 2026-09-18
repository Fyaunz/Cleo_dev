# pi-daemon/tests/test_servo.py
import unittest
import sys
import os
from typing import cast
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hal.servo import ServoController


class ServoTestBase(unittest.TestCase):
    """Builds a ServoController with the dynamixel_sdk fully mocked out.

    The SDK classes are patched at the hal.servo module level, so no serial
    port is opened and every write is captured instead of sent.
    """

    def setUp(self):
        patchers = [
            patch('hal.servo.PortHandler'),
            patch('hal.servo.PacketHandler'),
            patch('hal.servo.GroupSyncWrite'),
            patch('hal.servo.GroupSyncRead'),
        ]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)

        self.servo = ServoController(device_name='/dev/fake', servo_ids=[1, 2, 3, 4, 5])
        # patch() swapped GroupSyncWrite for a MagicMock, but the type checker
        # still sees the real class -- cast so .reset_mock()/.call_args_list resolve.
        self.writer = cast(MagicMock, self.servo.sync_writer)
        self.writer.reset_mock()

    def written(self):
        """Decode captured addParam calls into {dxl_id: (accel, vel, pos)}."""
        out = {}
        for call in self.writer.addParam.call_args_list:
            dxl_id, data = call.args
            b = bytes(data)
            self.assertEqual(len(b), 12, "SyncWrite payload must be exactly 12 bytes")
            out[dxl_id] = (
                int.from_bytes(b[0:4], 'little', signed=True),   # Profile Acceleration
                int.from_bytes(b[4:8], 'little', signed=True),   # Profile Velocity
                int.from_bytes(b[8:12], 'little', signed=True),  # Goal Position
            )
        return out


class TestTickConversion(ServoTestBase):
    def test_degrees_to_ticks(self):
        self.assertEqual(self.servo._convert_to_ticks(0.0, "deg"), 0)
        self.assertEqual(self.servo._convert_to_ticks(180.0, "deg"), 2048)

    def test_radians_to_ticks(self):
        import math
        self.assertEqual(self.servo._convert_to_ticks(math.pi, "rad"), 2048)

    def test_ticks_passthrough(self):
        self.assertEqual(self.servo._convert_to_ticks(1234, "ticks"), 1234)

    def test_clamped_to_hardware_limits(self):
        """Out-of-range angles must clamp, never wrap or overflow the servo."""
        self.assertEqual(self.servo._convert_to_ticks(360.0, "deg"), 4095)
        self.assertEqual(self.servo._convert_to_ticks(720.0, "deg"), 4095)
        self.assertEqual(self.servo._convert_to_ticks(-10.0, "deg"), 0)
        self.assertEqual(self.servo._convert_to_ticks(-720.0, "deg"), 0)

    def test_unknown_unit_raises(self):
        with self.assertRaises(ValueError):
            self.servo._convert_to_ticks(1.0, "furlongs")


class TestWriteGoals(ServoTestBase):
    def test_packet_layout_and_broadcast(self):
        """Each servo gets one 12-byte accel|vel|pos param, then one broadcast."""
        self.servo._write_goals({1: 180.0, 2: 90.0}, accel=300, vel=1500)

        self.assertEqual(self.written(), {
            1: (300, 1500, 2048),
            2: (300, 1500, 1024),
        })
        self.writer.txPacket.assert_called_once()

    def test_clears_stale_params_first(self):
        """clearParam must precede addParam or params leak between calls."""
        self.servo._write_goals({1: 0.0})
        self.writer.clearParam.assert_called_once()

    def test_negative_profile_values_survive_roundtrip(self):
        """Signed little-endian packing must round-trip negatives intact."""
        self.servo._write_goals({1: 0.0}, accel=-5, vel=-9)
        self.assertEqual(self.written()[1][:2], (-5, -9))


class TestSetServos(ServoTestBase):
    def test_uses_max_speed_profile(self):
        """set_servos must send accel=0/vel=0 (max speed), not a timed profile."""
        self.servo.set_servos(m1=180.0, m2=90.0, m3=0.0)
        for accel, vel, _pos in self.written().values():
            self.assertEqual((accel, vel), (0, 0))

    def test_maps_angles_to_first_three_ids(self):
        self.servo.set_servos(m1=180.0, m2=90.0, m3=45.0)
        w = self.written()
        self.assertEqual(set(w), {1, 2, 3})
        self.assertEqual(w[1][2], 2048)
        self.assertEqual(w[2][2], 1024)
        self.assertEqual(w[3][2], 512)


class TestHeadMotorAngles(ServoTestBase):
    """Crank geometry is not a servo command: the zero and the mirroring differ."""

    def test_level_pose_sits_on_the_servo_zero(self):
        """The servo's zero is 180 deg (tick 2048), the crank's is 172.416 deg.

        Writing the raw crank angle would park a level head 7.6 deg off.
        """
        m1, m2 = self.servo.head_motor_angles(0.0, 0.0)

        self.assertAlmostEqual(m1, 180.0, places=6)
        self.assertAlmostEqual(m2, 180.0, places=6)

    def test_level_pose_is_not_the_raw_crank_angle(self):
        """Regression lock: the offset must actually be applied, not skipped."""
        crank = self.servo.solver.get_crank_angles(pitch_deg=0.0, roll_deg=0.0)

        self.assertNotAlmostEqual(crank["L"], 180.0, places=1,
                                  msg="geometry changed; this test no longer proves anything")
        self.assertAlmostEqual(self.servo.head_motor_angles(0.0, 0.0)[0], 180.0, places=6)

    def test_pitch_drives_the_two_servos_in_opposite_directions(self):
        """SIGMA: the servos are mounted mirrored, the legs solved in one frame.

        Pure pitch moves both cranks the same way in the world frame, so the two
        servo commands must straddle the zero. Without SIGMA the head would
        pitch when asked to roll.
        """
        m1, m2 = self.servo.head_motor_angles(10.0, 0.0)

        self.assertGreater(m1, 180.0)
        self.assertLess(m2, 180.0)
        self.assertAlmostEqual(m1 - 180.0, 180.0 - m2, places=6,
                               msg="a symmetric pose must displace both legs equally")

    def test_roll_drives_them_the_same_way(self):
        """The differential's other mode: roll is common, pitch is differential."""
        m1, m2 = self.servo.head_motor_angles(0.0, 10.0)

        self.assertLess(m1, 180.0)
        self.assertLess(m2, 180.0)

    def test_pitch_displaces_about_the_zero_in_both_directions(self):
        """Up and down straddle 180 deg by a similar, not identical, amount.

        The crank-rod linkage is nonlinear, so +-10 deg of pitch is 20.31 deg of
        crank one way and 18.58 deg the other. The delta below is what separates
        that real asymmetry from a missing home offset, which would show up as a
        13 deg difference.
        """
        up = self.servo.head_motor_angles(10.0, 0.0)
        down = self.servo.head_motor_angles(-10.0, 0.0)

        self.assertAlmostEqual(up[0] - 180.0, 180.0 - down[0], delta=2.5)
        self.assertAlmostEqual(up[1] - 180.0, 180.0 - down[1], delta=2.5)

    def test_go_to_writes_the_mapped_angles(self):
        """The mapping must reach the bus, not just exist."""
        m1, m2 = self.servo.head_motor_angles(10.0, 5.0)
        self.servo.go_to(yaw=0.0, pitch=10.0, roll=5.0, t_total=0, t_accel=0)

        w = self.written()
        self.assertEqual(w[1][2], self.servo._convert_to_ticks(m1, "deg"))
        self.assertEqual(w[2][2], self.servo._convert_to_ticks(m2, "deg"))

    def test_level_head_lands_on_tick_2048(self):
        self.servo.go_to(yaw=0.0, pitch=0.0, roll=0.0, t_total=0, t_accel=0)

        w = self.written()
        self.assertEqual(w[1][2], self.servo.solver.HOME_COUNT["L"])
        self.assertEqual(w[2][2], self.servo.solver.HOME_COUNT["R"])


class TestGoTo(ServoTestBase):
    def test_uses_timed_profile(self):
        """go_to maps t_accel->Profile Accel and t_total->Profile Velocity."""
        self.servo.go_to(yaw=0.0, pitch=0.0, roll=0.0, t_total=2000, t_accel=500)
        for accel, vel, _pos in self.written().values():
            self.assertEqual((accel, vel), (500, 2000))

    def test_yaw_is_offset_by_180(self):
        """m3 is the yaw motor and carries a +180 deg mounting offset."""
        self.servo.go_to(yaw=10.0, pitch=0.0, roll=0.0, t_total=0, t_accel=0)
        expected = self.servo._convert_to_ticks(190.0, "deg")
        self.assertEqual(self.written()[3][2], expected)

    def test_returns_targets_for_router_reply(self):
        """router.py interpolates this return value into its 'ok' reply."""
        targets = self.servo.go_to(yaw=0.0, pitch=0.0, roll=0.0, t_total=0, t_accel=0)
        self.assertIsInstance(targets, dict)
        self.assertEqual(set(targets), {1, 2, 3})

    def test_symmetric_pose_gives_equal_crank_angles(self):
        """Zero pitch/roll is symmetric, so m1 and m2 must land on the same tick."""
        self.servo.go_to(yaw=0.0, pitch=0.0, roll=0.0, t_total=0, t_accel=0)
        w = self.written()
        self.assertEqual(w[1][2], w[2][2])

    def test_impossible_pose_is_clamped_to_the_workspace_edge(self):
        """An unreachable pose moves as far as it can, in the direction asked.

        The solver clamps rather than raising, so the write goes ahead -- what
        must not happen is garbage on the bus, i.e. anything other than the
        clamped pose's own ticks.
        """
        pitch, roll, scale = self.servo.solver.clamp_pose(85.0, 85.0)
        self.assertLess(scale, 1.0, "test pose is not actually out of range")

        self.servo.go_to(yaw=0.0, pitch=85.0, roll=85.0, t_total=0, t_accel=0)
        clamped_write = self.written()
        self.writer.txPacket.assert_called_once()

        self.writer.reset_mock()
        self.servo.go_to(yaw=0.0, pitch=pitch, roll=roll, t_total=0, t_accel=0)

        self.assertEqual(clamped_write, self.written())


class TestEars(ServoTestBase):
    """The ears are servo_ids[3]/[4], one servo each, no linkage to solve."""

    def _ear_angle(self, side: int, deg: float) -> int:
        return self.servo._convert_to_ticks(
            self.servo.EAR_CENTER_DEG + self.servo.EAR_DIRECTION[side] * deg, "deg")

    def test_neutral_is_the_centre_tick(self):
        self.servo.set_ears(ear_left=0.0, ear_right=0.0)
        w = self.written()
        centre = self.servo._convert_to_ticks(self.servo.EAR_CENTER_DEG, "deg")
        self.assertEqual(w[4][2], centre)
        self.assertEqual(w[5][2], centre)

    def test_sides_are_mirrored(self):
        """The same positive angle must move both ears the same physical way."""
        self.servo.set_ears(ear_left=30.0, ear_right=30.0)
        w = self.written()
        self.assertEqual(w[4][2], self._ear_angle(0, 30.0))
        self.assertEqual(w[5][2], self._ear_angle(1, 30.0))
        self.assertNotEqual(w[4][2], w[5][2], "Mirrored mounting means opposite ticks")

    def test_one_ear_alone_leaves_the_other_out_of_the_packet(self):
        self.servo.set_ears(ear_left=20.0)
        self.assertEqual(set(self.written()), {4})

    def test_ears_do_not_touch_the_head_servos(self):
        self.servo.set_ears(ear_left=20.0, ear_right=20.0)
        self.assertEqual(set(self.written()), {4, 5})

    def test_ears_are_independent_of_pitch_and_roll(self):
        """The same ear angle must produce the same tick at any head pose."""
        self.servo.go_to(yaw=0.0, pitch=0.0, roll=0.0, t_total=0, t_accel=0, ear_left=25.0)
        flat = self.written()[4][2]

        self.writer.reset_mock()
        self.servo.go_to(yaw=30.0, pitch=10.0, roll=-8.0, t_total=0, t_accel=0, ear_left=25.0)
        tilted = self.written()[4][2]

        self.assertEqual(flat, tilted)

    def test_travel_is_clamped_not_raised(self):
        """The 100Hz loop calls this; refusing a write would drop the tick."""
        self.servo.set_ears(ear_left=500.0)
        self.assertEqual(self.written()[4][2], self._ear_angle(0, self.servo.EAR_RANGE_DEG))

    def test_go_to_carries_ears_in_the_same_packet(self):
        targets = self.servo.go_to(yaw=0.0, pitch=0.0, roll=0.0, t_total=0, t_accel=0,
                                   ear_left=15.0, ear_right=15.0)
        self.assertEqual(set(targets), {1, 2, 3, 4, 5})
        self.writer.txPacket.assert_called_once()

    def test_go_to_without_ears_omits_them(self):
        """An unmentioned ear must not be re-commanded by a head move."""
        self.servo.go_to(yaw=0.0, pitch=0.0, roll=0.0, t_total=0, t_accel=0)
        self.assertEqual(set(self.written()), {1, 2, 3})

    def test_go_to_ears_uses_the_timed_profile(self):
        self.servo.go_to_ears(ear_left=10.0, ear_right=10.0, t_total=800, t_accel=200)
        for accel, vel, _pos in self.written().values():
            self.assertEqual((accel, vel), (200, 800))

    def test_set_ears_uses_max_speed(self):
        self.servo.set_ears(ear_left=10.0, ear_right=10.0)
        for accel, vel, _pos in self.written().values():
            self.assertEqual((accel, vel), (0, 0))

    def test_no_ear_angles_writes_nothing(self):
        self.servo.set_ears()
        self.writer.txPacket.assert_not_called()

    def test_head_without_ear_servos_refuses_ear_commands(self):
        with patch('hal.servo.PortHandler'), patch('hal.servo.PacketHandler'), \
             patch('hal.servo.GroupSyncWrite'), patch('hal.servo.GroupSyncRead'):
            earless = ServoController(device_name='/dev/fake', servo_ids=[1, 2, 3])
        with self.assertRaises(ValueError):
            earless.set_ears(ear_left=10.0)


class TestSetServosGoToEquivalence(ServoTestBase):
    def test_same_packet_for_same_angles(self):
        """Regression lock for the _write_goals dedup: both paths must agree.

        go_to(..., t=0) and set_servos() differ only in how targets are derived,
        so identical motor angles must produce byte-identical packets.
        """
        targets = self.servo.go_to(yaw=0.0, pitch=0.0, roll=0.0, t_total=0, t_accel=0)
        via_go_to = self.written()

        self.writer.reset_mock()
        self.servo.set_servos(m1=targets[1], m2=targets[2], m3=targets[3])
        via_set_servos = self.written()

        self.assertEqual(via_go_to, via_set_servos)


if __name__ == '__main__':
    unittest.main()
