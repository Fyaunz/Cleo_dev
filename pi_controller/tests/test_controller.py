# pi-daemon/tests/test_controller.py
import unittest
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hal.controller import KinematicController
from hal.solver import KinematicSolver

# Create a dummy Mock ServoController so we don't need real serial hardware connected to test threading
class MockServoController:
    def __init__(self):
        # Mirrors the real ServoController: KinematicController reuses this solver.
        self.solver = KinematicSolver()
        self.writes = []

    def head_motor_angles(self, pitch, roll):
        # The real offset/mirroring mapping is asserted in test_servo; here only
        # the shape matters, so the crank angles go straight through.
        crank = self.solver.get_crank_angles(pitch_deg=pitch, roll_deg=roll)
        return crank["L"], crank["R"]

    def set_servos(self, m1, m2, m3, ear_left=None, ear_right=None):
        self.writes.append({'m1': m1, 'm2': m2, 'm3': m3,
                            'ear_left': ear_left, 'ear_right': ear_right})

    def set_ears(self, ear_left=None, ear_right=None):
        self.writes.append({'ear_left': ear_left, 'ear_right': ear_right})

    def shutdown(self): pass

class TestKinematicController(unittest.TestCase):
    def test_thread_lifecycle_and_telemetry(self):
        mock_servo_controller = MockServoController()
        controller = KinematicController(servo_controller=mock_servo_controller, hz=50)
        
        # 1. Start the thread
        controller.start()
        time.sleep(0.1) # Let it spin up
        
        # 2. Check initial telemetry state
        initial_telemetry = controller.get_telemetry()
        self.assertIn('is_moving', initial_telemetry)
        self.assertFalse(initial_telemetry['is_moving']) # Should be idle
        
        # 3. Send a movement command to the thread
        print("Sending target to background thread...")
        controller.set_target(yaw=20.0, pitch=10.0, roll=0.0, v_max=100.0, a_max=200.0, j_max=1000.0)
        time.sleep(0.05) # Wait 50ms (2.5 control loops)
        
        # 4. Verify the thread caught the instruction and updated its state
        active_telemetry = controller.get_telemetry()
        self.assertTrue(active_telemetry['is_moving'], "Thread failed to activate Ruckig loop.")
        self.assertNotEqual(active_telemetry['pitch'], 0.0, "Trajectory math is stagnant.")
        
        # 5. Stop the thread cleanly (verifies 'del' calls run to clear nanobind instances)
        print("Stopping thread...")
        controller.stop()
        self.assertFalse(controller.thread.is_alive(), "Thread failed to terminate cleanly.") # type: ignore


class TestEarTrajectory(unittest.TestCase):
    """The ears run a second Ruckig generator, independent of the head's."""

    def setUp(self):
        self.servo = MockServoController()
        self.controller = KinematicController(servo_controller=self.servo, hz=100)
        self.controller.start()
        self.addCleanup(self.controller.stop)
        time.sleep(0.1)

    def test_ears_move_without_the_head(self):
        self.controller.set_ear_target(ear_left=40.0, ear_right=40.0,
                                       v_max=200.0, a_max=1000.0, j_max=4000.0)
        time.sleep(0.05)

        telemetry = self.controller.get_telemetry()
        self.assertTrue(telemetry['ears_moving'], "Ear Ruckig loop never activated.")
        self.assertFalse(telemetry['is_moving'], "An ear move must not report the head moving.")
        self.assertNotEqual(telemetry['ear_left'], 0.0, "Ear trajectory is stagnant.")

    def test_ear_only_move_leaves_the_head_servos_alone(self):
        self.controller.set_ear_target(ear_left=30.0, ear_right=30.0,
                                       v_max=200.0, a_max=1000.0, j_max=4000.0)
        time.sleep(0.05)

        writes = [w for w in self.servo.writes if w['ear_left'] is not None]
        self.assertTrue(writes, "No ear write reached the servo controller.")
        for write in writes:
            self.assertNotIn('m1', write, "An ear-only move rewrote the head's goal positions.")

    def test_head_and_ears_share_one_bus_write(self):
        """Both moving must stay one SyncWrite, not two packets per tick."""
        self.controller.set_target(yaw=20.0, pitch=5.0, roll=0.0,
                                   v_max=100.0, a_max=400.0, j_max=1000.0)
        self.controller.set_ear_target(ear_left=30.0, ear_right=-30.0,
                                       v_max=200.0, a_max=1000.0, j_max=4000.0)
        time.sleep(0.05)

        combined = [w for w in self.servo.writes
                    if 'm1' in w and w['ear_left'] is not None]
        self.assertTrue(combined, "Head and ears did not travel in the same packet.")

    def test_head_move_does_not_command_idle_ears(self):
        """An idle ear must be absent from the packet, not written as 0.0.

        Sending an angle would re-issue a goal position every tick and drag an
        ear parked elsewhere back to neutral behind the caller's back.
        """
        self.controller.set_target(yaw=20.0, pitch=5.0, roll=0.0,
                                   v_max=100.0, a_max=400.0, j_max=1000.0)
        time.sleep(0.05)

        head_writes = [w for w in self.servo.writes if 'm1' in w]
        self.assertTrue(head_writes, "The head never moved.")
        for write in head_writes:
            self.assertIsNone(write['ear_left'])
            self.assertIsNone(write['ear_right'])


class TestDirectMoveResync(unittest.TestCase):
    """go_to, center_head and playback move the servos without Ruckig.

    sync_state is how the loop hears about it; without it the next go_to_ruckig
    plans from the pose Ruckig last held and the move opens with a jump.
    """

    def setUp(self):
        self.servo = MockServoController()
        self.controller = KinematicController(servo_controller=self.servo, hz=100)
        self.controller.start()
        self.addCleanup(self.controller.stop)
        time.sleep(0.05)

    def _head_writes(self):
        return [w for w in self.servo.writes if 'm1' in w]

    def test_synced_pose_is_not_driven_back(self):
        """Adopting the position alone would send the head back to the old target."""
        self.controller.sync_state(yaw=30.0, pitch=0.0, roll=0.0)
        time.sleep(0.05)

        telemetry = self.controller.get_telemetry()
        self.assertEqual(telemetry['yaw'], 30.0)
        self.assertFalse(telemetry['is_moving'], "The loop chased a pose it was already at.")
        self.assertFalse(self._head_writes(), "The loop wrote to the head after a direct move.")

    def test_next_ruckig_move_starts_from_the_synced_pose(self):
        self.controller.sync_state(yaw=30.0, pitch=0.0, roll=0.0)
        time.sleep(0.03)

        self.controller.set_target(yaw=40.0, pitch=0.0, roll=0.0,
                                   v_max=100.0, a_max=400.0, j_max=1000.0)
        time.sleep(0.05)

        writes = self._head_writes()
        self.assertTrue(writes, "The head never moved.")
        # m3 is yaw + 180: the first step must leave from 30 deg, not from 0.
        first_yaw = writes[0]['m3'] - 180.0
        self.assertAlmostEqual(first_yaw, 30.0, delta=1.0,
                               msg="Ruckig re-planned from a stale position.")

    def test_a_stale_duration_does_not_survive_a_resync(self):
        """A minimum_duration left over from go_to_ruckig would hold is_moving true."""
        self.controller.set_target(yaw=20.0, pitch=0.0, roll=0.0,
                                   v_max=100.0, a_max=400.0, j_max=1000.0, duration=5.0)
        time.sleep(0.03)
        self.controller.sync_state(yaw=5.0, pitch=0.0, roll=0.0)
        time.sleep(0.05)

        self.assertFalse(self.controller.get_telemetry()['is_moving'])

    def test_one_ear_syncs_without_disturbing_the_other(self):
        self.controller.sync_state(ear_left=0.0, ear_right=25.0)
        time.sleep(0.03)

        self.controller.sync_state(ear_left=45.0)
        time.sleep(0.05)

        telemetry = self.controller.get_telemetry()
        self.assertEqual(telemetry['ear_left'], 45.0)
        self.assertAlmostEqual(telemetry['ear_right'], 25.0, delta=0.5,
                               msg="Syncing one ear moved the other.")

    def test_next_ear_move_starts_from_the_synced_pose(self):
        self.controller.sync_state(ear_left=45.0, ear_right=0.0)
        time.sleep(0.03)

        self.controller.set_ear_target(ear_left=60.0, ear_right=0.0,
                                       v_max=200.0, a_max=1000.0, j_max=4000.0)
        time.sleep(0.05)

        writes = [w for w in self.servo.writes if w.get('ear_left') is not None]
        self.assertTrue(writes, "The ears never moved.")
        self.assertAlmostEqual(writes[0]['ear_left'], 45.0, delta=1.5,
                               msg="The ear generator re-planned from a stale position.")


if __name__ == '__main__':
    unittest.main()