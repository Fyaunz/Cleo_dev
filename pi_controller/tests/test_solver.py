# pi-daemon/tests/test_solver.py
import unittest
import sys
import os

# Ensure the parent directory is in the path so we can import hal
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hal.solver import KinematicSolver

class TestKinematicSolver(unittest.TestCase):
    def setUp(self):
        self.solver = KinematicSolver()

    def test_center_position(self):
        """When pitch and roll are 0, crank angles should be equal and deterministic."""
        angles = self.solver.get_crank_angles(pitch_deg=0.0, roll_deg=0.0)

        # Check that angles are valid numbers, one per leg
        self.assertEqual(set(angles), {"L", "R"})
        for value in angles.values():
            self.assertIsInstance(value, float)
        # In a perfectly symmetrical setup, 0 pitch/roll means left and right legs match
        self.assertAlmostEqual(angles["L"], angles["R"], places=2)

    def test_no_revolution_jumps_across_the_workspace(self):
        """Crank angles must vary continuously, with no phase jump or sign flip.

        The result is deliberately left unwrapped -- wrapping into [0, 360) is
        what would *introduce* a full-revolution discontinuity when a solution
        crosses the boundary -- so continuity, not range, is what to assert.
        A servo commanded across such a jump would take the long way round.
        """
        previous = None
        for step in range(-200, 201):
            pose = step * 0.1  # a diagonal sweep through (1.76, 1.76) and beyond
            angles = self.solver.get_crank_angles(pitch_deg=pose, roll_deg=pose)

            if previous is not None:
                for leg in ("L", "R"):
                    self.assertLess(
                        abs(angles[leg] - previous[leg]), 30.0,
                        f"leg {leg} jumps at pitch=roll={pose:.1f} deg: "
                        f"{previous[leg]:.2f} -> {angles[leg]:.2f}")
            previous = angles

    def test_unreachable_pose_raises_when_unclamped(self):
        """Extreme impossible angles must raise rather than return nonsense."""
        with self.assertRaises(ValueError):
            # An 85 degree tilt is far outside a 84.1mm rod and 22.2mm crank
            self.solver.get_crank_angles(pitch_deg=85.0, roll_deg=85.0, clamp=False)

    def test_impossible_pose_is_clamped_by_default(self):
        """Clamping is the protection now: the head goes as far as it can.

        The scale is applied to both axes together, so the head still faces the
        direction it was asked to -- clipping each axis separately would change
        the direction whenever either bound is active.
        """
        pitch, roll, scale = self.solver.clamp_pose(85.0, 85.0)

        self.assertLess(scale, 1.0)
        self.assertAlmostEqual(pitch / roll, 1.0, places=6, msg="tilt direction changed")

        # The clamp lands exactly on the envelope boundary, where admissibility
        # is a float-epsilon question: assert the boundary itself instead, a
        # hair inside admissible and a hair further out not.
        self.assertTrue(self.solver.is_admissible(pitch * 0.999, roll * 0.999))
        self.assertFalse(self.solver.is_admissible(pitch * 1.001, roll * 1.001))

        # The default path returns the clamped pose's angles instead of raising.
        self.assertEqual(self.solver.get_crank_angles(pitch_deg=85.0, roll_deg=85.0),
                         self.solver.get_crank_angles(pitch_deg=pitch, roll_deg=roll))

if __name__ == '__main__':
    unittest.main()