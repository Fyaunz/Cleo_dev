import math
import numpy as np

SIDES = ("L", "R")


class KinematicSolver:
    # ------------------------------------------------------------------ #
    # Geometry. All lengths in millimetres, all angles in degrees
    # ------------------------------------------------------------------ #

    # Rod attachment on the platform, measured at the level pose.
    P = {"L": np.array([43.5,  51.5, 0.0]),
         "R": np.array([43.5, -51.5, 0.0])}

    # Servo output shaft centre.
    S = {"L": np.array([43.5,  51.5, -84.1]),
         "R": np.array([43.5, -51.5, -84.1])}

    U = {"L": np.array([1.0, 0.0, 0.0]), "R": np.array([1.0, 0.0, 0.0])}
    V = {"L": np.array([0.0, 0.0, 1.0]), "R": np.array([0.0, 0.0, 1.0])}

    CRANK_LENGTH = 22.2      # r, servo horn length
    ROD_LENGTH = 84.1        # l, rod length between joint centres

    BRANCH = {"L": +1, "R": +1}

    COUNTS_PER_REV = 4096
    HOME_COUNT = {"L": 2048, "R": 2048}
    SIGMA = {"L": +1, "R": -1}

    POSE_CONSTRAINTS = ((0.82, 22.2), (-0.80, 22.3))

    def __init__(self, **overrides):
        """Instance copies of the class geometry, with optional overrides."""
        for name in ("P", "S", "U", "V", "BRANCH", "HOME_COUNT", "SIGMA"):
            setattr(self, name, dict(getattr(type(self), name)))
        for name in ("CRANK_LENGTH", "ROD_LENGTH", "POSE_CONSTRAINTS",
                     "COUNTS_PER_REV"):
            setattr(self, name, getattr(type(self), name))
        for key, value in overrides.items():
            setattr(self, key, value)

        for k in SIDES:
            u, v = self.U[k], self.V[k]
            if abs(u @ v) > 1e-9 or abs(u @ u - 1) > 1e-9 or abs(v @ v - 1) > 1e-9:
                raise ValueError(f"crank basis for leg {k} is not orthonormal")

        self.alpha_home = {k: self._crank_angle(k, self._rotation(0.0, 0.0))
                           for k in SIDES}
        if any(a is None for a in self.alpha_home.values()):
            raise ValueError("level pose is not reachable -- check geometry")

    # ------------------------------------------------------------------ #
    # Core geometry
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rotation(pitch_rad, roll_rad):
        """
        R = Ry(pitch) @ Rx(roll): roll applied first, then pitch.
        """
        cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
        cr, sr = math.cos(roll_rad), math.sin(roll_rad)
        Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        return Ry @ Rx

    def _coefficients(self, k, R):
        """A, B, C for the leg constraint A cos(a) + B sin(a) = C."""
        d = R @ self.P[k] - self.S[k]
        r = self.CRANK_LENGTH
        A = 2.0 * r * (d @ self.U[k])
        B = 2.0 * r * (d @ self.V[k])
        C = d @ d + r * r - self.ROD_LENGTH ** 2
        return A, B, C

    def _crank_angle(self, k, R):
        """
        Crank angle in radians, or None if this leg cannot reach the pose.
        """
        A, B, C = self._coefficients(k, R)
        rho = math.hypot(A, B)
        if rho < 1e-9 or abs(C) > rho:
            return None
        return math.atan2(B, A) + self.BRANCH[k] * math.acos(C / rho)

    def _crank_tip(self, k, alpha_rad):
        return (self.S[k] + self.CRANK_LENGTH
                * (math.cos(alpha_rad) * self.U[k]
                   + math.sin(alpha_rad) * self.V[k]))

    # ------------------------------------------------------------------ #
    # Workspace
    # ------------------------------------------------------------------ #

    def clamp_pose(self, pitch_deg, roll_deg):
        """Scale a commanded pose toward the level pose until admissible.

        Returns (pitch, roll, scale) with scale <= 1. Scaling preserves the direction of the 
        commanded tilt: the head still faces where it was asked to, only less far.
        """
        s = 1.0
        for k, c in self.POSE_CONSTRAINTS:
            denominator = abs(roll_deg) - k * pitch_deg
            if denominator > 0.0:
                s = min(s, c / denominator)
        return s * pitch_deg, s * roll_deg, s

    def is_admissible(self, pitch_deg, roll_deg):
        """True if the pose satisfies the operating envelope."""
        return all(abs(roll_deg) <= k * pitch_deg + c
                   for k, c in self.POSE_CONSTRAINTS)

    def is_reachable(self, pitch_deg, roll_deg):
        """
        True if both legs can close, ignoring the safety margin.
        """
        R = self._rotation(math.radians(pitch_deg), math.radians(roll_deg))
        return all(self._crank_angle(k, R) is not None for k in SIDES)

    # ------------------------------------------------------------------ #
    # Inverse kinematics
    # ------------------------------------------------------------------ #

    def get_crank_angles(self, pitch_deg, roll_deg, clamp=True):
        """
        Geometric crank angles in degrees, as a dict keyed 'L' and 'R'.
        """
        if clamp:
            pitch_deg, roll_deg, _ = self.clamp_pose(pitch_deg, roll_deg)

        R = self._rotation(math.radians(pitch_deg), math.radians(roll_deg))
        angles = {}
        for k in SIDES:
            alpha = self._crank_angle(k, R)
            if alpha is None:
                # Should be unreachable once clamp_pose has run.
                raise ValueError(
                    f"leg {k}: pose ({pitch_deg:.2f}, {roll_deg:.2f}) deg is "
                    f"outside the workspace. If clamping was enabled, "
                    f"POSE_CONSTRAINTS no longer match the geometry -- refit."
                )
            angles[k] = math.degrees(alpha)
        return angles

    # ------------------------------------------------------------------ #
    # Forward kinematics
    # ------------------------------------------------------------------ #

    def forward_kinematics(self, alpha_deg, guess_deg=(0.0, 0.0),
                           tol=1e-12, max_iter=50):
        """
        Platform pose (pitch, roll) in degrees from the two crank angles.
        """
        q = np.radians(np.asarray(guess_deg, dtype=float))
        alpha = {k: math.radians(alpha_deg[k]) for k in SIDES}

        for _ in range(max_iter):
            R = self._rotation(q[0], q[1])
            axis_roll = np.array([math.cos(q[0]), 0.0, -math.sin(q[0])])
            g = np.zeros(2)
            J = np.zeros((2, 2))
            for i, k in enumerate(SIDES):
                T = R @ self.P[k]
                s = T - self._crank_tip(k, alpha[k])
                g[i] = s @ s - self.ROD_LENGTH ** 2
                J[i, 0] = 2.0 * s @ np.cross([0.0, 1.0, 0.0], T)
                J[i, 1] = 2.0 * s @ np.cross(axis_roll, T)
            dq = np.linalg.solve(J, -g)
            q += dq
            if np.linalg.norm(dq) < tol:
                return float(np.degrees(q[0])), float(np.degrees(q[1]))
        raise RuntimeError("forward kinematics did not converge")

    # ------------------------------------------------------------------ #
    # Analysis and verification
    # ------------------------------------------------------------------ #

    def residual(self, k, pitch_deg, roll_deg, alpha_deg):
        """||T - C|| - l, in mm. Zero when the geometry is measured correctly."""
        R = self._rotation(math.radians(pitch_deg), math.radians(roll_deg))
        tip = self._crank_tip(k, math.radians(alpha_deg))
        return float(np.linalg.norm(R @ self.P[k] - tip) - self.ROD_LENGTH)

    def jacobians(self, pitch_deg, roll_deg):
        """
        Compute the Jacobians (J_q, J_alpha) at the given pose.
        """
        pitch = math.radians(pitch_deg)
        R = self._rotation(pitch, math.radians(roll_deg))
        axis_roll = np.array([math.cos(pitch), 0.0, -math.sin(pitch)])
        Jq = np.zeros((2, 2))
        Ja = np.zeros(2)
        for i, k in enumerate(SIDES):
            alpha = self._crank_angle(k, R)
            if alpha is None:
                raise ValueError(f"leg {k}: pose is outside the workspace")
            T = R @ self.P[k]
            s = T - self._crank_tip(k, alpha)
            t = self.CRANK_LENGTH * (-math.sin(alpha) * self.U[k]
                                     + math.cos(alpha) * self.V[k])
            Jq[i, 0] = s @ np.cross([0.0, 1.0, 0.0], T)
            Jq[i, 1] = s @ np.cross(axis_roll, T)
            Ja[i] = s @ t
        return Jq, np.diag(Ja)

    def self_check(self, verbose=True):
        """Consistency checks. Run once after changing any geometry."""
        problems = []

        for k in SIDES:
            res = self.residual(k, 0.0, 0.0, math.degrees(self.alpha_home[k]))
            if abs(res) > 1e-6:
                problems.append(f"leg {k}: home residual {res:.4f} mm")
            if verbose:
                print(f"  leg {k}: home crank angle "
                      f"{math.degrees(self.alpha_home[k]):8.3f} deg, "
                      f"residual {res:+.2e} mm")

        for pitch, roll in [(0, 0), (10, 5), (-10, 5), (20, -8), (5, 15)]:
            if not self.is_admissible(pitch, roll):
                continue
            angles = self.get_crank_angles(pitch, roll, clamp=False)
            back = self.forward_kinematics(angles, guess_deg=(0.0, 0.0))
            err = max(abs(back[0] - pitch), abs(back[1] - roll))
            if err > 1e-6:
                problems.append(f"IK/FK round trip at ({pitch}, {roll}): "
                                f"error {err:.2e} deg")

        Jq, _ = self.jacobians(0.0, 0.0)
        cond = np.linalg.cond(Jq)
        if verbose:
            print(f"  level pose: det(Jq) = {np.linalg.det(Jq):.3e}, "
                  f"cond(Jq) = {cond:.3f}")
        if cond > 10.0:
            problems.append(f"level pose poorly conditioned: cond = {cond:.1f}")

        if verbose:
            print("  OK" if not problems else "  PROBLEMS:")
            for p in problems:
                print(f"    - {p}")
        return problems


if __name__ == "__main__":
    solver = KinematicSolver()
    print("self check:")
    solver.self_check()

    print("\npose -> crank angles -> goal counts")
    print(f"{'pitch':>7}{'roll':>7}{'scale':>8}"
          f"{'alpha L':>10}{'alpha R':>10}{'count L':>10}{'count R':>10}")
    for pitch, roll in [(0, 0), (10, 0), (-10, 0), (0, 10), (0, 15),
                        (7, 20), (0, 25), (30, 15), (45, 0)]:
        p, r, s = solver.clamp_pose(pitch, roll)
        a = solver.get_crank_angles(pitch, roll)
       # print(f"{pitch:+7}{roll:+7}{s:8.2f}"
         #     f"{a['L']:10.2f}{a['R']:10.2f}{c['L']:10d}{c['R']:10d}")