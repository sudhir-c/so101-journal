"""
Offline verification for the placo IK sidecar — NO motors, no dashboard.

Run in the IK venv (it needs placo, not lerobot):
    .venv-ik/bin/python teleop/robot/scripts/verify_ik.py

Loads the kinematics-only URDF, prints the joint/frame names, then round-trips
FK -> IK -> FK across a grid of small XYZ offsets and several wrist_roll values,
asserting the tip is reached within tolerance and that wrist_roll stays pinned
(proving the middle-joint freeze). Also checks that an out-of-reach target is
correctly reported unsolvable.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from teleop.robot.ik_service import ARM, PlacoArm  # noqa: E402

URDF = REPO / "teleop/robot/urdf/so101_kinematics.urdf"
TIP = "gripper_frame_link"
TOL_MM = 5.0
# Mirror of teleop.robot.control.JOINT_LIMITS (control.py needs lerobot, absent here).
LIMITS = {
    "shoulder_pan": (-102.3, 102.3), "shoulder_lift": (-102.7, 102.7),
    "elbow_flex": (-96.2, 96.2), "wrist_flex": (-100.5, 100.5),
    "wrist_roll": (-180.0, 180.0), "gripper": (0.0, 100.0),
}


def main() -> int:
    arm = PlacoArm(str(URDF), TIP, TOL_MM, LIMITS)
    print("joint_names:", list(arm.robot.joint_names()))
    print("frame_names:", list(arm.robot.frame_names()))

    home = {j: 0.0 for j in ARM + ["wrist_roll", "gripper"]}
    p0 = np.asarray(arm.fk_tip_mm(home))
    print("tip @ home (mm):", np.round(p0, 1))

    worst_err = 0.0
    worst_roll = 0.0
    for roll in (0.0, 30.0, -45.0):
        for dx in (-40, 0, 40):
            for dy in (-40, 0, 40):
                for dz in (-40, 0, 40):
                    target = p0 + np.array([dx, dy, dz], float)
                    sol, err = arm.solve(home, target, roll)
                    worst_err = max(worst_err, err)
                    reached_roll = np.rad2deg(arm.robot.get_joint("wrist_roll"))
                    worst_roll = max(worst_roll, abs(reached_roll - roll))
    print(f"worst reach err over grid: {worst_err:.3f} mm  (tol {TOL_MM})")
    print(f"worst wrist_roll drift from pinned: {worst_roll:.4f} deg")

    _, far_err = arm.solve(home, (p0 + [800, 0, 0]).tolist(), 0.0)
    print(f"far target err: {far_err:.1f} mm -> solvable={far_err <= TOL_MM} (expect False)")

    ok = worst_err <= TOL_MM and worst_roll < 1e-3 and far_err > TOL_MM
    print("VERIFY", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
