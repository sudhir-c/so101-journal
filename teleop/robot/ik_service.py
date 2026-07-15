"""
Placo inverse-kinematics sidecar for the SO-101.

Runs in the isolated `.venv-ik` (placo + numpy 2.3.5) because placo can't be
installed into the dashboard's main venv without bumping numpy and disturbing
torch/LeRobot. It is intentionally STANDALONE — imports only numpy + placo, and
NOT teleop.robot.control — so it loads in the minimal IK venv. The dashboard
(main venv) spawns it via `teleop/robot/kinematics.py` and pipes requests.

Protocol: one JSON object per line on stdin → one JSON object per line on stdout.
  {"op":"fk","q":{joint:deg,...}}
      -> {"tip_mm":[x,y,z]}
  {"op":"ik","seed":{six joints in deg},"target_mm":[x,y,z],"wrist_roll":deg}
      -> {"sol":{4 arm joints deg}|null, "solvable":bool, "err_mm":float}

Config comes from argv: urdf_path, tip_frame, tol_mm, joint_limits_json.

Kinematics: solve the 4 arm joints (shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex) for a tip *position* (orientation unconstrained). wrist_roll and
gripper are HARD-locked so the solver can't exploit them — wrist_roll to the
live value, gripper to 0 (the tip frame is jaw-independent, so gripper is
irrelevant to the tip). Unsolvable = clamp the solution to the real joint
limits, run FK, and check the tip is within tol of the target.
"""

import glob
import json
import os
import sys

# cmeel ships placo.so under a `cmeel.prefix/...` dir that a generated `.pth`
# file is supposed to add to sys.path at startup — but that activation is flaky
# and intermittently doesn't fire (placo.so is present yet `import placo` fails).
# Add the prefix explicitly so this always works, whether launched via the
# client subprocess or imported directly (verify_ik.py). Version-agnostic glob.
for _prefix in glob.glob(
    os.path.join(sys.prefix, "lib", "*", "site-packages", "cmeel.prefix", "lib", "*", "site-packages")
):
    if _prefix not in sys.path:
        sys.path.insert(0, _prefix)

import numpy as np
import placo

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
ALL5 = ARM + ["wrist_roll"]
IK_ITERS = 60  # placo velocity-IK steps per solve; plenty to converge a static target


class PlacoArm:
    def __init__(self, urdf_path: str, tip_frame: str, tol_mm: float, limits: dict):
        self.tip = tip_frame
        self.tol = tol_mm
        self.limits = limits  # {joint: (lo, hi)} degrees / percent
        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)  # base is fixed
        # Position-only frame task on the tip (orientation weight 0).
        self.frame = self.solver.add_frame_task(self.tip, np.eye(4))
        self.frame.configure(self.tip, "soft", 1.0, 0.0)
        # HARD-lock wrist_roll + gripper so they are never used to reach the target.
        self.joints = self.solver.add_joints_task()
        self.joints.set_joints({"wrist_roll": 0.0, "gripper": 0.0})
        self.joints.configure("joints", "hard")
        # Small regularization keeps the redundant (4-DOF for 3-DOF task) null space stable.
        self.solver.add_regularization_task(1e-5)

    def _apply(self, q_deg: dict) -> None:
        for name in ALL5:
            self.robot.set_joint(name, np.deg2rad(q_deg.get(name, 0.0)))
        self.robot.set_joint("gripper", 0.0)
        self.robot.update_kinematics()

    def fk_tip_mm(self, q_deg: dict) -> list:
        self._apply(q_deg)
        return (self.robot.get_T_world_frame(self.tip)[:3, 3] * 1000.0).tolist()

    def _clamp(self, name: str, value: float) -> float:
        lo, hi = self.limits[name]
        return max(lo, min(hi, value))

    def solve(self, seed: dict, target_mm, roll_deg: float):
        # Seed the solver from the current pose.
        for name in ALL5:
            self.robot.set_joint(name, np.deg2rad(seed.get(name, 0.0)))
        self.robot.set_joint("gripper", 0.0)
        # Position target; pin wrist_roll to the live value this solve.
        T = np.eye(4)
        T[:3, 3] = np.asarray(target_mm, dtype=float) / 1000.0
        self.frame.T_world_frame = T
        self.joints.set_joints({"wrist_roll": np.deg2rad(roll_deg), "gripper": 0.0})
        for _ in range(IK_ITERS):
            self.solver.solve(True)
            self.robot.update_kinematics()
        # Clamp to the REAL joint limits, then FK-check reachability under those limits.
        sol = {n: self._clamp(n, float(np.rad2deg(self.robot.get_joint(n)))) for n in ARM}
        reached = self.fk_tip_mm({**sol, "wrist_roll": roll_deg})
        err = float(np.linalg.norm(np.asarray(reached) - np.asarray(target_mm, dtype=float)))
        return sol, err


def main() -> None:
    urdf_path, tip_frame, tol_s, limits_s = sys.argv[1:5]
    tol = float(tol_s)
    limits = {k: tuple(v) for k, v in json.loads(limits_s).items()}
    arm = PlacoArm(urdf_path, tip_frame, tol, limits)
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "fk":
                resp = {"tip_mm": arm.fk_tip_mm(req["q"])}
            elif op == "ik":
                sol, err = arm.solve(req["seed"], req["target_mm"], req.get("wrist_roll", 0.0))
                solvable = err <= tol
                resp = {"sol": sol if solvable else None, "solvable": solvable, "err_mm": err}
            else:
                resp = {"error": f"unknown op {op!r}"}
        except Exception as e:  # never crash the loop; report and continue
            resp = {"error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
