"""
Client for the placo IK sidecar (`teleop/robot/ik_service.py`, run in `.venv-ik`).

placo can't go in the dashboard's main venv without bumping numpy (which would
disturb torch/LeRobot), so the kinematics run in an isolated venv as a small
subprocess. This client spawns it and exchanges JSON-line requests. If the
sidecar venv or URDF is missing, construction raises and the dashboard falls
back to slider-only mode — IK is strictly additive.

Everything is in degrees (matching RobotController) and millimeters (tip space).
"""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

from teleop.robot.control import JOINT_LIMITS

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
IK_VENV_PY = REPO / ".venv-ik" / "bin" / "python"
IK_SERVICE = HERE / "ik_service.py"
URDF = HERE / "urdf" / "so101_kinematics.urdf"

TIP_FRAME = "gripper_frame_link"  # fixed (jaw-independent) end-effector tip frame
TOL_MM = 5.0                      # reachability tolerance for "unsolvable"

# Joints IK solves vs. holds (see ik_service). Exposed for the server/UI.
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
PASSTHROUGH_JOINTS = ["wrist_roll", "gripper"]


class ArmIKClient:
    """Spawns and talks to the placo IK sidecar. Thread-safe (one lock around the pipe)."""

    def __init__(self):
        if not IK_VENV_PY.exists():
            raise FileNotFoundError(f"IK venv not found: {IK_VENV_PY} (create .venv-ik with placo)")
        if not URDF.exists():
            raise FileNotFoundError(f"URDF not found: {URDF}")
        limits = {k: list(v) for k, v in JOINT_LIMITS.items()}
        self._proc = subprocess.Popen(
            [str(IK_VENV_PY), "-u", str(IK_SERVICE), str(URDF), TIP_FRAME, str(TOL_MM), json.dumps(limits)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._lock = threading.Lock()
        ready = self._proc.stdout.readline()
        if "ready" not in ready:
            raise RuntimeError(f"IK sidecar failed to start: {ready!r}")

    def _rpc(self, req: dict) -> dict:
        with self._lock:
            if self._proc.poll() is not None:
                raise RuntimeError("IK sidecar has exited")
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("IK sidecar closed the pipe")
        resp = json.loads(line)
        if "error" in resp:
            raise RuntimeError(f"IK sidecar: {resp['error']}")
        return resp

    def fk_tip_mm(self, q: dict[str, float]) -> list[float]:
        """Forward kinematics: tip position (mm, base frame) for a full joint dict."""
        return self._rpc({"op": "fk", "q": q})["tip_mm"]

    def solve(self, seed: dict[str, float], target_mm, wrist_roll: float):
        """Solve the 4 arm joints for the tip target (mm). wrist_roll is held.

        Returns (sol|None, solvable, err_mm). `sol` is the clamped 4-joint dict
        when solvable, else None (so the caller commands no arm motion)."""
        r = self._rpc({
            "op": "ik",
            "seed": seed,
            "target_mm": list(target_mm),
            "wrist_roll": wrist_roll,
        })
        return r["sol"], r["solvable"], r["err_mm"]

    def close(self) -> None:
        try:
            self._proc.terminate()
            self._proc.wait(timeout=2)
        except Exception:  # noqa: BLE001 - best-effort shutdown
            self._proc.kill()
