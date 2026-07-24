"""
Gymnasium environment: SO-101 end-effector reaching on real hardware.

A deliberately simple reach task (free reward, free reset) that validates the
whole RL loop end-to-end and is structured so a later grasp task can swap the
reward / termination / reset without touching step().

Design:
  * Actions are small joint DELTAS on the 4 position-relevant joints, pushed
    through RobotController.step() — the existing safety core (clamp to limits +
    velocity limit). wrist_roll (overload history) and gripper are held fixed.
  * Observation is LEAN — only what the policy needs: 6 joint positions
    (normalized) + the target xyz (normalized). Nothing reward-only.
  * `info` is a RICH side channel (not seen by the policy): fk tip xyz, distance,
    raw joints, gripper width, and per-motor current/load/temperature. Future
    grasp rewards read from here without changing the observation.
  * Reward / termination / reset live in isolated methods (`_compute_reward`,
    `_is_terminated`, `_reset_task`) so `step()` is task-agnostic.

Dry run (no SB3):  .venv/bin/python -m teleop.rl_reach.so101_reach_env
"""

from __future__ import annotations

import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from teleop.robot.control import JOINT_LIMITS, JOINT_NAMES, RobotController

from .fk import CHAIN_JOINTS, fk

# ── Hardware ────────────────────────────────────────────────────────────────
PORT = "/dev/tty.usbmodem5B3D0486331"
ROBOT_ID = "spectre"

# ── Task / safety config (tune here) ────────────────────────────────────────
# Joints the policy controls (deltas). The rest are held fixed.
ACTION_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]

MAX_DELTA_DEG = 3.0          # per-step action bound (deg); safety core caps at 180 deg/s
CONTROL_HZ = 10.0            # control-loop rate
MAX_STEPS = 200              # episode step limit (truncation)

# Measured workspace bounds for the gripper tip (metres, base frame).
MIN_EE = np.array([0.1366, -0.0548, 0.0591])
MAX_EE = np.array([0.3099, 0.1090, 0.2297])
EE_MARGIN = 0.005            # shrink the allowed box this much (m) for safety

# Reward weights.
W_DIST = 1.0                 # per-metre distance penalty (dense; keeps ep_rew ≈ -avg_dist·steps)
SUCCESS_THRESH = 0.03        # m — within this counts as reaching
SUCCESS_BONUS = 10.0         # decisive terminal reward: dwarfs a single -dist step, so
                             # crossing the threshold (and terminating) is clearly worth it
W_ACTION = 0.01              # penalty on ‖action‖² to discourage thrashing
W_PROGRESS = 15.0            # potential-based shaping weight. Shaping = Φ(dist)−Φ(prev),
SHAPE_SCALE = 0.08           # with Φ = W_PROGRESS·(1−tanh(dist/SHAPE_SCALE)): a directional
                             # gradient that STEEPENS near the goal (tanh), so closing the
                             # final cm is worth disproportionately more — this is what
                             # centres the policy inside SUCCESS_THRESH instead of parking
                             # just outside. Potential-based ⇒ telescopes to a constant, so
                             # it can't be farmed by camping near the goal.

# Re-home: at the START of each episode, smoothly move the arm back to a
# consistent central pose so it can't drift into a corner across the free-reset
# episodes (which stalls learning). The pose is given in CARTESIAN xyz and solved
# to joint angles ONCE at env init via the placo IK sidecar — the step-loop stays
# IK-free. Set HOME_XYZ = None to disable re-homing.
HOME_XYZ = ((MIN_EE + MAX_EE) / 2.0).copy()   # box centre (m); set to any in-box spot
REHOME_TOL_DEG = 1.5         # "home reached" tolerance per joint
REHOME_MAX_STEPS = 50        # cap the ramp (safety)


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


class SO101ReachEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, controller: RobotController | None = None, connect: bool = True,
                 rehome: bool = True):
        super().__init__()
        self.controller = controller or RobotController(port=PORT, robot_id=ROBOT_ID)
        self._owns_controller = controller is None
        if connect and not self.controller.is_connected:
            self.controller.connect()

        self.action_space = spaces.Box(-1.0, 1.0, (len(ACTION_JOINTS),), dtype=np.float32)
        # obs = 6 joint positions (normalized) + target xyz (normalized) = 9
        self.observation_space = spaces.Box(-1.0, 1.0, (len(JOINT_NAMES) + 3,), dtype=np.float32)

        self._target = np.zeros(3)
        self._last_pos: dict[str, float] = {}
        self._prev_dist = 0.0        # distance at the previous step (progress shaping)
        self._step_count = 0
        # Solve the Cartesian home pose to joint targets once (IK sidecar).
        self._home_arm = self._solve_home() if (rehome and HOME_XYZ is not None) else None

    # ── normalization ─────────────────────────────────────────────────────
    @staticmethod
    def _norm_joint(name: str, v: float) -> float:
        lo, hi = JOINT_LIMITS[name]
        return _clip(2.0 * (v - lo) / (hi - lo) - 1.0, -1.0, 1.0)

    def _norm_target(self) -> np.ndarray:
        return np.clip(2.0 * (self._target - MIN_EE) / (MAX_EE - MIN_EE) - 1.0, -1.0, 1.0)

    def _obs(self, pos: dict[str, float]) -> np.ndarray:
        joints = [self._norm_joint(n, pos[n]) for n in JOINT_NAMES]
        return np.array([*joints, *self._norm_target()], dtype=np.float32)

    # ── EE workspace guard ────────────────────────────────────────────────
    @staticmethod
    def _violation(xyz: np.ndarray) -> float:
        """How far the tip is outside the (margin-shrunk) box; 0 if inside."""
        low = np.maximum(0.0, (MIN_EE + EE_MARGIN) - xyz)
        high = np.maximum(0.0, xyz - (MAX_EE - EE_MARGIN))
        return float(np.sum(low + high))

    def _predict_tip(self, base_pos: dict[str, float], targets: dict[str, float]) -> np.ndarray:
        q = {n: targets.get(n, base_pos[n]) for n in CHAIN_JOINTS}
        return fk(q)

    # ── task hooks (override for grasp) ───────────────────────────────────
    def _reset_task(self) -> None:
        self._target = self.np_random.uniform(MIN_EE, MAX_EE)

    @staticmethod
    def _phi(dist: float) -> float:
        """Shaping potential: high near the goal, saturating far away."""
        return W_PROGRESS * (1.0 - float(np.tanh(dist / SHAPE_SCALE)))

    def _compute_reward(self, obs, action, info) -> float:
        r = -W_DIST * info["distance"]
        r += self._phi(info["distance"]) - self._phi(self._prev_dist)   # tanh progress shaping
        if info["is_success"]:
            r += SUCCESS_BONUS
        r -= W_ACTION * float(np.sum(np.square(action)))
        return r

    def _is_terminated(self, info) -> bool:
        return info["is_success"]

    # ── info assembly ─────────────────────────────────────────────────────
    def _make_info(self, pos: dict[str, float], rejected: bool) -> dict:
        tip = fk(pos)
        dist = float(np.linalg.norm(tip - self._target))
        info = {
            "tip_xyz": tip,
            "target_xyz": self._target.copy(),
            "distance": dist,
            "is_success": dist < SUCCESS_THRESH,
            "joints_deg": dict(pos),
            "gripper": pos["gripper"],         # width proxy (0-100 %)
            "action_rejected": rejected,
        }
        try:
            info["telemetry"] = self.controller.read_telemetry()
        except Exception:  # noqa: BLE001 - telemetry is best-effort
            info["telemetry"] = {}
        return info

    # ── re-home (episode reset to a consistent central pose) ──────────────
    def _solve_home(self) -> dict[str, float]:
        """Solve HOME_XYZ to arm-joint targets once, via the placo IK sidecar.
        IK runs only here (init), never in the step loop."""
        from teleop.robot.kinematics import ArmIKClient  # lazy: only needed if re-homing

        q = self.controller.get_positions()
        ik = ArmIKClient()
        try:
            sol, solvable, err = ik.solve(q, np.asarray(HOME_XYZ) * 1000.0, q["wrist_roll"])
        finally:
            ik.close()
        if not solvable or sol is None:
            raise RuntimeError(
                f"HOME_XYZ {np.round(HOME_XYZ, 3)} is not reachable (IK err {err:.1f} mm) — "
                "pick an in-box spot for HOME_XYZ"
            )
        return {j: float(sol[j]) for j in ACTION_JOINTS}

    def _home(self) -> None:
        """Smoothly rate-limit the arm to the home pose (velocity-limited by step())."""
        for _ in range(REHOME_MAX_STEPS):
            pos = self.controller.get_positions()
            if all(abs(pos[j] - self._home_arm[j]) <= REHOME_TOL_DEG for j in ACTION_JOINTS):
                break
            self.controller.step(dict(self._home_arm))   # clamps + velocity-limits
            time.sleep(1.0 / CONTROL_HZ)
        self._last_pos = self.controller.get_positions()

    # ── gym API ───────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._home_arm is not None:
            self._home()                         # re-center so the arm can't drift into a corner
        self._reset_task()                       # sample a new target
        self._last_pos = self.controller.get_positions()
        self._step_count = 0
        obs = self._obs(self._last_pos)
        info = self._make_info(self._last_pos, rejected=False)
        self._prev_dist = info["distance"]      # baseline for progress shaping
        return obs, info

    def step(self, action):
        t0 = time.perf_counter()
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        base = self._last_pos
        targets = {j: base[j] + float(action[i]) * MAX_DELTA_DEG for i, j in enumerate(ACTION_JOINTS)}

        # EE-bounds guard: reject a step that would push the tip further out of
        # the workspace box (allows moving back in; blocks moving further out).
        rejected = self._violation(self._predict_tip(base, targets)) > self._violation(fk(base)) + 1e-9
        self.controller.step({} if rejected else targets)   # safety core clamps + velocity-limits

        # Hold the control period so the motion actually executes.
        dwell = (1.0 / CONTROL_HZ) - (time.perf_counter() - t0)
        if dwell > 0:
            time.sleep(dwell)

        self._last_pos = self.controller.get_positions()
        self._step_count += 1
        info = self._make_info(self._last_pos, rejected)
        obs = self._obs(self._last_pos)
        reward = self._compute_reward(obs, action, info)
        self._prev_dist = info["distance"]      # advance progress baseline
        terminated = self._is_terminated(info)
        truncated = self._step_count >= MAX_STEPS
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._owns_controller and self.controller.is_connected:
            self.controller.disconnect()          # torque off


def _dry_run():
    """Reset, take random actions through the safety core, print obs/reward/info.
    No SB3 — confirms safe motion + sane numbers at the control rate."""
    env = SO101ReachEnv()
    try:
        obs, info = env.reset(seed=0)
        print(f"target {np.round(info['target_xyz'], 3)} m  start dist {info['distance'] * 1000:.0f} mm")
        for i in range(40):
            a = env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            print(f"{i:3d} dist={info['distance'] * 1000:5.0f}mm  r={r:+.3f}  "
                  f"rejected={info['action_rejected']!s:5}  tip={np.round(info['tip_xyz'], 3)}")
            if term or trunc:
                print("  -> episode end; reset")
                obs, info = env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    _dry_run()
