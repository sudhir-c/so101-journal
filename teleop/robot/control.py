"""
Robot control safety layer for SO-101 follower arm.

Intentionally decoupled from HTTP — a future pose-estimator client
(hand tracking, etc.) can call set_joints() directly without going
through the FastAPI layer.
"""

import logging
import threading
import time
from typing import Optional

from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

logger = logging.getLogger(__name__)

# Motor order matches IDs 1–6 in the spectre arm
JOINT_NAMES: list[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Per-joint safe ranges derived from spectre.json calibration.
#
# For DEGREES joints (shoulder_pan … wrist_roll):
#   mid = (range_min + range_max) / 2
#   limit = ±(range_max − mid) × (360 / 4096)
#
# Gripper uses MotorNormMode.RANGE_0_100 regardless of use_degrees.
#
# These are the PHYSICAL limits from calibration — tighten them here
# to be more conservative before handing the arm to new code.
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan":  (-102.3, 102.3),
    "shoulder_lift": (-102.7, 102.7),
    "elbow_flex":    (-96.2,   96.2),
    "wrist_flex":    (-100.5, 100.5),
    "wrist_roll":    (-180.0, 180.0),
    "gripper":       (0.0,    100.0),
}

# Maximum joint SPEED, in degrees/second (or %/s for the gripper).
#
# This is a velocity limit, not a per-command step: each step() measures the
# real elapsed time since the previous command and caps motion to
# MAX_SPEED_DPS × dt. That makes the speed independent of how fast commands
# arrive — whether the transport is 20 Hz HTTP, an uncapped WebSocket, or a
# variable-rate pose estimator, the arm never exceeds this °/s. This is the
# primary anti-lunge safety knob: raise it for snappier motion, lower it for
# gentler motion.
MAX_SPEED_DPS: float = 180.0

# Cap on the elapsed time used for the velocity limit. After an idle gap (no
# commands for a while, or the very first command), dt could be large and a
# single step could lunge. Clamping dt bounds any single step to
# MAX_SPEED_DPS × MAX_DT degrees (here: 180 × 0.1 = 18°).
MAX_DT: float = 0.1


class RobotController:
    """
    Thread-safe wrapper around SOFollower with per-joint safety limits.

    Lifecycle:
        ctrl = RobotController(port=..., robot_id=...)
        ctrl.connect()           # arm goes limp briefly while configure() runs
        positions = ctrl.get_positions()
        ctrl.set_joint("shoulder_pan", 30.0)
        ctrl.set_joints({"shoulder_pan": 30.0, "elbow_flex": -20.0})
        ctrl.stop()              # reject further commands; arm holds position
        ctrl.resume()
        ctrl.disconnect()        # torque disabled on disconnect
    """

    def __init__(self, port: str, robot_id: str):
        self.port = port
        self.robot_id = robot_id
        self._robot: Optional[SOFollower] = None
        self._lock = threading.Lock()
        self._stopped = False
        self._last_step_t: Optional[float] = None  # monotonic time of last step()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        config = SOFollowerRobotConfig(
            port=self.port,
            id=self.robot_id,
            use_degrees=True,
            # Rate-limiting is done in software (see _compute_action) so we can
            # reuse one position read per tick. Leaving this None avoids a
            # redundant serial read inside send_action.
            max_relative_target=None,
            disable_torque_on_disconnect=True,
        )
        self._robot = SOFollower(config)
        # calibrate=False: spectre.json is already loaded into self._robot.calibration
        # in __init__ and was previously written to the motors during setup.
        # Avoids blocking on input() in a server context.
        self._robot.connect(calibrate=False)
        logger.info("Connected to %s on %s", self.robot_id, self.port)

    def disconnect(self) -> None:
        with self._lock:
            if self._robot and self._robot.is_connected:
                self._robot.disconnect()
            self._robot = None
        logger.info("Disconnected from %s", self.robot_id)

    @property
    def is_connected(self) -> bool:
        return self._robot is not None and self._robot.is_connected

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_positions(self) -> dict[str, float]:
        """Return current joint positions (degrees / %) freshly read from arm."""
        with self._lock:
            return self._read_positions_unlocked()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def step(self, targets: dict[str, float]) -> dict[str, float]:
        """
        The low-latency streaming path: read present position, command
        clamped + rate-limited goals toward `targets`, and return the
        positions — all in ONE serial read + ONE serial write.

        This is the primary entry point for streaming clients (the UI at
        20 Hz, and later the hand-tracking pose estimator). Returning
        positions lets the caller update its display without a second
        round-trip.

        Safety chain (per joint): validate name → clamp to JOINT_LIMITS →
        limit motion to MAX_SPEED_DPS × dt from PRESENT position → reject all
        when stopped.

        Returns the present positions read at the start of this step (i.e.
        one tick "behind" the commanded motion, which is imperceptible).
        """
        unknown = set(targets) - set(JOINT_NAMES)
        if unknown:
            raise ValueError(f"Unknown joints: {unknown}")
        with self._lock:
            self._require_connected()
            now = time.monotonic()
            # Elapsed time since last command, clamped so a post-idle step
            # can't lunge. None (first call) → treat as a full MAX_DT tick.
            dt = MAX_DT if self._last_step_t is None else min(now - self._last_step_t, MAX_DT)
            self._last_step_t = now

            present = self._read_positions_unlocked()
            if not self._stopped and targets:
                action = self._compute_action(targets, present, dt)
                if action:
                    self._robot.send_action(action)
            return present

    def set_joint(self, joint: str, angle: float) -> None:
        """Command a single joint (convenience wrapper around step())."""
        if joint not in JOINT_NAMES:
            raise ValueError(f"Unknown joint '{joint}'. Valid: {JOINT_NAMES}")
        self.step({joint: angle})

    def set_joints(self, angles: dict[str, float]) -> None:
        """Command multiple joints at once (convenience wrapper around step())."""
        self.step(angles)

    # ------------------------------------------------------------------
    # Safety controls
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """
        Reject all motion commands until resume() is called.
        The arm stays in position-control mode and holds its last
        commanded position — it does NOT go limp.
        """
        with self._lock:
            self._stopped = True
        logger.warning("STOP engaged — motion commands rejected")

    def resume(self) -> None:
        """Clear the stopped state and allow motion commands again."""
        with self._lock:
            self._stopped = False
        logger.info("STOP cleared — motion commands accepted")

    @property
    def is_stopped(self) -> bool:
        return self._stopped

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise RuntimeError("Robot not connected")

    def _read_positions_unlocked(self) -> dict[str, float]:
        """Read present positions. Caller must hold self._lock."""
        self._require_connected()
        obs = self._robot.get_observation()
        return {k.removesuffix(".pos"): v for k, v in obs.items() if k.endswith(".pos")}

    def _compute_action(
        self, targets: dict[str, float], present: dict[str, float], dt: float
    ) -> dict[str, float]:
        """
        Turn desired targets into a safe action dict, given the present
        position and the elapsed time since the last command. Each joint is
        clamped to its limit, then the motion is capped to
        MAX_SPEED_DPS × dt from where the joint actually is.
        Caller must hold self._lock.
        """
        step_cap = MAX_SPEED_DPS * dt
        action = {}
        for joint, tgt in targets.items():
            cur = present[joint]
            clamped = self._clamp(joint, tgt)
            delta = max(-step_cap, min(step_cap, clamped - cur))
            action[f"{joint}.pos"] = cur + delta
        return action

    @staticmethod
    def _clamp(joint: str, angle: float) -> float:
        lo, hi = JOINT_LIMITS[joint]
        return max(lo, min(hi, float(angle)))
