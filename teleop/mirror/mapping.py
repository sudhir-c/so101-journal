"""
Human-arm-angle → robot-joint-angle mapping for the live mirror.

A single affine map per joint turns a human joint angle into a robot target:

    robot = robot_min + (human - human_min)/(human_max - human_min)
            * (robot_max - robot_min)

The two calibration endpoints (human_min/human_max, captured per user) and the
robot endpoints together absorb range, zero-offset AND direction — there is no
separate sign/offset logic. Flip a joint by swapping its robot endpoints
(see INVERT) or by capturing the human extremes in the other order.

Everything the mapping produces still passes through the robot safety core
(`RobotController.step`: clamp → velocity-limit) before it reaches a motor.
"""

from __future__ import annotations

import json
from pathlib import Path

from teleop.robot.control import JOINT_LIMITS

# Joints this mirror drives from the human arm.
CONTROLLED: list[str] = ["shoulder_lift", "elbow_flex", "wrist_flex", "gripper"]

# Joints held at a fixed, safe constant (NOT driven). 0.0 is the calibrated
# centre of each joint's travel — the neutral, safe hold pose.
HELD: dict[str, float] = {"wrist_roll": 0.0, "shoulder_pan": 0.0}

# Robot endpoints per controlled joint = the joint's calibrated limits from
# spectre.json (exposed as JOINT_LIMITS). (robot_at_human_min, robot_at_human_max).
ROBOT_RANGE: dict[str, tuple[float, float]] = {j: JOINT_LIMITS[j] for j in CONTROLLED}

# Per-joint direction flip (affine joints only). If the mirror comes out
# reversed for a joint in the live preview, flip it here.
INVERT: dict[str, bool] = {
    "shoulder_lift": False,
    "elbow_flex": False,
    "wrist_flex": False,
    "gripper": False,
}

# Per-joint constant offset (degrees), added to the affine result then re-clamped.
OFFSET: dict[str, float] = {
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "gripper": 0.0,
}

# Pivot/gain mapping — used INSTEAD of the endpoint-affine for the joints listed
# here (calibration-independent). robot = robot_pivot + gain·(human − human_pivot).
# Tuned per joint as we bring each one online.
#   shoulder_lift: human 90° (arm horizontal) → robot 0°; arm UP (human ↑) drives
#                  the target DOWN (gain −1, i.e. 1:1 inverted).
#   elbow_flex:    human 90° (right angle) → robot 0°; straightening (human ↑)
#                  drives the target UP (gain +1, i.e. 1:1).
#   wrist_flex:    human 180° (hand in line) → robot 90°; bending the wrist
#                  (human ↓) sweeps the target down from 90 (gain +1), using the
#                  full range without clamping.
# gripper is NOT here — it uses the plain affine (human 0..1 → robot 0..100%).
PIVOT: dict[str, dict[str, float]] = {
    "shoulder_lift": {"human_pivot": 90.0, "robot_pivot": 0.0, "gain": -1.0},
    "elbow_flex": {"human_pivot": 90.0, "robot_pivot": 0.0, "gain": 1.0},
    "wrist_flex": {"human_pivot": 180.0, "robot_pivot": 90.0, "gain": 1.0},
}

# Placeholder human ranges used until you run calibration (Phase 2 overwrites
# these from your captured extremes). Degrees for the arm joints; 0..1 gripper.
#   shoulder_lift: hanging ~0 .. raised ~180
#   elbow_flex:    fully bent ~40 .. straight ~180
#   wrist_flex:    bent ~90 .. in line with forearm ~180
DEFAULT_HUMAN_RANGE: dict[str, tuple[float, float]] = {
    "shoulder_lift": (10.0, 160.0),
    "elbow_flex": (40.0, 178.0),
    "wrist_flex": (95.0, 178.0),
    "gripper": (0.0, 1.0),
}

CALIBRATION_PATH = Path(__file__).resolve().parent / "mirror_calibration.json"


def _affine(x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    if in_max == in_min:
        return out_min
    t = (x - in_min) / (in_max - in_min)
    return out_min + t * (out_max - out_min)


def map_human_to_robot(
    human: dict[str, float | None],
    human_range: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Map human joint values to robot targets for the controlled joints.

    `human` maps joint → value (or None when that joint isn't tracked this
    frame). Joints that are None are omitted from the result — the caller
    should HOLD the last good target for them, never send a guess.

    Every returned target is clamped to the joint's JOINT_LIMITS as a final
    guard (the robot safety core clamps again downstream).
    """
    targets: dict[str, float] = {}
    for joint in CONTROLLED:
        value = human.get(joint)
        if value is None:
            continue
        if joint in PIVOT:
            p = PIVOT[joint]
            robot = p["robot_pivot"] + p["gain"] * (value - p["human_pivot"])
        else:
            hmin, hmax = human_range[joint]
            rmin, rmax = ROBOT_RANGE[joint]
            if INVERT[joint]:
                rmin, rmax = rmax, rmin
            robot = _affine(value, hmin, hmax, rmin, rmax) + OFFSET[joint]
        lo, hi = JOINT_LIMITS[joint]
        targets[joint] = max(lo, min(hi, robot))
    return targets


# --------------------------------------------------------------------------- #
# Calibration persistence (human_min/human_max per controlled joint)
# --------------------------------------------------------------------------- #

def load_human_range(path: Path = CALIBRATION_PATH) -> dict[str, tuple[float, float]]:
    """Load saved human calibration, falling back to DEFAULT_HUMAN_RANGE for any
    joint that isn't present. Returns a full range dict for all CONTROLLED joints."""
    ranges = {j: DEFAULT_HUMAN_RANGE[j] for j in CONTROLLED}
    if path.is_file():
        try:
            saved = json.loads(path.read_text())
            for j in CONTROLLED:
                if j in saved and len(saved[j]) == 2:
                    ranges[j] = (float(saved[j][0]), float(saved[j][1]))
        except (ValueError, KeyError, TypeError):
            pass  # corrupt file → defaults
    return ranges


def save_human_range(ranges: dict[str, tuple[float, float]], path: Path = CALIBRATION_PATH) -> None:
    path.write_text(json.dumps({j: list(ranges[j]) for j in ranges}, indent=2))


def has_calibration(path: Path = CALIBRATION_PATH) -> bool:
    return path.is_file()
