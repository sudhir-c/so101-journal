"""
Forward kinematics for the SO-101, from scratch in numpy — no placo, no IK.

Parses the kinematic chain base_link -> gripper_frame_link straight out of the
URDF and composes homogeneous transforms. `fk(joint_deg)` returns the gripper
tip position (metres, base frame). Pure function, no hardware.

Convention: URDF joint <origin> is `T = Translate(xyz) · Rz(yaw)·Ry(pitch)·Rx(roll)`
(fixed-axis rpy), then each revolute joint rotates about its axis (all +z on this
arm) by the joint angle. Input angles are DEGREES, matching RobotController's
`get_positions()` (converted to radians internally).

Run `python -m teleop.rl_reach.fk` to self-test against the placo FK sidecar.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

URDF_PATH = Path(__file__).resolve().parents[1] / "robot" / "urdf" / "so101_new_calib.urdf"
BASE_LINK = "base_link"
TIP_LINK = "gripper_frame_link"
# The actuated joints on the base->tip chain, in order (gripper is off-chain).
CHAIN_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def _rpy_to_R(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _origin_T(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rpy_to_R(*rpy)
    T[:3, 3] = xyz
    return T


def _rot_axis(axis: np.ndarray, theta: float) -> np.ndarray:
    """Homogeneous rotation of `theta` (rad) about a unit `axis` (Rodrigues)."""
    a = axis / np.linalg.norm(axis)
    c, s = np.cos(theta), np.sin(theta)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    return T


def _load_chain(path: Path = URDF_PATH):
    """Return the base->tip chain as a list of (joint_name, type, origin_T, axis)."""
    root = ET.parse(str(path)).getroot()
    by_child = {}  # child_link -> (name, type, parent_link, origin_T, axis)
    for j in root.findall("joint"):
        origin = j.find("origin")
        xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
        rpy = np.array([float(v) for v in origin.get("rpy", "0 0 0").split()])
        axis_el = j.find("axis")
        axis = np.array([float(v) for v in axis_el.get("xyz").split()]) if axis_el is not None else np.array([0, 0, 1.0])
        by_child[j.find("child").get("link")] = (
            j.get("name"), j.get("type"), j.find("parent").get("link"), _origin_T(xyz, rpy), axis,
        )
    chain = []
    link = TIP_LINK
    while link != BASE_LINK:
        name, jtype, parent, origin_T, axis = by_child[link]
        chain.append((name, jtype, origin_T, axis))
        link = parent
    chain.reverse()
    return chain


_CHAIN = _load_chain()


def fk_chain(joint_deg) -> list[list[float]]:
    """Positions (metres, base frame) of every joint on the chain, base→tip.

    Returns one xyz per joint (each joint's origin after its fixed <origin>
    transform) plus the tip as the final point — i.e. the vertices of the arm
    skeleton for the given joint angles. `fk()` is `fk_chain(...)[-1]`.

    `joint_deg` is a mapping {joint_name: degrees} (the chain joints are used;
    gripper is ignored) or a sequence in CHAIN_JOINTS order.
    """
    if not isinstance(joint_deg, dict):
        joint_deg = {n: float(joint_deg[i]) for i, n in enumerate(CHAIN_JOINTS)}
    points = [[0.0, 0.0, 0.0]]        # base_link origin
    T = np.eye(4)
    for name, jtype, origin_T, axis in _CHAIN:
        T = T @ origin_T
        points.append(T[:3, 3].tolist())   # this joint's location
        if jtype == "revolute":
            T = T @ _rot_axis(axis, np.deg2rad(joint_deg[name]))
    return points


def fk(joint_deg) -> np.ndarray:
    """Gripper tip position (metres, base frame) for the given joint angles.

    `joint_deg` is a mapping {joint_name: degrees} (the chain joints are used;
    gripper is ignored) or a sequence in CHAIN_JOINTS order.
    """
    return np.array(fk_chain(joint_deg)[-1])


def _self_test() -> int:
    home = {n: 0.0 for n in CHAIN_JOINTS}
    tip = fk(home)
    print(f"home tip: {np.round(tip, 4)} m  ({np.round(tip * 1000, 1)} mm)")

    # Ground-truth: compare against the existing placo FK (same URDF), offline.
    try:
        import random

        from teleop.robot.control import JOINT_LIMITS
        from teleop.robot.kinematics import ArmIKClient
    except Exception as e:  # noqa: BLE001
        print("placo validation skipped (import):", e)
        return 0

    ik = ArmIKClient()
    try:
        worst = 0.0
        for _ in range(200):
            q = {n: random.uniform(*JOINT_LIMITS[n]) for n in CHAIN_JOINTS}
            mine_mm = fk(q) * 1000.0
            placo_mm = np.array(ik.fk_tip_mm({**q, "gripper": 0.0}))
            worst = max(worst, float(np.linalg.norm(mine_mm - placo_mm)))
    finally:
        ik.close()
    print(f"max FK diff vs placo over 200 random poses: {worst:.4f} mm")
    ok = worst < 1.0
    print("FK VALIDATION", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
