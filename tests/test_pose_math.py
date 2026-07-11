"""Drive compute_arm_state with synthetic landmarks posed at the sanity targets.

These are the acceptance criteria from the spec, expressed as code:

    shoulder_lift  hanging down 0 deg | horizontal 90 deg | straight up 180 deg
    elbow_flex     straight 180 deg   | right angle 90 deg
    wrist_flex     in line 180 deg    | bent 90 deg
    gripper        pinched 0.0        | wide open 1.0

Run:  .venv/bin/python -m pytest tests/ -q
   or .venv/bin/python tests/test_pose_math.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_pose.pose_math import (  # noqa: E402
    GripperConfig,
    HandIdx,
    PoseIdx,
    compute_arm_state,
)

# A deliberately non-square frame: if we ever regress on aspect-ratio correction,
# these tests fail. On 16:9, angles read straight off normalized coords are skewed.
W, H = 1280, 720


@dataclass
class LM:
    """Minimal stand-in for a MediaPipe NormalizedLandmark."""

    x: float
    y: float
    visibility: float = 1.0


def _px_to_norm(x: float, y: float) -> LM:
    """Author landmarks in intuitive pixel space; store them normalized."""
    return LM(x / W, y / H)


def make_pose(shoulder, elbow, wrist, side="right", visibility=1.0):
    """33-slot pose list with only the tracked arm's three joints filled in."""
    lms = [LM(0.0, 0.0, 0.0) for _ in range(33)]
    s_i, e_i, w_i = PoseIdx.for_side(side)
    for idx, (px, py) in ((s_i, shoulder), (e_i, elbow), (w_i, wrist)):
        lm = _px_to_norm(px, py)
        lm.visibility = visibility
        lms[idx] = lm
    return lms


def make_hand(wrist, middle_mcp, thumb_tip, index_tip):
    lms = [LM(0.0, 0.0) for _ in range(21)]
    lms[HandIdx.WRIST] = _px_to_norm(*wrist)
    lms[HandIdx.MIDDLE_FINGER_MCP] = _px_to_norm(*middle_mcp)
    lms[HandIdx.THUMB_TIP] = _px_to_norm(*thumb_tip)
    lms[HandIdx.INDEX_FINGER_TIP] = _px_to_norm(*index_tip)
    return lms


def state_for(pose=None, hand=None, side="right", gcfg=None):
    return compute_arm_state(
        pose, hand, side=side, image_size=(W, H), gripper_config=gcfg
    )


def close(a, b, tol=1.0):
    return a is not None and abs(a - b) <= tol


# --------------------------------------------------------------------------- #
# 1. shoulder_lift  (image y grows DOWNWARD)
# --------------------------------------------------------------------------- #
def test_shoulder_lift_hanging_down_is_0():
    # elbow directly below shoulder
    s = state_for(make_pose((600, 200), (600, 400), (600, 600)))
    assert close(s.shoulder_lift, 0.0), s.shoulder_lift


def test_shoulder_lift_horizontal_is_90():
    # elbow straight out to the side, same height as shoulder
    s = state_for(make_pose((600, 300), (900, 300), (1100, 300)))
    assert close(s.shoulder_lift, 90.0), s.shoulder_lift


def test_shoulder_lift_straight_up_is_180():
    # elbow directly above shoulder
    s = state_for(make_pose((600, 500), (600, 300), (600, 100)))
    assert close(s.shoulder_lift, 180.0), s.shoulder_lift


def test_shoulder_lift_45_degrees():
    # down-and-out at 45 deg: equal +x and +y from the shoulder
    s = state_for(make_pose((600, 300), (800, 500), (900, 600)))
    assert close(s.shoulder_lift, 45.0), s.shoulder_lift


def test_shoulder_lift_is_unsigned_across_body_midline():
    """Swinging the arm to the other side must not flip the sign."""
    right = state_for(make_pose((600, 300), (900, 300), (1000, 300)))
    left = state_for(make_pose((600, 300), (300, 300), (200, 300)))
    assert close(right.shoulder_lift, 90.0)
    assert close(left.shoulder_lift, 90.0)


# --------------------------------------------------------------------------- #
# 2. elbow_flex
# --------------------------------------------------------------------------- #
def test_elbow_flex_straight_arm_is_180():
    s = state_for(make_pose((600, 200), (600, 400), (600, 600)))
    assert close(s.elbow_flex, 180.0), s.elbow_flex


def test_elbow_flex_right_angle_is_90():
    # upper arm points down, forearm points out horizontally
    s = state_for(make_pose((600, 200), (600, 400), (800, 400)))
    assert close(s.elbow_flex, 90.0), s.elbow_flex


def test_elbow_flex_fully_folded_is_near_0():
    # forearm doubled back alongside the upper arm
    s = state_for(make_pose((600, 200), (600, 400), (600, 210)))
    assert s.elbow_flex is not None and s.elbow_flex < 10.0, s.elbow_flex


def test_elbow_flex_45_degrees():
    s = state_for(make_pose((600, 200), (600, 400), (600 + 200, 400 - 200)))
    assert close(s.elbow_flex, 45.0), s.elbow_flex


# --------------------------------------------------------------------------- #
# Aspect-ratio correction: the bug that silently skews every angle
# --------------------------------------------------------------------------- #
def test_angles_are_aspect_ratio_corrected():
    """A true 45 deg pixel-space angle must read 45 deg on a 16:9 frame.

    If we forgot to rescale normalized coords by (w, h), this comes out ~28 deg.
    """
    s = state_for(make_pose((600, 300), (800, 500), (900, 600)))
    assert close(s.shoulder_lift, 45.0, tol=0.5), s.shoulder_lift

    # And the same geometry measured with image_size=(1,1) SHOULD be wrong,
    # which is what proves the correction is doing real work.
    skewed = compute_arm_state(
        make_pose((600, 300), (800, 500), (900, 600)),
        None,
        side="right",
        image_size=(1, 1),
    )
    assert not close(skewed.shoulder_lift, 45.0, tol=2.0), (
        "expected uncorrected normalized coords to be skewed, but they matched"
    )


# --------------------------------------------------------------------------- #
# 3. wrist_flex
# --------------------------------------------------------------------------- #
def test_wrist_flex_in_line_with_forearm_is_180():
    # forearm runs elbow(600,300) -> wrist(600,500); hand continues to (600,600)
    pose = make_pose((600, 100), (600, 300), (600, 500))
    hand = make_hand(
        wrist=(600, 500), middle_mcp=(600, 600), thumb_tip=(560, 640), index_tip=(640, 660)
    )
    s = state_for(pose, hand)
    assert close(s.wrist_flex, 180.0), s.wrist_flex


def test_wrist_flex_bent_90():
    # forearm points down; hand turns off horizontally
    pose = make_pose((600, 100), (600, 300), (600, 500))
    hand = make_hand(
        wrist=(600, 500), middle_mcp=(720, 500), thumb_tip=(760, 470), index_tip=(780, 520)
    )
    s = state_for(pose, hand)
    assert close(s.wrist_flex, 90.0), s.wrist_flex


def test_wrist_flex_bent_135():
    pose = make_pose((600, 100), (600, 300), (600, 500))
    d = 100 / math.sqrt(2)
    hand = make_hand(
        wrist=(600, 500),
        middle_mcp=(600 + d, 500 + d),  # 45 deg off the forearm axis
        thumb_tip=(700, 600),
        index_tip=(720, 620),
    )
    s = state_for(pose, hand)
    assert close(s.wrist_flex, 135.0), s.wrist_flex


def test_wrist_flex_needs_the_arm():
    """No pose -> no forearm vector -> wrist_flex must be None, not a guess."""
    hand = make_hand((600, 500), (600, 600), (560, 640), (640, 660))
    s = state_for(None, hand)
    assert s.wrist_flex is None
    assert s.gripper is not None  # but the gripper only needs the hand


# --------------------------------------------------------------------------- #
# 4. gripper
# --------------------------------------------------------------------------- #
def test_gripper_pinched_is_0():
    # thumb and index tips touching -> ratio ~0 -> clamps to 0.0
    hand = make_hand(
        wrist=(600, 500), middle_mcp=(600, 400), thumb_tip=(650, 380), index_tip=(651, 381)
    )
    s = state_for(None, hand)
    assert s.gripper == 0.0, s.gripper


def test_gripper_wide_open_is_1():
    # tips separated by well over the palm length (palm here = 100 px)
    hand = make_hand(
        wrist=(600, 500), middle_mcp=(600, 400), thumb_tip=(500, 380), index_tip=(700, 380)
    )
    s = state_for(None, hand)
    assert s.gripper == 1.0, s.gripper


def test_gripper_is_scale_invariant():
    """Moving the same hand closer to the camera must not change the reading.

    This is the whole point of dividing by palm length.
    """
    small = make_hand((600, 500), (600, 450), (620, 430), (655, 430))
    # exactly 2x the size about the wrist
    big = make_hand((600, 500), (600, 400), (640, 360), (710, 360))
    a = state_for(None, small).gripper
    b = state_for(None, big).gripper
    assert a is not None and b is not None
    assert abs(a - b) < 1e-6, (a, b)


def test_gripper_thresholds_are_configurable():
    hand = make_hand(
        wrist=(600, 500), middle_mcp=(600, 400), thumb_tip=(600, 380), index_tip=(650, 380)
    )
    ratio = state_for(None, hand).gripper_raw_ratio
    assert close(ratio, 0.5, tol=0.01), ratio

    # A config whose midpoint sits exactly at this ratio must read 0.5.
    tight = GripperConfig(closed_ratio=0.0, open_ratio=1.0)
    assert close(state_for(None, hand, gcfg=tight).gripper, 0.5, tol=0.01)

    # Narrow the band so this same ratio now saturates open.
    loose = GripperConfig(closed_ratio=0.1, open_ratio=0.4)
    assert state_for(None, hand, gcfg=loose).gripper == 1.0


# --------------------------------------------------------------------------- #
# Confidence / graceful degradation
# --------------------------------------------------------------------------- #
def test_low_visibility_marks_arm_untracked():
    s = state_for(make_pose((600, 200), (600, 400), (600, 600), visibility=0.2))
    assert s.arm_visible is False
    assert close(s.arm_confidence, 0.2, tol=0.01)
    # values are still computed; the caller decides whether to trust them
    assert s.elbow_flex is not None


def test_arm_confidence_is_the_weakest_joint():
    pose = make_pose((600, 200), (600, 400), (600, 600))
    pose[PoseIdx.RIGHT_WRIST].visibility = 0.3  # one bad joint poisons the chain
    s = state_for(pose)
    assert close(s.arm_confidence, 0.3, tol=0.01), s.arm_confidence
    assert s.arm_visible is False


def test_no_landmarks_yields_all_none():
    s = state_for(None, None)
    assert s.shoulder_lift is None and s.elbow_flex is None
    assert s.wrist_flex is None and s.gripper is None
    assert s.arm_visible is False and s.hand_visible is False
    assert s.ok is False


def test_left_side_is_tracked_independently():
    """Filling only the LEFT arm must produce values for side='left' and none for right."""
    # Collinear down-and-out: shoulder -> elbow -> wrist all on one line, so the
    # arm is genuinely straight (180 deg) and lifted 45 deg away from vertical.
    pose = make_pose((600, 200), (400, 400), (200, 600), side="left")
    left = state_for(pose, side="left")
    assert close(left.elbow_flex, 180.0), left.elbow_flex
    assert close(left.shoulder_lift, 45.0), left.shoulder_lift
    # the right-arm slots were never filled, so they sit at (0,0) with visibility 0
    right = state_for(pose, side="right")
    assert right.arm_visible is False


def test_state_supports_dict_access_and_as_dict():
    s = state_for(make_pose((600, 200), (600, 400), (800, 400)))
    assert close(s["elbow_flex"], 90.0)
    d = s.as_dict()
    assert set(("shoulder_lift", "elbow_flex", "wrist_flex", "gripper")) <= set(d)


# --------------------------------------------------------------------------- #
def _main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
