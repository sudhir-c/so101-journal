"""Arm + hand joint-angle math, in 2D image-plane pixel space.

This module is the reusable core of the project and is deliberately free of any
video, MediaPipe, or serving dependencies -- it needs only ``numpy`` and the
stdlib.  A later project can ``from arm_pose.pose_math import compute_arm_state``
and feed it landmarks from any source.

Design notes
------------
* **Everything is 2D.**  We use only ``(x, y)`` and ignore MediaPipe's ``z``:
  monocular depth is unreliable, and the operator deliberately holds the arm
  parallel to the image plane so the in-plane projection carries the signal.

* **Image y grows downward.**  Angles are computed with that convention baked in,
  so "straight down" is ``(0, +1)``.

* **Aspect-ratio correction matters.**  MediaPipe returns coordinates normalized
  independently by width and height.  On a 16:9 frame those axes have different
  pixel scales, so an angle measured directly from normalized coordinates is
  *skewed*.  Callers pass ``image_size=(w, h)`` and we scale back into true
  pixels before measuring any angle.  Pass ``image_size=(1, 1)`` if your
  landmarks are already in pixel units.

* **No wrist_roll.**  Forearm rotation about its own axis is essentially
  unobservable from a single in-plane camera -- rolling the forearm barely moves
  the 2D landmarks -- so any value we produced would be noise.  It is left out on
  purpose.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

__all__ = [
    "PoseIdx",
    "HandIdx",
    "GripperConfig",
    "QualityConfig",
    "ArmState",
    "compute_arm_state",
    "angle_between",
    "vector_angle_from_down",
]


# --------------------------------------------------------------------------- #
# Landmark indices
# --------------------------------------------------------------------------- #
class PoseIdx:
    """MediaPipe Pose landmark indices (33-point topology)."""

    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16

    @classmethod
    def for_side(cls, side: str) -> tuple[int, int, int]:
        """Return ``(shoulder, elbow, wrist)`` indices for ``side``."""
        s = _norm_side(side)
        if s == "left":
            return cls.LEFT_SHOULDER, cls.LEFT_ELBOW, cls.LEFT_WRIST
        return cls.RIGHT_SHOULDER, cls.RIGHT_ELBOW, cls.RIGHT_WRIST


class HandIdx:
    """MediaPipe Hand landmark indices (21-point topology)."""

    WRIST = 0
    THUMB_TIP = 4
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9  # middle knuckle -- our hand-direction + scale reference


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GripperConfig:
    """Thresholds mapping a normalized pinch distance onto a 0..1 openness.

    The raw measurement is::

        ratio = |thumb_tip - index_tip| / |hand_wrist - middle_mcp|

    Dividing by the palm length (wrist -> middle knuckle) makes the value scale
    invariant, so moving your hand toward or away from the camera does not change
    the reading.

    ``closed_ratio`` and ``open_ratio`` are the tuning knobs.  Measure your own
    hand with the on-screen ``raw`` readout and adjust if your pinch never quite
    reaches 0.0 or your spread never quite reaches 1.0.
    """

    closed_ratio: float = 0.15  # ratio at/below this -> gripper 0.0 (pinched shut)
    open_ratio: float = 1.10  # ratio at/above this -> gripper 1.0 (wide open)

    def to_openness(self, ratio: float) -> float:
        """Map a raw pinch ratio onto 0..1, clamped."""
        span = self.open_ratio - self.closed_ratio
        if span <= 0:  # misconfigured; degrade gracefully rather than divide by ~0
            return 0.0
        return float(np.clip((ratio - self.closed_ratio) / span, 0.0, 1.0))


@dataclass(frozen=True)
class QualityConfig:
    """Confidence floors below which we declare tracking unreliable."""

    min_arm_visibility: float = 0.6
    min_hand_confidence: float = 0.5


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class ArmState:
    """The four computed values plus the confidence needed to trust them.

    The four headline values are ``shoulder_lift``, ``elbow_flex``, ``wrist_flex``
    (all degrees) and ``gripper`` (0..1).  Any of them is ``None`` when the
    landmarks it needs were not available, so a consumer can always distinguish
    "not tracked" from a real zero.
    """

    # --- the four values -------------------------------------------------- #
    shoulder_lift: float | None = None  # deg: hanging down 0, horizontal 90, up 180
    elbow_flex: float | None = None  # deg: straight 180, right angle 90
    wrist_flex: float | None = None  # deg: in line with forearm 180, bent 90
    gripper: float | None = None  # 0..1: pinched 0, wide open 1

    # --- tracking quality -------------------------------------------------- #
    arm_visible: bool = False
    hand_visible: bool = False
    arm_confidence: float = 0.0  # min visibility across shoulder/elbow/wrist
    hand_confidence: float = 0.0
    landmark_visibility: dict[str, float] = field(default_factory=dict)

    # --- diagnostics ------------------------------------------------------- #
    gripper_raw_ratio: float | None = None  # pre-threshold pinch ratio
    side: str = "right"

    @property
    def ok(self) -> bool:
        """True only when both arm and hand are confidently tracked."""
        return self.arm_visible and self.hand_visible

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access: ``state["elbow_flex"]``."""
        return getattr(self, key)


# --------------------------------------------------------------------------- #
# Geometry primitives
# --------------------------------------------------------------------------- #
def angle_between(v1: np.ndarray, v2: np.ndarray) -> float | None:
    """Unsigned angle between two 2D vectors, in degrees, on [0, 180].

    Uses ``atan2(|cross|, dot)`` rather than ``acos(dot / |a||b|)``: the acos form
    loses precision and can blow past its domain near 0 deg and 180 deg -- exactly
    the straight-arm case we care most about reading correctly.
    """
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < _EPS or n2 < _EPS:
        return None
    cross = float(v1[0] * v2[1] - v1[1] * v2[0])
    dot = float(v1[0] * v2[0] + v1[1] * v2[1])
    return math.degrees(math.atan2(abs(cross), dot))


def vector_angle_from_down(v: np.ndarray) -> float | None:
    """Angle of ``v`` away from straight-down, in degrees, on [0, 180].

    Image y grows downward, so "down" is ``(0, +1)``.  Unsigned, so it does not
    matter whether the arm swings out to the left or the right of the body.
    """
    return angle_between(v, _DOWN)


_EPS = 1e-6
_DOWN = np.array([0.0, 1.0])


# --------------------------------------------------------------------------- #
# Landmark access helpers (duck-typed, so we don't depend on MediaPipe types)
# --------------------------------------------------------------------------- #
def _norm_side(side: str) -> str:
    s = str(side).strip().lower()
    if s not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    return s


def _xy(landmark: Any, scale: np.ndarray) -> np.ndarray:
    """Pull ``(x, y)`` off a landmark and scale normalized -> pixel units.

    Accepts anything with ``.x``/``.y`` (MediaPipe) or an indexable pair.
    """
    if hasattr(landmark, "x") and hasattr(landmark, "y"):
        x, y = landmark.x, landmark.y
    else:
        x, y = landmark[0], landmark[1]
    return np.array([float(x) * scale[0], float(y) * scale[1]], dtype=float)


def _visibility(landmark: Any) -> float:
    """Landmark visibility on 0..1; 1.0 when the field is absent or unset.

    MediaPipe populates ``visibility`` for *pose* landmarks but leaves it unset
    (``None``) for *hand* landmarks, so absence must not be read as "invisible".
    """
    v = getattr(landmark, "visibility", None)
    if v is None:
        return 1.0
    return float(v)


def _get(landmarks: Sequence[Any] | None, idx: int) -> Any | None:
    if landmarks is None:
        return None
    try:
        if idx >= len(landmarks):
            return None
        return landmarks[idx]
    except TypeError:
        return None


# --------------------------------------------------------------------------- #
# The public entry point
# --------------------------------------------------------------------------- #
def compute_arm_state(
    pose_landmarks: Sequence[Any] | None,
    hand_landmarks: Sequence[Any] | None,
    side: str = "right",
    image_size: tuple[int, int] = (1, 1),
    gripper_config: GripperConfig | None = None,
    quality_config: QualityConfig | None = None,
    hand_confidence: float | None = None,
) -> ArmState:
    """Compute arm + gripper joint values from 2D landmarks.

    Parameters
    ----------
    pose_landmarks:
        MediaPipe Pose landmark list (33 entries) for one person, or ``None``.
    hand_landmarks:
        MediaPipe Hand landmark list (21 entries) for the hand belonging to the
        tracked arm, or ``None``.
    side:
        Which of the operator's arms to track: ``"left"`` or ``"right"``.  This is
        an *anatomical* side and must already account for any mirroring of the
        frame -- see ``arm_pose.tracker``, which resolves that before calling us.
    image_size:
        ``(width, height)`` in pixels, used to undo MediaPipe's per-axis
        normalization so angles are measured in true pixel space.  Pass
        ``(1, 1)`` if the landmarks are already in pixels.
    gripper_config, quality_config:
        Tuning knobs; sensible defaults are used when omitted.
    hand_confidence:
        Detector-supplied confidence for the hand (MediaPipe does not fill in
        per-landmark visibility for hands).  Defaults to 1.0 when a hand is
        present.

    Returns
    -------
    ArmState
        Dataclass carrying ``shoulder_lift``, ``elbow_flex``, ``wrist_flex``,
        ``gripper`` and the confidence needed to know whether to trust them.
        Supports ``state["elbow_flex"]`` and ``state.as_dict()``.
    """
    side = _norm_side(side)
    gcfg = gripper_config or GripperConfig()
    qcfg = quality_config or QualityConfig()
    scale = np.array([float(image_size[0]), float(image_size[1])], dtype=float)

    state = ArmState(side=side)

    # ---------------- arm: shoulder / elbow / wrist ------------------------ #
    s_i, e_i, w_i = PoseIdx.for_side(side)
    shoulder_lm = _get(pose_landmarks, s_i)
    elbow_lm = _get(pose_landmarks, e_i)
    wrist_lm = _get(pose_landmarks, w_i)

    have_arm = None not in (shoulder_lm, elbow_lm, wrist_lm)

    if have_arm:
        vis = {
            "shoulder": _visibility(shoulder_lm),
            "elbow": _visibility(elbow_lm),
            "wrist": _visibility(wrist_lm),
        }
        state.landmark_visibility = vis
        # The chain is only as trustworthy as its least-visible joint.
        state.arm_confidence = min(vis.values())
        state.arm_visible = state.arm_confidence >= qcfg.min_arm_visibility

        shoulder = _xy(shoulder_lm, scale)
        elbow = _xy(elbow_lm, scale)
        wrist = _xy(wrist_lm, scale)

        # 1. shoulder_lift -- how far the upper arm is raised away from hanging.
        #    U points shoulder -> elbow.  Measured against straight-down, so:
        #    hanging 0 deg, horizontal 90 deg, straight up 180 deg.
        state.shoulder_lift = vector_angle_from_down(elbow - shoulder)

        # 2. elbow_flex -- interior angle at the elbow.  Both vectors emanate
        #    from the joint, so a straight arm reads 180 deg and a right angle 90.
        state.elbow_flex = angle_between(shoulder - elbow, wrist - elbow)

    # ---------------- hand: wrist_flex + gripper --------------------------- #
    hand_wrist_lm = _get(hand_landmarks, HandIdx.WRIST)
    middle_mcp_lm = _get(hand_landmarks, HandIdx.MIDDLE_FINGER_MCP)
    thumb_tip_lm = _get(hand_landmarks, HandIdx.THUMB_TIP)
    index_tip_lm = _get(hand_landmarks, HandIdx.INDEX_FINGER_TIP)

    have_hand = None not in (hand_wrist_lm, middle_mcp_lm, thumb_tip_lm, index_tip_lm)

    if have_hand:
        state.hand_confidence = 1.0 if hand_confidence is None else float(hand_confidence)
        state.hand_visible = state.hand_confidence >= qcfg.min_hand_confidence

        hand_wrist = _xy(hand_wrist_lm, scale)
        middle_mcp = _xy(middle_mcp_lm, scale)
        thumb_tip = _xy(thumb_tip_lm, scale)
        index_tip = _xy(index_tip_lm, scale)

        # 3. wrist_flex -- interior angle at the wrist, same convention as the
        #    elbow: both vectors emanate from the joint, so hand-in-line-with-
        #    forearm reads 180 deg and a 90 deg bend reads 90 deg.
        #
        #    NOTE: we take the *pose* elbow and the *hand* wrist here.  The hand
        #    model gives a far crisper wrist point than the pose model does, and
        #    mixing them is safe because both live in the same normalized frame.
        if have_arm:
            elbow_px = _xy(elbow_lm, scale)
            forearm = elbow_px - hand_wrist  # wrist -> elbow (back up the arm)
            hand_dir = middle_mcp - hand_wrist  # wrist -> knuckles (out the hand)
            state.wrist_flex = angle_between(forearm, hand_dir)

        # 4. gripper -- thumb/index pinch, normalized by palm length so the value
        #    is invariant to how near the hand is to the camera.
        palm_len = float(np.linalg.norm(middle_mcp - hand_wrist))
        if palm_len > _EPS:
            pinch = float(np.linalg.norm(thumb_tip - index_tip))
            ratio = pinch / palm_len
            state.gripper_raw_ratio = ratio
            state.gripper = gcfg.to_openness(ratio)

    return state
