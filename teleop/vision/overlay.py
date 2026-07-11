"""OpenCV overlay: draws the tracked arm, the hand, the four values, and quality.

Pure drawing -- takes landmarks + an ArmState and paints on a BGR frame.  Nothing
here computes anything, so the math stays in one place.
"""

from __future__ import annotations

import cv2
import numpy as np

from .pose_math import ArmState, HandIdx, PoseIdx

__all__ = ["draw_overlay"]

# BGR
_GREEN = (80, 220, 120)
_AMBER = (60, 190, 250)
_RED = (70, 70, 240)
_WHITE = (245, 245, 245)
_DIM = (170, 170, 170)
_CYAN = (220, 200, 90)
_MAGENTA = (200, 110, 240)
_SHADOW = (25, 25, 25)

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# The hand skeleton, as index pairs into the 21-point topology.
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (5, 9), (9, 10), (10, 11), (11, 12),      # middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (0, 17),                                  # palm base
]


def _px(lm, w: int, h: int) -> tuple[int, int]:
    return int(lm.x * w), int(lm.y * h)


def _text(img, s, org, scale=0.6, color=_WHITE, thick=1):
    """Text with a thin dark outline so it stays readable over any background.

    The outline is drawn concentrically (same origin, heavier stroke) rather than
    offset -- an offset shadow reads as ghosted double-vision at these sizes.
    """
    cv2.putText(img, s, org, _FONT, scale, _SHADOW, thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, _FONT, scale, color, thick, cv2.LINE_AA)


def _fmt_deg(v: float | None) -> str:
    return "--" if v is None else f"{v:5.1f}deg"


def _fmt_grip(v: float | None) -> str:
    return "--" if v is None else f"{v:4.2f}"


def _draw_arm(img, pose_landmarks, mp_side: str, ok: bool) -> None:
    h, w = img.shape[:2]
    s_i, e_i, w_i = PoseIdx.for_side(mp_side)
    try:
        shoulder = _px(pose_landmarks[s_i], w, h)
        elbow = _px(pose_landmarks[e_i], w, h)
        wrist = _px(pose_landmarks[w_i], w, h)
    except (IndexError, AttributeError):
        return

    color = _GREEN if ok else _AMBER
    for a, b in ((shoulder, elbow), (elbow, wrist)):
        cv2.line(img, a, b, _SHADOW, 9, cv2.LINE_AA)
        cv2.line(img, a, b, color, 4, cv2.LINE_AA)

    for pt, label in ((shoulder, "shoulder"), (elbow, "elbow"), (wrist, "wrist")):
        cv2.circle(img, pt, 10, _SHADOW, -1, cv2.LINE_AA)
        cv2.circle(img, pt, 8, color, -1, cv2.LINE_AA)
        cv2.circle(img, pt, 8, _WHITE, 1, cv2.LINE_AA)
        _text(img, label, (pt[0] + 13, pt[1] + 5), 0.45, _DIM, 1)


def _draw_hand(img, hand_landmarks, gripper: float | None) -> None:
    h, w = img.shape[:2]
    pts = [_px(lm, w, h) for lm in hand_landmarks]

    for a, b in _HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], _SHADOW, 5, cv2.LINE_AA)
        cv2.line(img, pts[a], pts[b], _CYAN, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 3, _CYAN, -1, cv2.LINE_AA)

    # Emphasize the two landmarks the gripper is actually derived from, and draw
    # the pinch span between them so the number on screen has a visible cause.
    thumb = pts[HandIdx.THUMB_TIP]
    index = pts[HandIdx.INDEX_FINGER_TIP]

    if gripper is None:
        pinch_color = _AMBER
    else:
        # green when open, red when pinched shut
        pinch_color = (
            int(_RED[0] + (_GREEN[0] - _RED[0]) * gripper),
            int(_RED[1] + (_GREEN[1] - _RED[1]) * gripper),
            int(_RED[2] + (_GREEN[2] - _RED[2]) * gripper),
        )

    cv2.line(img, thumb, index, _SHADOW, 6, cv2.LINE_AA)
    cv2.line(img, thumb, index, pinch_color, 3, cv2.LINE_AA)
    for p, label in ((thumb, "thumb"), (index, "index")):
        cv2.circle(img, p, 11, _SHADOW, -1, cv2.LINE_AA)
        cv2.circle(img, p, 9, pinch_color, -1, cv2.LINE_AA)
        cv2.circle(img, p, 9, _WHITE, 1, cv2.LINE_AA)
        _text(img, label, (p[0] + 13, p[1] + 5), 0.45, _WHITE, 1)

    # Wrist -> middle knuckle: the hand-direction vector behind wrist_flex, and
    # the scale reference behind the gripper ratio.
    cv2.line(img, pts[HandIdx.WRIST], pts[HandIdx.MIDDLE_FINGER_MCP],
             _MAGENTA, 2, cv2.LINE_AA)


def _draw_panel(img, state: ArmState, phase: int, fps: float, mp_side: str) -> None:
    """Value readouts + tracking-quality banner, top-left."""
    h, w = img.shape[:2]

    rows: list[tuple[str, str, bool]] = []
    if phase >= 3:
        rows.append(("shoulder_lift", _fmt_deg(state.shoulder_lift), state.shoulder_lift is not None))
        rows.append(("elbow_flex", _fmt_deg(state.elbow_flex), state.elbow_flex is not None))
    if phase >= 4:
        rows.append(("wrist_flex", _fmt_deg(state.wrist_flex), state.wrist_flex is not None))
        rows.append(("gripper", _fmt_grip(state.gripper), state.gripper is not None))

    # Warnings that need their own line at the bottom of the panel.
    warnings: list[str] = []
    if phase >= 2 and not state.arm_visible:
        warnings.append("arm not fully in frame")
    if phase >= 4 and not state.hand_visible:
        warnings.append("hand not detected")

    pad = 14
    line_h = 30
    warn_h = 20
    # Header + quality line + one row each + a line per warning. Sizing the panel
    # from its contents is what stops the warning colliding with the last row.
    panel_h = 74 + line_h * len(rows) + warn_h * len(warnings) + 10
    panel_w = 340

    panel = img[0:panel_h, 0:panel_w].copy()
    panel[:] = (30, 28, 26)
    cv2.addWeighted(panel, 0.62, img[0:panel_h, 0:panel_w], 0.38, 0,
                    img[0:panel_h, 0:panel_w])

    y = pad + 14
    _text(img, f"phase {phase}  |  {state.side.upper()} arm  |  {fps:4.1f} fps",
          (pad, y), 0.5, _DIM, 1)

    # --- tracking quality: the "is this trustworthy" line ------------------ #
    y += 27
    if phase >= 2:
        if phase >= 4:
            good = state.arm_visible and state.hand_visible
            partial = state.arm_visible or state.hand_visible
        else:
            good = state.arm_visible
            partial = state.arm_visible

        if good:
            q_color, q_text = _GREEN, "TRACKING OK"
        elif partial:
            q_color, q_text = _AMBER, "PARTIAL"
        else:
            q_color, q_text = _RED, "NO TRACKING"

        cv2.circle(img, (pad + 6, y - 5), 6, q_color, -1, cv2.LINE_AA)
        _text(img, q_text, (pad + 20, y), 0.56, q_color, 1)

        detail = f"arm {state.arm_confidence:.2f}"
        if phase >= 4:
            detail += f"  hand {state.hand_confidence:.2f}"
        _text(img, detail, (pad + 165, y), 0.45, _DIM, 1)

    # --- the four values ---------------------------------------------------- #
    y += 10
    for name, value, present in rows:
        y += line_h
        color = _WHITE if present else _RED
        _text(img, f"{name}", (pad, y), 0.55, _DIM, 1)
        _text(img, value, (pad + 195, y), 0.62, color, 2)

    for msg in warnings:
        y += warn_h
        _text(img, msg, (pad, y), 0.44, _AMBER, 1)


def draw_overlay(
    frame_bgr: np.ndarray,
    result,  # TrackResult | None
    phase: int = 4,
    fps: float = 0.0,
) -> np.ndarray:
    """Annotate ``frame_bgr`` in place and return it.

    ``phase`` mirrors the build phases so each stage can be verified on its own:
      1 raw video, 2 + arm skeleton, 3 + shoulder_lift/elbow_flex,
      4 + hand, wrist_flex and gripper.
    """
    if phase <= 1 or result is None:
        if phase <= 1:
            h = frame_bgr.shape[0]
            _text(frame_bgr, "phase 1: raw camera feed", (14, h - 18), 0.6, _WHITE, 2)
        return frame_bgr

    state: ArmState = result.state

    if phase >= 2 and result.pose_landmarks is not None:
        _draw_arm(frame_bgr, result.pose_landmarks, result.mp_side, state.arm_visible)

    if phase >= 4 and result.hand_landmarks is not None:
        _draw_hand(frame_bgr, result.hand_landmarks, state.gripper)

    _draw_panel(frame_bgr, state, phase, fps, result.mp_side)
    return frame_bgr
