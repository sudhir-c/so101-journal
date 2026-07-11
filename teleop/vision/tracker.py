"""MediaPipe wrapper: frame in, (pose landmarks, hand landmarks, ArmState) out.

API note (this bit its way into the design)
-------------------------------------------
MediaPipe 0.10.35 -- the version pinned in this project's venv -- has **removed**
the legacy ``mp.solutions`` namespace entirely.  ``mp.solutions.pose``,
``mp.solutions.hands`` and ``mp.solutions.holistic`` no longer exist; the module
top level is just ``Image``, ``ImageFormat`` and ``tasks``.  Everything here
therefore uses the current **Tasks API** (``PoseLandmarker`` / ``HandLandmarker``),
which also means the model weights are *not* shipped inside the wheel and must be
downloaded as ``.task`` bundles -- see ``models/`` and the README.

We run Pose and Hands as two separate landmarkers rather than the Tasks
``HolisticLandmarker``, because that keeps independent confidence signals and
independent tuning for each, which we want.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from .pose_math import (
    ArmState,
    GripperConfig,
    HandIdx,
    PoseIdx,
    QualityConfig,
    compute_arm_state,
)
from .smoothing import ArmStateSmoother

__all__ = ["TrackerConfig", "ArmTracker", "TrackResult"]

_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


@dataclass
class TrackerConfig:
    """Tuning for the tracker. Model paths default to the repo's ``models/`` dir."""

    side: str = "right"  # which of *your* arms to track (anatomical)
    mirror: bool = True  # selfie view: flip frame horizontally

    pose_model: Path = _MODEL_DIR / "pose_landmarker_full.task"
    hand_model: Path = _MODEL_DIR / "hand_landmarker.task"

    # Detection thresholds handed to MediaPipe.
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_pose_tracking_confidence: float = 0.5
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_hand_tracking_confidence: float = 0.5

    gripper: GripperConfig = None  # type: ignore[assignment]
    quality: QualityConfig = None  # type: ignore[assignment]
    smooth: bool = True

    def __post_init__(self) -> None:
        if self.gripper is None:
            self.gripper = GripperConfig()
        if self.quality is None:
            self.quality = QualityConfig()
        self.pose_model = Path(self.pose_model)
        self.hand_model = Path(self.hand_model)


@dataclass
class TrackResult:
    """Everything one processed frame produced."""

    state: ArmState
    pose_landmarks: Sequence[Any] | None  # full 33-point list, or None
    hand_landmarks: Sequence[Any] | None  # the 21-point list for the tracked hand
    mp_side: str  # the side we actually asked MediaPipe for (mirror-resolved)


def _opposite(side: str) -> str:
    return "left" if side == "right" else "right"


class ArmTracker:
    """Stateful, single-threaded MediaPipe tracker.

    MediaPipe landmarker objects are **not thread-safe** and ``VIDEO`` running
    mode requires monotonically increasing timestamps, so exactly one thread must
    own an instance and call :meth:`process` in frame order.
    """

    def __init__(self, config: TrackerConfig | None = None):
        self.config = config or TrackerConfig()
        for p in (self.config.pose_model, self.config.hand_model):
            if not p.exists():
                raise FileNotFoundError(
                    f"MediaPipe model bundle not found: {p}\n"
                    "The Tasks API does not ship weights in the wheel. "
                    "Run scripts/download_models.sh to fetch them."
                )

        self._pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.config.pose_model)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=self.config.min_pose_detection_confidence,
                min_pose_presence_confidence=self.config.min_pose_presence_confidence,
                min_tracking_confidence=self.config.min_pose_tracking_confidence,
                output_segmentation_masks=False,
            )
        )
        # Detect both hands: we choose between them by proximity to the tracked
        # wrist (see _pick_hand), which is sturdier than trusting handedness.
        self._hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.config.hand_model)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=self.config.min_hand_detection_confidence,
                min_hand_presence_confidence=self.config.min_hand_presence_confidence,
                min_tracking_confidence=self.config.min_hand_tracking_confidence,
            )
        )
        self._smoother = ArmStateSmoother() if self.config.smooth else None

    # ------------------------------------------------------------------ #
    @property
    def mp_side(self) -> str:
        """The side to ask MediaPipe for, accounting for the mirror flip.

        MediaPipe labels landmarks by the *anatomy it sees*.  When we horizontally
        flip the frame for a natural selfie view, the person in the image is a
        mirror image of the real you -- so the arm that is really your right arm
        now looks, to the model, like a left arm.  Tracking your true right arm
        under mirroring therefore means reading MediaPipe's LEFT indices.
        """
        if self.config.mirror:
            return _opposite(self.config.side)
        return self.config.side

    # ------------------------------------------------------------------ #
    def _pick_hand(
        self,
        hand_result: Any,
        pose_landmarks: Sequence[Any] | None,
    ) -> tuple[Sequence[Any] | None, float]:
        """Choose which detected hand belongs to the arm we're tracking.

        Rather than trusting MediaPipe's handedness label (which flips under
        mirroring and misfires when the hand is rotated), we take the hand whose
        wrist landmark sits closest to the *pose* wrist landmark. Geometry beats
        classification here.

        Returns ``(landmarks, confidence)``.
        """
        hands = getattr(hand_result, "hand_landmarks", None) or []
        if not hands:
            return None, 0.0

        handedness = getattr(hand_result, "handedness", None) or []

        def score_of(i: int) -> float:
            try:
                return float(handedness[i][0].score)
            except (IndexError, AttributeError, TypeError):
                return 1.0

        # Without a pose wrist to compare against, fall back to the most confident.
        _, _, w_i = PoseIdx.for_side(self.mp_side)
        pose_wrist = None
        if pose_landmarks is not None and w_i < len(pose_landmarks):
            lm = pose_landmarks[w_i]
            pose_wrist = np.array([lm.x, lm.y], dtype=float)

        if pose_wrist is None:
            best = max(range(len(hands)), key=score_of)
            return hands[best], score_of(best)

        def dist_to_pose_wrist(i: int) -> float:
            hw = hands[i][HandIdx.WRIST]
            return float(np.hypot(hw.x - pose_wrist[0], hw.y - pose_wrist[1]))

        best = min(range(len(hands)), key=dist_to_pose_wrist)

        # If the nearest hand is still miles from the arm's wrist, it is somebody
        # else's hand (or the other hand) -- better to report nothing than to bolt
        # the wrong hand onto the arm. 0.25 is in normalized units, i.e. a quarter
        # of the frame, which is very generous but still rules out the far hand.
        if dist_to_pose_wrist(best) > 0.25:
            return None, 0.0

        return hands[best], score_of(best)

    # ------------------------------------------------------------------ #
    def process(self, rgb_frame: np.ndarray, timestamp_ms: int) -> TrackResult:
        """Run pose + hand inference on one RGB frame.

        ``rgb_frame`` must be RGB (not BGR) and already mirrored if mirroring is
        on -- the capture layer owns that flip so that what we analyze is exactly
        what gets drawn and shown.

        ``timestamp_ms`` must increase monotonically across calls (VIDEO mode).
        """
        h, w = rgb_frame.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        pose_result = self._pose.detect_for_video(mp_image, timestamp_ms)
        hand_result = self._hands.detect_for_video(mp_image, timestamp_ms)

        pose_landmarks = None
        if pose_result.pose_landmarks:
            pose_landmarks = pose_result.pose_landmarks[0]

        hand_landmarks, hand_conf = self._pick_hand(hand_result, pose_landmarks)

        state = compute_arm_state(
            pose_landmarks,
            hand_landmarks,
            side=self.mp_side,
            image_size=(w, h),
            gripper_config=self.config.gripper,
            quality_config=self.config.quality,
            hand_confidence=hand_conf if hand_landmarks is not None else None,
        )
        # Report the side the *operator* asked for, not the mirror-resolved one.
        state.side = self.config.side

        if self._smoother is not None:
            state = self._smoother(state, timestamp_ms / 1000.0)

        return TrackResult(
            state=state,
            pose_landmarks=pose_landmarks,
            hand_landmarks=hand_landmarks,
            mp_side=self.mp_side,
        )

    def close(self) -> None:
        self._pose.close()
        self._hands.close()

    def __enter__(self) -> "ArmTracker":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
