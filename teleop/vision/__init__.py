"""Arm + hand pose estimation from a single webcam, in 2D image-plane space.

The reusable core is :func:`teleop.vision.pose_math.compute_arm_state`, which has
no dependency on OpenCV, MediaPipe or the web server -- import it directly:

    from teleop.vision import compute_arm_state
"""

from .pose_math import (
    ArmState,
    GripperConfig,
    HandIdx,
    PoseIdx,
    QualityConfig,
    angle_between,
    compute_arm_state,
    vector_angle_from_down,
)
from .smoothing import ArmStateSmoother, OneEuroFilter

__all__ = [
    "compute_arm_state",
    "ArmState",
    "GripperConfig",
    "QualityConfig",
    "PoseIdx",
    "HandIdx",
    "angle_between",
    "vector_angle_from_down",
    "ArmStateSmoother",
    "OneEuroFilter",
]

__version__ = "0.1.0"
