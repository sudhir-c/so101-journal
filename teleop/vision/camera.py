"""Webcam capture and device discovery (macOS / AVFoundation friendly).

Kept separate from the tracker and the server so the vision math never has to
care where pixels came from.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["CameraInfo", "list_cameras", "format_camera_table", "Camera"]

_IS_MAC = platform.system() == "Darwin"

# ---------------------------------------------------------------------------- #
# CONFIG: which camera to use.
#
# This machine exposes three devices (built-in camera, an OBS virtual camera, and
# an iPhone Continuity camera).  Index 1 was verified to be the built-in webcam
# pointed at the operator.  Do NOT treat that as gospel: macOS reshuffles capture
# indices as virtual cameras register and Continuity Camera connects/disconnects,
# and the built-in is not always 0.
#
# Run `--list-cameras` to see which indices deliver frames, or `--snapshot` to
# dump one JPEG per camera and simply look at which one is you.
# ---------------------------------------------------------------------------- #
DEFAULT_CAMERA_INDEX = 1


@dataclass
class CameraInfo:
    index: int
    width: int
    height: int
    works: bool  # opened AND actually delivered a frame
    blank: bool = False  # delivered frames, but they carry no image (see below)


def macos_camera_names() -> list[str]:
    """Camera device names known to macOS.

    Deliberately returned as an *unordered set of names*, *not* zipped against
    OpenCV indices.  ``system_profiler``'s enumeration order does not reliably
    match AVFoundation's capture-index order -- observed directly on this machine,
    where the orders disagreed between runs.  Pinning a name to an index would
    confidently point you at the wrong camera, which is worse than staying quiet.
    Use ``snapshot_cameras()`` to find out which index is actually you.
    """
    if not _IS_MAC:
        return []
    try:
        out = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []

    names: list[str] = []
    for raw in out.splitlines():
        line = raw.rstrip()
        # Device names are indented exactly 4 spaces and end with ':'
        if line.endswith(":") and len(line) - len(line.lstrip()) == 4:
            name = line.strip().rstrip(":")
            if name and name != "Camera":
                names.append(name)
    return names


def _is_blank(frame: np.ndarray) -> bool:
    """Is this frame effectively empty?

    An idle OBS Virtual Camera happily *delivers* frames -- they are just solid
    black.  So "did we get a frame?" is not enough to identify a usable camera;
    without this check the picker cheerfully recommends the virtual cam and you
    stare at a black stream wondering why nothing tracks.  Near-zero variance and
    near-zero brightness means there is no image here.
    """
    return bool(frame.std() < 3.0 and frame.mean() < 12.0)


def list_cameras(max_index: int = 5) -> list[CameraInfo]:
    """Probe camera indices and report which ones actually deliver a real image."""
    backend = cv2.CAP_AVFOUNDATION if _IS_MAC else cv2.CAP_ANY

    found: list[CameraInfo] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        if not cap.isOpened():
            cap.release()
            continue
        # Let auto-exposure settle; a real camera's first frame is often black too.
        frame = None
        for _ in range(5):
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
        cap.release()

        works = frame is not None
        h, w = (frame.shape[:2] if works else (0, 0))
        found.append(
            CameraInfo(
                index=i,
                width=w,
                height=h,
                works=works,
                blank=works and _is_blank(frame),
            )
        )
    return found


def format_camera_table(cams: list[CameraInfo]) -> str:
    """Human-readable listing for startup / --list-cameras."""
    if not cams:
        return "  (no cameras found)"
    lines = []
    for c in cams:
        if not c.works:
            status = "opens but NO FRAMES"
        elif c.blank:
            status = "delivers only BLACK frames  <- virtual camera, not you"
        else:
            status = f"live image  ({c.width}x{c.height})"
        lines.append(f"  --camera {c.index}    {status}")

    names = macos_camera_names()
    if names:
        lines.append("")
        lines.append("  Cameras macOS knows about (NOT necessarily in index order):")
        for n in names:
            lines.append(f"    - {n}")
    return "\n".join(lines)


def snapshot_cameras(out_dir: str = "/tmp", max_index: int = 5) -> list[str]:
    """Grab one frame from every working camera so you can *see* which is which.

    This is the reliable way to answer "which index is pointed at me?", given the
    index<->name ordering cannot be trusted.
    """
    import os

    backend = cv2.CAP_AVFOUNDATION if _IS_MAC else cv2.CAP_ANY
    written: list[str] = []
    for cam in list_cameras(max_index):
        if not cam.works:
            continue
        cap = cv2.VideoCapture(cam.index, backend)
        # Discard a few frames so auto-exposure settles before we save one.
        frame = None
        for _ in range(5):
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
        cap.release()
        if frame is None:
            continue
        path = os.path.join(out_dir, f"camera_{cam.index}.jpg")
        cv2.imwrite(path, frame)
        written.append(path)
    return written


class Camera:
    """Thin, resilient webcam reader.

    Owns the horizontal flip for selfie view, so that the frame we analyze is
    byte-for-byte the frame we display -- otherwise the drawn skeleton would not
    line up with the person.
    """

    def __init__(
        self,
        index: int = DEFAULT_CAMERA_INDEX,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        mirror: bool = True,
    ):
        self.index = index
        self.mirror = mirror

        backend = cv2.CAP_AVFOUNDATION if _IS_MAC else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(index, backend)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {index}.\n"
                f"Available cameras:\n{format_camera_table(list_cameras())}"
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        # Keep latency down: we always want the freshest frame, never a backlog.
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass  # not supported on every backend; harmless

        frame = None
        for _ in range(5):  # let auto-exposure settle before judging the image
            ok, f = self._cap.read()
            if ok and f is not None:
                frame = f
        if frame is None:
            self._cap.release()
            raise RuntimeError(
                f"Camera {index} opened but returned no frames. On macOS this is "
                f"usually an inactive virtual camera (e.g. OBS), or Terminal lacks "
                f"camera permission (System Settings > Privacy & Security > Camera).\n"
                f"Available cameras:\n{format_camera_table(list_cameras())}"
            )

        if _is_blank(frame):
            print(
                f"\n  WARNING: camera {index} is delivering solid black frames -- this "
                f"is almost certainly a virtual camera (e.g. OBS), not your webcam.\n"
                f"  Nothing will track. Run `--snapshot` to find the index that is you.\n"
            )

        self.height, self.width = frame.shape[:2]

    def read(self) -> np.ndarray | None:
        """Return the next BGR frame (already mirrored if enabled), or None."""
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        if self.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
