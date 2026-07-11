"""FastAPI server: annotated MJPEG stream + live joint values.

Architecture
------------
A single background worker thread owns the camera *and* the MediaPipe tracker,
and publishes the latest annotated JPEG + latest ArmState into a tiny lock-guarded
slot.  HTTP handlers only ever read that slot.

This matters for two reasons:
  * MediaPipe landmarker objects are not thread-safe and VIDEO running mode needs
    monotonically increasing timestamps -- one owner thread guarantees both.
  * Inference cost stays constant no matter how many browser tabs are watching.

Run:
    .venv/bin/python server.py --list-cameras
    .venv/bin/python server.py --camera 1 --side right
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from arm_pose.camera import (
    DEFAULT_CAMERA_INDEX,
    Camera,
    format_camera_table,
    list_cameras,
    snapshot_cameras,
)
from arm_pose.overlay import draw_overlay
from arm_pose.pose_math import ArmState, GripperConfig, QualityConfig
from arm_pose.tracker import ArmTracker, TrackerConfig

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Gripper thresholds live here (and in arm_pose/pose_math.py:GripperConfig).
# Raise CLOSED if a full pinch never quite reads 0.00; lower OPEN if a wide
# spread never quite reaches 1.00.  The `raw` value on the web UI is the number
# these are compared against, so tune by watching it.
# --------------------------------------------------------------------------- #
GRIPPER_CLOSED_RATIO = 0.15
GRIPPER_OPEN_RATIO = 1.10

# Confidence floors below which the UI shows the tracking warning.
MIN_ARM_VISIBILITY = 0.6
MIN_HAND_CONFIDENCE = 0.5


class Pipeline:
    """Owns the capture+inference thread and the latest published frame/state."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._state: ArmState = ArmState(side=args.side)
        self._fps: float = 0.0
        self._error: str | None = None
        self._frame_event = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        a = self.args
        try:
            camera = Camera(
                index=a.camera,
                width=a.width,
                height=a.height,
                fps=a.fps,
                mirror=not a.no_mirror,
            )
        except RuntimeError as exc:
            self._error = str(exc)
            print(f"\n[camera error] {exc}\n", file=sys.stderr)
            return

        tracker = None
        if a.phase >= 2:
            tracker = ArmTracker(
                TrackerConfig(
                    side=a.side,
                    mirror=not a.no_mirror,
                    gripper=GripperConfig(
                        closed_ratio=a.gripper_closed,
                        open_ratio=a.gripper_open,
                    ),
                    quality=QualityConfig(
                        min_arm_visibility=MIN_ARM_VISIBILITY,
                        min_hand_confidence=MIN_HAND_CONFIDENCE,
                    ),
                    smooth=not a.no_smooth,
                )
            )

        print(f"[pipeline] camera {a.camera} -> {camera.width}x{camera.height}, "
              f"phase {a.phase}, tracking {a.side.upper()} arm, "
              f"mirror={'off' if a.no_mirror else 'on'}")

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), a.jpeg_quality]
        t0 = time.perf_counter()
        fps_ema = 0.0
        last = t0

        try:
            while not self._stop.is_set():
                frame = camera.read()
                if frame is None:
                    time.sleep(0.005)
                    continue

                now = time.perf_counter()
                dt = now - last
                last = now
                if dt > 0:
                    inst = 1.0 / dt
                    fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst

                result = None
                if tracker is not None:
                    # VIDEO mode wants a monotonic ms timestamp.
                    ts_ms = int((now - t0) * 1000.0)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = tracker.process(rgb, ts_ms)

                annotated = draw_overlay(frame, result, phase=a.phase, fps=fps_ema)

                ok, buf = cv2.imencode(".jpg", annotated, encode_params)
                if not ok:
                    continue

                with self._lock:
                    self._jpeg = buf.tobytes()
                    self._fps = fps_ema
                    if result is not None:
                        self._state = result.state
                self._frame_event.set()
        finally:
            camera.release()
            if tracker is not None:
                tracker.close()

    # ------------------------------------------------------------------ #
    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def snapshot(self) -> dict:
        with self._lock:
            s = self._state
            fps = self._fps
            err = self._error
        return {
            "error": err,
            "fps": round(fps, 1),
            "phase": self.args.phase,
            "side": s.side,
            "shoulder_lift": None if s.shoulder_lift is None else round(s.shoulder_lift, 1),
            "elbow_flex": None if s.elbow_flex is None else round(s.elbow_flex, 1),
            "wrist_flex": None if s.wrist_flex is None else round(s.wrist_flex, 1),
            "gripper": None if s.gripper is None else round(s.gripper, 3),
            "gripper_raw_ratio": (
                None if s.gripper_raw_ratio is None else round(s.gripper_raw_ratio, 3)
            ),
            "arm_visible": s.arm_visible,
            "hand_visible": s.hand_visible,
            "arm_confidence": round(s.arm_confidence, 2),
            "hand_confidence": round(s.hand_confidence, 2),
            "landmark_visibility": {
                k: round(v, 2) for k, v in s.landmark_visibility.items()
            },
        }

    def mjpeg(self):
        """Yield multipart JPEG frames, waiting on the producer rather than polling."""
        boundary = b"--frame\r\n"
        while not self._stop.is_set():
            self._frame_event.wait(timeout=1.0)
            self._frame_event.clear()
            jpeg = self.latest_jpeg()
            if jpeg is None:
                continue
            yield boundary + b"Content-Type: image/jpeg\r\n"
            yield f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
            yield jpeg + b"\r\n"


def build_app(args: argparse.Namespace) -> FastAPI:
    pipeline = Pipeline(args)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pipeline.start()
        yield
        pipeline.stop()

    app = FastAPI(title="Arm Pose Viz", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse((HERE / "static" / "index.html").read_text())

    @app.get("/stream")
    async def stream() -> StreamingResponse:
        return StreamingResponse(
            pipeline.mjpeg(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    @app.get("/api/state")
    async def state() -> JSONResponse:
        return JSONResponse(pipeline.snapshot())

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Webcam arm + hand pose visualization (no robot, no hardware).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--list-cameras", action="store_true",
                   help="probe camera indices, print them, and exit")
    p.add_argument("--snapshot", action="store_true",
                   help="save one JPEG per working camera to /tmp and exit, so you "
                        "can SEE which index is pointed at you")
    # ---- CONFIG: camera index. On this Mac index 0 is the OBS virtual camera,
    #      which opens but yields no frames; the built-in FaceTime cam is 1.
    p.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX,
                   help="camera index (run --list-cameras to see options)")
    p.add_argument("--side", choices=("left", "right"), default="right",
                   help="which of YOUR arms to track")
    p.add_argument("--phase", type=int, choices=(1, 2, 3, 4), default=4,
                   help="1 raw video | 2 +arm skeleton | 3 +shoulder/elbow | 4 +hand")

    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--jpeg-quality", type=int, default=80)

    p.add_argument("--no-mirror", action="store_true",
                   help="disable selfie-view horizontal flip")
    p.add_argument("--no-smooth", action="store_true",
                   help="disable One-Euro smoothing (to see the raw jitter)")

    # ---- CONFIG: gripper thresholds ------------------------------------- #
    p.add_argument("--gripper-closed", type=float, default=GRIPPER_CLOSED_RATIO,
                   help="pinch ratio at/below which gripper reads 0.0")
    p.add_argument("--gripper-open", type=float, default=GRIPPER_OPEN_RATIO,
                   help="pinch ratio at/above which gripper reads 1.0")

    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()

    if args.list_cameras:
        print("\nAvailable cameras:")
        print(format_camera_table(list_cameras()))
        print("\nNot sure which index is you?  Run:  python server.py --snapshot\n")
        return

    if args.snapshot:
        print("\nGrabbing one frame from each working camera...")
        paths = snapshot_cameras()
        if not paths:
            print("  no working cameras found")
            return
        for path in paths:
            print(f"  {path}")
        print("\nOpen those and see which one is you, then:  --camera <index>\n")
        return

    print(f"\n  Arm pose viz  ->  http://{args.host}:{args.port}\n")
    uvicorn.run(build_app(args), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
