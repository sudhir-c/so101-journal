"""
Live mirror: MediaPipe arm pose → SO-101 follower (single process).

Links the two standalone tools by IMPORTING them — neither is modified:
  * vision:  teleop.vision  (Camera, ArmTracker, draw_overlay, ArmState)
  * robot:   teleop.robot.control.RobotController  (the safety core)

This process is the SOLE owner of the follower's serial port while it runs.

Phase 1 (current): PREVIEW ONLY. Shows the annotated webcam, live human angles,
the mapped robot TARGET angles, the robot's ACTUAL read-back angles, a 2D
schematic pose preview (target + actual), and a tracking-quality light. The
robot does NOT move. ENABLE is inert until Phase 3; STOP/RESUME already control
holding torque so you can go limp / re-hold at any time.

Run (from the repo root):
    .venv/bin/python -m teleop.mirror.server --camera 1 --side right
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
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from teleop.mirror import mapping
from teleop.robot.control import JOINT_NAMES, RobotController
from teleop.vision.camera import DEFAULT_CAMERA_INDEX, Camera, macos_camera_names
from teleop.vision.overlay import draw_overlay
from teleop.vision.pose_math import ArmState, GripperConfig, QualityConfig
from teleop.vision.tracker import ArmTracker, TrackerConfig

HERE = Path(__file__).resolve().parent

# ── Robot configuration ─────────────────────────────────────────────────────
ROBOT_PORT = "/dev/tty.usbmodem5B3D0486331"
ROBOT_ID = "spectre"

# Tracking-quality floors (mirror the vision tool's defaults).
MIN_ARM_VISIBILITY = 0.6
MIN_HAND_CONFIDENCE = 0.5

# How often to read the robot's actual joint positions (Hz) in PREVIEW. When
# enabled, every frame commands+reads, so this only paces the idle read.
ROBOT_READ_HZ = 15.0

# On ENABLE, interpolate from the arm's current pose to the mapped target pose
# over this many seconds before live tracking begins (avoids a lurch).
RAMP_SECONDS = 1.5


def _round(v, n=1):
    return None if v is None else round(float(v), n)


class MirrorPipeline:
    """Capture + inference + robot-readback thread; publishes the latest frame
    and a full state snapshot. Phase 1 commands no motion."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._lock = threading.Lock()

        self._jpeg: bytes | None = None
        self._fps = 0.0
        self._error: str | None = None
        self._frame_event = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # Published state.
        self._state = ArmState(side=args.side)
        self._targets: dict[str, float] = {}     # last GOOD mapped robot targets
        self._actual: dict[str, float] = {}       # last robot read-back
        self._tracking_ok = False

        # Motion control.
        self._enabled = False
        self._ramp: dict | None = None  # {start, goal, t0} during ramp-on-enable

        # Camera selection (switchable live from the UI).
        self._camera_index = args.camera
        self._switch_to: int | None = None

        # Mapping calibration (human_min/human_max per joint).
        self.human_range = mapping.load_human_range()

        # Robot connection (read-only in Phase 1).
        self.robot = RobotController(port=ROBOT_PORT, robot_id=ROBOT_ID)
        self._last_robot_read = 0.0

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        self.robot.connect()
        self._thread = threading.Thread(target=self._run, name="mirror", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        try:
            self.robot.disconnect()
        except Exception:  # nosec - best-effort cleanup
            pass

    # ------------------------------------------------------------------ #
    def _open_camera(self, index: int) -> Camera:
        a = self.args
        return Camera(index=index, width=a.width, height=a.height,
                      fps=a.fps, mirror=not a.no_mirror)

    def _run(self) -> None:
        a = self.args
        camera: Camera | None = None
        current_index: int | None = None

        tracker = ArmTracker(TrackerConfig(
            side=a.side,
            mirror=not a.no_mirror,
            gripper=GripperConfig(closed_ratio=a.gripper_closed, open_ratio=a.gripper_open),
            quality=QualityConfig(
                min_arm_visibility=MIN_ARM_VISIBILITY,
                min_hand_confidence=MIN_HAND_CONFIDENCE,
            ),
            smooth=not a.no_smooth,
        ))

        print(f"[mirror] tracking {a.side.upper()} arm, PREVIEW (no motion), "
              f"starting camera index {a.camera}")

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), a.jpeg_quality]
        t0 = time.perf_counter()
        fps_ema = 0.0
        last = t0

        try:
            while not self._stop.is_set():
                # (Re)open the camera on first pass or when a switch is requested.
                with self._lock:
                    pending = self._switch_to
                    self._switch_to = None
                desired = pending if pending is not None else (
                    current_index if current_index is not None else a.camera)
                if camera is None or desired != current_index:
                    if camera is not None:
                        camera.release()
                        camera = None
                    try:
                        camera = self._open_camera(desired)
                        current_index = desired
                        with self._lock:
                            self._camera_index = current_index
                            self._error = None
                        print(f"[mirror] camera -> index {current_index} "
                              f"({camera.width}x{camera.height})")
                    except RuntimeError as exc:
                        with self._lock:
                            self._error = f"camera {desired}: {exc}"
                        camera = None
                        time.sleep(0.5)
                        continue

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

                ts_ms = int((now - t0) * 1000.0)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = tracker.process(rgb, ts_ms)
                state = result.state

                annotated = draw_overlay(frame, result, phase=4, fps=fps_ema)
                ok, buf = cv2.imencode(".jpg", annotated, encode_params)

                # Map human → robot targets, holding last-good on lost tracking.
                tracking_ok = state.ok
                if tracking_ok:
                    human = {
                        "shoulder_lift": state.shoulder_lift,
                        "elbow_flex": state.elbow_flex,
                        "wrist_flex": state.wrist_flex,
                        "gripper": state.gripper,
                    }
                    new_targets = mapping.map_human_to_robot(human, self.human_range)
                else:
                    new_targets = {}  # HOLD: never send angles from a bad frame

                # Publish the frame + update last-good targets first.
                with self._lock:
                    if ok:
                        self._jpeg = buf.tobytes()
                    self._fps = fps_ema
                    self._state = state
                    self._tracking_ok = tracking_ok
                    if new_targets:
                        self._targets.update(new_targets)  # update only fresh joints
                    enabled = self._enabled

                # Drive the robot (or just read it, in preview).
                stopped = self.robot.is_stopped
                torque = self.robot.is_torque_enabled
                actual = None
                try:
                    if enabled and not stopped and torque:
                        # Command through the safety core (clamp + velocity-limit).
                        # step() returns the present positions, so no extra read.
                        cmd = self._command_for_frame(now)
                        if cmd:
                            actual = self.robot.step(cmd)
                    if actual is None and now - self._last_robot_read >= 1.0 / ROBOT_READ_HZ:
                        actual = self.robot.get_positions()
                        self._last_robot_read = now
                except Exception:  # serial hiccup → keep last known
                    actual = None

                if actual is not None:
                    with self._lock:
                        self._actual = actual
                self._frame_event.set()
        finally:
            if camera is not None:
                camera.release()
            tracker.close()

    # ------------------------------------------------------------------ #
    # Motion control
    # ------------------------------------------------------------------ #
    def _command_for_frame(self, now: float) -> dict[str, float] | None:
        """The joint command for this frame: a ramp interpolation right after
        ENABLE, otherwise the live mapped targets. Held joints ride along at
        their fixed constant. Returns None if nothing to command yet."""
        with self._lock:
            ramp = self._ramp
            targets = dict(self._targets)
            if ramp is not None:
                alpha = (now - ramp["t0"]) / RAMP_SECONDS
                if alpha < 1.0:
                    start, goal = ramp["start"], ramp["goal"]
                    return {j: start[j] + alpha * (goal[j] - start[j]) for j in goal}
                self._ramp = None  # ramp done → fall through to live tracking
        # Live: last-good controlled targets (held on lost tracking) + held joints.
        cmd = {j: targets[j] for j in mapping.CONTROLLED if j in targets}
        cmd.update(mapping.HELD)
        return cmd or None

    def enable(self) -> None:
        """Begin mirroring: ensure torque, snapshot a ramp from the arm's actual
        pose to the current mapped target, then live-track once the ramp ends."""
        if not self.robot.is_connected:
            return
        if not self.robot.is_torque_enabled:
            self.robot.enable_torque()
        self.robot.resume()
        try:
            actual = self.robot.get_positions()
        except Exception:
            return
        with self._lock:
            goal = {j: self._targets[j] for j in mapping.CONTROLLED if j in self._targets}
            goal.update(mapping.HELD)
            start = {j: actual.get(j, goal[j]) for j in goal}
            self._ramp = {"start": start, "goal": goal, "t0": time.perf_counter()}
            self._enabled = True

    def disable(self) -> None:
        """Stop mirroring (arm holds its current pose; torque stays on)."""
        with self._lock:
            self._enabled = False
            self._ramp = None

    def emergency_stop(self) -> None:
        self.robot.stop()            # reject any future motion commands
        self.robot.disable_torque()  # go limp
        with self._lock:
            self._enabled = False
            self._ramp = None

    def resume(self) -> None:
        self.robot.enable_torque()   # re-hold current pose (no snap)
        self.robot.resume()          # stays in preview until ENABLE again

    def switch_camera(self, index: int) -> None:
        with self._lock:
            self._switch_to = int(index)

    def current_camera_index(self) -> int:
        with self._lock:
            return self._camera_index

    # ------------------------------------------------------------------ #
    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def snapshot(self) -> dict:
        with self._lock:
            s = self._state
            targets = dict(self._targets)
            actual = dict(self._actual)
            fps = self._fps
            tracking_ok = self._tracking_ok
            enabled = self._enabled
            ramping = self._ramp is not None
            camera_index = self._camera_index
            err = self._error
        # Held joints are shown as their fixed constants.
        target_full = {**targets, **mapping.HELD}
        return {
            "error": err,
            "fps": round(fps, 1),
            "camera_index": camera_index,
            "mode": "enabled" if enabled else "preview",
            "enabled": enabled,
            "ramping": ramping,
            "stopped": self.robot.is_stopped,
            "torque": self.robot.is_torque_enabled,
            "connected": self.robot.is_connected,
            "tracking_ok": tracking_ok,
            "quality": {
                "arm_visible": s.arm_visible,
                "hand_visible": s.hand_visible,
                "arm_confidence": round(s.arm_confidence, 2),
                "hand_confidence": round(s.hand_confidence, 2),
                "landmark_visibility": {k: round(v, 2) for k, v in s.landmark_visibility.items()},
            },
            "human": {
                "shoulder_lift": _round(s.shoulder_lift),
                "elbow_flex": _round(s.elbow_flex),
                "wrist_flex": _round(s.wrist_flex),
                "gripper": _round(s.gripper, 3),
            },
            "target": {j: _round(target_full.get(j)) for j in JOINT_NAMES},
            "actual": {j: _round(actual.get(j)) for j in JOINT_NAMES},
            "controlled": mapping.CONTROLLED,
        }

    def mjpeg(self):
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


class MonitorFeed:
    """A second, processing-free camera feed (e.g. pointed at the arm). Off by
    default; switchable live. Runs its own thread and publishes JPEGs."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._event = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._index: int | None = None     # None = off
        self._desired: int | None = None
        self._dirty = False
        self._error: str | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def switch(self, index: int | None) -> None:
        with self._lock:
            self._desired = index
            self._dirty = True

    def current(self) -> int | None:
        with self._lock:
            return self._index

    def _run(self) -> None:
        a = self.args
        camera: Camera | None = None
        current: int | None = None
        encode = [int(cv2.IMWRITE_JPEG_QUALITY), a.jpeg_quality]
        while not self._stop.is_set():
            with self._lock:
                dirty, desired = self._dirty, self._desired
                self._dirty = False
            if dirty and desired != current:
                if camera is not None:
                    camera.release()
                    camera = None
                current = desired
                with self._lock:
                    self._index = current
                    self._error = None
                if desired is not None:
                    try:  # monitor view of the arm: no selfie mirror
                        camera = Camera(index=desired, width=a.width, height=a.height,
                                        fps=a.fps, mirror=False)
                    except RuntimeError as exc:
                        with self._lock:
                            self._error = f"monitor camera {desired}: {exc}"
                        camera = None
            if camera is None:
                time.sleep(0.1)
                continue
            frame = camera.read()
            if frame is None:
                time.sleep(0.01)
                continue
            ok, buf = cv2.imencode(".jpg", frame, encode)
            if ok:
                with self._lock:
                    self._jpeg = buf.tobytes()
                self._event.set()
        if camera is not None:
            camera.release()

    def mjpeg(self):
        boundary = b"--frame\r\n"
        while not self._stop.is_set():
            self._event.wait(timeout=1.0)
            self._event.clear()
            with self._lock:
                jpeg = self._jpeg
            if jpeg is None:
                continue
            yield boundary + b"Content-Type: image/jpeg\r\n"
            yield f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
            yield jpeg + b"\r\n"


class CameraReq(BaseModel):
    index: int
    slot: str = "track"   # "track" (pose camera) or "monitor" (arm view)


def build_app(args: argparse.Namespace) -> FastAPI:
    pipeline = MirrorPipeline(args)
    monitor = MonitorFeed(args)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pipeline.start()
        monitor.start()
        yield
        monitor.stop()
        pipeline.stop()

    app = FastAPI(title="SO-101 Mirror Teleop", lifespan=lifespan)

    @app.get("/")
    def index():
        return FileResponse(HERE / "static" / "index.html")

    @app.get("/video")
    def video():
        return StreamingResponse(
            pipeline.mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.get("/video2")
    def video2():
        return StreamingResponse(
            monitor.mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.get("/api/state")
    def state():
        return JSONResponse(pipeline.snapshot())

    @app.post("/api/stop")
    def stop():
        pipeline.emergency_stop()
        return {"ok": True, "stopped": True, "torque": pipeline.robot.is_torque_enabled}

    @app.post("/api/resume")
    def resume():
        pipeline.resume()
        return {"ok": True, "stopped": False, "torque": pipeline.robot.is_torque_enabled}

    @app.get("/api/cameras")
    def cameras():
        # Indices are the reliable handle; device names are shown only as an
        # unordered hint (macOS doesn't map names→indices reliably). Pick the
        # index whose live feed shows what you want.
        return {
            "indices": list(range(5)),
            "names": macos_camera_names(),
            "current_track": pipeline.current_camera_index(),
            "current_monitor": monitor.current(),
        }

    @app.post("/api/camera")
    def set_camera(req: CameraReq):
        if req.slot == "monitor":
            # index < 0 means "off".
            monitor.switch(None if req.index < 0 else req.index)
        else:
            pipeline.switch_camera(req.index)
        return {"ok": True, "index": req.index, "slot": req.slot}

    @app.post("/api/enable")
    def enable():
        pipeline.enable()
        return {"ok": True, "enabled": pipeline._enabled}

    @app.post("/api/disable")
    def disable():
        pipeline.disable()
        return {"ok": True, "enabled": False}

    return app


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SO-101 live mirror teleop (Phase 1: preview).")
    p.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    p.add_argument("--side", choices=["left", "right"], default="right")
    p.add_argument("--no-mirror", action="store_true", help="disable selfie-view flip")
    p.add_argument("--no-smooth", action="store_true", help="disable One-Euro smoothing")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--gripper-closed", type=float, default=0.15)
    p.add_argument("--gripper-open", type=float, default=1.10)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(build_app(args), host=args.host, port=args.port, log_level="warning")
