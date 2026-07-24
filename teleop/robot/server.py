"""
SO-101 follower arm web server: manual joint control + streaming.

FastAPI over the RobotController safety layer in `control.py`. Endpoints:
  GET  /            — slider UI (static/index.html)
  GET  /health      — liveness + connection status
  GET  /positions   — current joint angles (degrees / %)
  GET  /limits      — per-joint safe ranges
  WS   /ws          — streaming control (UI + future pose estimator)
  POST /step        — one command+read round-trip (HTTP form of a /ws frame)
  POST /joint/{name}, /joints — single / batch joint commands
  POST /stop, /resume         — safety flag

Run (from the repo root):  .venv/bin/python -m teleop.robot.server
"""

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .control import JOINT_LIMITS, JOINT_NAMES, RobotController
from .kinematics import PASSTHROUGH_JOINTS, TIP_FRAME, ArmIKClient
from ..vision.camera import macos_camera_names
from ..vision.camera_feed import CameraFeed

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# ── Robot configuration ────────────────────────────────────────────────────────
PORT = "/dev/tty.usbmodem5B3D0486331"
ROBOT_ID = "spectre"

controller = RobotController(port=PORT, robot_id=ROBOT_ID)


# ── Inverse kinematics (position mode) ──────────────────────────────────────────
# Additive: if the IK sidecar/URDF can't load, `client` stays None and the
# dashboard runs slider-only. The origin is the tip position (mm) at startup.

class IKState:
    def __init__(self):
        self.client: ArmIKClient | None = None
        self.origin_mm: list[float] | None = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.client is not None

    def build(self) -> None:
        client = ArmIKClient()
        self.origin_mm = client.fk_tip_mm(controller.get_positions())
        self.client = client  # set last so `available` flips only when fully ready

    def close(self) -> None:
        if self.client is not None:
            self.client.close()

    def set_origin(self, q: dict) -> None:
        with self._lock:
            self.origin_mm = self.client.fk_tip_mm(q)

    def tip_rel(self, q: dict) -> list[float]:
        """Tip position relative to the origin (mm), for a full joint dict."""
        with self._lock:
            origin = self.origin_mm
        tip = self.client.fk_tip_mm(q)
        return [tip[i] - origin[i] for i in range(3)]


ik_state = IKState()


def ik_step(pose: dict, passthrough: dict) -> dict:
    """Blocking pose→motion step (runs in a threadpool). Solve the arm for the
    tip target, command it through the safety core, and report the resulting
    tip + solvability. On unsolvable, the arm holds (only wrist_roll/gripper
    passthrough still applies)."""
    q = controller.get_positions()
    with ik_state._lock:
        origin = ik_state.origin_mm
    target_abs = [origin[i] + float(pose[k]) for i, k in enumerate(("x", "y", "z"))]
    # Freeze the IK at the arm's current wrist_roll (roll barely affects the tip).
    sol, solvable, _err = ik_state.client.solve(q, target_abs, q["wrist_roll"])
    passthrough = {k: v for k, v in passthrough.items() if k in PASSTHROUGH_JOINTS}
    targets = {**sol, **passthrough} if solvable else dict(passthrough)
    positions = controller.step(targets)
    tip = ik_state.client.fk_tip_mm(positions)
    return {
        "positions": positions,
        "tip": {"x": tip[0] - origin[0], "y": tip[1] - origin[1], "z": tip[2] - origin[2]},
        "solvable": solvable,
    }


def tip_dict(positions: dict) -> dict:
    """Tip position relative to origin as {x,y,z} (for slider-mode replies)."""
    t = ik_state.tip_rel(positions)
    return {"x": t[0], "y": t[1], "z": t[2]}


# ── Optional robot-facing camera feed ───────────────────────────────────────────
# A passthrough webcam (point it at the arm) so you can watch the robot while
# driving it. Off by default, switchable live from the UI; independent of the
# serial port. `CameraFeed` lives in teleop.vision.camera_feed (shared with the
# RL eval visualizer).

camera_feed = CameraFeed()


# ── App lifecycle ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    controller.connect()
    try:
        ik_state.build()
        logger.info("IK ready (tip frame %s, origin_mm=%s)", TIP_FRAME, ik_state.origin_mm)
    except Exception as e:  # noqa: BLE001 - IK is optional; degrade to slider-only
        logger.warning("IK unavailable — running slider-only mode: %s", e)
    camera_feed.start()
    yield
    # Release the arm FIRST and unconditionally — cutting torque on shutdown must
    # never be blocked by camera / IK-sidecar cleanup errors. disconnect() honors
    # disable_torque_on_disconnect=True, so the arm goes limp here.
    try:
        controller.disconnect()
    except Exception:
        logger.exception("robot disconnect failed")
    for label, cleanup in (("camera", camera_feed.stop), ("IK sidecar", ik_state.close)):
        try:
            cleanup()
        except Exception:
            logger.exception("%s shutdown failed", label)


app = FastAPI(title="SO-101 arm controller", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Phase 1: read-only endpoints ───────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "ok": True,
        "connected": controller.is_connected,
        "stopped": controller.is_stopped,
        "torque": controller.is_torque_enabled,
        "ik_available": ik_state.available,
        "tip_frame": TIP_FRAME,
    }


@app.get("/positions")
def get_positions():
    try:
        return controller.get_positions()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/limits")
def get_limits():
    """Return per-joint safe ranges so the UI can set slider bounds."""
    return {
        joint: {"min": lo, "max": hi}
        for joint, (lo, hi) in JOINT_LIMITS.items()
    }


# ── Phase 2: joint commands ────────────────────────────────────────────────────

class JointCmd(BaseModel):
    angle: float


class JointsCmd(BaseModel):
    angles: dict[str, float]


@app.post("/joint/{name}")
def set_joint(name: str, cmd: JointCmd):
    """
    Command a single joint. The commanded angle is clamped to the joint's
    safe range and velocity-limited (MAX_SPEED_DPS) from the arm's current
    position, both enforced in control.py.
    """
    if name not in JOINT_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown joint '{name}'")
    try:
        controller.set_joint(name, cmd.angle)
        return {"ok": True, "stopped": controller.is_stopped}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/joints")
def set_joints(cmd: JointsCmd):
    """Command multiple joints at once (same safety chain as /joint)."""
    try:
        controller.set_joints(cmd.angles)
        return {"ok": True, "stopped": controller.is_stopped}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/step")
def step(cmd: JointsCmd):
    """
    Low-latency streaming endpoint: command all joints toward `angles` and
    return the resulting positions in one round-trip. This is what the UI
    (and later the pose estimator) polls at high rate.
    """
    try:
        positions = controller.step(cmd.angles)
        return {"positions": positions, "stopped": controller.is_stopped}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    """
    Streaming control channel (the low-latency path; UI and future pose
    estimator use this).

    Protocol — client sends JSON messages, one per desired update:
        {"angles": {"wrist_roll": 12.3, ...}}    # slider mode: joint targets
        {"pose": {"x","y","z"}, "angles": {...}}  # IK mode: tip target (mm, rel
                                                  #   to origin) + roll/gripper
        {"cmd": "stop"} / {"cmd": "resume"}       # safety controls

    Server replies to every message with:
        {"positions": {joint: float}, "stopped": bool,
         "tip"?: {x,y,z}, "solvable"?: bool}      # tip/solvable in IK mode

    The blocking serial step() (and the IK solve) run in a threadpool and share
    RobotController's lock with the REST endpoints, so concurrent access is safe.
    """
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_json()
            cmd = msg.get("cmd")
            if cmd == "stop":
                controller.stop()
            elif cmd == "resume":
                controller.resume()
            pose = msg.get("pose")
            angles = msg.get("angles", {})
            try:
                if pose is not None and ik_state.available:
                    # IK mode: solve the arm for the tip target, hold on unsolvable.
                    reply = await run_in_threadpool(ik_step, pose, angles)
                    reply["stopped"] = controller.is_stopped
                else:
                    # Slider mode (or IK idle empty frames): step() ignores
                    # targets while stopped, always returns fresh positions.
                    positions = await run_in_threadpool(controller.step, angles)
                    reply = {"positions": positions, "stopped": controller.is_stopped}
                    if ik_state.available:
                        reply["tip"] = await run_in_threadpool(tip_dict, positions)
            except (ValueError, RuntimeError) as e:
                await websocket.send_json({"error": str(e)})
                continue
            reply["torque"] = controller.is_torque_enabled
            await websocket.send_json(reply)
    except WebSocketDisconnect:
        # Client vanished — the arm simply stops receiving commands and holds
        # its last position. No torque change, no lunge.
        pass


@app.post("/stop")
def stop():
    controller.stop()
    return {"ok": True, "stopped": True}


@app.post("/resume")
def resume():
    controller.resume()
    return {"ok": True, "stopped": False}


@app.post("/release")
def release():
    """Cut torque — the arm goes limp and can be moved by hand."""
    try:
        controller.disable_torque()
        return {"ok": True, "torque": controller.is_torque_enabled}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/hold")
def hold():
    """Re-enable torque, holding the arm's current (possibly hand-moved) pose."""
    try:
        controller.enable_torque()
        return {"ok": True, "torque": controller.is_torque_enabled}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/ik/reset_origin")
def reset_origin():
    """Set the IK origin (0,0,0) to the arm's current tip position."""
    if not ik_state.available:
        raise HTTPException(status_code=503, detail="IK unavailable")
    try:
        ik_state.set_origin(controller.get_positions())
        return {"ok": True, "origin_mm": ik_state.origin_mm}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ── Robot camera feed ───────────────────────────────────────────────────────────

class CameraReq(BaseModel):
    index: int   # negative = off


@app.get("/video")
def video():
    return StreamingResponse(
        camera_feed.mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/cameras")
def cameras():
    # Indices are the reliable handle; device names are an unordered hint
    # (macOS doesn't map names→indices). Pick the index that shows the arm.
    return {
        "indices": list(range(5)),
        "names": macos_camera_names(),
        "current": camera_feed.current(),
    }


@app.post("/api/camera")
def set_camera(req: CameraReq):
    camera_feed.switch(None if req.index < 0 else req.index)
    return {"ok": True, "index": req.index}


# ── Static frontend ────────────────────────────────────────────────────────────

@app.get("/")
def serve_ui():
    return FileResponse(HERE / "static" / "index.html")


# ── Entry point ──────────────────────────────────────────────────────────────
# Run with:  python -m teleop.robot.server
# (never with uvicorn --reload — it watches .venv and reconnects the serial port)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
