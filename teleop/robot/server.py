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
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .control import JOINT_LIMITS, JOINT_NAMES, RobotController

HERE = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# ── Robot configuration ────────────────────────────────────────────────────────
PORT = "/dev/tty.usbmodem5B3D0486331"
ROBOT_ID = "spectre"

controller = RobotController(port=PORT, robot_id=ROBOT_ID)


# ── App lifecycle ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    controller.connect()
    yield
    controller.disconnect()


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
    return {"ok": True, "connected": controller.is_connected, "stopped": controller.is_stopped}


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
        {"angles": {"wrist_roll": 12.3, ...}}   # command toward these targets
        {"cmd": "stop"} / {"cmd": "resume"}      # safety controls

    Server replies to every message with:
        {"positions": {joint: float, ...}, "stopped": bool}

    The client streams continuously (send → await reply → send again), so the
    effective rate self-paces to whatever the serial bus sustains. The blocking
    serial step() runs in a threadpool and shares RobotController's lock with
    the REST endpoints, so concurrent access stays safe.
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
            # step() ignores targets internally while stopped, and always
            # returns fresh positions for the display.
            angles = msg.get("angles", {})
            try:
                positions = await run_in_threadpool(controller.step, angles)
            except (ValueError, RuntimeError) as e:
                await websocket.send_json({"error": str(e)})
                continue
            await websocket.send_json({"positions": positions, "stopped": controller.is_stopped})
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
