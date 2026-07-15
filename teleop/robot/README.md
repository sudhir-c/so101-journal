# SO-101 Arm Web Controller

Local web UI for manual joint control of the `spectre` SO-101 follower arm,
built as the first step toward hand-tracking teleoperation.

## Setup

Install the server dependencies once, into the existing venv (the venv is
uv-managed and has no `pip` — use `uv pip`):

```bash
uv pip install --python .venv/bin/python fastapi "uvicorn[standard]"
```

## Running

From the **repo root** (so `teleop` is importable as a package):

```bash
.venv/bin/python -m teleop.robot.server
```

Then open **http://localhost:8000/** for the slider UI (all 6 joints, live
readouts, STOP, Re-sync). The UI (`static/index.html`) is re-read on each
request, so editing it only needs a browser refresh; editing server code needs
a restart.

**Never run this under `uvicorn --reload`.** The reloader watches the whole
tree — including the thousands of files under `.venv/` — which triggers endless
reloads, and each reload re-runs startup, reconnecting the serial port and
making the arm go limp. The `python -m teleop.robot.server` entry point above
runs a single, non-reloading server. Always invoke it via `.venv/bin/python`
so it can't pick up a global Homebrew interpreter with the wrong NumPy/torch.

The arm connection is established at startup. This server is the sole owner
of the serial port — do **not** run `lerobot-teleoperate` or `lerobot-record`
while it is running.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Slider UI (`static/index.html`) |
| GET | `/health` | Liveness — `{ok, connected, stopped}` |
| GET | `/positions` | Current joint angles (degrees / %) |
| GET | `/limits` | Per-joint safe ranges for slider bounds |
| WS | `/ws` | **Streaming control** — the UI and future pose estimator use this |
| POST | `/step` | One-shot command+read (HTTP equivalent of a `/ws` frame) |
| POST | `/joint/{name}` | Command one joint — body `{"angle": float}` |
| POST | `/joints` | Command many — body `{"angles": {joint: float}}` |
| POST | `/stop` | Reject motion commands (arm holds position) |
| POST | `/resume` | Allow motion commands again |
| POST | `/ik/reset_origin` | Set the IK origin (0,0,0) to the current tip |

### `/ws` protocol

Client sends, one message per update:
```json
{"angles": {"wrist_roll": 12.3, "elbow_flex": -5.0}}       // slider mode: joints
{"pose": {"x": 30, "y": 0, "z": -20}, "angles": {...}}      // IK mode: tip (mm)
{"cmd": "stop"}    {"cmd": "resume"}
```
Server replies with `{"positions": {joint: float}, "stopped": bool}`, plus
`"tip": {x,y,z}` (+ `"solvable": bool` for pose frames) when IK is available.

The UI streams continuously (send → await reply → send), so the command rate
self-paces to whatever the serial bus sustains — no fixed poll rate. On
disconnect the arm simply stops receiving commands and holds position.

## Position (IK) mode

The UI toggles between **Sliders** and **Position (IK)**. In IK mode you set an
**X/Y/Z target for the end-effector tip** (mm, relative to an origin that
defaults to the tip at startup and is resettable) and the arm solves its way
there. IK drives the 4 arm joints (`shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex`); `wrist_roll` + `gripper` stay on their sliders. Unreachable
targets → the arm holds (badge shows "UNREACHABLE"). Every solution still goes
through `RobotController.step()` (clamp + velocity-limit + STOP).

**The solver runs in a separate venv.** placo (the IK library) can't be
installed into the main venv without bumping numpy and disturbing torch/LeRobot,
so `teleop/robot/kinematics.py` spawns `teleop/robot/ik_service.py` as a
subprocess in an isolated **`.venv-ik`**. If that venv or the URDF is missing,
the dashboard runs slider-only (the IK toggle is disabled). Recreate the venv:

```bash
uv venv --python 3.12 .venv-ik
uv pip install --python .venv-ik/bin/python --link-mode=copy placo
```

Kinematics come from the SO-ARM100 URDF (`urdf/so101_new_calib.urdf`, meshes
stripped into `urdf/so101_kinematics.urdf`); tip frame `gripper_frame_link`.
Verify offline (no motors): `.venv-ik/bin/python teleop/robot/scripts/verify_ik.py`.

## Safety limits

Limits are enforced in `control.py`, not `server.py`, so they apply
to any client (HTTP or future pose-estimator).

### Per-joint clamping (`JOINT_LIMITS`)

Derived from `spectre.json` calibration via `±(range_max−mid)×(360/4096)`:

| Joint | Min | Max | Unit |
|-------|-----|-----|------|
| shoulder_pan | −102.3 | 102.3 | degrees |
| shoulder_lift | −102.7 | 102.7 | degrees |
| elbow_flex | −96.2 | 96.2 | degrees |
| wrist_flex | −100.5 | 100.5 | degrees |
| wrist_roll | −180.0 | 180.0 | degrees |
| gripper | 0.0 | 100.0 | % |

To tighten a limit, edit `JOINT_LIMITS` in `control.py`.

### Velocity limiting (`MAX_SPEED_DPS`)

`MAX_SPEED_DPS = 180.0` — max joint **speed** in degrees/second (%/s for the
gripper). This is a true velocity limit, not a per-command step: each `step()`
measures the real elapsed time `dt` since the previous command and caps motion
to `MAX_SPEED_DPS × dt`, enforced against the **freshly-read present position**.

Because it's time-based, the speed is identical whether commands arrive at
20 Hz, over an uncapped WebSocket, or from a variable-rate pose estimator.
This is the main anti-lunge knob — **raise `MAX_SPEED_DPS` for snappier motion.**

`MAX_DT = 0.1` caps the elapsed time used, so after an idle gap (or the first
command) a single step can't lunge more than `MAX_SPEED_DPS × MAX_DT` (= 18°).

To change: edit `MAX_SPEED_DPS` (and optionally `MAX_DT`) in `control.py`.

### STOP

`POST /stop` sets a flag that silently rejects all joint commands.
The arm stays in position-control mode and holds position — it does **not** go
limp. `POST /resume` clears the flag.

## Architecture

```
control.py                # Safety layer — no HTTP dependency
  RobotController
    .connect()
    .get_positions()      → dict[joint, float]
    .step(targets)        → positions   # read + clamp + rate-limit + write, one pass
    .set_joint(j, v)      # convenience wrapper around step()
    .set_joints({…})      # convenience wrapper around step()
    .stop() / .resume()

server.py                 # FastAPI — /ws stream + REST wrappers around RobotController
static/index.html         # Vanilla JS UI (WebSocket client)
```

`step()` is the core: one serial read → clamp to limits → velocity-limit →
one serial write, returning positions. Every path (WS, REST, UI) goes through
it, so the safety chain is identical everywhere.

The future hand-tracking client either connects to `/ws` or imports
`RobotController` and calls `step()` directly — same safety path.

## On connect

The arm briefly goes limp during `connect()` while LeRobot's `configure()`
runs (torque disabled → PID set → torque re-enabled). Keep the arm in a
stable resting pose before starting the server.

## Calibration

The `spectre` calibration is loaded automatically from
`~/.cache/huggingface/lerobot/calibration/robots/so_follower/spectre.json`.
The server uses `connect(calibrate=False)` to avoid blocking on interactive
prompts. If the motors drift, run `lerobot-calibrate` separately (server stopped).
