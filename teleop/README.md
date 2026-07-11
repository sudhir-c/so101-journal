# teleop

Teleoperating the SO-101 arm from a human arm. Two halves that are **deliberately
not yet connected** — each runs and is tested on its own:

| half | what it does | hardware | entry point |
|------|--------------|----------|-------------|
| [`robot/`](robot/) | serial control + safety layer + slider UI for the `spectre` follower arm | the SO-101 | `python -m teleop.robot.server` |
| [`vision/`](vision/) | webcam → arm/hand pose → joint values, streamed to the browser | a webcam only | `python -m teleop.vision.server` |

The end goal is to feed vision-derived joint angles into the robot's control
layer. **That retarget link does not exist yet** — the two halves share no code
path. This directory just organizes both APIs side by side so that step is easy
to add later (see "Next step").

```
teleop/
  robot/      control.py · server.py · static/index.html · README.md · setup.txt
  vision/     pose_math.py · smoothing.py · tracker.py · camera.py · overlay.py
              server.py · static/index.html · README.md
  models/     MediaPipe .task bundles (gitignored — fetch with scripts/download_models.sh)
  scripts/    download_models.sh · cam_viz.py · teleop.sh
  tests/      test_pose_math.py
```

## Prerequisites

Everything runs from the project's existing uv-managed venv. The venv has **no
`pip`** — use `uv pip`. Run all commands **from the repo root** (the parent of
this directory), so `teleop` imports as a package.

```bash
# one-time: server deps + MediaPipe model bundles
uv pip install --python .venv/bin/python fastapi "uvicorn[standard]"
./teleop/scripts/download_models.sh
```

## Run the robot control UI (needs the arm)

```bash
.venv/bin/python -m teleop.robot.server        # → http://localhost:8000/
```

Manual per-joint sliders with clamping + velocity limiting, STOP, and live
readouts. This process is the **sole owner of the serial port** — don't run
`lerobot-teleoperate`/`lerobot-record` alongside it. Details: [`robot/README.md`](robot/README.md).

## Run the arm visualization UI (needs only a webcam)

```bash
.venv/bin/python -m teleop.vision.server --snapshot          # find your camera index
.venv/bin/python -m teleop.vision.server --camera 1 --side right   # → http://127.0.0.1:8080/
```

No robot, no serial, no LeRobot. Details: [`vision/README.md`](vision/README.md).

## Tests

```bash
.venv/bin/python teleop/tests/test_pose_math.py
```

## Next step: the retarget layer (not built)

`vision/` emits *raw human* joint angles; `robot/` expects angles in the SO-101's
signed, zero-centred ranges (`JOINT_LIMITS`). The units, signs, zero points and
even the set of joints differ. The future link is a small `retarget` module that
maps an `ArmState` from `teleop.vision` onto the `{joint: angle}` dict that
`teleop.robot.control.RobotController.step()` already accepts, clamping on the way
out. Keep it out of `vision/pose_math.py` — that module's independence (numpy
only) is what makes it reusable.
