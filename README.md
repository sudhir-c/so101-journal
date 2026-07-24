# so101-journal

https://github.com/user-attachments/assets/48a9cb43-b508-4f25-87ad-baa328d1d13c

Personal journal **and** working code for my [SO-101 robot arm](https://github.com/TheRobotStudio/SO-ARM100)
— a 5-DOF + gripper follower arm (nicknamed `spectre`). The repo has two halves:

- **[`journal/`](./journal/)** — the narrative write-up of the project, entry by
  entry (teleop, cameras, first policies, moving-object pick-and-place, …). Start
  at [`journal/00_intro.md`](./journal/00_intro.md).
- **[`teleop/`](./teleop/)** — the engineering: real-time arm control, webcam pose
  teleoperation, a live vision→robot mirror, and an on-hardware RL reaching setup.

## What's in `teleop/`

Run everything as modules **from the repo root** (so `teleop` imports as a
package), using the project venv explicitly (`.venv/bin/python`). Full details and
safety notes are in [`teleop/README.md`](./teleop/README.md) and [`CLAUDE.md`](./CLAUDE.md).

| module | what it does | run | port |
|---|---|---|---|
| `teleop.robot` | serial control of the follower arm; web slider + IK dashboard | `python -m teleop.robot.server` | 8000 |
| `teleop.vision` | webcam → arm/hand pose → joint values (MediaPipe Tasks; **no hardware**) | `python -m teleop.vision.server --camera 1` | 8080 |
| `teleop.mirror` | live: your arm (vision) → the follower (robot), one process | `python -m teleop.mirror.server` | 8090 |
| `teleop.rl_reach` | from-scratch FK + Gymnasium env + SB3 SAC to train the real arm to reach an xyz target | `python -m teleop.rl_reach.train_reach` / `eval_reach` | — |

Only **one** of these may own the serial port at a time. `teleop/scripts/` holds
helpers (`download_models.sh` fetches the MediaPipe `.task` bundles — required
before vision/mirror run); `teleop/tests/` holds the pose-math regression tests.

## Setup

- Python 3.12, a `uv`-managed venv at `.venv/` (macOS arm64). Always invoke it
  explicitly (`.venv/bin/python`); it holds torch + LeRobot + the vision/RL stack.
- First-time vision/mirror: `./teleop/scripts/download_models.sh`.
- The RL IK sidecar and MediaPipe weights are fetched, not committed — see
  [`CLAUDE.md`](./CLAUDE.md) for the exact environment gotchas.

Top-level [`scripts/`](./scripts/) holds journal video tooling (e.g.
`video_compressor.py`).
