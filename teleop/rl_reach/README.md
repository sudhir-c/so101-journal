# RL reaching on the SO-101

A self-contained SB3 (SAC) + custom Gymnasium setup that trains the real `spectre`
arm to move its gripper tip to a target xyz. Deliberately simple (free reward, free
reset) — a shakedown of the whole on-hardware RL loop, structured so a later grasp
task can swap the reward / termination / reset / observation cleanly.

Everything runs in the main venv (`.venv`) from the repo root, e.g.
`.venv/bin/python -m teleop.rl_reach.<script>`. The env is the **sole owner of the
serial port** while it runs — don't run the teleop dashboard / mirror / lerobot at
the same time.

## Files
- `fk.py` — from-scratch numpy forward kinematics (parses the URDF; no placo/IK at
  runtime). `fk(joint_deg) -> gripper_xyz` (metres).
- `so101_reach_env.py` — the Gymnasium env: lean observation, rich `info`, and
  swappable `_compute_reward` / `_is_terminated` / `_reset_task`.
- `train_reach.py` — SAC on MPS with resumable checkpoints.
- `eval_reach.py` — roll out a trained policy.
- Reuses `teleop.robot.control.RobotController` (the existing safety core) for ALL
  motion — clamp to `JOINT_LIMITS` + velocity limit + STOP. The only addition to it
  was a read-only `read_telemetry()`.

## Run it, phase by phase
1. FK (no hardware): `python -m teleop.rl_reach.fk` — self-tests against the placo
   FK (< 1 mm). Then sanity-check physically: compare `fk(get_positions())` to a
   tape-measured tip at a couple of poses.
2. Env dry run (hardware, no learning): `python -m teleop.rl_reach.so101_reach_env`
   — resets, takes random actions through the safety core at 10 Hz, prints
   obs/reward/info. Confirm the motion is safe and the numbers look sane.
   Position the arm INSIDE the workspace box first (see EE bounds below).
3. Train: `python -m teleop.rl_reach.train_reach --timesteps 20000`. Checkpoints
   (model + replay buffer) land in `checkpoints/` every `--save-freq` steps;
   re-running resumes from the latest (`--fresh` to restart). Ctrl-C saves an
   interrupt checkpoint and releases the arm.
4. Eval: `python -m teleop.rl_reach.eval_reach --model checkpoints/reach_sac_final`.
   By default this opens a live **3D visualizer** at http://127.0.0.1:8091 (robot
   camera feed + a three.js scene of the arm skeleton, the sampled target, and the
   end-effector, with the error line/distance and a fading EE trail). One-time setup:
   `./teleop/rl_reach/scripts/download_three.sh` (fetches vendored three.js r128 —
   gitignored). Pick the arm-facing camera from the dropdown, or pass `--camera N`.
   The visualizer is **display only** — eval stays the sole serial owner. Flags:
   `--no-viz` (terminal-only, original behavior), `--viz-port`, `--no-browser`.
   Reuses the shared `teleop.vision.camera_feed.CameraFeed` (also used by the robot
   dashboard) via `teleop/rl_reach/viz_server.py`.

## Where the knobs live (top of `so101_reach_env.py`)
- `ACTION_JOINTS` — the joints the policy moves (shoulder_pan, shoulder_lift,
  elbow_flex, wrist_flex). wrist_roll (overload history) and gripper are held fixed.
- `MAX_DELTA_DEG = 2.0` — per-step action bound (deg). Safety core hard-caps at
  180 deg/s regardless.
- `CONTROL_HZ = 10`, `MAX_STEPS = 200`.
- `MIN_EE` / `MAX_EE` / `EE_MARGIN` — measured workspace box; a step whose predicted
  tip would go further outside the box is rejected (arm holds).
- `HOME_XYZ` — Cartesian pose (m) the arm re-homes to at the START of each episode,
  solved to joint angles ONCE via the placo IK sidecar (the step-loop stays IK-free).
  Defaults to the box centre. This prevents the arm drifting into a corner across the
  free-reset episodes (which otherwise stalls learning). Set `HOME_XYZ = None` to
  disable re-homing. Because it re-homes, the arm no longer needs to be positioned
  inside the box by hand before training.
- Reward: `W_DIST`, `SUCCESS_THRESH`, `SUCCESS_BONUS`, `W_ACTION`.
- Hardware: `PORT`, `ROBOT_ID`.

## Safety
- Every action goes through `RobotController.step()` (clamp + velocity limit); never
  bypassed. The EE guard keeps the tip in the measured workspace.
- `env.close()` (and train's try/finally) disconnects → torque off. `stop()` /
  `disable_torque()` on the controller remain available as immediate stops.
- Reaching reset does not move the arm (no lurch); the target is just re-sampled.

## Extending to a grasp task
The split is designed for this: keep `step()` as-is and override the three hooks.
- `_reset_task()` — sample/place the object; optionally home the arm smoothly (a
  `_home()` rate-limited move hook is the place for it) instead of the no-op reset.
- `_compute_reward(obs, action, info)` — read grasp signals straight from `info`
  (which already carries fk tip xyz, gripper width, and per-motor current / load /
  temperature every step) WITHOUT touching the observation. Motor current is the
  natural contact/grasp signal.
- `_is_terminated(info)` — success when grasped/lifted.
- If the grasp policy needs the gripper, add it to `ACTION_JOINTS`. The lean
  observation only changes if the policy genuinely needs a new input.
