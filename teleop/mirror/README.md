# Mirror teleop — human arm → SO-101 (live)

A third, standalone entry point that **links** the two existing tools: it captures
your webcam, computes your arm angles with the vision module, maps them to robot
joint targets, and (from Phase 3) drives the follower through the robot safety
core. One process owns the follower's serial port while it runs.

**The two originals are untouched and still run independently:**
- slider teleop: `python -m teleop.robot.server`
- pose visualizer: `python -m teleop.vision.server`

This mirror **imports** their modules (`teleop.vision`, `teleop.robot.control`) —
it does not fork or modify them. The one change to the robot module was purely
additive (new `disable_torque()` / `enable_torque()` methods for the go-limp STOP;
`stop()`/`resume()`/`step()` are unchanged, so the slider UI behaves exactly as before).

## Launch

From the **repo root**, with nothing else holding the serial port
(stop `teleop.robot.server` / any `lerobot-*` first — single owner):

```bash
.venv/bin/python -m teleop.mirror.server --camera 1 --side right
```

Then open **http://localhost:8090/**. Use `--camera N` to pick your webcam
(run the visualizer's `--snapshot` if unsure which index is you).

## Status: Phase 1 — PREVIEW ONLY (the robot does not move)

The UI shows: annotated webcam feed, live **human** angles, mapped robot
**target** angles, the robot's **actual** read-back angles, a 2D schematic pose
preview (blue = target, grey = actual), and a tracking-quality light.

- **STOP** already works: it disables torque (arm goes limp) and rejects motion;
  the same button becomes **RESUME** (re-hold at the current pose, no snap).
- **ENABLE** is disabled until Phase 3 (which adds ramp-on-enable + live motion).

Coming next: Phase 2 (calibration capture), Phase 3 (ENABLE + ramp, driving
gripper + elbow first), Phase 4 (add shoulder_lift + wrist_flex, tune).

## The mapping (affine, per joint)

`robot = robot_min + (human − human_min)/(human_max − human_min) · (robot_max − robot_min)`

The calibration endpoints absorb range, zero-offset **and** direction — there is
no separate sign logic. All targets pass through the robot safety core
(`RobotController.step`: clamp → velocity-limit) before reaching a motor.

- **Controlled joints:** `shoulder_lift, elbow_flex, wrist_flex, gripper`.
- **Held joints (fixed, not driven):** `wrist_roll`, `shoulder_pan` → `0.0`
  (calibrated centre / neutral).
- **Robot ranges** = the calibrated limits from `spectre.json` (via
  `JOINT_LIMITS`). Because your calibration is symmetric (`±`), a human extreme
  maps onto a robot extreme — so a straight arm may not look "straight" on the
  robot. Validate this in the preview; flip any reversed joint with `INVERT`.

## Where the knobs live

| What | File | Name |
|------|------|------|
| Controlled / held joints | `mapping.py` | `CONTROLLED`, `HELD` |
| Robot target range per joint | `mapping.py` | `ROBOT_RANGE` (= `JOINT_LIMITS`) |
| Per-joint direction flip | `mapping.py` | `INVERT` |
| Human range (until calibrated) | `mapping.py` | `DEFAULT_HUMAN_RANGE` |
| Saved calibration file | `mapping.py` | `CALIBRATION_PATH` (`mirror_calibration.json`) |
| Robot port / id | `server.py` | `ROBOT_PORT`, `ROBOT_ID` |
| Tracking-quality floors | `server.py` | `MIN_ARM_VISIBILITY`, `MIN_HAND_CONFIDENCE` |
| Robot read-back rate | `server.py` | `ROBOT_READ_HZ` |
| Gripper pinch thresholds | CLI | `--gripper-closed`, `--gripper-open` |

## Safety notes

- One process owns the serial port; don't run another LeRobot tool meanwhile.
- Every command goes through the safety core — clamp + velocity limit are never
  bypassed. Bad/again-lost frames HOLD the last good target (no guessing).
- Smoothing is the vision One-Euro filter (applied to the human angles inside the
  tracker); disable with `--no-smooth`.
