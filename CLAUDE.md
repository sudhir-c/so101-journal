# so-101-arm

SO-101 robot arm project. The teleoperation code lives under `teleop/` in three
parts — two standalone tools plus a **mirror** that links them:

| | what | entry point (run from repo root) |
|---|---|---|
| **robot** | serial control of the `spectre` SO-101 follower arm | `python -m teleop.robot.server` — `teleop/robot/{control,server}.py`, `teleop/robot/README.md` |
| **vision** | webcam → arm/hand pose → joint values. **No hardware at all.** | `python -m teleop.vision.server` — `teleop/vision/`, `teleop/vision/README.md` |
| **mirror** | live: your arm (vision) → the follower (robot), one process | `python -m teleop.mirror.server` — `teleop/mirror/`, `teleop/mirror/README.md` |

The mirror **imports** the other two (it does not fork them), so **robot** and
**vision** still run independently and unchanged. The mirror is the sole serial
owner while it runs. See "Mirror" below.

`teleop/README.md` is the overview + how-to-run for all three. `teleop/scripts/`
holds helpers: `download_models.sh` (fetch MediaPipe `.task` bundles — required
before vision/mirror run), `cam_viz.py` (quick OpenCV camera preview), and
`teleop.sh` (LeRobot **physical leader→follower** teleop: `so101_leader`
"phantom" → `spectre` — a *separate* serial owner, not the mirror).

(Outside `teleop/`: `journal/` is the write-up, the `*_policy/` dirs and
`outputs/` are training artifacts, top-level `scripts/` holds journal video
tooling, e.g. `video_compressor.py`.)

---

## Commits

- **Never add Claude as a co-author.** Do not append a `Co-Authored-By: Claude`
  trailer (or any Claude/AI attribution) to commit messages.

---

## Environment — read this before installing anything

- Python 3.12, venv at `.venv/`, macOS arm64 (M4 Pro).
- **The venv is uv-managed and has NO `pip` module.** `python -m pip install ...`
  fails with `No module named pip`. Use:
  ```bash
  uv pip install --python .venv/bin/python <pkg>
  ```
- **Always invoke the venv explicitly** (`.venv/bin/python`, or `python -m uvicorn`).
  A bare `python`/`uvicorn`/`pip` may resolve to Homebrew's and import the wrong
  numpy/torch.
- The venv holds torch + LeRobot. Before adding packages, `--dry-run` the install
  and confirm it only *adds* — a resolver downgrade of `numpy`/`protobuf` would
  break the policy training setup.

---

## Vision half (`teleop/vision/`)

Run from the repo root (so `teleop` imports as a package):

```bash
.venv/bin/python -m teleop.vision.server --snapshot                  # which camera index is you?
.venv/bin/python -m teleop.vision.server --camera 1 --side right     # → http://127.0.0.1:8080
```

Four values, all computed in **2D image-plane pixel space** (MediaPipe `z` is
ignored — monocular depth is unreliable):

| value | range | convention |
|---|---|---|
| `shoulder_lift` | 0–180° | hanging 0 · horizontal 90 · straight up 180 |
| `elbow_flex` | 0–180° | **straight = 180** · right angle 90 |
| `wrist_flex` | 0–180° | **in line = 180** · bent 90 |
| `gripper` | 0.0–1.0 | pinched 0 · wide open 1 |

`wrist_roll` is **deliberately not implemented** — forearm rotation about its own
axis barely moves the 2D landmarks under a single in-plane camera, so any value
would be noise. Do not "add" it without a second camera or an IMU.

### Gotchas that will cost you an hour each

1. **`mp.solutions` DOES NOT EXIST.** mediapipe 0.10.35 removed the legacy API
   entirely — `pose`, `hands` **and `holistic`** are all gone; the module top
   level is just `Image` / `ImageFormat` / `tasks`. Every old tutorial fails here.
   Use the **Tasks API** (`PoseLandmarker`, `HandLandmarker`, `VIDEO` mode).
   Holistic-in-one-pass is not available, so Pose and Hands run as two landmarkers.

2. **Model weights are not in the wheel.** The Tasks API needs `.task` bundles.
   They're gitignored (22 MB). A fresh clone must run:
   ```bash
   ./teleop/scripts/download_models.sh
   ```

3. **Camera indices are unstable, and index 0 is not you.** This Mac exposes an
   OBS Virtual Camera, the built-in camera, and an iPhone Continuity camera.
   `system_profiler`'s device order does **not** match OpenCV's capture indices,
   and the mapping was observed changing between runs. An idle OBS virtual camera
   **returns black frames rather than failing**, so "does index N work?" ≠ "is
   index N me?". Never hardcode a name→index map. Use `--snapshot` and *look*.
   Empirically **index 1 = the built-in webcam** (the default).

4. **Angles must be aspect-ratio corrected.** MediaPipe normalizes x by width and
   y by height *independently*, so on a 16:9 frame the axes have different pixel
   scales and an angle read straight off normalized coords is skewed (a true 45°
   reads ~28°). `compute_arm_state(..., image_size=(w, h))` undoes this. Pinned by
   a regression test — if you touch the math, keep it passing.

5. **Mirroring flips handedness.** The feed is a selfie-view flip by default, so
   MediaPipe — which labels by the anatomy it *sees* — perceives your right arm as
   a left arm. `ArmTracker.mp_side` resolves this. Pass the operator's *anatomical*
   side; don't second-guess it.

6. The hand is matched to the arm by **proximity to the pose wrist**, not by
   MediaPipe's handedness label (which flips under mirroring and misfires on
   rotated hands).

### Architecture — keep this seam

`teleop/vision/pose_math.py` is the **reusable core**: `compute_arm_state()` is
pure, stateless, and imports **nothing but numpy** — no cv2, no mediapipe, no
FastAPI. `teleop/vision/__init__.py` deliberately does *not* import `tracker.py`
so that `from teleop.vision import compute_arm_state` stays clean. **Don't break
that** — it is the whole point of the module.

Smoothing is separate and stateful (`smoothing.py`, a One-Euro filter — not an
EMA, because an EMA forces a choice between jitter-at-rest and lag-in-motion).

```
teleop/vision/pose_math.py   ← reusable, numpy-only. compute_arm_state()
teleop/vision/smoothing.py   ← One-Euro filter (stateful, kept out of the math)
teleop/vision/tracker.py     ← MediaPipe Tasks wrapper; mirroring, hand↔arm matching
teleop/vision/camera.py      ← capture, camera discovery, blank-feed detection
teleop/vision/overlay.py     ← OpenCV drawing
teleop/vision/server.py      ← FastAPI: MJPEG /stream + /api/state
teleop/tests/test_pose_math.py ← 23 tests. Run: .venv/bin/python teleop/tests/test_pose_math.py
```

Nothing is ever silently faked: a value whose landmarks are missing comes back
`None`/`null`, never a fake zero. Confidence is exposed so callers can gate on it;
arm confidence is the **weakest** of shoulder/elbow/wrist, so one joint drifting
out of frame correctly drags the whole reading down.

---

## Mirror (`teleop/mirror/`)

The retarget layer that links vision → robot. One process: capture thread
(`Camera` + `ArmTracker` + `draw_overlay`) + `RobotController` + FastAPI, on
port **8090**. Imports both halves; modifies neither.

- `mapping.py` — the human→robot map, **per joint**. Arm joints use a
  **pivot/gain** form `robot = robot_pivot + gain·(human − human_pivot)` (in
  `PIVOT`); the gripper uses the plain affine (`0..1 → 0..100%`). `wrist_roll` and
  `shoulder_pan` are **held at 0** (`HELD`) — not driven. Every target is clamped
  to `JOINT_LIMITS`. Current tuned values: shoulder_lift `pivot 90→0, gain −1`;
  elbow_flex `90→0, +1`; wrist_flex `180→90, +1`.
- `server.py` — drives the robot only when **ENABLEd**: a `RAMP_SECONDS` (1.5s)
  ramp from the arm's actual pose to the mapped target, then live tracking, every
  frame through `RobotController.step()` (clamp + velocity-limit). Launches in
  **PREVIEW** (no motion). Hold-on-lost-tracking re-sends the last good target,
  never a bad-frame guess. Two cameras: tracking (drives pose) + optional
  **monitor** passthrough (`/video2`, point it at the arm).
- STOP = **go limp** via the additive `RobotController.disable_torque()` /
  `enable_torque()` (see robot half). So an extended arm can drop on STOP.

Human-vs-robot conventions the map bridges (units, sign, zero point differ):
`shoulder_lift` human 0=hanging/180=up vs robot ±102.7 zero-centred; `elbow_flex`
human 180=straight vs robot 0≈straight ±96.2; `gripper` 0..1 vs 0..100%.

**Not yet built:** Phase 2 **calibration capture** (record real per-joint human
min/max → `mirror_calibration.json`; today `mapping.DEFAULT_HUMAN_RANGE` holds
placeholders, and the pivot joints are calibration-independent anyway). Also
`shoulder_pan` has no vision source, and `shoulder_lift` is unsigned. Keep robot
semantics OUT of `teleop/vision/pose_math.py` (numpy-only reusable core).

---

## Robot half (`teleop/robot/`)

See `teleop/robot/README.md`. Key hazards repeated here because they bite hard:

- **Never run uvicorn with `--reload`** — it watches `.venv/` (thousands of files),
  reloads endlessly, and each reload reconnects the serial port and makes the arm
  go limp. The `python -m teleop.robot.server` entry point runs a single,
  non-reloading server.
- **Sole owner of the serial port.** Only one of `teleop.robot.server`,
  `teleop.mirror.server`, or any `lerobot-*` command may hold it at a time.
- `control.py`'s `stop()`/`resume()`/`step()` are the stable core (the slider UI
  depends on their exact behavior — `stop()` **freezes**, does not go limp). The
  mirror's go-limp STOP uses the **additive** `disable_torque()`/`enable_torque()`;
  don't change `stop()` semantics to add torque-off — keep it additive.
- Ports: robot UI **8000**, vision **8080**, mirror **8090**.
