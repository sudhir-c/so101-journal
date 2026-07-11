# so-101-arm

SO-101 robot arm project. Two halves, currently **not connected to each other**:

| | what | entry point |
|---|---|---|
| **robot** | serial control of the `spectre` SO-101 follower arm | `robot_control.py`, `backend.py` (FastAPI + slider UI), `CONTROLLER.md` |
| **vision** | webcam → arm/hand pose → joint values. **No hardware at all.** | `server.py`, `arm_pose/`, `ARM_POSE_README.md` |

The end goal is teleoperation: mirror a human arm onto the robot. The vision half
computes human joint angles; the robot half accepts robot joint commands. **The
retarget layer between them does not exist yet** — see "Next step" below.

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

## Vision half (`arm_pose/` + `server.py`)

```bash
.venv/bin/python server.py --snapshot                  # which camera index is you?
.venv/bin/python server.py --camera 1 --side right     # → http://127.0.0.1:8080
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
   ./scripts/download_models.sh
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

`arm_pose/pose_math.py` is the **reusable core**: `compute_arm_state()` is pure,
stateless, and imports **nothing but numpy** — no cv2, no mediapipe, no FastAPI.
`arm_pose/__init__.py` deliberately does *not* import `tracker.py` so that
`from arm_pose import compute_arm_state` stays clean. **Don't break that** — it is
the whole point of the module.

Smoothing is separate and stateful (`smoothing.py`, a One-Euro filter — not an
EMA, because an EMA forces a choice between jitter-at-rest and lag-in-motion).

```
arm_pose/pose_math.py   ← reusable, numpy-only. compute_arm_state()
arm_pose/smoothing.py   ← One-Euro filter (stateful, kept out of the math)
arm_pose/tracker.py     ← MediaPipe Tasks wrapper; mirroring, hand↔arm matching
arm_pose/camera.py      ← capture, camera discovery, blank-feed detection
arm_pose/overlay.py     ← OpenCV drawing
server.py               ← FastAPI: MJPEG /stream + /api/state
tests/test_pose_math.py ← 23 tests. Run: .venv/bin/python tests/test_pose_math.py
```

Nothing is ever silently faked: a value whose landmarks are missing comes back
`None`/`null`, never a fake zero. Confidence is exposed so callers can gate on it;
arm confidence is the **weakest** of shoulder/elbow/wrist, so one joint drifting
out of frame correctly drags the whole reading down.

---

## Next step: the retarget layer (not built)

The vision values are **raw human joint angles** and are **NOT normalized to the
SO-101's ranges**. Feeding them into `robot_control.set_joints()` directly would be
unsafe. The mismatch is in units, sign, zero point, *and* cardinality:

| | vision emits | SO-101 `JOINT_LIMITS` |
|---|---|---|
| `shoulder_lift` | 0…180° **unsigned**, 0 = hanging | −102.7…+102.7 **signed**, zero-centred |
| `elbow_flex` | 0…180°, **180 = straight** | −96.2…+96.2, **0 ≈ straight** |
| `wrist_flex` | 0…180° | −100.5…+100.5 |
| `gripper` | 0.0…1.0 | 0…100 (percent) |
| `shoulder_pan` | — *not produced* | −102.3…+102.3 |
| `wrist_roll` | — *excluded by design* | −180…+180 |

A hanging arm reads `shoulder_lift: 0°` in vision, but `0` on the robot is
*mid-range* — sent raw, the arm would snap to the middle of its travel. And
`elbow_flex: 180` (straight) would slam into the `+96.2` limit.

Two things to solve before 5-DOF mirroring works:

- **`shoulder_lift` is unsigned** — an arm swung out to the right and one swung
  across the body both read 90°. You need a signed lift, or a separate estimate.
- **`shoulder_pan` has no vision source.** Likely derivable from the
  shoulder-to-shoulder axis, but it is not computed today.

Build this as a **new module** (e.g. `retarget.py`) that takes an `ArmState` and
returns the `{joint: angle}` dict `robot_control.step()` already accepts, clamping
to `JOINT_LIMITS` on the way out. **Do not put robot semantics into
`pose_math.py`** — its independence is what makes it reusable.

---

## Robot half

See `CONTROLLER.md`. Key hazards repeated here because they bite hard:

- **Never run uvicorn with `--reload`** — it watches `.venv/` (thousands of files),
  reloads endlessly, and each reload reconnects the serial port and makes the arm
  go limp.
- The backend is the **sole owner of the serial port**. Do not run
  `lerobot-teleoperate` / `lerobot-record` while it is up.
- `backend.py` (robot UI) defaults to port 8000; `server.py` (vision) to 8080. They
  can run together, but only one process can hold the webcam at a time.
