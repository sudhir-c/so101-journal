# Arm + hand pose visualization from a webcam

Standalone computer vision: point a webcam at yourself, and this tracks one arm
and its hand, computes four joint values a robot arm would need, and streams the
annotated video to your browser with the values overlaid.

**No robot, no serial ports, no LeRobot, no hardware.** Pure CV + visualization.

The reusable half is `arm_pose/pose_math.py` -- `compute_arm_state()` depends on
nothing but `numpy`, so a later project can import it as-is.

---

## Quick start

Everything runs out of the existing project venv. Use the venv's `python`
explicitly -- a bare `pip`/`python` on this machine points somewhere else.

```bash
cd /Users/sudhirc/Desktop/Projects/so-101-arm

# 1. Which camera is actually pointed at you?  (see "Camera" below -- this matters)
.venv/bin/python server.py --snapshot

# 2. Run it
.venv/bin/python server.py --camera 1 --side right
```

Then open **<http://127.0.0.1:8000>**.

---

## Camera: pick the right index, don't guess

> **This machine has three cameras and the indices are not stable.**

`system_profiler` lists *OBS Virtual Camera*, *MacBook Pro Camera*, and *Sudhir's
iPhone Camera* -- but **that order does not match OpenCV's capture indices**, and
the mapping was observed changing between runs as OBS and Continuity Camera come
and go. Two traps follow from this:

- The conventional "camera 0" is **not** your webcam here. Index 0 has variously
  been an inactive virtual camera and a downward desk view.
- An idle OBS Virtual Camera **does return frames** -- they're just solid black.
  So "does index N work?" is not the same question as "is index N me?"

Because of that, the tool never claims to know which name goes with which index.
It gives you two ways to find out for real:

```bash
.venv/bin/python server.py --list-cameras   # which indices deliver a live image
.venv/bin/python server.py --snapshot       # dumps /tmp/camera_N.jpg -- just LOOK
```

At the time of writing, **`--camera 1` is the built-in MacBook camera pointed at
you**, and that is the default. If you ever get a black or wrong stream, re-run
`--snapshot`. The server also prints a loud warning if the camera you picked is
delivering black frames.

The camera index is defined as `DEFAULT_CAMERA_INDEX` in
[`arm_pose/camera.py`](arm_pose/camera.py) and overridable with `--camera`.

---

## Which arm

```bash
.venv/bin/python server.py --side right    # default
.venv/bin/python server.py --side left
```

`--side` is **your anatomical arm**, which is what you actually care about.

The video is mirrored by default (selfie view), so raising your right arm raises
the arm on the right of the picture. Mirroring means MediaPipe -- which labels
landmarks by the anatomy it *sees* -- perceives your right arm as a left arm, so
the code flips the side internally before asking for landmarks. You don't have to
think about it; just say which of *your* arms to track. Pass `--no-mirror` to
disable the flip.

---

## The four values

All computed in **2D image-plane pixel space**. MediaPipe's `z` is deliberately
ignored -- monocular depth is unreliable, and holding your arm parallel to the
image plane is what makes the 2D projection meaningful.

| value | how | sanity check |
|---|---|---|
| `shoulder_lift` | angle of (elbow − shoulder) from straight-down | hanging `0°` · horizontal `90°` · straight up `180°` |
| `elbow_flex` | angle at the elbow between (shoulder − elbow) and (wrist − elbow) | straight `180°` · right angle `90°` |
| `wrist_flex` | angle at the wrist between (elbow − wrist) and (middle-knuckle − wrist) | in line with forearm `180°` · bent `90°` |
| `gripper` | ‖thumb tip − index tip‖ ÷ palm length, mapped to 0–1 | pinched `0.00` · wide open `1.00` |

**`wrist_roll` is deliberately not implemented.** Forearm rotation about its own
axis barely moves the 2D landmarks under a single in-plane camera, so any value
would be noise dressed up as a measurement.

### One note on the `wrist_flex` spec

The brief defined the forearm vector as `F = wrist − elbow` *and* asked for
"in line with forearm ≈ 180°". Those two are inconsistent: with `F` pointing
elbow→wrist, a hand held in line with the forearm gives **0°**, not 180°.

The implementation follows the **sanity targets**, which are also what makes
`wrist_flex` consistent with `elbow_flex`: both measure the interior angle at a
joint, with both vectors pointing *away* from that joint. So we use
`elbow − wrist` and `middle_mcp − wrist`, giving straight = 180°, bent = 90°.
(If you ever want the other convention, it's just `180° − wrist_flex`.)

---

## Tuning the gripper

The raw measurement is scale-invariant by construction:

```
ratio = ‖thumb_tip − index_tip‖ / ‖hand_wrist − middle_mcp‖    # ÷ palm length
```

Dividing by palm length means moving your hand toward or away from the camera
does not change the reading. That ratio is then mapped onto 0–1 by two
thresholds:

```python
# arm_pose/pose_math.py
class GripperConfig:
    closed_ratio: float = 0.15   # at/below this -> 0.0  (pinched shut)
    open_ratio:   float = 1.10   # at/above this -> 1.0  (wide open)
```

They're mirrored as constants at the top of [`server.py`](server.py)
(`GRIPPER_CLOSED_RATIO` / `GRIPPER_OPEN_RATIO`) and overridable per-run:

```bash
.venv/bin/python server.py --gripper-closed 0.20 --gripper-open 0.95
```

**How to tune:** the web UI shows the live **raw pinch ratio** under the gripper
bar. Pinch fully and note the ratio -> that's your `closed_ratio`. Spread thumb
and index wide and note it -> that's your `open_ratio`.

---

## Build phases

The `--phase` flag lets you validate each layer on its own:

```bash
.venv/bin/python server.py --phase 1    # raw webcam only        -> confirm the camera
.venv/bin/python server.py --phase 2    # + arm skeleton         -> confirm arm tracking
.venv/bin/python server.py --phase 3    # + shoulder_lift, elbow_flex
.venv/bin/python server.py --phase 4    # + hand, wrist_flex, gripper   (default)
```

---

## Tracking quality

Both the video overlay and the web UI show a traffic light:

- **green — TRACKING OK** — everything this phase needs is confidently detected
- **amber — PARTIAL** — arm or hand present but not both (the UI names which)
- **red — NO TRACKING** — nothing trustworthy

Arm confidence is the **lowest** visibility across shoulder/elbow/wrist, so a
single joint drifting out of frame correctly drags the whole reading down rather
than being averaged away. Per-landmark visibility is exposed on `/api/state`.

Values still get computed at low confidence -- they're just flagged. Nothing is
silently faked: any value whose landmarks are missing comes back `null`, so you
can always tell "not tracked" from a real zero.

> **Practical tip:** sit far enough back that your whole arm -- shoulder *and*
> elbow *and* wrist -- is inside the frame. Sitting close enough that your elbow
> drops below the bottom edge is the single most common reason for a red light.

---

## Reusing the math (the point of all this)

`compute_arm_state` is pure, stateless, and imports nothing but `numpy` -- no
OpenCV, no MediaPipe, no FastAPI:

```python
from arm_pose import compute_arm_state

state = compute_arm_state(
    pose_landmarks,          # 33 MediaPipe pose landmarks (or None)
    hand_landmarks,          # 21 MediaPipe hand landmarks (or None)
    side="right",
    image_size=(1280, 720),  # REQUIRED for correct angles -- see below
)

state.shoulder_lift   # deg, or None
state.elbow_flex      # deg, or None
state.wrist_flex      # deg, or None
state.gripper         # 0.0-1.0, or None
state.arm_visible     # bool
state.arm_confidence  # 0-1
state.as_dict()       # plain dict
state["elbow_flex"]   # dict-style access also works
```

It duck-types the landmarks: anything with `.x` / `.y` (and optionally
`.visibility`) works, as does a plain `(x, y)` pair.

**`image_size` is not optional in spirit.** MediaPipe normalizes x by width and y
by height *independently*, so on a 16:9 frame the two axes have different pixel
scales and any angle read straight off normalized coordinates is skewed (a true
45° reads as ~28°). Passing `image_size` undoes that. Pass `(1, 1)` only if your
landmarks are already in pixels. There's a regression test pinning this.

Smoothing is kept separate (and stateful) in `arm_pose/smoothing.py`, so the math
stays pure:

```python
from arm_pose import ArmStateSmoother
smoother = ArmStateSmoother()
state = smoother(state, timestamp_seconds)
```

It's a **One-Euro filter**, not a plain EMA: an EMA forces you to choose between
jitter when still and lag when moving. One-Euro adapts its cutoff to the signal's
speed, so readings sit still when you hold a pose but keep up when you move.
`wrist_flex` gets the heaviest smoothing -- it rides on small hand landmarks with
a short lever arm, so a few pixels of noise swing it by degrees. `--no-smooth`
turns it off if you want to see the raw jitter.

---

## Layout

```
arm_pose/
  pose_math.py    <- THE reusable module. numpy only. compute_arm_state().
  smoothing.py    <- One-Euro filter (stateful, kept out of the math)
  tracker.py      <- MediaPipe Tasks wrapper; mirroring + hand<->arm matching
  camera.py       <- capture, camera discovery, blank-feed detection
  overlay.py      <- OpenCV drawing (skeleton, values, quality light)
server.py         <- FastAPI: MJPEG stream + /api/state
static/index.html <- the page you watch
models/           <- MediaPipe .task bundles (downloaded, not in the wheel)
tests/            <- the sanity targets above, as executable tests
```

`arm_pose/__init__.py` intentionally does **not** import `tracker.py`, so
`from arm_pose import compute_arm_state` never drags MediaPipe or OpenCV into a
consuming project.

### Endpoints

| route | what |
|---|---|
| `/` | the page |
| `/stream` | annotated MJPEG |
| `/api/state` | live JSON: the four values, confidences, per-landmark visibility |

---

## MediaPipe API note (this will bite you otherwise)

This project pins **mediapipe 0.10.35**, which has **removed `mp.solutions`
entirely**. `mp.solutions.pose`, `mp.solutions.hands` and `mp.solutions.holistic`
**no longer exist** -- the module's whole top level is now `Image`, `ImageFormat`
and `tasks`. Every tutorial written against the old API will fail here.

Consequences:

- We use the **Tasks API** (`PoseLandmarker`, `HandLandmarker`) in `VIDEO`
  running mode, which also gives free temporal tracking between frames.
- **Holistic-in-one-pass is not on the table**, so Pose and Hands run as two
  separate landmarkers. That's fine, and it buys independent confidence signals
  and independent tuning for each.
- The Tasks API does **not ship model weights in the wheel**. The `.task` bundles
  must be downloaded:

```bash
./scripts/download_models.sh
```

The hand is matched to the arm by **proximity** -- we take whichever detected hand
has its wrist nearest the pose wrist -- rather than by trusting MediaPipe's
handedness label, which flips under mirroring and misfires on rotated hands.

---

## Install

Already done in the project venv, but for the record:

```bash
uv pip install --python .venv/bin/python mediapipe opencv-python fastapi 'uvicorn[standard]'
```

Note this venv is **uv-managed and has no `pip` module**, so `python -m pip
install` fails with `No module named pip`; use `uv pip install --python
.venv/bin/python`. The install only *adds* packages -- `numpy`, `protobuf` and
`torch` are untouched, so the existing LeRobot setup in this venv is unaffected.

## Tests

```bash
.venv/bin/python tests/test_pose_math.py
```

23 tests encoding the sanity targets above, plus scale-invariance of the gripper,
aspect-ratio correction, and graceful degradation when landmarks go missing.
