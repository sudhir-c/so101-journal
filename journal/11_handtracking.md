# 11 - Arm & Hand Tracking

The slider dashboard was a good first step, but the real goal was to drive Spectre with my own arm. Before I could mirror anything, though, I needed to actually measure my arm — turn a webcam image of me into a handful of joint angles I could later hand to the robot. So this entry is the pure computer-vision half: no robot, no serial port, just a webcam and some geometry.

I built it as a standalone tool — point it at yourself and it streams the annotated video to the browser with your joint angles drawn on top, live. That way I could get the vision rock-solid on its own before wiring it to any motors.

## What MediaPipe is doing

[MediaPipe](https://developers.google.com/mediapipe) is Google's open-source framework for real-time perception on video. It ships pre-trained neural-network models for things like body pose, hands, and face tracking that run on-device — CPU or GPU, fast enough for live video — and it hands you clean, labeled keypoints instead of raw pixels. I lean on two of its models: a **Pose Landmarker** and a **Hand Landmarker** (their weights come as `.task` bundles you download once).

Every frame, the Pose Landmarker takes the raw RGB image and returns **33 body landmarks** — nose, shoulders, elbows, wrists, hips, and so on — each as an (x, y) position normalized to the frame (0–1), plus a rough depth z and a *visibility* score for how confident it is that the point is really there. The Hand Landmarker does the same for a hand, returning **21 points**: the wrist, and the joints and tips of every finger. Under the hood these are Google's on-device landmark models (the body one is based on their BlazePose research); from my side, I feed in a frame and get keypoints out.

There's a nice efficiency trick built in. Rather than scanning the whole image every frame, MediaPipe runs a heavier detector *once* to find the person (or hand), then on following frames just **tracks** the keypoints forward from where they were, only re-running the full detector when it loses the target. Running it in this "video" mode makes it both faster and steadier frame-to-frame — the catch is it wants a monotonically increasing timestamp with each frame so it knows their order.

So MediaPipe handles the genuinely hard part — turning pixels into labeled body points. What's left for me is geometry.

## From landmarks to joint angles

Once I have the keypoints, computing a joint angle is just 2D vector math. Everything is computed in the **image plane** — the (x, y) pixels — and I deliberately ignore MediaPipe's z, because monocular depth off a single webcam is too noisy to trust. I rely on holding my arm roughly parallel to the camera instead.

Take the elbow. I grab three landmarks — shoulder, elbow, wrist — and measure the interior angle at the elbow, between the vector pointing back up the upper arm and the vector pointing out along the forearm:

```python
upper = shoulder - elbow                 # back up the upper arm
fore  = wrist - elbow                     # out along the forearm
elbow_flex = degrees(atan2(norm(cross(upper, fore)), dot(upper, fore)))
```

I use `atan2(|cross|, dot)` rather than the textbook `acos(dot / |a||b|)` because acos goes numerically unstable near 0° and 180° — which is exactly the straight-arm and fully-folded poses I care most about. The other joints are the same idea with different landmarks: `shoulder_lift` is the angle of the upper arm away from straight-down, and `wrist_flex` is the angle at the wrist between the forearm and the direction to the middle knuckle.

That gives me four values:

- **shoulder_lift** — upper arm from straight down (hanging 0°, horizontal 90°, straight up 180°)
- **elbow_flex** — the angle at the elbow (straight 180°, right angle 90°)
- **wrist_flex** — the angle at the wrist (in line with the forearm 180°, bent 90°)
- **gripper** — how open my hand is, 0 (pinched) to 1 (wide open)

The gripper is a small trick of its own: I take the distance between my thumb tip and index tip and divide it by my palm length (wrist to middle knuckle). Dividing by palm length makes it scale-invariant, so moving my hand toward or away from the camera doesn't change the reading — only actually pinching does.

```
ratio = ‖thumb_tip − index_tip‖ / ‖wrist − middle_knuckle‖
```

I deliberately *didn't* implement wrist roll (rotating the forearm about its own axis). It barely moves any of the 2D landmarks under a single flat camera, so any number I computed would be garbage dressed up as a measurement — better to leave it out than fake it.

All of this geometry lives in one pure function — no OpenCV, no MediaPipe, no web server, just numpy — so it's trivially reusable and testable on its own:

```python
from teleop.vision import compute_arm_state

state = compute_arm_state(pose_landmarks, hand_landmarks, side="right", image_size=(1280, 720))
state.elbow_flex   # degrees, or None if it couldn't be measured
```

The `image_size` there isn't decorative: MediaPipe normalizes x by the width and y by the height *independently*, so on a 16:9 frame the two axes end up with different pixel scales, and an angle read straight off the normalized coordinates comes out skewed. Passing the real frame size back in rescales the axes so the geometry is honest.

https://github.com/user-attachments/assets/PLACEHOLDER-tracking

## Being honest about uncertainty

One principle I stuck to: never silently fake a value. If the landmarks a value needs aren't visible, it comes back `None`, never a fake zero — so a consumer can always tell "not tracked" from a real reading of zero. The tool also exposes a confidence for the arm, taken as the *weakest* of the shoulder/elbow/wrist visibilities, so one joint drifting out of frame correctly drags the whole reading down rather than getting averaged away. That matters a lot for the next step, where these numbers are about to start moving a real motor.

There's also a One-Euro filter smoothing the outputs — it adapts how hard it smooths based on how fast the value is moving, so it kills jitter when I hold still without adding lag when I move quickly. An ordinary moving average forces you to pick one or the other, and neither choice feels good on a live readout.

With my arm reliably measured, the last piece is to map these human angles onto Spectre's joints and let it move — the full product, next entry.
