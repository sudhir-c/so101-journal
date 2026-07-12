# 12 - The Full Mirror

This is where the two halves come together: the slider dashboard's safety core from [entry 10](10_teleopdashboard.md) and the arm tracking from [entry 11](11_handtracking.md), wired into a single program that lets me move Spectre by moving my own arm in front of a webcam.

Structurally it's one process that imports both of the earlier pieces without changing either — the standalone slider dashboard and the standalone tracker still run on their own exactly as before. The mirror just borrows their guts: the tracker to measure my arm, and the robot controller (with all its safety checks) to move Spectre. Since it holds the same serial port, only one of the three can run at a time.

## Bridging two different arms

My arm and Spectre's joints don't share units, sign, or zero point. My elbow reads 180° when it's straight; Spectre's elbow is about 0° when straight and can bend both ways. So each joint gets a tiny linear map that I tuned by eye:

```python
# teleop/mirror/mapping.py  — robot = robot_pivot + gain * (human - human_pivot)
"shoulder_lift": {"human_pivot": 90.0,  "robot_pivot": 0.0,  "gain": -1.0},  # arm horizontal -> 0°, raise -> negative
"elbow_flex":    {"human_pivot": 90.0,  "robot_pivot": 0.0,  "gain":  1.0},
"wrist_flex":    {"human_pivot": 180.0, "robot_pivot": 90.0, "gain":  1.0},
```

Two numbers per joint absorb range, offset, and direction all at once — if a joint mirrors backwards, I flip the gain and it's fixed. I drive shoulder lift, elbow, wrist, and the gripper; wrist roll and the base rotation I just pin at a fixed safe angle for now.

I tuned every one of these in a **preview mode** that shows my webcam, the live human angles, the mapped robot targets, and a little 2D stick-figure of the pose the robot *would* take — all without moving a single motor. Getting the mapping right on a cartoon before trusting it on real hardware saved me from a few would-be lurches. I brought the joints online one at a time — shoulder, then elbow, then wrist and gripper — validating each in the preview before adding the next.

## Making it safe to actually move

The moment webcam data starts driving a real motor is exactly where you want to be paranoid. On top of the clamping and velocity limit already baked into the robot controller from entry 10, the mirror adds a few of its own:

**It starts in preview and never moves until I say so.** ENABLE is an explicit, deliberate action — nothing twitches on launch.

**Ramp on enable.** My arm is almost never in the same pose as the robot when I hit go. So instead of snapping, it interpolates from Spectre's current pose to my mapped pose over about 1.5 seconds, *then* begins live tracking:

```python
alpha = (now - ramp.t0) / RAMP_SECONDS                    # 0 → 1 over ~1.5s
cmd   = {j: start[j] + alpha * (goal[j] - start[j]) for j in goal}
robot.step(cmd)                                           # still through the same safety core
```

**Hold on lost tracking.** If MediaPipe loses my arm or hand — I step out of frame, the light drops — the mirror keeps re-sending the last *good* pose rather than a zero or a guess, so the arm freezes and then picks back up smoothly when I return. Those `None`s from entry 11 never reach a motor.

**STOP goes limp.** Unlike the slider dashboard, where STOP freezes the arm in place, here the big red button cuts torque entirely so the arm goes slack and I can grab it — and re-enabling stages the current position as the goal first, so it holds where it is instead of snapping back to a stale target.

I also added a second, optional camera feed to the dashboard pointed at the robot itself, so I can watch my arm and Spectre side by side while I drive.



https://github.com/user-attachments/assets/b141c2d3-faf0-436c-873e-30b0ae0a26b6



This process was more of a "for-fun" exploration than something strictly related to robot learning, but I thought it was cool so I pursued it. I will try to come back and expand on this work as I get more ideas for other interesting things to do.
