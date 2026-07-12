# 10 - Slider Teleop Dashboard

I wanted a way to drive Spectre using my own actual hand, just for fun. As a first step I wanted to get some basic teleoperation. So I built a web dashboard: six sliders, one per joint, a live readout of where the arm actually is, and a big STOP button.

The whole thing is one Python process that owns the serial connection to Spectre and serves a FastAPI backend plus a static browser UI. It runs separately from the leader-follower workflow — no `lerobot-teleoperate`, no Phantom — but it's built on the same LeRobot `SO101Follower` class underneath, and it talks to the same Feetech servo bus, so only one of them can hold the port at a time.

## How it works

I wrapped LeRobot's follower in a small controller that exposes basically one method for driving motion. You hand it a dict of target joint angles, and it reads the current position, works out a safe move, and writes it — one serial read, one serial write:

```python
# teleop/robot/control.py
def step(self, targets: dict[str, float]) -> dict[str, float]:
    with self._lock:
        present = self._read_positions_unlocked()
        if not self._stopped and targets:
            action = self._compute_action(targets, present, dt)  # clamp + rate-limit
            self._robot.send_action(action)
        return present
```

The browser talks to this over a WebSocket. As I drag a slider, the page streams the target angle for that joint; the backend runs it through `step()` and streams back the arm's live positions so the readouts stay honest. The message shape is dead simple — `{"angles": {"elbow_flex": 12.3}}` going out, `{"positions": {...}}` coming back.

That last part had a subtle bug that took me a while to pin down. My first version clocked itself off the WebSocket replies — send, wait for the reply, send again — which on localhost spins thousands of times a second. Chrome quietly throttles a loop like that when the tab isn't focused, so my slider drags weren't getting sent... until I opened the dev tools, which disabled the throttling and made the arm suddenly lurch to catch up. Very confusing to debug on real hardware 😅. The fix was to drive the sending from a fixed 30 Hz timer instead, decoupled from the replies, which behaves predictably.

https://github.com/user-attachments/assets/PLACEHOLDER-sliders

## Safety features

This drives real hardware, and a bad command can slam a joint into a hardstop or trip an overload fault, so most of the actual effort went into the safety layer. Every command — from a slider or anything else — funnels through that same `step()` and can't skip the checks.

**Per-joint clamping.** Each target is clamped to that joint's calibrated min/max, read straight from the same `spectre.json` calibration LeRobot generated. The arm physically can't be told to go past its own range of motion.

**A real velocity limit.** This is the one I'm happiest with. Rather than capping how far a joint moves per *command* (which makes the speed depend on how fast messages arrive), I cap it by *time* — degrees per second — measured against the actual elapsed time since the last command, and always relative to where the joint *currently* is:

```python
step_cap = MAX_SPEED_DPS * dt          # 180°/s, scaled by real elapsed time
delta    = max(-step_cap, min(step_cap, clamped - cur))
action[f"{joint}.pos"] = cur + delta   # never more than a hair from where it is now
```

Because it clamps against the present position every tick, yanking a slider across its whole range doesn't fling the arm — it just glides there at a bounded speed.

**No jerk on startup.** When the page loads, the sliders initialize to the arm's *actual* current pose, and the loop only sends commands while I'm actively dragging. So opening the dashboard doesn't snap the arm to some default — it sits exactly where it is until I move a slider.

**Hold on bad input.** If a command is missing or a frame is junk, the controller holds the last good position instead of sending a guess. Nothing half-formed reaches a motor.

**STOP.** The big red button flips a flag that makes the controller reject every incoming command, so the arm freezes wherever it is and stays there — it stays in position-control mode and holds, it doesn't go slack. There's also a re-sync button to snap the sliders back to reality if I ever bump the arm by hand.

This slider dashboard ended up being the backbone for something more fun — driving the arm with my *own* arm through a webcam — but that's the next entry.
