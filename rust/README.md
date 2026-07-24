# Rust STS3215 driver + force-feedback teleop

Standalone Rust talking directly to the Feetech STS3215 servo bus. No LeRobot,
no Python.

- **`sts3215/`** — driver library for the servo protocol. Reusable on its own.
- **`ff-teleop/`** — bilateral force-feedback teleoperation built on the driver.

## Status

Phases 1–8 (plus 2.5) done: the full driver, the force-feedback control loop
(`control.rs`), and a validated one-joint bilateral run on real hardware.
106 tests, all offline; the loop is also tested end-to-end against a pair of
mock buses. Phases 3, 4, 6, 8 validated on the arms.

At 20 Hz on one joint, measured cycle *work* was 0.8 / 1.1 / 1.4 ms
(min/mean/max) over 204 cycles with zero errors — the bus is idle ~98% of each
50 ms period, so there is large headroom for higher rates (phase 9).

Beyond the original 9 phases, the loop now also **mirrors** (leader → follower
position tracking), making it real bilateral teleop — validated on one joint
(wrist_flex): the follower tracks the leader and its resistance is felt back on
the leader. Direction was correct as-mapped (both arms `drive_mode: 0`, no
inversion needed). Scaling the mirror to all joints is the remaining work.

Phases 3–4 are validated on real hardware: reading one servo, and a bounded
`move-joint` write (wrist_flex moved +100 counts and held, then `limp`
released it). `move-joint` and `limp` are the write path.

### Motion safety (`move-joint`)

- **Relative only.** You command a signed `--delta` from the joint's *present*
  position; there is no absolute-jump form.
- **Two independent clamps** (`plan_step`, exhaustively unit-tested): `--delta`
  is capped to `±--max-step` (default 100 counts ≈ 8.8°) *before* anything is
  sent, and the result is clamped into the encoder's `0..=4095`.
- **No jump on torque-enable.** The goal is pinned to the present position,
  torque is enabled (so it holds where it is), *then* the real target is
  written. Torque is never enabled against a stale goal.
- **Nonzero profile speed** is mandatory — on this firmware a zero
  `Goal_Velocity` can mean "as fast as possible".
- **Fail limp.** Ctrl-C or any bus fault disables torque before returning. On
  success it holds at the target; `--limp-after` releases instead.
- `limp` is the standalone STOP path: disable torque on one joint.

## Build and test

The toolchain was installed with `--no-modify-path`, so `cargo` may not be on
your `PATH`. Either `. "$HOME/.cargo/env"` first, or use the full path.

```bash
cd rust
cargo build
cargo test            # 16 tests, all offline
cargo clippy --all-targets
```

### Offline builds cannot touch hardware

`serialport` is behind the `serial` feature, and `serial.rs` is the only module
that can open a port:

```bash
cargo test -p sts3215 --no-default-features
```

That build does not link `serialport` at all — verify with
`cargo tree -p sts3215 --no-default-features | grep serialport` (no matches).
This is the offline test path: it is not merely that hardware access is
*avoided*, it is not compiled in.

## Running

Serial ports are **single-owner**. Nothing else — no `lerobot-*` command, no
`teleop.robot.server`, no `teleop.mirror.server` — may hold these ports while
this runs.

Port paths are required arguments with no defaults, so nothing can connect to
the wrong device by falling back to a baked-in path.

```bash
# read-only; opens nothing
cargo run -p ff-teleop -- list-ports

# validate config; opens nothing, moves nothing
cargo run -p ff-teleop -- check \
    --leader-port   /dev/tty.usbmodem5B3D0466471 \
    --follower-port /dev/tty.usbmodem5B3D0486331 \
    --rate-hz 20

# opens the bus, but only ever reads
cargo run -p ff-teleop -- ping --port /dev/tty.usbmodem5B3D0486331 --id 3
cargo run -p ff-teleop -- read-position --port /dev/tty.usbmodem5B3D0486331 --id 3 --watch
cargo run -p ff-teleop -- read-load --port /dev/tty.usbmodem5B3D0486331 --id 3 --watch
cargo run -p ff-teleop -- scan --port /dev/tty.usbmodem5B3D0486331 --watch   # all 6 joints
```

| arm | role | port | supply |
|---|---|---|---|
| phantom | leader (you hold it) | `/dev/tty.usbmodem5B3D0466471` | 5V |
| spectre | follower (it moves) | `/dev/tty.usbmodem5B3D0486331` | 12V |

## Protocol

Dynamixel-1.0-style framing, protocol version 0, **1 Mbps**, 8N1:

```
request:  FF FF  ID  LEN  INST  PARAM...  CHK
status:   FF FF  ID  LEN  ERR   PARAM...  CHK

LEN = param count + 2
CHK = !(sum of bytes from ID through the last param) & 0xFF
```

Instructions: `PING=1  READ=2  WRITE=3  SYNC_READ=0x82  SYNC_WRITE=0x83`.
Broadcast id `0xFE`; ids `0..=252` assignable; `253` reserved.

Multi-byte values are **little-endian** — this is where the STS series diverges
from Feetech's older SCS series. A big-endian read returns plausible-looking
garbage rather than failing outright, so it is an expensive mistake to make.

### Registers

Transcribed from the Feetech STS/SMS e-manual, cross-checked against two
independent copies already vendored in this repo:

- `rl-sim/lerobot/src/lerobot/motors/feetech/tables.py` (`STS_SMS_SERIES_CONTROL_TABLE`)
- `.venv/lib/python3.12/site-packages/scservo_sdk/` (Feetech's own SDK)

| register | addr | size | use |
|---|---|---|---|
| `Torque_Enable` | 40 | 1 | STOP / go-limp path |
| `Goal_Position` | 42 | 2 | commanded position |
| `Goal_Velocity` | 46 | 2 | speed cap |
| `Torque_Limit` | 48 | 2 | the force-feedback knob |
| `Present_Position` | 56 | 2 | joint angle |
| `Present_Load` | 60 | 2 | load feedback, sign-magnitude bit 10 |
| `Present_Voltage` | 62 | 1 | monitoring |
| `Present_Temperature` | 63 | 1 | monitoring |
| `Present_Current` | 69 | 2 | alternative force signal |

Resolution is 4096 counts/rev. Several fields use **sign-magnitude**, not two's
complement — sign bit 10 for `Present_Load`, 11 for `Homing_Offset`, 15 for
velocity. Decoding one as two's complement gives a plausible wrong number
rather than an obvious failure, so the `SignMag<BIT>` type handles it in one
place.

Registers carry their value type and access mode in the type, so writing to
`Present_Position`, or writing a `u16` to the one-byte `Torque_Enable`, are
compile errors. Both are pinned by `compile_fail` doctests in `registers.rs`.

### Golden test vectors

The expected byte strings in `protocol.rs` are not hand-computed. They are the
literal output of Feetech's own SDK (`scservo_sdk`, vendored in `.venv`), run
against a stub port that records writes instead of opening hardware:

```bash
.venv/bin/python rust/tools/gen_golden_vectors.py
```

Re-run that to regenerate or audit them. It opens nothing.

## `Present_Load` reflects motor effort, not external force

Validated on hardware (phase 6): a **limp** servo reads load ≈ 0 no matter how
hard the joint is pushed, because `Present_Load` reports the motor's own PWM
duty / current, not the force on the output shaft. A limp motor exerts nothing,
so it reads nothing. Load only becomes a usable force signal when torque is
**ON and the servo is holding a position** — then external force creates a
position error the servo fights, and load rises with it. Measured: holding
against a mere 6-count error produced −36 load; the same joint limp read 0.

This shapes Part 2 directly: the **follower must have torque enabled and be
holding position** for its load to mean anything. Reading load off a limp
follower would feed the leader pure zero.

## Two channels: mirror + resistance

The `run` loop now does real bilateral teleop, not just force reflection:

1. **Position (leader → follower).** Each cycle the follower is commanded to
   track the leader's pose, mapped range-to-range through both arms'
   calibration (`calib.rs` `PositionMap`, since the two arms have different
   calibrated travel). The follower goal is slew-limited (`--follower-max-step`,
   default 40 counts/cycle), so it walks toward the leader and one bad leader
   read can only nudge it.
2. **Force (follower → leader).** Because the follower is tracking under torque,
   when it meets resistance and can't keep up, its `Present_Load` rises — and
   that drives the leader's `Torque_Limit` (stiffness). See below.

A **load deadband** (`--deadband`) is essential here: the follower always
registers some load just tracking the leader (its own inertia, friction,
gravity), and without the deadband that effort stiffens the leader whenever you
move it — it feels "always stiff". Only load *beyond* the deadband — genuine
external resistance — adds stiffness. Measured free-tracking load is ~±20–60;
forcing a joint is ±150–700, so a deadband around 120 cleanly separates them.
Pair it with a low `--base-limit` so the leader is slack at rest.

So: move the leader, the follower mirrors it; when the follower pushes into
something, you feel it as resistance on the leader.

Calibration is read from the LeRobot JSON files (`--leader-calib`,
`--follower-calib`, defaulting to `~/.cache/huggingface/lerobot/calibration/…`).

## Force feedback: what it actually is

The STS3215 runs its own internal position loop; it is not a torque-controlled
motor. So this is a *resistance* rendering, not true haptics:

1. Leader torque enabled, `Goal_Position` continuously set to the leader's own
   `Present_Position` — a "stay where you are" null command.
2. `Torque_Limit` set proportional to the follower's measured `Present_Load`.

Near-zero load leaves the leader effectively limp and free to move. Rising load
raises the limit, so the leader increasingly resists being moved. It can oppose
your motion but cannot push back with a commanded force, and it has no
directionality. The leader also runs at 5V, so the ceiling on renderable force
is set by hardware, not code.

`Torque_Limit` is therefore both the effect and the primary safety clamp.

## Safety

- Port paths are required arguments — no defaults, no autodetection.
- Clamps seed from the real calibration in
  `~/.cache/huggingface/lerobot/calibration/`, applied as the last step before
  any write so no code path bypasses them. Note both arms' `wrist_roll` is
  recorded as an uncalibrated full 0–4095 and has no meaningful clamp.
- Homing offsets differ per arm, so leader↔follower mapping goes through
  normalized units, never raw ticks.
- Per-cycle rate limiting on commanded deltas.
- Bus errors hold last-good state and never write; repeated errors trip
  torque-off.
- Torque-off on `Drop` and on Ctrl-C, so panic and interrupt both land limp.
- Loop rate is clamped to 1–200 Hz and starts at 20 Hz. Raise only after
  measuring jitter.

## Phases

| # | mode | deliverable |
|---|---|---|
| 1 | offline | workspace, transport trait, mock, CLI skeleton ✅ |
| 2 | offline | framing, checksum, parsing + golden-vector tests ✅ |
| 2.5 | offline | `Bus` transactions: deadlines, retries, fail-safe ✅ |
| 3 | **hardware** | read `Present_Position`, one servo ✅ |
| 4 | **hardware** | write `Goal_Position`, one joint, small range ✅ |
| 5 | offline | `Present_Load`, sync-read/write ✅ |
| 6 | **hardware** | load reads + 6-joint sync ops ✅ |
| 7 | offline | control loop, mapping, clamps, STOP — against the mock ✅ |
| 8 | **hardware** | force feedback, one joint, 20 Hz ✅ |
| 9 | **hardware** | all joints, timing study, rate tuning |
