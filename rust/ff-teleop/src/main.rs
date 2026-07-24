//! Bilateral force-feedback teleoperation for the SO-101.
//!
//! Read commands (`ping`, `read-position`) and single-joint motion
//! (`move-joint`, `limp`) are in place. The bilateral control loop arrives in
//! phase 7. Every motion path clamps the target twice and fails limp.

mod calib;
mod control;

use clap::{Args, Parser, Subcommand};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use sts3215::registers::{
    ACCELERATION, GOAL_POSITION, GOAL_VELOCITY, PRESENT_LOAD, PRESENT_POSITION,
    PRESENT_TEMPERATURE, PRESENT_VOLTAGE, RESOLUTION, TORQUE_ENABLE,
};
use sts3215::serial::SerialTransport;
use sts3215::{Bus, ServoId, Transport};

/// Absolute encoder bounds. A commanded position is always clamped into this
/// range regardless of what the caller asks for — the servo's own hard limit.
const POSITION_MIN: u16 = 0;
const POSITION_MAX: u16 = RESOLUTION - 1;

/// Plan a bounded relative move.
///
/// Two independent clamps, both of which must hold:
/// 1. the requested `delta` is capped to `±max_step`, so a single command can
///    never order a large lunge — even a typo of `--delta 100000`;
/// 2. the resulting target is clamped into `[POSITION_MIN, POSITION_MAX]`, so
///    it can never drive past the encoder's range.
///
/// Pure and total, so it is exhaustively unit-tested with no hardware.
fn plan_step(present: u16, delta: i32, max_step: u16) -> u16 {
    let capped = delta.clamp(-(max_step as i32), max_step as i32);
    let target = present as i32 + capped;
    target.clamp(POSITION_MIN as i32, POSITION_MAX as i32) as u16
}

/// Encoder counts to degrees, for human-readable logging.
fn counts_to_deg(counts: u16) -> f32 {
    f32::from(counts) * 360.0 / f32::from(RESOLUTION)
}

/// Loop rates outside this band are refused. The upper bound is deliberately
/// conservative — a bilateral force loop is easiest to destabilise by running
/// it faster than the bus can actually service, and 1 Mbps with four
/// transactions per cycle does not leave much headroom.
const MIN_RATE_HZ: f64 = 1.0;
const MAX_RATE_HZ: f64 = 200.0;

#[derive(Parser, Debug)]
#[command(name = "ff-teleop", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// List serial devices the OS reports. Read-only: opens nothing.
    ListPorts,

    /// Print the resolved configuration and exit without touching hardware.
    ///
    /// Use this to confirm the leader and follower paths are the right way
    /// round before any phase that actually drives the arms.
    Check(RunArgs),

    /// Probe whether a servo answers. Sends only a PING; writes nothing.
    Ping(ProbeArgs),

    /// Read a servo's present position. Read-only — writes nothing, changes no
    /// torque state, so the arm stays exactly as limp or as held as it was.
    ReadPosition(ReadArgs),

    /// Read a servo's present load, signed. Read-only. This is the signal the
    /// force-feedback loop reflects: positive/negative shows load direction,
    /// magnitude is 0..1023. Push the joint by hand and watch it respond.
    ReadLoad(LoadArgs),

    /// Sync-read every joint's position and load in one request per register.
    /// Read-only. Exercises the multi-servo path the control loop depends on.
    Scan(ScanArgs),

    /// Move ONE joint by a small relative amount. The first command that writes
    /// to hardware. Enables torque, moves, then holds at the target; run `limp`
    /// to release. Ctrl-C or any fault goes limp immediately.
    MoveJoint(MoveArgs),

    /// Disable a servo's torque — go limp. The STOP path. The joint will drop
    /// under gravity, so support it first.
    Limp(ProbeArgs),

    /// Run the bilateral force-feedback loop: hold the follower, reflect its
    /// load as resistance on the leader. Drives BOTH arms. Ctrl-C goes limp.
    Run(RunFfArgs),
}

/// Arguments shared by every command that opens a bus.
#[derive(Args, Debug)]
struct ProbeArgs {
    /// Serial device of the arm to talk to. Required; there is no default.
    #[arg(long, value_name = "PATH")]
    port: String,

    /// Servo id, 1..=6 on an SO-101 (1 = shoulder_pan .. 6 = gripper).
    #[arg(long)]
    id: u8,

    /// Bus speed. The SO-101 servos ship at 1 Mbps.
    #[arg(long, default_value_t = 1_000_000)]
    baud: u32,

    /// Per-attempt read deadline.
    #[arg(long, default_value_t = 20)]
    timeout_ms: u64,
}

impl ProbeArgs {
    fn servo(&self) -> Result<ServoId, String> {
        ServoId::new(self.id).map_err(|e| e.to_string())
    }

    /// Open the bus. This is the point at which hardware is touched.
    fn open(&self) -> Result<Bus<SerialTransport>, String> {
        let timeout = Duration::from_millis(self.timeout_ms);
        let transport = SerialTransport::open_with(&self.port, self.baud, timeout)
            .map_err(|e| format!("could not open {}: {e}", self.port))?;
        Ok(Bus::new(transport).with_timeout(timeout))
    }
}

#[derive(Args, Debug)]
struct ReadArgs {
    #[command(flatten)]
    probe: ProbeArgs,

    /// Keep reading until interrupted, instead of reading once.
    #[arg(long)]
    watch: bool,

    /// Delay between reads in watch mode.
    #[arg(long, default_value_t = 100)]
    interval_ms: u64,
}

#[derive(Args, Debug)]
struct LoadArgs {
    #[command(flatten)]
    probe: ProbeArgs,

    /// Keep reading until interrupted, instead of reading once.
    #[arg(long)]
    watch: bool,

    /// Delay between reads in watch mode.
    #[arg(long, default_value_t = 100)]
    interval_ms: u64,
}

#[derive(Args, Debug)]
struct ScanArgs {
    /// Serial device of the arm. Required; there is no default.
    #[arg(long, value_name = "PATH")]
    port: String,

    /// Servo ids to scan, comma-separated. Defaults to the SO-101's 1..=6.
    #[arg(long, value_delimiter = ',', default_value = "1,2,3,4,5,6")]
    ids: Vec<u8>,

    #[arg(long, default_value_t = 1_000_000)]
    baud: u32,

    #[arg(long, default_value_t = 20)]
    timeout_ms: u64,

    /// Keep scanning until interrupted.
    #[arg(long)]
    watch: bool,

    /// Delay between scans in watch mode.
    #[arg(long, default_value_t = 200)]
    interval_ms: u64,
}

impl ScanArgs {
    fn servos(&self) -> Result<Vec<ServoId>, String> {
        self.ids
            .iter()
            .map(|&i| ServoId::new(i).map_err(|e| e.to_string()))
            .collect()
    }

    fn open(&self) -> Result<Bus<SerialTransport>, String> {
        let timeout = Duration::from_millis(self.timeout_ms);
        let transport = SerialTransport::open_with(&self.port, self.baud, timeout)
            .map_err(|e| format!("could not open {}: {e}", self.port))?;
        Ok(Bus::new(transport).with_timeout(timeout))
    }
}

#[derive(Args, Debug)]
struct RunFfArgs {
    /// Leader arm ("phantom") — the one you hold. Required.
    #[arg(long, value_name = "PATH")]
    leader_port: String,

    /// Follower arm ("spectre") — the one that's held and pushed. Required.
    #[arg(long, value_name = "PATH")]
    follower_port: String,

    /// Joints to drive, comma-separated ids. Start with ONE for phase 8.
    #[arg(long, value_delimiter = ',', default_value = "4")]
    joints: Vec<u8>,

    /// Control-loop rate. Start low; raise only after the jitter looks sane.
    #[arg(long, default_value_t = 20.0)]
    rate_hz: f64,

    /// Torque limit at zero load — the leader's resting slackness. Low so the
    /// leader feels free when the follower is not resisting anything.
    #[arg(long, default_value_t = 20)]
    base_limit: u16,

    /// Load magnitude below which no extra stiffness is added. The follower
    /// always registers some load just tracking the leader (its own inertia,
    /// friction, gravity); this filters that out so only genuine external
    /// resistance stiffens the leader. Set just above free-tracking load.
    #[arg(long, default_value_t = 120)]
    deadband: u16,

    /// Torque-limit units added per unit of follower load *beyond* the
    /// deadband.
    #[arg(long, default_value_t = 0.25)]
    gain: f32,

    /// Hard ceiling on leader torque limit — the primary force clamp.
    #[arg(long, default_value_t = 300)]
    max_limit: u16,

    /// Largest torque-limit change per cycle (slew limit).
    #[arg(long, default_value_t = 20)]
    max_delta: u16,

    /// Largest follower-goal change per cycle, in counts. Caps how fast the
    /// follower tracks the leader — and how far one bad leader read could move
    /// it. Start conservative.
    #[arg(long, default_value_t = 40)]
    follower_max_step: u16,

    /// Profile speed written to both arms' Goal_Velocity. Nonzero — a zero can
    /// mean full speed on this firmware. Higher = snappier tracking.
    #[arg(long, default_value_t = 300)]
    speed: u16,

    /// Profile acceleration written to both arms. Low = gentle.
    #[arg(long, default_value_t = 20)]
    accel: u8,

    /// Leader arm calibration JSON. `~` is expanded.
    #[arg(
        long,
        default_value = "~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/phantom.json"
    )]
    leader_calib: String,

    /// Follower arm calibration JSON. `~` is expanded.
    #[arg(
        long,
        default_value = "~/.cache/huggingface/lerobot/calibration/robots/so_follower/spectre.json"
    )]
    follower_calib: String,

    #[arg(long, default_value_t = 1_000_000)]
    baud: u32,

    #[arg(long, default_value_t = 15)]
    timeout_ms: u64,
}

#[derive(Args, Debug)]
struct MoveArgs {
    #[command(flatten)]
    probe: ProbeArgs,

    /// Signed relative move in encoder counts (4096 = one full turn, so ~11.4
    /// counts per degree). Positive and negative both allowed.
    #[arg(long, allow_hyphen_values = true)]
    delta: i32,

    /// Hard cap on move magnitude in counts, applied before anything is sent.
    /// Even `--delta 100000` moves at most this far. 100 counts ≈ 8.8°.
    #[arg(long, default_value_t = 100)]
    max_step: u16,

    /// Profile speed written to Goal_Velocity. MUST be nonzero: on this
    /// firmware a zero speed can mean "as fast as possible". Kept low.
    #[arg(long, default_value_t = 300)]
    speed: u16,

    /// Profile acceleration written to Acceleration. Low = gentle.
    #[arg(long, default_value_t = 20)]
    accel: u8,

    /// How long to wait for the joint to reach the target before reporting.
    #[arg(long, default_value_t = 2500)]
    settle_ms: u64,

    /// Go limp at the end instead of holding at the target.
    #[arg(long)]
    limp_after: bool,
}

#[derive(Parser, Debug)]
struct RunArgs {
    /// Serial device for the leader arm ("phantom") — the arm you hold.
    ///
    /// Required, with no default: naming the device explicitly every time is
    /// what stops this connecting to the wrong bus.
    #[arg(long, value_name = "PATH")]
    leader_port: String,

    /// Serial device for the follower arm ("spectre") — the arm that moves.
    #[arg(long, value_name = "PATH")]
    follower_port: String,

    /// Control loop rate in Hz. Start low; raise only once timing is measured.
    #[arg(long, default_value_t = 20.0)]
    rate_hz: f64,

    /// Upper bound on the resistance commanded to the leader, as a fraction of
    /// its torque range. The primary force-feedback safety clamp.
    #[arg(long, default_value_t = 0.25)]
    max_force: f64,
}

impl RunArgs {
    /// Reject configurations that are unsafe or self-evidently a mistake,
    /// before anything downstream can act on them.
    fn validate(&self) -> Result<(), String> {
        if self.leader_port == self.follower_port {
            return Err(format!(
                "leader and follower are both {:?}; they are separate buses",
                self.leader_port
            ));
        }
        if !(MIN_RATE_HZ..=MAX_RATE_HZ).contains(&self.rate_hz) {
            return Err(format!(
                "rate {} Hz is outside the supported {MIN_RATE_HZ}..={MAX_RATE_HZ} Hz",
                self.rate_hz
            ));
        }
        if !(0.0..=1.0).contains(&self.max_force) {
            return Err(format!(
                "max-force {} is outside 0.0..=1.0",
                self.max_force
            ));
        }
        Ok(())
    }

    fn period(&self) -> Duration {
        Duration::from_secs_f64(1.0 / self.rate_hz)
    }
}

fn main() -> std::process::ExitCode {
    match run(Cli::parse()) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        Command::ListPorts => {
            let ports = sts3215::serial::available_ports().map_err(|e| e.to_string())?;
            if ports.is_empty() {
                println!("no serial ports found");
            } else {
                for p in ports {
                    println!("{p}");
                }
            }
        }
        Command::Check(args) => {
            args.validate()?;
            println!("leader   (phantom): {}", args.leader_port);
            println!("follower (spectre): {}", args.follower_port);
            println!(
                "rate:               {} Hz  (period {:.1} ms)",
                args.rate_hz,
                args.period().as_secs_f64() * 1e3
            );
            println!("max force:          {:.2}", args.max_force);
            println!();
            println!("config OK — no port was opened, nothing was moved.");
        }
        Command::Ping(args) => {
            let servo = args.servo()?;
            let mut bus = args.open()?;
            match bus.ping(servo) {
                Ok(()) => println!("servo {servo} on {} responded", args.port),
                Err(e) => return Err(format!("servo {servo} did not respond: {e}")),
            }
        }
        Command::ReadPosition(args) => read_position(&args)?,
        Command::ReadLoad(args) => read_load(&args)?,
        Command::Scan(args) => scan(&args)?,
        Command::MoveJoint(args) => move_joint(&args)?,
        Command::Limp(args) => {
            let servo = args.servo()?;
            let mut bus = args.open()?;
            bus.set_torque(servo, false)
                .map_err(|e| format!("failed to disable torque on servo {servo}: {e}"))?;
            println!("servo {servo} is now limp (torque OFF)");
        }
        Command::Run(args) => run_force_feedback(&args)?,
    }
    Ok(())
}

fn run_force_feedback(args: &RunFfArgs) -> Result<(), String> {
    use control::{Config, ForceFeedback, ForceMap, LoopStats};

    if args.leader_port == args.follower_port {
        return Err("leader and follower are the same port; they are separate arms".into());
    }
    let joints: Vec<ServoId> = args
        .joints
        .iter()
        .map(|&n| ServoId::new(n).map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;

    // Load both arms' calibration and build a per-joint leader→follower map.
    let leader_calib = calib::load(&calib::expand_home(&args.leader_calib))?;
    let follower_calib = calib::load(&calib::expand_home(&args.follower_calib))?;
    let mut position_maps = Vec::with_capacity(joints.len());
    for &id in &joints {
        let l = leader_calib
            .get(&id.raw())
            .ok_or_else(|| format!("leader calibration has no joint id {}", id.raw()))?;
        let f = follower_calib
            .get(&id.raw())
            .ok_or_else(|| format!("follower calibration has no joint id {}", id.raw()))?;
        let map = calib::PositionMap::new(l, f)
            .ok_or_else(|| format!("joint id {} has a degenerate calibration range", id.raw()))?;
        position_maps.push(map);
    }

    let cfg = Config {
        joints,
        position_maps,
        follower_max_step: args.follower_max_step,
        rate_hz: args.rate_hz,
        map: ForceMap {
            base_limit: args.base_limit,
            deadband: args.deadband,
            gain: args.gain,
            max_limit: args.max_limit,
            max_delta: args.max_delta,
        },
        accel: args.accel,
        speed: args.speed,
        max_consecutive_errors: 5,
    };
    cfg.validate()?;

    let timeout = Duration::from_millis(args.timeout_ms);
    let open = |path: &str| -> Result<Bus<SerialTransport>, String> {
        SerialTransport::open_with(path, args.baud, timeout)
            .map(|t| Bus::new(t).with_timeout(timeout))
            .map_err(|e| format!("could not open {path}: {e}"))
    };
    let leader = open(&args.leader_port)?;
    let follower = open(&args.follower_port)?;

    let period = cfg.period();
    println!(
        "force feedback on {} joint(s) at {} Hz (period {:.1} ms)",
        cfg.joints.len(),
        args.rate_hz,
        period.as_secs_f64() * 1e3
    );
    println!("leader   {}\nfollower {}", args.leader_port, args.follower_port);
    println!(
        "move the LEADER by hand — the follower mirrors it. When the follower meets"
    );
    println!("resistance, the leader stiffens. Ctrl-C to stop (both go limp).\n");

    let mut ff = ForceFeedback::new(leader, follower, cfg);
    let stop = stop_flag();

    // Setup energises both arms; from here on any exit path must go limp.
    if let Err(e) = ff.setup() {
        return Err(format!("setup failed (arms set limp): {e}"));
    }

    let mut stats = LoopStats::default();
    let mut tick = 0u64;
    while !stop.load(Ordering::SeqCst) {
        let cycle_start = Instant::now();
        match ff.step() {
            Ok(report) => {
                stats.record(report.duration, report.clean);
                // Log at ~2 Hz to keep the console readable at 20+ Hz.
                if tick % (args.rate_hz.max(1.0) as u64 / 2).max(1) == 0 {
                    let flag = if report.clean { ' ' } else { '!' };
                    let cols: String = report
                        .joints
                        .iter()
                        .map(|j| {
                            format!("j{} f>{:4} L{:+5}>{:3}", j.id.raw(), j.follower_goal, j.load, j.limit)
                        })
                        .collect::<Vec<_>>()
                        .join("  ");
                    println!("{flag} {cols}");
                }
            }
            Err(e) => {
                ff.shutdown();
                return Err(format!("loop stopped after repeated bus errors (arms limp): {e}"));
            }
        }
        tick += 1;
        if let Some(rest) = period.checked_sub(cycle_start.elapsed()) {
            std::thread::sleep(rest);
        }
    }

    ff.shutdown();
    println!(
        "\nstopped. both arms limp. cycles {} ({} with errors), cycle work min/mean/max = {:.1}/{:.1}/{:.1} ms",
        stats.cycles,
        stats.error_cycles,
        stats.min().as_secs_f64() * 1e3,
        stats.mean().as_secs_f64() * 1e3,
        stats.max().as_secs_f64() * 1e3,
    );
    Ok(())
}

fn read_load(args: &LoadArgs) -> Result<(), String> {
    let servo = args.probe.servo()?;
    let mut bus = args.probe.open()?;
    let stop = stop_flag();

    let show = |load: sts3215::SignMag<10>| {
        // A little bar so direction and magnitude read at a glance.
        let mag = (load.0.unsigned_abs() as usize * 20 / 1024).min(20);
        let bar: String = std::iter::repeat_n('#', mag).collect();
        let sign = if load.0 < 0 { '-' } else { '+' };
        println!("load {sign}{:4}  {bar}", load.0.abs());
    };

    if !args.watch {
        let load = bus
            .read(servo, PRESENT_LOAD)
            .map_err(|e| format!("read failed: {e}"))?;
        show(load);
        return Ok(());
    }

    println!("watching servo {servo} load — push the joint by hand; Ctrl-C to stop");
    let interval = Duration::from_millis(args.interval_ms);
    while !stop.load(Ordering::SeqCst) {
        let started = Instant::now();
        match bus.read(servo, PRESENT_LOAD) {
            Ok(load) => show(load),
            Err(e) => eprintln!("read error: {e}"),
        }
        if let Some(rest) = interval.checked_sub(started.elapsed()) {
            std::thread::sleep(rest);
        }
    }
    Ok(())
}

fn scan(args: &ScanArgs) -> Result<(), String> {
    let servos = args.servos()?;
    let mut bus = args.open()?;
    let stop = stop_flag();

    let one_scan = |bus: &mut Bus<SerialTransport>| -> Result<(), String> {
        // One sync-read for all positions, one for all loads — two bus
        // requests total, regardless of joint count.
        let positions = bus
            .sync_read(&servos, PRESENT_POSITION)
            .map_err(|e| format!("sync-read positions failed: {e}"))?;
        let loads = bus
            .sync_read(&servos, PRESENT_LOAD)
            .map_err(|e| format!("sync-read loads failed: {e}"))?;

        println!("{:>4}  {:>10}  {:>8}", "id", "position", "load");
        for ((id, pos), (_, load)) in positions.iter().zip(&loads) {
            let pos_s = pos
                .as_ref()
                .map(|p| format!("{p} ({:.0}°)", counts_to_deg(*p)))
                .unwrap_or_else(|_| "—".into());
            let load_s = load
                .as_ref()
                .map(|l| format!("{:+}", l.0))
                .unwrap_or_else(|_| "—".into());
            println!("{:>4}  {:>10}  {:>8}", id.raw(), pos_s, load_s);
        }
        Ok(())
    };

    if !args.watch {
        return one_scan(&mut bus);
    }

    println!("scanning {} joints — Ctrl-C to stop\n", servos.len());
    let interval = Duration::from_millis(args.interval_ms);
    while !stop.load(Ordering::SeqCst) {
        let started = Instant::now();
        if let Err(e) = one_scan(&mut bus) {
            eprintln!("{e}");
        }
        println!();
        if let Some(rest) = interval.checked_sub(started.elapsed()) {
            std::thread::sleep(rest);
        }
    }
    Ok(())
}

/// Install a Ctrl-C handler that flips a shared flag. The move loop polls it
/// and shuts down cleanly (limp) rather than being killed mid-transaction.
fn stop_flag() -> Arc<AtomicBool> {
    let flag = Arc::new(AtomicBool::new(false));
    let handler_flag = Arc::clone(&flag);
    // A second Ctrl-C after this is armed will still hard-kill, which is the
    // behaviour we want as a last resort.
    let _ = ctrlc::set_handler(move || handler_flag.store(true, Ordering::SeqCst));
    flag
}

/// How a move ended, which decides the safe end state.
enum MoveEnd {
    /// Reached (or settled near) the target with no fault.
    Settled,
    /// Ctrl-C was pressed during the move.
    Interrupted,
}

fn move_joint(args: &MoveArgs) -> Result<(), String> {
    let servo = args.probe.servo()?;
    let mut bus = args.probe.open()?;
    let stop = stop_flag();

    // Read where the joint actually is, and plan a bounded move from there.
    let present = bus
        .read(servo, PRESENT_POSITION)
        .map_err(|e| format!("could not read present position: {e}"))?;
    let target = plan_step(present, args.delta, args.max_step);

    println!(
        "servo {servo}: present {present} counts ({:.1}°)",
        counts_to_deg(present)
    );
    println!(
        "target        {target} counts ({:.1}°)  —  net move {:+} counts",
        counts_to_deg(target),
        target as i32 - present as i32
    );
    if target == present {
        println!("already at target after clamping; nothing to do.");
        return Ok(());
    }

    // Do the move; on ANY failure, force the joint limp before returning, so a
    // fault never leaves a servo energised in an unknown state.
    match run_move(&mut bus, servo, present, target, args, &stop) {
        Ok(MoveEnd::Settled) if !args.limp_after => {
            let reached = bus.read(servo, PRESENT_POSITION).unwrap_or(target);
            println!("holding at {reached} counts (torque ON). Run `limp` to release.");
            Ok(())
        }
        Ok(MoveEnd::Settled) => {
            bus.set_torque(servo, false)
                .map_err(|e| format!("move done but failed to go limp: {e}"))?;
            println!("move complete; joint is now limp (torque OFF).");
            Ok(())
        }
        Ok(MoveEnd::Interrupted) => {
            let _ = bus.set_torque(servo, false);
            Err("interrupted — joint set limp".into())
        }
        Err(e) => {
            // Best-effort limp; report the original fault regardless.
            let _ = bus.set_torque(servo, false);
            Err(format!("move failed ({e}); joint set limp"))
        }
    }
}

/// The write sequence itself. Ordered so torque is only ever enabled while the
/// goal equals the present position — enabling torque then never causes a jump.
fn run_move<T: Transport>(
    bus: &mut Bus<T>,
    servo: ServoId,
    present: u16,
    target: u16,
    args: &MoveArgs,
    stop: &Arc<AtomicBool>,
) -> Result<MoveEnd, String> {
    let map = |what: &str, e: sts3215::Error| format!("{what}: {e}");

    // 1. Pin the goal to where the joint already is, so step 3 can't lunge.
    bus.write(servo, GOAL_POSITION, present)
        .map_err(|e| map("write goal=present", e))?;
    // 2. Gentle motion profile. Nonzero speed is a safety requirement here.
    bus.write(servo, ACCELERATION, args.accel)
        .map_err(|e| map("write acceleration", e))?;
    bus.write(servo, GOAL_VELOCITY, args.speed)
        .map_err(|e| map("write goal velocity", e))?;
    // 3. Enable torque — holds at `present`, no motion yet.
    bus.set_torque(servo, true)
        .map_err(|e| map("enable torque", e))?;
    // 4. Command the real target.
    bus.write(servo, GOAL_POSITION, target)
        .map_err(|e| map("write goal=target", e))?;

    // 5. Watch it converge, printing progress, until settled or timed out.
    let deadline = Instant::now() + Duration::from_millis(args.settle_ms);
    loop {
        if stop.load(Ordering::SeqCst) {
            return Ok(MoveEnd::Interrupted);
        }
        let pos = bus
            .read(servo, PRESENT_POSITION)
            .map_err(|e| map("read during settle", e))?;
        let remaining = target as i32 - pos as i32;
        println!("  at {pos:5} counts ({:+} to go)", remaining);
        if remaining.abs() <= 8 || Instant::now() >= deadline {
            return Ok(MoveEnd::Settled);
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn read_position(args: &ReadArgs) -> Result<(), String> {
    let servo = args.probe.servo()?;
    let mut bus = args.probe.open()?;

    // Report the torque state up front so it is unambiguous whether the arm is
    // holding itself or hanging limp before anyone lets go of it.
    match bus.read(servo, TORQUE_ENABLE) {
        Ok(0) => println!("servo {servo}: torque OFF (limp)"),
        Ok(_) => println!("servo {servo}: torque ON (holding)"),
        Err(e) => println!("servo {servo}: torque state unreadable ({e})"),
    }
    if let (Ok(v), Ok(t)) = (
        bus.read(servo, PRESENT_VOLTAGE),
        bus.read(servo, PRESENT_TEMPERATURE),
    ) {
        println!("servo {servo}: {:.1} V, {t} °C", f32::from(v) / 10.0);
    }
    println!();

    if !args.watch {
        let pos = bus
            .read(servo, PRESENT_POSITION)
            .map_err(|e| format!("read failed: {e}"))?;
        println!("present position: {pos} counts");
        return Ok(());
    }

    println!("watching servo {servo} — Ctrl-C to stop");
    let interval = Duration::from_millis(args.interval_ms);
    // Errors are reported and the loop continues: a single dropped reply on a
    // read-only watch is not worth aborting for, and the count makes a flaky
    // bus visible rather than silently smoothing over it.
    let mut errors = 0u32;
    loop {
        let started = Instant::now();
        match bus.read(servo, PRESENT_POSITION) {
            Ok(pos) => println!("{pos:5} counts"),
            Err(e) => {
                errors += 1;
                eprintln!("read error ({errors}): {e}");
            }
        }
        if let Some(rest) = interval.checked_sub(started.elapsed()) {
            std::thread::sleep(rest);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> RunArgs {
        RunArgs {
            leader_port: "/dev/ttyLEADER".into(),
            follower_port: "/dev/ttyFOLLOWER".into(),
            rate_hz: 20.0,
            max_force: 0.25,
        }
    }

    #[test]
    fn accepts_a_sane_config() {
        assert!(args().validate().is_ok());
    }

    #[test]
    fn rejects_one_port_used_for_both_arms() {
        let mut a = args();
        a.follower_port = a.leader_port.clone();
        assert!(a.validate().is_err());
    }

    #[test]
    fn rejects_rates_outside_the_supported_band() {
        for rate in [0.0, 0.5, 500.0, f64::NAN] {
            let mut a = args();
            a.rate_hz = rate;
            assert!(a.validate().is_err(), "{rate} Hz should be rejected");
        }
    }

    #[test]
    fn rejects_force_limits_outside_unit_range() {
        for f in [-0.1, 1.5, f64::NAN] {
            let mut a = args();
            a.max_force = f;
            assert!(a.validate().is_err(), "max_force {f} should be rejected");
        }
    }

    #[test]
    fn period_matches_rate() {
        let mut a = args();
        a.rate_hz = 50.0;
        assert_eq!(a.period(), Duration::from_millis(20));
    }

    // ---- plan_step: the motion safety clamps ----

    #[test]
    fn plan_step_makes_the_requested_move_when_it_is_small() {
        assert_eq!(plan_step(2000, 80, 100), 2080);
        assert_eq!(plan_step(2000, -80, 100), 1920);
    }

    #[test]
    fn plan_step_caps_an_oversized_delta() {
        // A fat-fingered delta cannot produce a large move.
        assert_eq!(plan_step(2000, 100_000, 100), 2100);
        assert_eq!(plan_step(2000, -100_000, 100), 1900);
        assert_eq!(plan_step(2000, i32::MAX, 100), 2100);
        assert_eq!(plan_step(2000, i32::MIN, 100), 1900);
    }

    #[test]
    fn plan_step_never_exceeds_the_encoder_range() {
        // Even a legal-size step near a limit is clamped to the hard bound.
        assert_eq!(plan_step(4090, 100, 100), POSITION_MAX);
        assert_eq!(plan_step(10, -100, 100), POSITION_MIN);
        assert_eq!(plan_step(POSITION_MAX, 50, 100), POSITION_MAX);
        assert_eq!(plan_step(POSITION_MIN, -50, 100), POSITION_MIN);
    }

    #[test]
    fn plan_step_is_a_no_op_for_zero_delta() {
        assert_eq!(plan_step(1234, 0, 100), 1234);
    }

    #[test]
    fn plan_step_result_is_always_in_range_for_any_input() {
        // Exhaustive-ish sweep: no combination escapes the bounds.
        for present in [0u16, 1, 500, 2048, 4094, POSITION_MAX] {
            for delta in [i32::MIN, -5000, -100, 0, 100, 5000, i32::MAX] {
                for max_step in [0u16, 1, 100, 4095] {
                    let t = plan_step(present, delta, max_step);
                    assert!((POSITION_MIN..=POSITION_MAX).contains(&t));
                    // And it never moves further than max_step from present.
                    assert!((t as i32 - present as i32).unsigned_abs() <= max_step as u32);
                }
            }
        }
    }

    #[test]
    fn counts_to_deg_maps_a_full_turn() {
        assert!((counts_to_deg(RESOLUTION - 1) - 360.0).abs() < 0.2);
        assert!((counts_to_deg(RESOLUTION / 2) - 180.0).abs() < 0.2);
        assert_eq!(counts_to_deg(0), 0.0);
    }
}
