//! The force-feedback control loop.
//!
//! # The two channels
//!
//! **Position (leader → follower):** each cycle the follower is commanded to
//! track the leader's pose, mapped range-to-range through both arms'
//! calibration ([`PositionMap`]). The follower's goal is slew-limited, so it
//! walks toward the leader rather than lunging.
//!
//! **Force (follower → leader):** Phase 6 established that `Present_Load`
//! reflects the *motor's own effort*, not shaft force — so it only means
//! anything while the servo is torque-enabled and working to hold/reach a
//! position. Because the follower is tracking the leader under torque, when it
//! meets resistance and cannot keep up, its load rises. That load drives the
//! leader's `Torque_Limit`: high load makes the leader stiff to move, low load
//! leaves it slack. The leader is null-commanded to its own present position
//! every cycle so it otherwise feels free.
//!
//! Together: move the leader, the follower follows; when the follower pushes
//! into something, you feel it as resistance on the leader. It is a resistance
//! rendering, not directional pushback (a `Torque_Limit` is a magnitude).
//!
//! The result is a *resistance* rendering, not true haptics: it can oppose
//! motion but cannot push back, and it is non-directional (a `Torque_Limit` is
//! a magnitude ceiling). See the crate README.
//!
//! # Testability
//!
//! [`ForceFeedback`] is generic over two [`Transport`]s, so the whole loop —
//! mapping, clamps, rate limiting, error handling — is exercised offline
//! against a pair of [`MockTransport`](sts3215::MockTransport)s. No hardware is
//! needed to test any of the logic below.

use crate::calib::PositionMap;
use std::time::{Duration, Instant};
use sts3215::registers::{
    ACCELERATION, GOAL_POSITION, GOAL_VELOCITY, PRESENT_LOAD, PRESENT_POSITION, TORQUE_ENABLE,
    TORQUE_LIMIT, TORQUE_LIMIT_MAX,
};
use sts3215::{Bus, Error, ServoId, SignMag, Transport};

/// Move `current` toward `target` by at most `max_delta`.
///
/// The per-cycle slew limit used for both the leader's torque limit and the
/// follower's tracked goal — no value the loop writes ever jumps, it walks.
pub fn saturating_step(current: u16, target: u16, max_delta: u16) -> u16 {
    if target >= current {
        current.saturating_add(max_delta).min(target)
    } else {
        current.saturating_sub(max_delta).max(target)
    }
}

/// Load → `Torque_Limit` mapping, with the safety clamps baked in.
///
/// Pure and fully unit-tested. Every value the loop ever writes to a leader's
/// `Torque_Limit` comes out of here, so the clamps here are the clamps that
/// matter.
#[derive(Debug, Clone, Copy)]
pub struct ForceMap {
    /// Torque limit commanded at zero load. Low, so the leader starts slack.
    pub base_limit: u16,
    /// Load magnitude below which no extra stiffness is added.
    ///
    /// The follower always registers *some* load just tracking the leader —
    /// its own inertia, friction, and gravity. Without a deadband that effort
    /// stiffens the leader whenever you move, which reads as "always stiff".
    /// Only load beyond this threshold — genuine external resistance — adds
    /// stiffness. Set it just above the follower's free-tracking load.
    pub deadband: u16,
    /// Torque-limit units added per unit of load magnitude *beyond* the
    /// deadband.
    pub gain: f32,
    /// Hard ceiling on the commanded torque limit. The primary force clamp —
    /// the leader can never be made stiffer than this however large the load.
    pub max_limit: u16,
    /// Largest change to the torque limit permitted in one cycle. Stops a load
    /// spike from slamming the leader stiff in a single step.
    pub max_delta: u16,
}

impl ForceMap {
    /// The torque limit a given load magnitude maps to, before rate limiting.
    ///
    /// `load_magnitude` is `Present_Load`'s magnitude, `0..=1023`. The result
    /// is clamped into `0..=max_limit`, so it is always a safe value to write.
    pub fn target_limit(&self, load_magnitude: u16) -> u16 {
        // Only load beyond the deadband contributes. Below it, the leader stays
        // at its slack base regardless of the follower's tracking effort.
        let excess = load_magnitude.saturating_sub(self.deadband);
        let raw = f32::from(self.base_limit) + self.gain * f32::from(excess);
        // clamp handles NaN defensively by returning the low bound.
        let clamped = raw.clamp(0.0, f32::from(self.max_limit));
        clamped as u16
    }

    /// Move `current` toward `target` by at most `max_delta`.
    ///
    /// This is the per-cycle slew limit: the loop never jumps the torque limit
    /// straight to the mapped target, it walks toward it.
    pub fn rate_limited(&self, current: u16, target: u16) -> u16 {
        saturating_step(current, target, self.max_delta)
    }
}

/// Everything the loop needs, validated up front.
#[derive(Debug, Clone)]
pub struct Config {
    /// Joints to drive, by id. Leader and follower share id numbering (both
    /// arms have servos 1..=6); a joint is id N on each bus.
    pub joints: Vec<ServoId>,
    /// Leader→follower position map per joint, parallel to `joints`.
    pub position_maps: Vec<PositionMap>,
    /// Largest change to a follower goal per cycle (its slew limit). Caps how
    /// fast the follower tracks — and how far a single bad leader read could
    /// move it.
    pub follower_max_step: u16,
    /// Control-loop rate. Start low; raise only after measuring jitter.
    pub rate_hz: f64,
    /// The load → torque-limit map and its clamps.
    pub map: ForceMap,
    /// Gentle motion profile written to both arms at setup.
    pub accel: u8,
    /// Nonzero — a zero `Goal_Velocity` can mean full speed on this firmware.
    pub speed: u16,
    /// Consecutive fully-failed cycles tolerated before the loop stops itself
    /// and goes limp.
    pub max_consecutive_errors: u32,
}

impl Config {
    /// Reject configurations that are unsafe or nonsensical before the loop
    /// can act on them.
    pub fn validate(&self) -> Result<(), String> {
        if self.joints.is_empty() {
            return Err("no joints selected".into());
        }
        if self.position_maps.len() != self.joints.len() {
            return Err(format!(
                "{} joints but {} position maps",
                self.joints.len(),
                self.position_maps.len()
            ));
        }
        if self.follower_max_step == 0 {
            return Err("follower_max_step must be nonzero, or the follower can never track".into());
        }
        if self.map.max_limit > TORQUE_LIMIT_MAX {
            return Err(format!(
                "max_limit {} exceeds the servo's {TORQUE_LIMIT_MAX}",
                self.map.max_limit
            ));
        }
        if self.map.base_limit > self.map.max_limit {
            return Err(format!(
                "base_limit {} exceeds max_limit {}",
                self.map.base_limit, self.map.max_limit
            ));
        }
        if self.speed == 0 {
            return Err("speed must be nonzero (0 can mean full speed)".into());
        }
        if !(self.rate_hz.is_finite() && self.rate_hz > 0.0) {
            return Err(format!("rate {} Hz is not a positive number", self.rate_hz));
        }
        Ok(())
    }

    /// The target period between cycle starts.
    pub fn period(&self) -> Duration {
        Duration::from_secs_f64(1.0 / self.rate_hz)
    }
}

/// One joint's state after a cycle, for telemetry.
#[derive(Debug, Clone, Copy)]
pub struct JointReport {
    /// The joint's id.
    pub id: ServoId,
    /// Follower load used this cycle (last-good if the read failed).
    pub load: i32,
    /// Torque limit commanded to the leader this cycle.
    pub limit: u16,
    /// Follower goal commanded this cycle (where the follower is being told to
    /// track to).
    pub follower_goal: u16,
    /// Whether this cycle's follower-load read succeeded.
    pub load_ok: bool,
    /// Whether this cycle's leader-position read succeeded.
    pub pos_ok: bool,
}

/// Telemetry from one cycle.
#[derive(Debug, Clone)]
pub struct StepReport {
    /// Per-joint state, in config order.
    pub joints: Vec<JointReport>,
    /// Wall time the cycle's bus work took.
    pub duration: Duration,
    /// True if every read and write this cycle succeeded.
    pub clean: bool,
}

/// Running loop-timing statistics, for the jitter study in phase 9.
#[derive(Debug, Clone, Default)]
pub struct LoopStats {
    /// Cycles counted.
    pub cycles: u64,
    /// Cycles in which at least one bus op failed.
    pub error_cycles: u64,
    min: Option<Duration>,
    max: Duration,
    sum: Duration,
}

impl LoopStats {
    /// Fold one cycle's work-duration in.
    pub fn record(&mut self, duration: Duration, clean: bool) {
        self.cycles += 1;
        if !clean {
            self.error_cycles += 1;
        }
        self.min = Some(self.min.map_or(duration, |m| m.min(duration)));
        self.max = self.max.max(duration);
        self.sum += duration;
    }

    /// Shortest cycle seen.
    pub fn min(&self) -> Duration {
        self.min.unwrap_or_default()
    }

    /// Longest cycle seen — the number that decides the safe rate ceiling.
    pub fn max(&self) -> Duration {
        self.max
    }

    /// Mean cycle work-duration.
    pub fn mean(&self) -> Duration {
        if self.cycles == 0 {
            Duration::ZERO
        } else {
            self.sum / self.cycles as u32
        }
    }
}

/// The bilateral force-feedback loop over a leader and a follower bus.
pub struct ForceFeedback<L: Transport, F: Transport> {
    leader: Bus<L>,
    follower: Bus<F>,
    cfg: Config,
    /// Current torque limit per joint, parallel to `cfg.joints`.
    limits: Vec<u16>,
    /// Current commanded follower goal per joint. Walks toward the mapped
    /// leader position under the slew limit.
    follower_goal: Vec<u16>,
    /// Last-good follower load per joint, held across a failed read.
    last_load: Vec<i32>,
    consecutive_errors: u32,
}

impl<L: Transport, F: Transport> ForceFeedback<L, F> {
    /// Wrap a leader and follower bus. `cfg` must already be [`Config::validate`]d.
    pub fn new(leader: Bus<L>, follower: Bus<F>, cfg: Config) -> Self {
        let n = cfg.joints.len();
        let base = cfg.map.base_limit;
        Self {
            leader,
            follower,
            cfg,
            limits: vec![base; n],
            // Filled from the follower's actual pose in setup(); until then the
            // follower is not commanded anywhere.
            follower_goal: vec![0; n],
            last_load: vec![0; n],
            consecutive_errors: 0,
        }
    }

    /// Prepare both arms: hold the follower in place so its load is real, and
    /// set the leader slack and null-commanded.
    ///
    /// On any failure it goes limp before returning — setup never leaves an arm
    /// energised in a half-configured state.
    pub fn setup(&mut self) -> Result<(), Error> {
        match self.try_setup() {
            Ok(()) => Ok(()),
            Err(e) => {
                self.shutdown();
                Err(e)
            }
        }
    }

    fn try_setup(&mut self) -> Result<(), Error> {
        for (i, &id) in self.cfg.joints.iter().enumerate() {
            // Follower: pin goal to present, gentle profile, torque on. Seed
            // the tracked goal with the actual pose so the first cycle steps
            // from *here*, never lunging from a stale zero.
            let fpos = self.follower.read(id, PRESENT_POSITION)?;
            self.follower_goal[i] = fpos;
            self.follower.write(id, GOAL_POSITION, fpos)?;
            self.follower.write(id, ACCELERATION, self.cfg.accel)?;
            self.follower.write(id, GOAL_VELOCITY, self.cfg.speed)?;
            self.follower.write(id, TORQUE_LIMIT, TORQUE_LIMIT_MAX)?;
            self.follower.set_torque(id, true)?;

            // Leader: pin goal to present, start at the slack base limit,
            // torque on → free to move, resistance governed by torque limit.
            let lpos = self.leader.read(id, PRESENT_POSITION)?;
            self.leader.write(id, GOAL_POSITION, lpos)?;
            self.leader.write(id, ACCELERATION, self.cfg.accel)?;
            self.leader.write(id, GOAL_VELOCITY, self.cfg.speed)?;
            self.leader.write(id, TORQUE_LIMIT, self.cfg.map.base_limit)?;
            self.leader.set_torque(id, true)?;
        }
        Ok(())
    }

    /// Run one control cycle.
    ///
    /// Reads the follower's loads and the leader's positions, computes each
    /// leader torque limit (mapped, then rate limited), then null-commands the
    /// leader goals and writes the limits. Individual read failures degrade to
    /// last-good for that joint; the cycle still completes.
    ///
    /// Returns `Err` only after `max_consecutive_errors` fully-failed cycles in
    /// a row — the signal for the caller to stop and go limp.
    pub fn step(&mut self) -> Result<StepReport, Error> {
        let started = Instant::now();
        let joints = self.cfg.joints.clone();

        // 1. Follower loads. A whole-read failure holds every joint's load.
        let loads = self.follower.sync_read(&joints, PRESENT_LOAD);
        let load_failed = loads.is_err();
        let load_results = loads.unwrap_or_default();

        // 2. Leader positions: drive both the follower's target (mirror) and
        //    the leader's own null command.
        let positions = self.leader.sync_read(&joints, PRESENT_POSITION);
        let pos_failed = positions.is_err();
        let pos_results = positions.unwrap_or_default();

        let mut reports = Vec::with_capacity(joints.len());
        let mut leader_goal_writes: Vec<(ServoId, u16)> = Vec::with_capacity(joints.len());
        let mut limit_writes: Vec<(ServoId, u16)> = Vec::with_capacity(joints.len());
        let mut follower_goal_writes: Vec<(ServoId, u16)> = Vec::with_capacity(joints.len());

        for (i, &id) in joints.iter().enumerate() {
            // Load → leader torque limit (mapped, then slew limited).
            let (load, load_ok) = match load_results.get(i) {
                Some((_, Ok(v))) => {
                    let raw: SignMag<10> = *v;
                    self.last_load[i] = raw.0;
                    (raw.0, true)
                }
                _ => (self.last_load[i], false),
            };
            let target = self.cfg.map.target_limit(load.unsigned_abs().min(u16::MAX as u32) as u16);
            self.limits[i] = self.cfg.map.rate_limited(self.limits[i], target);
            limit_writes.push((id, self.limits[i]));

            // Leader position drives both channels — only when it read cleanly.
            // A failed read must never command a stale or zero goal on either
            // arm; the follower simply holds its last commanded goal.
            let pos_ok = matches!(pos_results.get(i), Some((_, Ok(_))));
            if let Some((_, Ok(pos))) = pos_results.get(i) {
                // Mirror: walk the follower goal toward the mapped leader pose.
                let mapped = self.cfg.position_maps[i].map(*pos);
                self.follower_goal[i] =
                    saturating_step(self.follower_goal[i], mapped, self.cfg.follower_max_step);
                follower_goal_writes.push((id, self.follower_goal[i]));
                // Null-command the leader to its own present position.
                leader_goal_writes.push((id, *pos));
            }

            reports.push(JointReport {
                id,
                load,
                limit: self.limits[i],
                follower_goal: self.follower_goal[i],
                load_ok,
                pos_ok,
            });
        }

        // 3. Command the follower (mirror), then the leader (null goal + limit).
        let follower_write = if follower_goal_writes.is_empty() {
            Ok(())
        } else {
            self.follower.sync_write(GOAL_POSITION, &follower_goal_writes)
        };
        let goal_write = if leader_goal_writes.is_empty() {
            Ok(())
        } else {
            self.leader.sync_write(GOAL_POSITION, &leader_goal_writes)
        };
        let limit_write = self.leader.sync_write(TORQUE_LIMIT, &limit_writes);

        let clean = !load_failed
            && !pos_failed
            && follower_write.is_ok()
            && goal_write.is_ok()
            && limit_write.is_ok()
            && reports.iter().all(|r| r.load_ok && r.pos_ok);

        if clean {
            self.consecutive_errors = 0;
        } else {
            self.consecutive_errors += 1;
            if self.consecutive_errors >= self.cfg.max_consecutive_errors {
                // Surface the most informative underlying error we have.
                return Err(follower_write
                    .err()
                    .or(goal_write.err())
                    .or(limit_write.err())
                    .unwrap_or(Error::Transport(sts3215::TransportError::Timeout)));
            }
        }

        Ok(StepReport { joints: reports, duration: started.elapsed(), clean })
    }

    /// Disable torque on every joint of both arms — go limp.
    ///
    /// Best-effort and infallible: it is the safe-state path, called from error
    /// handlers and shutdown, so it must not itself be able to fail out. Errors
    /// are swallowed because there is nothing safer left to do.
    pub fn shutdown(&mut self) {
        let joints = self.cfg.joints.clone();
        let off: Vec<(ServoId, u8)> = joints.iter().map(|&id| (id, 0u8)).collect();
        // Try the one-shot broadcast first; fall back to per-joint so a single
        // bad servo cannot leave the rest energised.
        if self.follower.sync_write(TORQUE_ENABLE, &off).is_err() {
            for &id in &joints {
                let _ = self.follower.set_torque(id, false);
            }
        }
        if self.leader.sync_write(TORQUE_ENABLE, &off).is_err() {
            for &id in &joints {
                let _ = self.leader.set_torque(id, false);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sts3215::protocol;
    use sts3215::transport::{MockTransport, Reply};

    fn id(n: u8) -> ServoId {
        ServoId::new(n).unwrap()
    }

    fn map() -> ForceMap {
        // deadband 0 keeps the base mapping tests direct; deadband has its own
        // tests below.
        ForceMap { base_limit: 50, deadband: 0, gain: 0.25, max_limit: 300, max_delta: 20 }
    }

    // ---- ForceMap: the safety clamps ----

    #[test]
    fn target_limit_is_base_at_zero_load() {
        assert_eq!(map().target_limit(0), 50);
    }

    #[test]
    fn target_limit_rises_with_load() {
        assert_eq!(map().target_limit(100), 75); // 50 + 0.25*100
        assert_eq!(map().target_limit(400), 150);
    }

    #[test]
    fn target_limit_is_capped_at_max() {
        // 50 + 0.25*1023 = 305.75 -> clamped to 300.
        assert_eq!(map().target_limit(1023), 300);
        assert_eq!(map().target_limit(u16::MAX), 300);
    }

    #[test]
    fn target_limit_never_exceeds_max_for_any_load() {
        let m = map();
        for load in (0..=u16::MAX).step_by(37) {
            assert!(m.target_limit(load) <= m.max_limit);
        }
    }

    #[test]
    fn target_limit_treats_nan_gain_as_the_low_bound() {
        let m = ForceMap { gain: f32::NAN, ..map() };
        assert_eq!(m.target_limit(500), 0);
    }

    #[test]
    fn deadband_keeps_the_leader_slack_under_tracking_effort() {
        // Load at or below the deadband adds nothing — the leader stays at base.
        let m = ForceMap { base_limit: 20, deadband: 100, ..map() };
        assert_eq!(m.target_limit(0), 20);
        assert_eq!(m.target_limit(60), 20); // normal tracking effort: still slack
        assert_eq!(m.target_limit(100), 20); // right at the threshold
    }

    #[test]
    fn deadband_lets_only_excess_load_add_stiffness() {
        // Beyond the deadband, stiffness rises on the *excess*, not the raw load.
        let m = ForceMap { base_limit: 20, deadband: 100, gain: 0.25, ..map() };
        assert_eq!(m.target_limit(200), 45); // 20 + 0.25*(200-100)
        assert_eq!(m.target_limit(500), 120); // 20 + 0.25*(500-100)
    }

    #[test]
    fn rate_limit_walks_up_by_at_most_max_delta() {
        assert_eq!(map().rate_limited(50, 300), 70);
        assert_eq!(map().rate_limited(70, 300), 90);
    }

    #[test]
    fn rate_limit_walks_down_by_at_most_max_delta() {
        assert_eq!(map().rate_limited(300, 50), 280);
        assert_eq!(map().rate_limited(60, 50), 50); // within delta, lands exactly
    }

    #[test]
    fn rate_limit_lands_exactly_when_close() {
        assert_eq!(map().rate_limited(50, 60), 60);
        assert_eq!(map().rate_limited(50, 50), 50);
    }

    #[test]
    fn rate_limit_takes_many_cycles_to_reach_a_far_target() {
        // A full-scale spike cannot slam the leader stiff in one step.
        let m = map();
        let mut cur = 50u16;
        let mut cycles = 0;
        while cur != 300 {
            cur = m.rate_limited(cur, 300);
            cycles += 1;
            assert!(cycles < 100, "should converge");
        }
        assert_eq!(cycles, 13); // (300-50)/20 rounded up
    }

    // ---- Config validation ----

    /// A test position map: leader [1000,3000] → follower [1000,3000], so a
    /// leader position maps to itself (identity within the shared range) —
    /// keeps the mirror arithmetic easy to assert.
    fn test_map() -> PositionMap {
        let leader = crate::calib::JointCalib {
            id: 0,
            drive_mode: 0,
            homing_offset: 0,
            range_min: 1000,
            range_max: 3000,
        };
        PositionMap::new(&leader, &leader).unwrap()
    }

    fn cfg(joints: &[u8]) -> Config {
        Config {
            joints: joints.iter().map(|&n| id(n)).collect(),
            position_maps: vec![test_map(); joints.len()],
            follower_max_step: 50,
            rate_hz: 20.0,
            map: map(),
            accel: 20,
            speed: 300,
            max_consecutive_errors: 5,
        }
    }

    #[test]
    fn config_accepts_a_sane_setup() {
        assert!(cfg(&[4]).validate().is_ok());
        assert!(cfg(&[1, 2, 3, 4, 5, 6]).validate().is_ok());
    }

    #[test]
    fn config_rejects_dangerous_values() {
        let mut c = cfg(&[4]);
        c.joints.clear();
        assert!(c.validate().is_err());

        let mut c = cfg(&[4]);
        c.map.max_limit = 2000; // above the servo's 1000
        assert!(c.validate().is_err());

        let mut c = cfg(&[4]);
        c.map.base_limit = 500;
        c.map.max_limit = 300;
        assert!(c.validate().is_err());

        let mut c = cfg(&[4]);
        c.speed = 0;
        assert!(c.validate().is_err());

        let mut c = cfg(&[4]);
        c.rate_hz = 0.0;
        assert!(c.validate().is_err());
    }

    // ---- Loop, against paired mock buses ----

    /// Frame a status packet the way a servo would.
    fn status(id: u8, params: &[u8]) -> Vec<u8> {
        let mut p = vec![0xFF, 0xFF, id, (params.len() + 2) as u8, 0];
        p.extend_from_slice(params);
        let ck = protocol::checksum(&p[2..]);
        p.push(ck);
        p
    }

    /// A sync-read reply: one status packet per joint, concatenated.
    fn sync_reply(entries: &[(u8, u16)]) -> Vec<u8> {
        entries
            .iter()
            .flat_map(|(id, v)| status(*id, &v.to_le_bytes()))
            .collect()
    }

    /// Encode a load magnitude+sign as a raw `Present_Load` word (bit 10 sign).
    fn load_word(signed: i32) -> u16 {
        SignMag::<10>(signed).to_raw()
    }

    /// Expand one reply-bytes-per-cycle into a full mock script.
    ///
    /// Each cycle a bus makes `requests_per_cycle` requests; only the first is a
    /// read that gets these bytes, the rest are unacknowledged sync-writes
    /// (`Silence`). An empty byte vec models a silent/absent read.
    fn script(per_cycle: &[Vec<u8>], requests_per_cycle: usize) -> Vec<Reply> {
        per_cycle
            .iter()
            .flat_map(|bytes| {
                let read = if bytes.is_empty() {
                    Reply::Silence
                } else {
                    Reply::bytes(bytes.clone())
                };
                std::iter::once(read)
                    .chain(std::iter::repeat_n(Reply::Silence, requests_per_cycle - 1))
            })
            .collect()
    }

    /// The torque limit written in the last `TORQUE_LIMIT` sync-write, decoded
    /// from the packet the leader mock recorded.
    fn last_limit_write(leader: &MockTransport, joint_index: usize) -> u16 {
        // TORQUE_LIMIT sync-write: FF FF FE LEN 83 ADDR(48) WIDTH(2) [id lo hi]...
        let pkt = leader
            .written()
            .iter()
            .rev()
            .find(|p| p.len() >= 7 && p[4] == 0x83 && p[5] == 48)
            .expect("a torque-limit sync-write");
        let at = 7 + joint_index * 3 + 1; // skip id
        u16::from_le_bytes([pkt[at], pkt[at + 1]])
    }

    /// Build a loop over two lenient mocks from per-cycle reply bytes: one
    /// follower-load reply and one leader-position reply per cycle (empty vec =
    /// silent). The helper expands each to the real per-cycle request count —
    /// follower does 2 (read loads, write goals), leader does 3 (read pos,
    /// write goals, write limits). A short bus timeout keeps silent-read tests
    /// fast.
    fn loop_with(
        joints: &[u8],
        follower_loads: &[Vec<u8>],
        leader_positions: &[Vec<u8>],
    ) -> ForceFeedback<MockTransport, MockTransport> {
        let t = Duration::from_millis(2);
        let follower =
            Bus::new(MockTransport::new(script(follower_loads, 2)).lenient()).with_timeout(t);
        let leader =
            Bus::new(MockTransport::new(script(leader_positions, 3)).lenient()).with_timeout(t);
        ForceFeedback::new(leader, follower, cfg(joints))
    }

    #[test]
    fn a_clean_cycle_maps_load_to_a_rate_limited_limit() {
        // One joint. Follower load 400 -> target 150; from base 50, one cycle
        // of rate limiting reaches only 70.
        let mut ff = loop_with(
            &[4],
            &[sync_reply(&[(4, load_word(400))])],
            &[sync_reply(&[(4, 2532)])],
        );
        let report = ff.step().unwrap();

        assert!(report.clean);
        assert_eq!(report.joints[0].load, 400);
        assert_eq!(report.joints[0].limit, 70); // 50 + 20, walking toward 150
        assert_eq!(last_limit_write(ff.leader.transport(), 0), 70);
    }

    #[test]
    fn limit_converges_over_several_cycles_at_constant_load() {
        // Constant load 400 -> target 150. Should walk 50,70,90,...,150.
        let load = sync_reply(&[(4, load_word(400))]);
        let pos = sync_reply(&[(4, 2532)]);
        let mut ff = loop_with(&[4], &vec![load; 6], &vec![pos; 6]);
        let mut limit = 0;
        for _ in 0..6 {
            limit = ff.step().unwrap().joints[0].limit;
        }
        assert_eq!(limit, 150); // 50+20*5 = 150 reached by cycle 5
    }

    /// The follower goal written in the last GOAL_POSITION (addr 42) sync-write
    /// on the follower bus, for the given joint index.
    fn last_follower_goal(follower: &MockTransport, joint_index: usize) -> u16 {
        let pkt = follower
            .written()
            .iter()
            .rev()
            .find(|p| p.len() >= 8 && p[4] == 0x83 && p[5] == 42)
            .expect("a follower goal sync-write");
        let at = 8 + joint_index * 3;
        u16::from_le_bytes([pkt[at], pkt[at + 1]])
    }

    #[test]
    fn follower_goal_walks_toward_the_mapped_leader_position() {
        // test_map is identity on [1000,3000]. Leader sits at 2500, so the
        // follower should track toward 2500 at follower_max_step (50) per cycle,
        // starting from 0 (setup not called in this unit test).
        let load = sync_reply(&[(4, load_word(0))]);
        let pos = sync_reply(&[(4, 2500)]);
        let mut ff = loop_with(&[4], &vec![load; 3], &vec![pos; 3]);

        let g1 = ff.step().unwrap().joints[0].follower_goal;
        let g2 = ff.step().unwrap().joints[0].follower_goal;
        let g3 = ff.step().unwrap().joints[0].follower_goal;

        assert_eq!((g1, g2, g3), (50, 100, 150), "slew-limited walk toward 2500");
        // And the commanded goal reached the follower bus.
        assert_eq!(last_follower_goal(ff.follower.transport(), 0), 150);
    }

    #[test]
    fn follower_tracks_toward_a_moving_leader_and_stays_in_range() {
        // Leader sweeps; follower goal should chase it, never leaving [1000,3000]
        // (the follower range in test_map), and never jumping more than the slew.
        let sweeps = [1000u16, 3000, 2000, 3000, 1000];
        let loads: Vec<Vec<u8>> = sweeps.iter().map(|_| sync_reply(&[(4, load_word(0))])).collect();
        let positions: Vec<Vec<u8>> = sweeps.iter().map(|&p| sync_reply(&[(4, p)])).collect();
        let mut ff = loop_with(&[4], &loads, &positions);

        let mut prev = 0u16;
        for _ in &sweeps {
            let g = ff.step().unwrap().joints[0].follower_goal;
            assert!(g <= 3000, "never past follower max");
            assert!(g.abs_diff(prev) <= 50, "never jumps more than the slew limit");
            prev = g;
        }
    }

    #[test]
    fn null_commands_the_leader_to_its_own_present_position() {
        let mut ff = loop_with(
            &[4],
            &[sync_reply(&[(4, load_word(0))])],
            &[sync_reply(&[(4, 2500)])],
        );
        ff.step().unwrap();

        // A GOAL_POSITION (addr 42) sync-write carrying the read-back 2500.
        let goal = ff
            .leader
            .transport()
            .written()
            .iter()
            .find(|p| p.len() >= 10 && p[4] == 0x83 && p[5] == 42)
            .expect("a goal sync-write");
        let val = u16::from_le_bytes([goal[8], goal[9]]);
        assert_eq!(val, 2500, "leader goal should equal its present position");
    }

    #[test]
    fn holds_last_good_load_when_a_follower_read_fails() {
        // Cycle 1: load 400 seen. Cycle 2: follower silent -> hold 400.
        let mut ff = loop_with(
            &[4],
            &[sync_reply(&[(4, load_word(400))]), vec![]],
            &[sync_reply(&[(4, 2500)]), sync_reply(&[(4, 2500)])],
        );

        let r1 = ff.step().unwrap();
        assert_eq!(r1.joints[0].load, 400);

        let r2 = ff.step().unwrap();
        assert_eq!(r2.joints[0].load, 400, "load held across the failed read");
        assert!(!r2.joints[0].load_ok);
        assert!(!r2.clean);
    }

    #[test]
    fn stops_after_too_many_consecutive_failures() {
        // Follower always silent; every cycle fails. max_consecutive_errors=5.
        let mut ff = loop_with(
            &[4],
            &vec![vec![]; 10],
            &vec![sync_reply(&[(4, 2500)]); 10],
        );

        for _ in 0..4 {
            assert!(ff.step().is_ok(), "should tolerate up to the threshold");
        }
        assert!(ff.step().is_err(), "5th consecutive failure stops the loop");
    }

    #[test]
    fn a_clean_cycle_resets_the_error_count() {
        let mut ff = loop_with(
            &[4],
            &[vec![], vec![], sync_reply(&[(4, load_word(0))]), vec![]],
            &vec![sync_reply(&[(4, 2500)]); 4],
        );

        assert!(ff.step().is_ok()); // err 1
        assert!(ff.step().is_ok()); // err 2
        assert!(ff.step().unwrap().clean); // clean -> reset
        assert!(ff.step().is_ok()); // err 1 again, nowhere near threshold
    }

    #[test]
    fn shutdown_writes_torque_off_to_both_arms() {
        let mut ff = loop_with(&[4, 5], &[], &[]);
        ff.shutdown();

        let off_write = |t: &MockTransport| {
            t.written()
                .iter()
                .any(|p| p.len() >= 6 && p[4] == 0x83 && p[5] == 40) // sync-write TORQUE_ENABLE(40)
        };
        assert!(off_write(ff.leader.transport()), "leader torque-off");
        assert!(off_write(ff.follower.transport()), "follower torque-off");
    }
}
