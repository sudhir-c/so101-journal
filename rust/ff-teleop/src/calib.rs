//! Per-arm calibration and the leader→follower position map.
//!
//! The leader and follower are the same servo model but calibrated
//! independently — different encoder zeros and travel ranges — so a leader
//! position cannot be sent to the follower raw. [`PositionMap`] maps one arm's
//! travel onto the other's, range to range.
//!
//! Calibration is read from the LeRobot JSON files already on disk (see
//! [`load`]). The mapping math ([`PositionMap`]) is pure and unit-tested with
//! no files and no hardware.

use serde::Deserialize;
use std::collections::HashMap;

/// One joint's calibration, as stored in a LeRobot calibration file.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct JointCalib {
    /// Servo id.
    pub id: u8,
    /// 0 or 1 — whether this joint's sense of direction is flipped. If the two
    /// arms disagree here, the map inverts.
    #[serde(default)]
    pub drive_mode: i32,
    /// Encoder trim. Loaded so the file parses cleanly and to document the
    /// calibration; the range-to-range map does not need it (the ranges already
    /// bake in each arm's zero).
    #[serde(default)]
    #[allow(dead_code)]
    pub homing_offset: i32,
    /// Lower travel limit, encoder counts.
    pub range_min: u16,
    /// Upper travel limit, encoder counts.
    pub range_max: u16,
}

/// Load a LeRobot calibration file, indexed by servo id.
///
/// The file is a JSON object of joint-name → calibration; we reindex by id so
/// the loop can look joints up the way it addresses them.
pub fn load(path: &str) -> Result<HashMap<u8, JointCalib>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("reading calibration {path}: {e}"))?;
    let by_name: HashMap<String, JointCalib> = serde_json::from_str(&text)
        .map_err(|e| format!("parsing calibration {path}: {e}"))?;
    Ok(by_name.into_values().map(|j| (j.id, j)).collect())
}

/// Expand a leading `~/` to the user's home directory.
pub fn expand_home(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{home}/{rest}");
        }
    }
    path.to_string()
}

/// A linear map from one joint's travel range to another's.
///
/// Built from the two arms' calibrations for the same joint. Mapping clamps the
/// input to the leader's range and the output to the follower's, so it can
/// never command the follower outside its safe travel however wild the input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositionMap {
    leader_min: u16,
    leader_max: u16,
    follower_min: u16,
    follower_max: u16,
    /// True when the arms' drive modes differ, flipping the direction.
    invert: bool,
}

impl PositionMap {
    /// Build the map for one joint from both arms' calibration.
    ///
    /// Returns `None` if either range is degenerate (min ≥ max), which would
    /// make the map ill-defined.
    pub fn new(leader: &JointCalib, follower: &JointCalib) -> Option<Self> {
        if leader.range_min >= leader.range_max || follower.range_min >= follower.range_max {
            return None;
        }
        Some(PositionMap {
            leader_min: leader.range_min,
            leader_max: leader.range_max,
            follower_min: follower.range_min,
            follower_max: follower.range_max,
            invert: leader.drive_mode != follower.drive_mode,
        })
    }

    /// Map a raw leader position to the follower position it should track.
    ///
    /// The output is always within the follower's `[range_min, range_max]`.
    pub fn map(&self, leader_raw: u16) -> u16 {
        let clamped = leader_raw.clamp(self.leader_min, self.leader_max);
        let span = f64::from(self.leader_max - self.leader_min);
        let mut frac = f64::from(clamped - self.leader_min) / span;
        if self.invert {
            frac = 1.0 - frac;
        }
        let follower_span = f64::from(self.follower_max - self.follower_min);
        let mapped = f64::from(self.follower_min) + frac * follower_span;
        // Round, then clamp defensively against float edge effects.
        (mapped.round() as u16).clamp(self.follower_min, self.follower_max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calib(id: u8, drive_mode: i32, range_min: u16, range_max: u16) -> JointCalib {
        JointCalib { id, drive_mode, homing_offset: 0, range_min, range_max }
    }

    /// spectre (follower) and phantom (leader) wrist_flex, from the real files.
    fn wrist_map() -> PositionMap {
        let leader = calib(4, 0, 761, 3106); // phantom
        let follower = calib(4, 0, 888, 3175); // spectre
        PositionMap::new(&leader, &follower).unwrap()
    }

    #[test]
    fn maps_range_endpoints_to_range_endpoints() {
        let m = wrist_map();
        assert_eq!(m.map(761), 888); // leader min -> follower min
        assert_eq!(m.map(3106), 3175); // leader max -> follower max
    }

    #[test]
    fn maps_the_midpoint_to_the_midpoint() {
        let m = wrist_map();
        let leader_mid = (761 + 3106) / 2;
        let follower_mid: i32 = (888 + 3175) / 2;
        // Within a count of rounding.
        assert!((i32::from(m.map(leader_mid)) - follower_mid).abs() <= 1);
    }

    #[test]
    fn clamps_input_beyond_the_leader_range() {
        let m = wrist_map();
        assert_eq!(m.map(0), 888); // below leader min -> follower min
        assert_eq!(m.map(4095), 3175); // above leader max -> follower max
    }

    #[test]
    fn output_is_always_within_the_follower_range() {
        let m = wrist_map();
        for raw in (0..=4095u16).step_by(13) {
            let out = m.map(raw);
            assert!((888..=3175).contains(&out), "{raw} -> {out} out of range");
        }
    }

    #[test]
    fn preserves_direction_with_matching_drive_modes() {
        let m = wrist_map();
        // Increasing leader position -> increasing follower position.
        assert!(m.map(1000) < m.map(2000));
        assert!(m.map(2000) < m.map(3000));
    }

    #[test]
    fn inverts_direction_when_drive_modes_differ() {
        let leader = calib(4, 0, 761, 3106);
        let follower = calib(4, 1, 888, 3175); // opposite drive mode
        let m = PositionMap::new(&leader, &follower).unwrap();
        assert_eq!(m.map(761), 3175); // leader min -> follower MAX
        assert_eq!(m.map(3106), 888); // leader max -> follower MIN
        assert!(m.map(1000) > m.map(2000)); // direction flipped
    }

    #[test]
    fn rejects_a_degenerate_range() {
        let bad = calib(4, 0, 2000, 2000);
        let ok = calib(4, 0, 800, 3000);
        assert!(PositionMap::new(&bad, &ok).is_none());
        assert!(PositionMap::new(&ok, &bad).is_none());
    }

    #[test]
    fn parses_a_lerobot_calibration_object() {
        let json = r#"{
            "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 1894, "range_min": 859, "range_max": 3187},
            "wrist_flex":   {"id": 4, "drive_mode": 0, "homing_offset": -48,  "range_min": 888, "range_max": 3175}
        }"#;
        let by_name: HashMap<String, JointCalib> = serde_json::from_str(json).unwrap();
        let by_id: HashMap<u8, JointCalib> = by_name.into_values().map(|j| (j.id, j)).collect();
        assert_eq!(by_id[&1].range_min, 859);
        assert_eq!(by_id[&4].range_max, 3175);
        assert_eq!(by_id[&4].homing_offset, -48);
    }

    #[test]
    fn expand_home_leaves_absolute_paths_alone() {
        assert_eq!(expand_home("/dev/tty.foo"), "/dev/tty.foo");
        assert_eq!(expand_home("relative/path"), "relative/path");
    }
}
