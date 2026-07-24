//! The STS3215 register map.
//!
//! # Provenance
//!
//! Addresses are transcribed from the Feetech STS/SMS e-manual, cross-checked
//! against two independent copies vendored in this repository:
//!
//! - `rl-sim/lerobot/src/lerobot/motors/feetech/tables.py`
//!   (`STS_SMS_SERIES_CONTROL_TABLE`, mapped to `"sts3215"`)
//! - `.venv/lib/python3.12/site-packages/scservo_sdk/` (Feetech's own SDK)
//!
//! Nothing here is guessed.
//!
//! # Type safety
//!
//! A register carries both its value type and its access mode in its type, so
//! two whole classes of bug are compile errors rather than runtime surprises:
//!
//! ```compile_fail
//! # use sts3215::{registers::PRESENT_POSITION, protocol::write_reg, ServoId};
//! # let id = ServoId::new(1).unwrap();
//! // Present_Position is read-only, so no write packet can name it.
//! write_reg(id, PRESENT_POSITION, 2048u16).unwrap();
//! ```
//!
//! ```compile_fail
//! # use sts3215::{registers::TORQUE_ENABLE, protocol::write_reg, ServoId};
//! # let id = ServoId::new(1).unwrap();
//! // Torque_Enable is one byte, so a u16 value does not compile.
//! write_reg(id, TORQUE_ENABLE, 1u16).unwrap();
//! ```
//!
//! ```
//! # use sts3215::{registers::GOAL_POSITION, protocol::write_reg, ServoId};
//! # let id = ServoId::new(1).unwrap();
//! // Goal_Position is a writable 16-bit register, so this is fine.
//! let pkt = write_reg(id, GOAL_POSITION, 2048u16).unwrap();
//! assert_eq!(pkt.as_bytes(), &[0xFF, 0xFF, 0x01, 0x05, 0x03, 0x2A, 0x00, 0x08, 0xC4]);
//! ```

use std::marker::PhantomData;

/// Access marker: read-only.
#[derive(Debug, Clone, Copy)]
pub struct Ro;

/// Access marker: readable and writable.
#[derive(Debug, Clone, Copy)]
pub struct Rw;

/// A value that can live in a register.
///
/// All multi-byte STS registers are **little-endian**. This is the one place
/// that fact is encoded, so it cannot be got wrong per-call-site.
pub trait RegValue: Copy {
    /// Width in bytes on the wire.
    const WIDTH: usize;

    /// Decode from exactly `WIDTH` little-endian bytes.
    fn decode(bytes: &[u8]) -> Self;

    /// Encode into exactly `WIDTH` little-endian bytes.
    fn encode(self, out: &mut [u8]);
}

impl RegValue for u8 {
    const WIDTH: usize = 1;
    fn decode(bytes: &[u8]) -> Self {
        bytes[0]
    }
    fn encode(self, out: &mut [u8]) {
        out[0] = self;
    }
}

impl RegValue for u16 {
    const WIDTH: usize = 2;
    fn decode(bytes: &[u8]) -> Self {
        u16::from_le_bytes([bytes[0], bytes[1]])
    }
    fn encode(self, out: &mut [u8]) {
        out[..2].copy_from_slice(&self.to_le_bytes());
    }
}

/// A 16-bit value in **sign-magnitude** form, with the sign in bit `BIT`.
///
/// Feetech does not use two's complement for these fields. `Present_Load` puts
/// the sign in bit 10, position and velocity fields in bit 15. Decoding one as
/// two's complement yields a plausible-looking wrong number rather than an
/// obvious failure, which is exactly why it gets its own type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignMag<const BIT: u32>(pub i32);

impl<const BIT: u32> SignMag<BIT> {
    /// Decode a raw register word.
    pub const fn from_raw(raw: u16) -> Self {
        let sign = 1u16 << BIT;
        if raw & sign != 0 {
            // Magnitude is everything below the sign bit.
            SignMag(-((raw & !sign) as i32))
        } else {
            SignMag(raw as i32)
        }
    }

    /// Encode back to a raw register word. Magnitude is clamped to what the
    /// field can hold, so an out-of-range value cannot corrupt the sign bit.
    pub const fn to_raw(self) -> u16 {
        let sign = 1u16 << BIT;
        let max = (sign - 1) as i32;
        let mag = if self.0 < 0 { -self.0 } else { self.0 };
        let mag = if mag > max { max } else { mag } as u16;
        if self.0 < 0 { mag | sign } else { mag }
    }
}

impl<const BIT: u32> RegValue for SignMag<BIT> {
    const WIDTH: usize = 2;
    fn decode(bytes: &[u8]) -> Self {
        Self::from_raw(u16::from_le_bytes([bytes[0], bytes[1]]))
    }
    fn encode(self, out: &mut [u8]) {
        out[..2].copy_from_slice(&self.to_raw().to_le_bytes());
    }
}

/// A register: an address, a value type `T`, and an access mode `A`.
#[derive(Debug)]
pub struct Reg<T, A> {
    /// Address in the control table.
    pub address: u8,
    /// Human-readable name, matching the e-manual. Used in log lines.
    pub name: &'static str,
    _value: PhantomData<T>,
    _access: PhantomData<A>,
}

// Derived Copy/Clone would demand `T: Copy, A: Copy`; a register is plain data
// regardless of its markers, so implement them by hand.
impl<T, A> Clone for Reg<T, A> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T, A> Copy for Reg<T, A> {}

impl<T: RegValue, A> Reg<T, A> {
    const fn at(address: u8, name: &'static str) -> Self {
        Reg { address, name, _value: PhantomData, _access: PhantomData }
    }

    /// Width of this register's value on the wire, in bytes.
    pub const fn width(&self) -> usize {
        T::WIDTH
    }
}

macro_rules! regs {
    ($( $(#[$m:meta])* $name:ident : $ty:ty, $access:ty = $addr:expr, $label:literal );* $(;)?) => {
        $(
            $(#[$m])*
            pub const $name: Reg<$ty, $access> = Reg::at($addr, $label);
        )*
    };
}

regs! {
    // ---- EPROM (persistent; writes wear the cell — do not touch in a loop) ----
    /// Servo id, 0..=252.
    ID: u8, Rw = 5, "ID";
    /// Baud rate index (not the rate itself).
    BAUD_RATE: u8, Rw = 6, "Baud_Rate";
    /// Delay before the servo replies.
    RETURN_DELAY_TIME: u8, Rw = 7, "Return_Delay_Time";
    /// Which instructions produce a status packet.
    RESPONSE_STATUS_LEVEL: u8, Rw = 8, "Response_Status_Level";
    /// Lower travel limit, in encoder counts.
    MIN_POSITION_LIMIT: u16, Rw = 9, "Min_Position_Limit";
    /// Upper travel limit, in encoder counts.
    MAX_POSITION_LIMIT: u16, Rw = 11, "Max_Position_Limit";
    /// Shutdown temperature, °C.
    MAX_TEMPERATURE_LIMIT: u8, Rw = 13, "Max_Temperature_Limit";
    /// Upper supply voltage limit, decivolts.
    MAX_VOLTAGE_LIMIT: u8, Rw = 14, "Max_Voltage_Limit";
    /// Lower supply voltage limit, decivolts.
    MIN_VOLTAGE_LIMIT: u8, Rw = 15, "Min_Voltage_Limit";
    /// Ceiling for [`TORQUE_LIMIT`].
    MAX_TORQUE_LIMIT: u16, Rw = 16, "Max_Torque_Limit";
    /// Position-loop proportional gain.
    P_COEFFICIENT: u8, Rw = 21, "P_Coefficient";
    /// Position-loop derivative gain.
    D_COEFFICIENT: u8, Rw = 22, "D_Coefficient";
    /// Position-loop integral gain.
    I_COEFFICIENT: u8, Rw = 23, "I_Coefficient";
    /// Zero-point trim, sign-magnitude with the sign in bit 11.
    HOMING_OFFSET: SignMag<11>, Rw = 31, "Homing_Offset";
    /// 0 = position, 1 = velocity, 2 = PWM, 3 = step (confirm against firmware).
    OPERATING_MODE: u8, Rw = 33, "Operating_Mode";

    // ---- SRAM (volatile; these are the control-loop registers) ----
    /// 0 = limp, 1 = holding. The go-limp path.
    TORQUE_ENABLE: u8, Rw = 40, "Torque_Enable";
    /// Trapezoidal acceleration limit.
    ACCELERATION: u8, Rw = 41, "Acceleration";
    /// Commanded position, encoder counts.
    GOAL_POSITION: u16, Rw = 42, "Goal_Position";
    /// Time-based move duration.
    GOAL_TIME: u16, Rw = 44, "Goal_Time";
    /// Speed cap for position moves.
    GOAL_VELOCITY: u16, Rw = 46, "Goal_Velocity";
    /// Torque ceiling, 0..=1000. The force-feedback knob.
    TORQUE_LIMIT: u16, Rw = 48, "Torque_Limit";
    /// EPROM write lock.
    LOCK: u8, Rw = 55, "Lock";

    /// Measured position, encoder counts.
    PRESENT_POSITION: u16, Ro = 56, "Present_Position";
    /// Measured velocity, sign-magnitude with the sign in bit 15.
    PRESENT_VELOCITY: SignMag<15>, Ro = 58, "Present_Velocity";
    /// Measured load, sign-magnitude with the sign in bit 10. Force feedback
    /// reads this.
    PRESENT_LOAD: SignMag<10>, Ro = 60, "Present_Load";
    /// Supply voltage, decivolts.
    PRESENT_VOLTAGE: u8, Ro = 62, "Present_Voltage";
    /// Case temperature, °C.
    PRESENT_TEMPERATURE: u8, Ro = 63, "Present_Temperature";
    /// Hardware status flags.
    STATUS: u8, Ro = 65, "Status";
    /// Nonzero while executing a move.
    MOVING: u8, Ro = 66, "Moving";
    /// Measured current. Candidate alternative force signal to [`PRESENT_LOAD`].
    PRESENT_CURRENT: u16, Ro = 69, "Present_Current";
}

/// Encoder counts per revolution.
pub const RESOLUTION: u16 = 4096;

/// Largest value [`TORQUE_LIMIT`] accepts.
pub const TORQUE_LIMIT_MAX: u16 = 1000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addresses_match_the_vendored_control_table() {
        // Spot-check against STS_SMS_SERIES_CONTROL_TABLE.
        assert_eq!(TORQUE_ENABLE.address, 40);
        assert_eq!(GOAL_POSITION.address, 42);
        assert_eq!(GOAL_VELOCITY.address, 46);
        assert_eq!(TORQUE_LIMIT.address, 48);
        assert_eq!(PRESENT_POSITION.address, 56);
        assert_eq!(PRESENT_VELOCITY.address, 58);
        assert_eq!(PRESENT_LOAD.address, 60);
        assert_eq!(PRESENT_VOLTAGE.address, 62);
        assert_eq!(PRESENT_TEMPERATURE.address, 63);
        assert_eq!(PRESENT_CURRENT.address, 69);
    }

    #[test]
    fn widths_match_the_control_table() {
        assert_eq!(TORQUE_ENABLE.width(), 1);
        assert_eq!(GOAL_POSITION.width(), 2);
        assert_eq!(PRESENT_LOAD.width(), 2);
        assert_eq!(PRESENT_TEMPERATURE.width(), 1);
    }

    #[test]
    fn u16_round_trips_little_endian() {
        let mut buf = [0u8; 2];
        2048u16.encode(&mut buf);
        assert_eq!(buf, [0x00, 0x08], "little-endian: low byte first");
        assert_eq!(u16::decode(&buf), 2048);
    }

    #[test]
    fn sign_magnitude_load_decodes_both_directions() {
        // Sign is bit 10, so magnitudes run 0..=1023.
        assert_eq!(SignMag::<10>::from_raw(0), SignMag(0));
        assert_eq!(SignMag::<10>::from_raw(500), SignMag(500));
        assert_eq!(SignMag::<10>::from_raw(1023), SignMag(1023));
        // 0x400 is the sign bit alone: negative zero.
        assert_eq!(SignMag::<10>::from_raw(0x400), SignMag(0));
        assert_eq!(SignMag::<10>::from_raw(0x400 | 500), SignMag(-500));
        assert_eq!(SignMag::<10>::from_raw(0x400 | 1023), SignMag(-1023));
    }

    #[test]
    fn sign_magnitude_is_not_twos_complement() {
        // The bug this type exists to prevent: as two's complement 0x4064
        // would be a large positive number, not -100.
        assert_eq!(SignMag::<10>::from_raw(0x400 | 100), SignMag(-100));
        assert_ne!(SignMag::<10>::from_raw(0x464).0, 0x464);
    }

    #[test]
    fn sign_magnitude_round_trips() {
        for v in [-1023, -500, -1, 0, 1, 500, 1023] {
            assert_eq!(SignMag::<10>::from_raw(SignMag::<10>(v).to_raw()).0, v);
        }
        for v in [-32767, -1, 0, 1, 32767] {
            assert_eq!(SignMag::<15>::from_raw(SignMag::<15>(v).to_raw()).0, v);
        }
    }

    #[test]
    fn sign_magnitude_clamps_rather_than_corrupting_the_sign_bit() {
        // An over-range magnitude must saturate, never spill into the sign bit.
        assert_eq!(SignMag::<10>(9999).to_raw(), 1023);
        assert_eq!(SignMag::<10>(-9999).to_raw(), 0x400 | 1023);
    }

    #[test]
    fn sign_magnitude_decodes_from_wire_bytes_little_endian() {
        // 0x0464 on the wire, low byte first.
        assert_eq!(SignMag::<10>::decode(&[0x64, 0x04]), SignMag(-100));
    }
}
