//! A driver for Feetech STS3215 smart serial servos.
//!
//! # Scope
//!
//! The STS3215 is a *smart* servo: it runs its own internal position loop and
//! is commanded over a half-duplex serial bus by reading and writing registers.
//! It is not a directly torque-controlled motor. This crate speaks that
//! register protocol; it does not and cannot offer true torque control.
//!
//! # Layering
//!
//! ```text
//!   bus         register-level operations (read / write / sync)   [phase 2+]
//!   protocol    framing, checksums, parsing — pure, no I/O        [phase 2]
//!   transport   bytes in, bytes out                               [this phase]
//! ```
//!
//! Only [`serial`] can open hardware, and only when the `serial` feature is
//! enabled. Everything else is pure logic over byte slices, exercised offline
//! against [`transport::MockTransport`].
//!
//! # Protocol summary
//!
//! Framing follows Dynamixel protocol 1.0:
//!
//! ```text
//!   request:  FF FF  ID  LEN  INST  PARAM...  CHK
//!   status:   FF FF  ID  LEN  ERR   PARAM...  CHK
//!
//!   LEN = param count + 2
//!   CHK = !(sum of bytes from ID through the last param) & 0xFF
//! ```
//!
//! Multi-byte register values are **little-endian**, which is where the STS
//! series diverges from Feetech's older SCS series. Reading a 16-bit register
//! big-endian yields plausible-looking garbage rather than an obvious failure,
//! so this is worth stating loudly.
//!
//! Register addresses are transcribed from the Feetech STS/SMS e-manual as
//! vendored in this repository; see `registers.rs` for provenance.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod bus;
pub mod error;
pub mod protocol;
pub mod registers;
pub mod transport;

#[cfg(feature = "serial")]
pub mod serial;

pub use bus::{Bus, Response};
pub use error::{Error, Result, TransportError};
pub use protocol::{Instruction, Packet, Parsed, StatusPacket};
pub use registers::{Reg, RegValue, Ro, Rw, SignMag};
pub use transport::{MockTransport, Reply, Transport};

/// Broadcast address: every servo acts, none replies.
pub const BROADCAST_ID: u8 = 0xFE;

/// Highest assignable servo id. 253 is reserved and 254 is broadcast.
pub const MAX_ID: u8 = 0xFC;

/// Identifies one servo on a bus.
///
/// Constructing this is the only way to address a servo, so an out-of-range id
/// is rejected once, at the boundary, rather than becoming a malformed packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ServoId(u8);

impl ServoId {
    /// The broadcast address. Valid for writes; never for reads.
    pub const BROADCAST: ServoId = ServoId(BROADCAST_ID);

    /// Create an id, rejecting anything above [`MAX_ID`].
    ///
    /// Use [`ServoId::BROADCAST`] for the broadcast address — it is deliberately
    /// not reachable through this constructor, so a broadcast is always a
    /// visible choice at the call site.
    pub const fn new(id: u8) -> Result<Self, Error> {
        if id > MAX_ID {
            return Err(Error::InvalidId(id));
        }
        Ok(ServoId(id))
    }

    /// The raw byte, for packet construction.
    pub const fn raw(self) -> u8 {
        self.0
    }

    /// Whether this is the broadcast address.
    pub const fn is_broadcast(self) -> bool {
        self.0 == BROADCAST_ID
    }
}

impl std::fmt::Display for ServoId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_broadcast() {
            f.write_str("broadcast")
        } else {
            write!(f, "#{}", self.0)
        }
    }
}

impl TryFrom<u8> for ServoId {
    type Error = Error;
    fn try_from(id: u8) -> Result<Self, Error> {
        ServoId::new(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_the_so101_joint_ids() {
        // shoulder_pan .. gripper
        for id in 1..=6u8 {
            assert_eq!(ServoId::new(id).unwrap().raw(), id);
        }
    }

    #[test]
    fn rejects_reserved_and_broadcast_ids() {
        assert!(matches!(ServoId::new(253), Err(Error::InvalidId(253))));
        assert!(matches!(ServoId::new(254), Err(Error::InvalidId(254))));
        assert!(matches!(ServoId::new(255), Err(Error::InvalidId(255))));
    }

    #[test]
    fn broadcast_is_only_reachable_deliberately() {
        assert!(ServoId::BROADCAST.is_broadcast());
        assert!(!ServoId::new(1).unwrap().is_broadcast());
    }
}
