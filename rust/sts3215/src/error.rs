//! Error types.
//!
//! Every fallible operation returns a `Result` — this crate drives real
//! hardware, so bus faults are values to be handled, never panics.

use thiserror::Error;

/// A fault at the byte-transport layer, below any knowledge of packets.
#[derive(Debug, Error)]
pub enum TransportError {
    /// The underlying port reported an I/O failure.
    #[error("serial I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// No bytes arrived within the read deadline.
    #[error("read timed out")]
    Timeout,

    /// The port was closed, or was never opened.
    #[error("transport is not open")]
    NotOpen,

    /// A mock transport was driven in a way its script did not anticipate.
    /// Only ever produced in tests.
    #[error("mock transport: {0}")]
    Mock(String),
}

/// A fault at or above the packet layer.
#[derive(Debug, Error)]
pub enum Error {
    /// Propagated from the byte transport.
    #[error(transparent)]
    Transport(#[from] TransportError),

    /// Response checksum did not match the bytes received. The response is
    /// discarded — a corrupt packet is never partially believed.
    #[error("checksum mismatch: computed {computed:#04x}, packet carried {received:#04x}")]
    Checksum {
        /// Checksum computed over the received bytes.
        computed: u8,
        /// Checksum byte the packet actually carried.
        received: u8,
    },

    /// No `FF FF` frame start was found in the received bytes.
    #[error("no packet header found in {scanned} byte(s)")]
    BadHeader {
        /// How many bytes were searched before giving up.
        scanned: usize,
    },

    /// The packet ended before its declared length.
    #[error("truncated packet: declared {declared} byte(s), received {received}")]
    Truncated {
        /// Length the packet's own header declared.
        declared: usize,
        /// Bytes actually received.
        received: usize,
    },

    /// A servo replied, but with a nonzero error byte. Bit meanings are
    /// servo-defined (voltage / angle / overheat / overload etc.).
    #[error("servo {id} reported status error {status:#010b}")]
    ServoStatus {
        /// Servo that reported the fault.
        id: u8,
        /// Raw status byte, one flag per bit.
        status: u8,
    },

    /// A reply arrived from a servo other than the one addressed.
    #[error("expected reply from servo {expected}, got {actual}")]
    WrongResponder {
        /// Servo the request was addressed to.
        expected: u8,
        /// Servo the reply claimed to come from.
        actual: u8,
    },

    /// Servo IDs are 0..=252; 253 is reserved and 254 is broadcast.
    #[error("invalid servo id {0} (valid range is 0..=252)")]
    InvalidId(u8),

    /// A broadcast was used where a single addressable servo is required —
    /// e.g. a read, which cannot be answered by every servo at once.
    #[error("broadcast id cannot be used for {0}")]
    BroadcastNotAllowed(&'static str),

    /// A packet would exceed the protocol's maximum framed length.
    #[error("packet would be {len} bytes, maximum is {max}")]
    PacketTooLong {
        /// Framed length the packet would have had.
        len: usize,
        /// Maximum the protocol permits.
        max: usize,
    },

    /// A sync operation was given no servos to address.
    #[error("sync operation requires at least one servo")]
    EmptySync,

    /// A payload did not match the width the register declares.
    #[error("register width mismatch: register is {expected} byte(s), got {actual}")]
    WidthMismatch {
        /// Width the register declares.
        expected: usize,
        /// Width of the payload supplied.
        actual: usize,
    },
}

/// Crate result type, defaulting to this crate's [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;
