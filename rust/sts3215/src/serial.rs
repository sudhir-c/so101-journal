//! The real serial [`Transport`]. Compiled only with the `serial` feature.
//!
//! This is the sole module in the crate that can open a hardware port. Nothing
//! here has a default path: the caller must always name the device explicitly,
//! so no code path can connect to "whatever was plugged in".

use crate::error::TransportError;
use crate::transport::Transport;
use serialport::{ClearBuffer, DataBits, FlowControl, Parity, SerialPort, StopBits};
use std::io::ErrorKind;
use std::time::Duration;

/// Bus speed the SO-101 servos are configured for.
pub const DEFAULT_BAUD: u32 = 1_000_000;

/// Default per-read timeout. Generous relative to a 1 Mbps round-trip; the
/// control loop enforces its own, tighter deadline on top.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_millis(20);

/// An open serial port carrying an STS3215 bus.
pub struct SerialTransport {
    port: Box<dyn SerialPort>,
    path: String,
}

impl SerialTransport {
    /// Open `path` at [`DEFAULT_BAUD`] in 8N1, no flow control.
    ///
    /// These ports are single-owner. Opening one while another process (a
    /// LeRobot command, the Python teleop server) holds it will fail, and it
    /// should — two writers on a half-duplex bus corrupt each other's traffic.
    pub fn open(path: &str) -> Result<Self, TransportError> {
        Self::open_with(path, DEFAULT_BAUD, DEFAULT_TIMEOUT)
    }

    /// Open with an explicit baud rate and read timeout.
    pub fn open_with(
        path: &str,
        baud: u32,
        timeout: Duration,
    ) -> Result<Self, TransportError> {
        let port = serialport::new(path, baud)
            .data_bits(DataBits::Eight)
            .parity(Parity::None)
            .stop_bits(StopBits::One)
            .flow_control(FlowControl::None)
            .timeout(timeout)
            .open()
            .map_err(|e| TransportError::Io(std::io::Error::other(e)))?;

        Ok(Self { port, path: path.to_string() })
    }

    /// The device path this transport was opened on. Useful in log lines, where
    /// confusing the leader and follower buses is an easy and costly mistake.
    pub fn path(&self) -> &str {
        &self.path
    }
}

impl Transport for SerialTransport {
    fn write_all(&mut self, data: &[u8]) -> Result<(), TransportError> {
        std::io::Write::write_all(&mut self.port, data)?;
        // Push the request out now: on a half-duplex bus the reply cannot begin
        // until the request has physically left.
        std::io::Write::flush(&mut self.port)?;
        Ok(())
    }

    fn read(&mut self, buf: &mut [u8]) -> Result<usize, TransportError> {
        match std::io::Read::read(&mut self.port, buf) {
            Ok(n) => Ok(n),
            // Per the `Transport` contract, "nothing yet" is `Ok(0)`; the
            // caller owns the deadline and decides when that becomes a timeout.
            Err(e) if e.kind() == ErrorKind::TimedOut => Ok(0),
            Err(e) if e.kind() == ErrorKind::Interrupted => Ok(0),
            Err(e) => Err(TransportError::Io(e)),
        }
    }

    fn clear_input(&mut self) -> Result<(), TransportError> {
        self.port
            .clear(ClearBuffer::Input)
            .map_err(|e| TransportError::Io(std::io::Error::other(e)))
    }

    fn set_timeout(&mut self, timeout: Duration) -> Result<(), TransportError> {
        self.port
            .set_timeout(timeout)
            .map_err(|e| TransportError::Io(std::io::Error::other(e)))
    }
}

/// List serial devices the OS currently reports.
///
/// Read-only: enumerating does not open anything, so this is safe to call while
/// another process owns a port.
pub fn available_ports() -> Result<Vec<String>, TransportError> {
    serialport::available_ports()
        .map(|ports| ports.into_iter().map(|p| p.port_name).collect())
        .map_err(|e| TransportError::Io(std::io::Error::other(e)))
}
