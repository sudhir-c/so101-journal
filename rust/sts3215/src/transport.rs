//! The byte-level transport seam.
//!
//! Everything above this module — packet framing, checksums, register access —
//! is pure logic over `&[u8]` and can be exercised with no serial port in
//! existence. That is the point of the trait: the offline test suite drives the
//! real driver code against [`MockTransport`], so packet construction and
//! response parsing are covered before any hardware is powered on.

use crate::error::TransportError;
use std::collections::VecDeque;
use std::time::Duration;

/// A bidirectional stream of bytes to a servo bus.
///
/// The STS3215 bus is half-duplex: a request is written, then the addressed
/// servo replies on the same wire. Implementations are responsible only for
/// moving bytes; they know nothing about packets.
pub trait Transport {
    /// Write the entire buffer, or fail.
    fn write_all(&mut self, data: &[u8]) -> Result<(), TransportError>;

    /// Read whatever bytes are currently available into `buf`.
    ///
    /// Returns the number of bytes read. Returning `Ok(0)` means "nothing
    /// available yet", not end-of-stream — the caller polls against its own
    /// deadline and decides when to give up. A genuine deadline expiry is the
    /// caller's [`TransportError::Timeout`] to raise, not the transport's.
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, TransportError>;

    /// Discard any buffered inbound bytes.
    ///
    /// Called before each transaction so that a late reply from a previous,
    /// timed-out request cannot be mistaken for the answer to this one. This is
    /// a real failure mode on a shared bus and the reason it is on the trait.
    fn clear_input(&mut self) -> Result<(), TransportError>;

    /// Set the blocking read timeout, where the implementation has one.
    fn set_timeout(&mut self, timeout: Duration) -> Result<(), TransportError>;
}

/// What a [`MockTransport`] should do when the bus is next read.
#[derive(Debug, Clone)]
pub enum Reply {
    /// Return these bytes. Use to model a well-formed — or deliberately
    /// malformed — status packet.
    Bytes(Vec<u8>),
    /// Return nothing at all, modelling an absent or unpowered servo. The
    /// caller should surface a timeout.
    Silence,
}

impl Reply {
    /// Convenience for the common case.
    pub fn bytes(b: impl Into<Vec<u8>>) -> Self {
        Reply::Bytes(b.into())
    }
}

/// A scripted, hardware-free [`Transport`] for offline tests.
///
/// Queue the replies the bus should produce, run driver code against it, then
/// inspect [`MockTransport::written`] to assert on the exact bytes that would
/// have gone out on the wire.
///
/// ```
/// use sts3215::transport::{MockTransport, Reply, Transport};
///
/// let mut bus = MockTransport::new([Reply::bytes([0xFF, 0xFF, 0x01, 0x02, 0x00, 0xFC])]);
/// bus.write_all(&[0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB]).unwrap();
///
/// let mut buf = [0u8; 16];
/// let n = bus.read(&mut buf).unwrap();
/// assert_eq!(&buf[..n], &[0xFF, 0xFF, 0x01, 0x02, 0x00, 0xFC]);
/// assert_eq!(bus.written(), &[vec![0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB]]);
/// ```
#[derive(Debug, Default)]
pub struct MockTransport {
    /// Replies still to be served, in order.
    replies: VecDeque<Reply>,
    /// Bytes staged by the most recent reply, awaiting `read`.
    inbound: VecDeque<u8>,
    /// Every `write_all` call, in order, kept whole so tests can assert on
    /// individual packets rather than one flattened byte soup.
    written: Vec<Vec<u8>>,
    timeout: Option<Duration>,
    /// If true, a `write_all` with no scripted reply left is an error rather
    /// than silence. Defaults to true so an unexpected extra bus transaction
    /// fails loudly instead of looking like a dead servo.
    strict: bool,
}

impl MockTransport {
    /// A mock that will serve `replies` in order, one per request written.
    pub fn new(replies: impl IntoIterator<Item = Reply>) -> Self {
        Self {
            replies: replies.into_iter().collect(),
            inbound: VecDeque::new(),
            written: Vec::new(),
            timeout: None,
            strict: true,
        }
    }

    /// A mock that answers nothing — models a bus with no servos on it.
    pub fn silent() -> Self {
        let mut m = Self::new([]);
        m.strict = false;
        m
    }

    /// Allow requests beyond the scripted replies; they simply get silence.
    pub fn lenient(mut self) -> Self {
        self.strict = false;
        self
    }

    /// Queue one more reply.
    pub fn push_reply(&mut self, reply: Reply) {
        self.replies.push_back(reply);
    }

    /// Every request written, in order, one `Vec` per `write_all` call.
    pub fn written(&self) -> &[Vec<u8>] {
        &self.written
    }

    /// The single request written, or panic. For the many tests that perform
    /// exactly one transaction.
    pub fn sole_request(&self) -> &[u8] {
        assert_eq!(
            self.written.len(),
            1,
            "expected exactly one request on the bus, saw {}",
            self.written.len()
        );
        &self.written[0]
    }

    /// Replies scripted but never consumed. A nonzero count at the end of a
    /// test usually means the driver made fewer bus round-trips than expected.
    pub fn unused_replies(&self) -> usize {
        self.replies.len()
    }

    /// The timeout the driver last requested, if any.
    pub fn timeout(&self) -> Option<Duration> {
        self.timeout
    }
}

impl Transport for MockTransport {
    fn write_all(&mut self, data: &[u8]) -> Result<(), TransportError> {
        self.written.push(data.to_vec());

        match self.replies.pop_front() {
            Some(Reply::Bytes(b)) => self.inbound.extend(b),
            // Silence: stage nothing, so the caller sees a timeout.
            Some(Reply::Silence) => {}
            None if self.strict => {
                return Err(TransportError::Mock(format!(
                    "request #{} written but no reply was scripted: {:02X?}",
                    self.written.len(),
                    data
                )));
            }
            None => {}
        }
        Ok(())
    }

    fn read(&mut self, buf: &mut [u8]) -> Result<usize, TransportError> {
        let n = buf.len().min(self.inbound.len());
        for slot in buf.iter_mut().take(n) {
            *slot = self.inbound.pop_front().expect("length checked above");
        }
        Ok(n)
    }

    fn clear_input(&mut self) -> Result<(), TransportError> {
        self.inbound.clear();
        Ok(())
    }

    fn set_timeout(&mut self, timeout: Duration) -> Result<(), TransportError> {
        self.timeout = Some(timeout);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_each_request_separately() {
        let mut bus = MockTransport::silent();
        bus.write_all(&[1, 2, 3]).unwrap();
        bus.write_all(&[4, 5]).unwrap();
        assert_eq!(bus.written(), &[vec![1, 2, 3], vec![4, 5]]);
    }

    #[test]
    fn serves_scripted_replies_in_order() {
        let mut bus = MockTransport::new([Reply::bytes([0xAA]), Reply::bytes([0xBB, 0xCC])]);
        let mut buf = [0u8; 8];

        bus.write_all(&[0]).unwrap();
        let n = bus.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], &[0xAA]);

        bus.write_all(&[0]).unwrap();
        let n = bus.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], &[0xBB, 0xCC]);

        assert_eq!(bus.unused_replies(), 0);
    }

    #[test]
    fn silence_yields_no_bytes() {
        let mut bus = MockTransport::new([Reply::Silence]);
        bus.write_all(&[0]).unwrap();
        let mut buf = [0u8; 8];
        assert_eq!(bus.read(&mut buf).unwrap(), 0);
    }

    #[test]
    fn read_drains_across_calls_when_buffer_is_short() {
        // The framing layer reads a header, then the body. A short buffer must
        // not lose the bytes it could not hold.
        let mut bus = MockTransport::new([Reply::bytes([1, 2, 3, 4])]);
        bus.write_all(&[0]).unwrap();

        let mut buf = [0u8; 2];
        assert_eq!(bus.read(&mut buf).unwrap(), 2);
        assert_eq!(buf, [1, 2]);
        assert_eq!(bus.read(&mut buf).unwrap(), 2);
        assert_eq!(buf, [3, 4]);
        assert_eq!(bus.read(&mut buf).unwrap(), 0);
    }

    #[test]
    fn clear_input_discards_a_stale_reply() {
        let mut bus = MockTransport::new([Reply::bytes([0xDE, 0xAD])]);
        bus.write_all(&[0]).unwrap();
        bus.clear_input().unwrap();

        let mut buf = [0u8; 8];
        assert_eq!(bus.read(&mut buf).unwrap(), 0);
    }

    #[test]
    fn strict_mock_rejects_an_unscripted_request() {
        let mut bus = MockTransport::new([]);
        let err = bus.write_all(&[0xFF]).unwrap_err();
        assert!(matches!(err, TransportError::Mock(_)));
    }

    #[test]
    fn lenient_mock_allows_an_unscripted_request() {
        let mut bus = MockTransport::new([]).lenient();
        assert!(bus.write_all(&[0xFF]).is_ok());
    }
}
