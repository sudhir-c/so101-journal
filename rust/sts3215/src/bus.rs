//! The transaction layer: request out, reply in, with deadlines and retries.
//!
//! [`Bus`] is generic over [`Transport`], so the whole of it — timeouts,
//! checksum failures, wrong responders, retry behaviour — is exercised offline
//! against [`MockTransport`](crate::MockTransport).
//!
//! # Failure policy
//!
//! Every operation returns `Result`. The bus never invents a value: a read that
//! could not be completed is an error, never a plausible-looking default. Which
//! errors are retried is deliberate — see [`Bus::with_retries`].
//!
//! Holding last-known-good state on error is the *caller's* policy, not the
//! bus's, because only the control loop knows what "last good" means.

use crate::error::{Error, Result, TransportError};
use crate::protocol::{self, Packet, Parsed, StatusPacket};
use crate::registers::{Reg, RegValue, Rw};
use crate::transport::Transport;
use crate::{ServoId, registers};
use std::time::{Duration, Instant};

/// Default per-transaction deadline.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_millis(20);

/// Default retry count for transient bus faults.
pub const DEFAULT_RETRIES: u8 = 2;

/// Largest reply payload [`Response`] will carry.
const MAX_RESPONSE_PARAMS: usize = 32;

/// Bytes pulled from the transport per poll.
const READ_CHUNK: usize = 64;

/// An owned copy of a status packet.
///
/// Owned rather than borrowed so a transaction can loop over its receive
/// buffer and still hand a result back to the caller.
#[derive(Debug, Clone, Copy)]
pub struct Response {
    /// Servo that replied.
    pub id: u8,
    /// Servo-defined error flags; zero means healthy.
    pub error: u8,
    params: [u8; MAX_RESPONSE_PARAMS],
    len: usize,
}

impl Response {
    fn from_packet(p: &StatusPacket<'_>) -> Result<Self> {
        if p.params.len() > MAX_RESPONSE_PARAMS {
            return Err(Error::PacketTooLong {
                len: p.params.len(),
                max: MAX_RESPONSE_PARAMS,
            });
        }
        let mut params = [0u8; MAX_RESPONSE_PARAMS];
        params[..p.params.len()].copy_from_slice(p.params);
        Ok(Response { id: p.id, error: p.error, params, len: p.params.len() })
    }

    /// The reply payload.
    pub fn params(&self) -> &[u8] {
        &self.params[..self.len]
    }

    /// Turn a nonzero error byte into an [`Error`]; healthy replies pass.
    pub fn check_status(&self) -> Result<()> {
        if self.error != 0 {
            return Err(Error::ServoStatus { id: self.id, status: self.error });
        }
        Ok(())
    }

    /// Decode the payload as a typed register value.
    pub fn value<V: RegValue>(&self) -> Result<V> {
        if self.len != V::WIDTH {
            return Err(Error::WidthMismatch { expected: V::WIDTH, actual: self.len });
        }
        Ok(V::decode(self.params()))
    }
}

/// Whether a fault is worth retrying.
///
/// Transient wire problems are; a servo reporting a genuine fault is not —
/// retrying an overload or over-temperature condition just hammers a servo
/// that is already in trouble.
fn is_transient(e: &Error) -> bool {
    matches!(
        e,
        Error::Transport(TransportError::Timeout)
            | Error::Checksum { .. }
            | Error::BadHeader { .. }
            | Error::Truncated { .. }
            | Error::WrongResponder { .. }
    )
}

/// A servo bus: framed transactions over some [`Transport`].
#[derive(Debug)]
pub struct Bus<T: Transport> {
    transport: T,
    rx: Vec<u8>,
    timeout: Duration,
    retries: u8,
}

impl<T: Transport> Bus<T> {
    /// Wrap a transport with default timeout and retry policy.
    pub fn new(transport: T) -> Self {
        Bus {
            transport,
            rx: Vec::with_capacity(256),
            timeout: DEFAULT_TIMEOUT,
            retries: DEFAULT_RETRIES,
        }
    }

    /// Set the per-attempt deadline.
    ///
    /// This bounds one attempt, not the whole call: with retries, worst-case
    /// wall time is `timeout * (retries + 1)`. A control loop must budget for
    /// that, not for `timeout` alone.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set how many times a transient fault is retried. Zero means one attempt.
    pub fn with_retries(mut self, retries: u8) -> Self {
        self.retries = retries;
        self
    }

    /// Worst-case duration of one transaction, across all retries. Use this to
    /// check a control loop's period is actually achievable.
    pub fn worst_case_duration(&self) -> Duration {
        self.timeout * (self.retries as u32 + 1)
    }

    /// Borrow the underlying transport.
    pub fn transport(&self) -> &T {
        &self.transport
    }

    /// Mutably borrow the underlying transport.
    pub fn transport_mut(&mut self) -> &mut T {
        &mut self.transport
    }

    /// Unwrap back to the transport.
    pub fn into_transport(self) -> T {
        self.transport
    }

    /// Send a packet and wait for the addressed servo's reply.
    pub fn transact(&mut self, packet: &Packet, expect: ServoId) -> Result<Response> {
        let mut attempt = 0;
        loop {
            match self.try_transact(packet, expect) {
                Ok(r) => return Ok(r),
                Err(e) if is_transient(&e) && attempt < self.retries => {
                    attempt += 1;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Send a packet that produces no reply — a broadcast sync-write.
    ///
    /// Not retried: with no acknowledgement there is no way to know whether the
    /// first attempt landed, and blindly repeating a motion command is worse
    /// than reporting the failure upward.
    pub fn send(&mut self, packet: &Packet) -> Result<()> {
        self.transport.clear_input()?;
        self.transport.write_all(packet.as_bytes())?;
        Ok(())
    }

    fn try_transact(&mut self, packet: &Packet, expect: ServoId) -> Result<Response> {
        // Drop anything already buffered. A late reply to a previous,
        // timed-out request would otherwise be read as the answer to this one,
        // making every subsequent read one transaction stale.
        self.transport.clear_input()?;
        self.rx.clear();
        self.transport.write_all(packet.as_bytes())?;

        let deadline = Instant::now() + self.timeout;
        let mut chunk = [0u8; READ_CHUNK];
        // Remembered so an expired deadline can report *why* parsing never
        // succeeded, rather than a bare timeout.
        let mut last_parse_error: Option<Error> = None;

        loop {
            let n = self.transport.read(&mut chunk)?;
            if n > 0 {
                self.rx.extend_from_slice(&chunk[..n]);

                match protocol::parse_status(&self.rx) {
                    Ok(Parsed::Complete { packet, .. }) => {
                        if packet.id != expect.raw() {
                            return Err(Error::WrongResponder {
                                expected: expect.raw(),
                                actual: packet.id,
                            });
                        }
                        packet.check_status()?;
                        return Response::from_packet(&packet);
                    }
                    // Not a whole packet yet; keep reading.
                    Ok(Parsed::NeedMore) => {}
                    // Junk so far. More bytes may still resolve it, so keep
                    // reading until the deadline, but remember the reason.
                    Err(e) => last_parse_error = Some(e),
                }
            }

            if Instant::now() >= deadline {
                return Err(last_parse_error
                    .unwrap_or(Error::Transport(TransportError::Timeout)));
            }
        }
    }

    /// Probe whether a servo is present and answering.
    pub fn ping(&mut self, id: ServoId) -> Result<()> {
        let pkt = protocol::ping(id)?;
        self.transact(&pkt, id)?;
        Ok(())
    }

    /// Read one register from one servo.
    pub fn read<V: RegValue, A>(&mut self, id: ServoId, reg: Reg<V, A>) -> Result<V> {
        let pkt = protocol::read_reg(id, reg)?;
        self.transact(&pkt, id)?.value::<V>()
    }

    /// Write one register to one servo.
    ///
    /// Takes `Reg<V, Rw>`, so a read-only register cannot be named here.
    /// Requires the servo's status reply, matching what LeRobot's Feetech bus
    /// does. If writes time out on real hardware, suspect the servo's
    /// `Response_Status_Level`.
    pub fn write<V: RegValue>(
        &mut self,
        id: ServoId,
        reg: Reg<V, Rw>,
        value: V,
    ) -> Result<()> {
        let pkt = protocol::write_reg(id, reg, value)?;
        self.transact(&pkt, id)?;
        Ok(())
    }

    /// Enable or disable a servo's torque.
    ///
    /// Disabling goes limp — an arm held against gravity will drop. Enabling
    /// makes the servo hold whatever `Goal_Position` currently says, which may
    /// not be where the arm is now; read and re-write the present position
    /// first if that matters.
    pub fn set_torque(&mut self, id: ServoId, on: bool) -> Result<()> {
        self.write(id, registers::TORQUE_ENABLE, u8::from(on))
    }

    /// Read the same register from several servos in one request.
    ///
    /// Returns one entry per requested id, **in the order asked**, each an
    /// independent `Result`: a servo that does not answer, replies corrupt, or
    /// flags a fault fails only its own entry. The whole cycle is not lost to
    /// one bad joint — the caller holds last-good for just that joint. The
    /// outer `Result` fails only if the request itself could not be sent.
    ///
    /// Not retried: a partial result is more useful than a retry that stalls
    /// the control loop, and per-id errors already localise the failure.
    pub fn sync_read<V: RegValue, A>(
        &mut self,
        ids: &[ServoId],
        reg: Reg<V, A>,
    ) -> Result<Vec<(ServoId, Result<V>)>> {
        let raw = self.sync_read_raw(ids, reg.address, V::WIDTH as u8)?;
        Ok(raw
            .into_iter()
            .map(|(id, r)| (id, r.and_then(|resp| resp.check_status().and(resp.value::<V>()))))
            .collect())
    }

    fn sync_read_raw(
        &mut self,
        ids: &[ServoId],
        address: u8,
        len: u8,
    ) -> Result<Vec<(ServoId, Result<Response>)>> {
        // Validates non-empty and no broadcast in the id list.
        let pkt = protocol::sync_read(ids, address, len)?;
        self.transport.clear_input()?;
        self.rx.clear();
        self.transport.write_all(pkt.as_bytes())?;

        let deadline = Instant::now() + self.timeout;
        let mut chunk = [0u8; READ_CHUNK];
        // (id, response) for each servo that answered, first reply wins.
        let mut collected: Vec<(u8, Response)> = Vec::with_capacity(ids.len());

        loop {
            // Drain every complete packet currently buffered.
            loop {
                match protocol::parse_status(&self.rx) {
                    Ok(Parsed::Complete { packet, consumed }) => {
                        let known = ids.iter().any(|i| i.raw() == packet.id);
                        let seen = collected.iter().any(|(id, _)| *id == packet.id);
                        if known && !seen {
                            let resp = Response::from_packet(&packet)?;
                            collected.push((packet.id, resp));
                        }
                        self.rx.drain(..consumed);
                    }
                    Ok(Parsed::NeedMore) => break,
                    // A corrupt reply from one servo must not swallow the
                    // others: drop one byte and resync on the next header.
                    Err(_) => {
                        if self.rx.is_empty() {
                            break;
                        }
                        self.rx.remove(0);
                    }
                }
            }

            if collected.len() == ids.len() || Instant::now() >= deadline {
                break;
            }
            let n = self.transport.read(&mut chunk)?;
            if n > 0 {
                self.rx.extend_from_slice(&chunk[..n]);
            }
        }

        // Assemble results in the requested order; a missing reply is a
        // per-id timeout, not a failure of the whole read.
        Ok(ids
            .iter()
            .map(|id| {
                let found = collected
                    .iter()
                    .find(|(cid, _)| *cid == id.raw())
                    .map(|(_, resp)| *resp)
                    .ok_or(Error::Transport(TransportError::Timeout));
                (*id, found)
            })
            .collect())
    }

    /// Write per-servo values to the same register in one broadcast request.
    ///
    /// One packet drives every listed joint at once — this is what the control
    /// loop uses to command all six goals per cycle. No servo replies, so like
    /// [`Bus::send`] it is not retried: repeating an unacknowledged motion
    /// command is worse than reporting the failure upward.
    pub fn sync_write<V: RegValue>(
        &mut self,
        reg: Reg<V, Rw>,
        entries: &[(ServoId, V)],
    ) -> Result<()> {
        let pkt = protocol::sync_write_reg(reg, entries)?;
        self.send(&pkt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registers::*;
    use crate::transport::{MockTransport, Reply};

    fn id(n: u8) -> ServoId {
        ServoId::new(n).unwrap()
    }

    /// Frame a status packet the way a servo would.
    fn status(id: u8, error: u8, params: &[u8]) -> Vec<u8> {
        let mut p = vec![0xFF, 0xFF, id, (params.len() + 2) as u8, error];
        p.extend_from_slice(params);
        let ck = protocol::checksum(&p[2..]);
        p.push(ck);
        p
    }

    /// A bus that fails fast, for tests that expect an error.
    fn quick(t: MockTransport) -> Bus<MockTransport> {
        Bus::new(t).with_timeout(Duration::from_millis(1)).with_retries(0)
    }

    #[test]
    fn reads_a_position() {
        let mut bus = Bus::new(MockTransport::new([Reply::bytes(status(1, 0, &[0x00, 0x08]))]));
        assert_eq!(bus.read(id(1), PRESENT_POSITION).unwrap(), 2048);

        // The request on the wire must be the golden read packet.
        assert_eq!(
            bus.transport().sole_request(),
            &[0xFF, 0xFF, 0x01, 0x04, 0x02, 0x38, 0x02, 0xBE]
        );
    }

    #[test]
    fn reads_a_negative_load() {
        let mut bus = Bus::new(MockTransport::new([Reply::bytes(status(3, 0, &[0x64, 0x04]))]));
        assert_eq!(bus.read(id(3), PRESENT_LOAD).unwrap(), SignMag(-100));
    }

    #[test]
    fn writes_a_goal_position() {
        let mut bus = Bus::new(MockTransport::new([Reply::bytes(status(1, 0, &[]))]));
        bus.write(id(1), GOAL_POSITION, 2048).unwrap();
        assert_eq!(
            bus.transport().sole_request(),
            &[0xFF, 0xFF, 0x01, 0x05, 0x03, 0x2A, 0x00, 0x08, 0xC4]
        );
    }

    #[test]
    fn torque_off_sends_the_documented_packet() {
        let mut bus = Bus::new(MockTransport::new([Reply::bytes(status(1, 0, &[]))]));
        bus.set_torque(id(1), false).unwrap();
        assert_eq!(
            bus.transport().sole_request(),
            &[0xFF, 0xFF, 0x01, 0x04, 0x03, 0x28, 0x00, 0xCF]
        );
    }

    #[test]
    fn ping_succeeds_on_a_clean_reply() {
        let mut bus = Bus::new(MockTransport::new([Reply::bytes(status(4, 0, &[]))]));
        assert!(bus.ping(id(4)).is_ok());
    }

    // ---- failure behaviour ----

    #[test]
    fn silence_becomes_a_timeout_not_a_value() {
        let mut bus = quick(MockTransport::new([Reply::Silence]));
        assert!(matches!(
            bus.read(id(1), PRESENT_POSITION),
            Err(Error::Transport(TransportError::Timeout))
        ));
    }

    #[test]
    fn a_corrupt_reply_is_an_error_not_a_plausible_number() {
        let mut corrupt = status(1, 0, &[0x00, 0x08]);
        *corrupt.last_mut().unwrap() ^= 0xFF;

        let mut bus = quick(MockTransport::new([Reply::bytes(corrupt)]));
        assert!(matches!(bus.read(id(1), PRESENT_POSITION), Err(Error::Checksum { .. })));
    }

    #[test]
    fn a_reply_from_the_wrong_servo_is_rejected() {
        // Servo 2 answering a question put to servo 1 must never be believed.
        let mut bus = quick(MockTransport::new([Reply::bytes(status(2, 0, &[0x00, 0x08]))]));
        assert!(matches!(
            bus.read(id(1), PRESENT_POSITION),
            Err(Error::WrongResponder { expected: 1, actual: 2 })
        ));
    }

    #[test]
    fn a_servo_error_flag_surfaces() {
        let mut bus = quick(MockTransport::new([Reply::bytes(status(1, 0b0010_0000, &[0, 0]))]));
        assert!(matches!(
            bus.read(id(1), PRESENT_POSITION),
            Err(Error::ServoStatus { id: 1, status: 0b0010_0000 })
        ));
    }

    #[test]
    fn a_short_reply_is_a_width_mismatch_not_a_truncated_value() {
        let mut bus = quick(MockTransport::new([Reply::bytes(status(1, 0, &[0x00]))]));
        assert!(matches!(
            bus.read(id(1), PRESENT_POSITION),
            Err(Error::WidthMismatch { expected: 2, actual: 1 })
        ));
    }

    // ---- retry policy ----

    #[test]
    fn retries_a_transient_fault_then_succeeds() {
        let mut corrupt = status(1, 0, &[0x00, 0x08]);
        *corrupt.last_mut().unwrap() ^= 0xFF;

        let mut bus = Bus::new(MockTransport::new([
            Reply::bytes(corrupt),
            Reply::Silence,
            Reply::bytes(status(1, 0, &[0x00, 0x08])),
        ]))
        .with_timeout(Duration::from_millis(1))
        .with_retries(2);

        assert_eq!(bus.read(id(1), PRESENT_POSITION).unwrap(), 2048);
        assert_eq!(bus.transport().written().len(), 3, "should have made 3 attempts");
    }

    #[test]
    fn gives_up_after_the_retry_budget() {
        let mut bus = Bus::new(MockTransport::new(Vec::<Reply>::new()).lenient())
            .with_timeout(Duration::from_millis(1))
            .with_retries(2);

        assert!(bus.read(id(1), PRESENT_POSITION).is_err());
        assert_eq!(bus.transport().written().len(), 3, "1 attempt + 2 retries");
    }

    #[test]
    fn does_not_retry_a_genuine_servo_fault() {
        // Overload is a real condition; retrying just hammers a struggling
        // servo. One attempt, then report.
        let mut bus = Bus::new(MockTransport::new([Reply::bytes(status(1, 0b0010_0000, &[0, 0]))]))
            .with_timeout(Duration::from_millis(1))
            .with_retries(3);

        assert!(matches!(
            bus.read(id(1), PRESENT_POSITION),
            Err(Error::ServoStatus { .. })
        ));
        assert_eq!(bus.transport().written().len(), 1, "must not retry a real fault");
    }

    #[test]
    fn worst_case_duration_accounts_for_retries() {
        let bus = Bus::new(MockTransport::silent())
            .with_timeout(Duration::from_millis(20))
            .with_retries(2);
        assert_eq!(bus.worst_case_duration(), Duration::from_millis(60));
    }

    // ---- stale-data hygiene ----

    /// A transport that returns exactly one byte per read, modelling a reply
    /// trickling in over the wire. Forces the accumulate-and-reparse path to
    /// be walked for every byte of the packet.
    struct Dribble {
        inbound: Vec<u8>,
        pos: usize,
        writes: usize,
    }

    impl Transport for Dribble {
        fn write_all(&mut self, _data: &[u8]) -> Result<(), TransportError> {
            self.writes += 1;
            Ok(())
        }
        fn read(&mut self, buf: &mut [u8]) -> Result<usize, TransportError> {
            if self.pos >= self.inbound.len() || buf.is_empty() {
                return Ok(0);
            }
            buf[0] = self.inbound[self.pos];
            self.pos += 1;
            Ok(1)
        }
        fn clear_input(&mut self) -> Result<(), TransportError> {
            Ok(())
        }
        fn set_timeout(&mut self, _t: Duration) -> Result<(), TransportError> {
            Ok(())
        }
    }

    #[test]
    fn a_reply_arriving_one_byte_at_a_time_is_reassembled() {
        let bus = Bus::new(Dribble {
            inbound: status(1, 0, &[0x00, 0x08]),
            pos: 0,
            writes: 0,
        });
        let mut bus = bus.with_timeout(Duration::from_millis(50));
        assert_eq!(bus.read(id(1), PRESENT_POSITION).unwrap(), 2048);
        assert_eq!(bus.transport().writes, 1, "no retry should have been needed");
    }

    #[test]
    fn leading_line_noise_does_not_break_a_read() {
        let mut noisy = vec![0x00, 0x11, 0x22];
        noisy.extend_from_slice(&status(1, 0, &[0x00, 0x08]));
        let mut bus = Bus::new(MockTransport::new([Reply::bytes(noisy)]));
        assert_eq!(bus.read(id(1), PRESENT_POSITION).unwrap(), 2048);
    }

    #[test]
    fn every_transaction_clears_stale_input_first() {
        let mut bus = Bus::new(MockTransport::new([
            Reply::bytes(status(1, 0, &[0x00, 0x08])),
            Reply::bytes(status(1, 0, &[0xE8, 0x03])),
        ]));
        assert_eq!(bus.read(id(1), PRESENT_POSITION).unwrap(), 2048);
        assert_eq!(bus.read(id(1), PRESENT_POSITION).unwrap(), 1000);
    }

    // ---- sync read ----

    /// Concatenate several servos' status packets, as a sync-read reply arrives.
    fn concat(parts: &[Vec<u8>]) -> Vec<u8> {
        parts.iter().flatten().copied().collect()
    }

    #[test]
    fn sync_read_returns_a_value_per_servo_in_order() {
        let ids: Vec<_> = (1..=6).map(id).collect();
        let reply = concat(&[
            status(1, 0, &[0x00, 0x08]), // 2048
            status(2, 0, &[0xE8, 0x03]), // 1000
            status(3, 0, &[0x28, 0x0C]), // 3112
            status(4, 0, &[0x00, 0x02]), // 512
            status(5, 0, &[0x00, 0x08]), // 2048
            status(6, 0, &[0xC4, 0x09]), // 2500
        ]);
        let mut bus = Bus::new(MockTransport::new([Reply::bytes(reply)]));

        let got = bus.sync_read(&ids, PRESENT_POSITION).unwrap();
        let values: Vec<u16> = got.iter().map(|(_, r)| *r.as_ref().unwrap()).collect();
        assert_eq!(values, [2048, 1000, 3112, 512, 2048, 2500]);
        // Order matches the request, id by id.
        assert_eq!(got.iter().map(|(id, _)| id.raw()).collect::<Vec<_>>(), [1, 2, 3, 4, 5, 6]);

        // And it went out as the golden sync-read packet.
        assert_eq!(
            bus.transport().sole_request(),
            &[0xFF, 0xFF, 0xFE, 0x0A, 0x82, 0x38, 0x02, 1, 2, 3, 4, 5, 6, 0x26]
        );
    }

    #[test]
    fn sync_read_decodes_signed_loads() {
        let ids: Vec<_> = (1..=3).map(id).collect();
        let reply = concat(&[
            status(1, 0, &[0x64, 0x00]), // +100
            status(2, 0, &[0x64, 0x04]), // -100 (sign bit 10 set)
            status(3, 0, &[0x00, 0x00]), // 0
        ]);
        let mut bus = Bus::new(MockTransport::new([Reply::bytes(reply)]));

        let got = bus.sync_read(&ids, PRESENT_LOAD).unwrap();
        let loads: Vec<i32> = got.iter().map(|(_, r)| r.as_ref().unwrap().0).collect();
        assert_eq!(loads, [100, -100, 0]);
    }

    #[test]
    fn sync_read_reports_a_missing_servo_as_its_own_timeout() {
        // Servo 2 is silent; 1 and 3 answer. Only 2's entry fails.
        let ids: Vec<_> = (1..=3).map(id).collect();
        let reply = concat(&[status(1, 0, &[0x00, 0x08]), status(3, 0, &[0x28, 0x0C])]);
        let mut bus = quick(MockTransport::new([Reply::bytes(reply)]));

        let got = bus.sync_read(&ids, PRESENT_POSITION).unwrap();
        assert_eq!(*got[0].1.as_ref().unwrap(), 2048);
        assert!(matches!(
            got[1].1,
            Err(Error::Transport(TransportError::Timeout))
        ));
        assert_eq!(*got[2].1.as_ref().unwrap(), 3112);
    }

    #[test]
    fn sync_read_isolates_one_corrupt_reply() {
        // Servo 2's checksum is wrong; the resync must still deliver 1 and 3.
        let mut bad = status(2, 0, &[0xE8, 0x03]);
        *bad.last_mut().unwrap() ^= 0xFF;
        let ids: Vec<_> = (1..=3).map(id).collect();
        let reply = concat(&[status(1, 0, &[0x00, 0x08]), bad, status(3, 0, &[0x28, 0x0C])]);
        let mut bus = quick(MockTransport::new([Reply::bytes(reply)]));

        let got = bus.sync_read(&ids, PRESENT_POSITION).unwrap();
        assert_eq!(*got[0].1.as_ref().unwrap(), 2048);
        assert!(got[1].1.is_err(), "corrupt servo 2 must fail its own entry");
        assert_eq!(*got[2].1.as_ref().unwrap(), 3112);
    }

    #[test]
    fn sync_read_surfaces_a_per_servo_fault_flag() {
        let ids: Vec<_> = (1..=2).map(id).collect();
        let reply = concat(&[
            status(1, 0, &[0x00, 0x08]),
            status(2, 0b0010_0000, &[0x00, 0x00]), // overload flag
        ]);
        let mut bus = quick(MockTransport::new([Reply::bytes(reply)]));

        let got = bus.sync_read(&ids, PRESENT_POSITION).unwrap();
        assert_eq!(*got[0].1.as_ref().unwrap(), 2048);
        assert!(matches!(got[1].1, Err(Error::ServoStatus { id: 2, .. })));
    }

    #[test]
    fn sync_read_rejects_an_empty_id_list() {
        let mut bus = Bus::new(MockTransport::silent());
        assert!(matches!(bus.sync_read(&[], PRESENT_POSITION), Err(Error::EmptySync)));
    }

    #[test]
    fn sync_read_reassembles_a_reply_arriving_byte_by_byte() {
        let ids: Vec<_> = (1..=2).map(id).collect();
        let bus = Bus::new(Dribble {
            inbound: concat(&[status(1, 0, &[0x00, 0x08]), status(2, 0, &[0xE8, 0x03])]),
            pos: 0,
            writes: 0,
        });
        let mut bus = bus.with_timeout(Duration::from_millis(50));
        let got = bus.sync_read(&ids, PRESENT_POSITION).unwrap();
        assert_eq!(*got[0].1.as_ref().unwrap(), 2048);
        assert_eq!(*got[1].1.as_ref().unwrap(), 1000);
    }

    // ---- sync write ----

    #[test]
    fn sync_write_emits_the_golden_broadcast_and_expects_no_reply() {
        let entries: Vec<(ServoId, u16)> = vec![
            (id(1), 2048),
            (id(2), 1000),
            (id(3), 3000),
            (id(4), 512),
            (id(5), 2048),
            (id(6), 2500),
        ];
        // Silent: a sync-write is a broadcast, so nothing answers.
        let mut bus = Bus::new(MockTransport::silent());
        bus.sync_write(GOAL_POSITION, &entries).unwrap();

        assert_eq!(
            bus.transport().sole_request(),
            &[
                0xFF, 0xFF, 0xFE, 0x16, 0x83, 0x2A, 0x02, //
                0x01, 0x00, 0x08, 0x02, 0xE8, 0x03, 0x03, 0xB8, 0x0B, //
                0x04, 0x00, 0x02, 0x05, 0x00, 0x08, 0x06, 0xC4, 0x09, 0x9A,
            ]
        );
    }

    #[test]
    fn sync_write_does_not_wait_for_or_consume_a_reply() {
        // Even if a stray packet is present, sync_write must not read it.
        let entries: Vec<(ServoId, u16)> = vec![(id(1), 100), (id(2), 200)];
        let mut bus = Bus::new(MockTransport::silent());
        assert!(bus.sync_write(GOAL_POSITION, &entries).is_ok());
        // One write, zero reads implied by completing instantly with no reply.
        assert_eq!(bus.transport().written().len(), 1);
    }

    #[test]
    fn sync_write_rejects_an_empty_entry_list() {
        let none: [(ServoId, u16); 0] = [];
        let mut bus = Bus::new(MockTransport::silent());
        assert!(matches!(bus.sync_write(GOAL_POSITION, &none), Err(Error::EmptySync)));
    }
}
