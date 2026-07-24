//! Packet framing, checksums, and response parsing.
//!
//! Pure logic over byte slices — this module performs no I/O and holds no
//! state, so all of it is exercised offline. The golden test vectors are the
//! literal bytes Feetech's own SDK emits, captured by running its packet
//! builders against a stub port.
//!
//! ```text
//!   request:  FF FF  ID  LEN  INST  PARAM...  CHK
//!   status:   FF FF  ID  LEN  ERR   PARAM...  CHK
//!
//!   LEN = param count + 2
//!   CHK = !(sum of bytes from ID through the last param) & 0xFF
//! ```

use crate::error::{Error, Result};
use crate::registers::{Reg, RegValue, Rw};
use crate::{BROADCAST_ID, ServoId};

/// Largest packet the bus will build or accept.
pub const MAX_PACKET_LEN: usize = 250;

/// Bytes of framing overhead around the parameters: `FF FF ID LEN INST` in
/// front, and the trailing checksum. Six, not five — forgetting the
/// instruction byte truncates every packet by one and lands the checksum on
/// top of the last parameter.
const OVERHEAD: usize = 6;

/// Protocol instruction codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Instruction {
    /// Probe for a servo's presence.
    Ping = 0x01,
    /// Read a span of registers.
    Read = 0x02,
    /// Write a span of registers.
    Write = 0x03,
    /// Stage a write, applied on [`Instruction::Action`].
    RegWrite = 0x04,
    /// Apply all staged writes.
    Action = 0x05,
    /// Read one register span from many servos in a single request.
    SyncRead = 0x82,
    /// Write one register span to many servos in a single request.
    SyncWrite = 0x83,
}

/// The checksum over an already-framed packet body.
///
/// Sums bytes from `ID` through the last parameter — that is, everything except
/// the two header bytes and the checksum slot itself — then inverts.
pub fn checksum(id_through_params: &[u8]) -> u8 {
    !id_through_params
        .iter()
        .fold(0u8, |acc, &b| acc.wrapping_add(b))
}

/// A framed request, built on the stack.
///
/// Fixed-capacity so the control loop allocates nothing per cycle.
#[derive(Debug, Clone)]
pub struct Packet {
    buf: [u8; MAX_PACKET_LEN],
    len: usize,
}

impl Packet {
    /// The framed bytes, ready to write to the bus.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf[..self.len]
    }

    /// Total framed length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the packet carries no bytes. Never true for a built packet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Frame `id`, `instruction` and `params` into a complete packet.
    fn build(id: u8, instruction: Instruction, params: &[u8]) -> Result<Self> {
        let total = params.len() + OVERHEAD;
        if total > MAX_PACKET_LEN {
            return Err(Error::PacketTooLong { len: total, max: MAX_PACKET_LEN });
        }

        let mut buf = [0u8; MAX_PACKET_LEN];
        buf[0] = 0xFF;
        buf[1] = 0xFF;
        buf[2] = id;
        // LEN counts the instruction, the params, and the checksum.
        buf[3] = (params.len() + 2) as u8;
        buf[4] = instruction as u8;
        buf[5..5 + params.len()].copy_from_slice(params);
        buf[total - 1] = checksum(&buf[2..total - 1]);

        Ok(Packet { buf, len: total })
    }
}

/// Probe whether a servo is present.
pub fn ping(id: ServoId) -> Result<Packet> {
    Packet::build(id.raw(), Instruction::Ping, &[])
}

/// Read `len` bytes starting at `address` from one servo.
///
/// Broadcast is rejected: every servo would answer at once and collide.
pub fn read(id: ServoId, address: u8, len: u8) -> Result<Packet> {
    if id.is_broadcast() {
        return Err(Error::BroadcastNotAllowed("read"));
    }
    Packet::build(id.raw(), Instruction::Read, &[address, len])
}

/// Read one typed register from one servo.
pub fn read_reg<T: RegValue, A>(id: ServoId, reg: Reg<T, A>) -> Result<Packet> {
    read(id, reg.address, T::WIDTH as u8)
}

/// Write raw bytes to a register span.
pub fn write(id: ServoId, address: u8, data: &[u8]) -> Result<Packet> {
    let mut params = [0u8; MAX_PACKET_LEN];
    params[0] = address;
    let n = data.len();
    if n + OVERHEAD + 1 > MAX_PACKET_LEN {
        return Err(Error::PacketTooLong { len: n + OVERHEAD + 1, max: MAX_PACKET_LEN });
    }
    params[1..1 + n].copy_from_slice(data);
    Packet::build(id.raw(), Instruction::Write, &params[..1 + n])
}

/// Write one typed register to one servo.
///
/// Takes `Reg<T, Rw>`, so a read-only register is rejected at compile time.
pub fn write_reg<T: RegValue>(id: ServoId, reg: Reg<T, Rw>, value: T) -> Result<Packet> {
    let mut data = [0u8; 4];
    value.encode(&mut data);
    write(id, reg.address, &data[..T::WIDTH])
}

/// Read the same register span from several servos in one request.
///
/// Each addressed servo replies with its own status packet, in the order the
/// ids were listed. This is a Feetech extension — Dynamixel 1.0 has no
/// sync-read.
pub fn sync_read(ids: &[ServoId], address: u8, len: u8) -> Result<Packet> {
    if ids.is_empty() {
        return Err(Error::EmptySync);
    }
    if let Some(bad) = ids.iter().find(|i| i.is_broadcast()) {
        let _ = bad;
        return Err(Error::BroadcastNotAllowed("sync_read"));
    }

    let mut params = [0u8; MAX_PACKET_LEN];
    params[0] = address;
    params[1] = len;
    for (slot, id) in params[2..2 + ids.len()].iter_mut().zip(ids) {
        *slot = id.raw();
    }
    Packet::build(BROADCAST_ID, Instruction::SyncRead, &params[..2 + ids.len()])
}

/// Read one typed register from several servos in one request.
pub fn sync_read_reg<T: RegValue, A>(ids: &[ServoId], reg: Reg<T, A>) -> Result<Packet> {
    sync_read(ids, reg.address, T::WIDTH as u8)
}

/// Write per-servo values to the same register in one request.
///
/// Sent to the broadcast address; no servo replies.
pub fn sync_write_reg<T: RegValue>(
    reg: Reg<T, Rw>,
    entries: &[(ServoId, T)],
) -> Result<Packet> {
    if entries.is_empty() {
        return Err(Error::EmptySync);
    }
    if entries.iter().any(|(id, _)| id.is_broadcast()) {
        return Err(Error::BroadcastNotAllowed("sync_write entry"));
    }

    let stride = 1 + T::WIDTH;
    let param_len = 2 + entries.len() * stride;
    if param_len + OVERHEAD > MAX_PACKET_LEN {
        return Err(Error::PacketTooLong {
            len: param_len + OVERHEAD,
            max: MAX_PACKET_LEN,
        });
    }

    let mut params = [0u8; MAX_PACKET_LEN];
    params[0] = reg.address;
    params[1] = T::WIDTH as u8;
    for (i, (id, value)) in entries.iter().enumerate() {
        let at = 2 + i * stride;
        params[at] = id.raw();
        value.encode(&mut params[at + 1..at + 1 + T::WIDTH]);
    }

    Packet::build(BROADCAST_ID, Instruction::SyncWrite, &params[..param_len])
}

/// A parsed status packet, borrowing its parameters from the read buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatusPacket<'a> {
    /// Servo that replied.
    pub id: u8,
    /// Servo-defined error flags; zero means healthy.
    pub error: u8,
    /// Payload bytes, if the instruction returns data.
    pub params: &'a [u8],
}

impl StatusPacket<'_> {
    /// Convert a nonzero error byte into an [`Error`].
    pub fn check_status(&self) -> Result<()> {
        if self.error != 0 {
            return Err(Error::ServoStatus { id: self.id, status: self.error });
        }
        Ok(())
    }

    /// Decode the payload as a typed register value.
    pub fn value<T: RegValue>(&self) -> Result<T> {
        if self.params.len() != T::WIDTH {
            return Err(Error::WidthMismatch {
                expected: T::WIDTH,
                actual: self.params.len(),
            });
        }
        Ok(T::decode(self.params))
    }
}

/// Result of attempting to parse a status packet from a partial buffer.
#[derive(Debug)]
pub enum Parsed<'a> {
    /// A complete, checksum-valid packet, plus how many bytes it consumed
    /// from the front of the buffer (including any leading junk skipped).
    Complete {
        /// The packet.
        packet: StatusPacket<'a>,
        /// Bytes consumed from the front of the input.
        consumed: usize,
    },
    /// The buffer holds a valid prefix but not yet a whole packet. Read more
    /// bytes and call again — this is not an error.
    NeedMore,
}

/// Parse one status packet from the front of `buf`.
///
/// Leading bytes that are not a frame start are skipped: on a half-duplex bus
/// a little line noise before a reply is normal. A checksum failure, by
/// contrast, is a hard error — a corrupt packet is never partially believed.
pub fn parse_status(buf: &[u8]) -> Result<Parsed<'_>> {
    // Find `FF FF` followed by a byte that could be an id. Ids max out at 0xFE,
    // so a third 0xFF means we are still inside run-in padding.
    let mut start = None;
    for i in 0..buf.len().saturating_sub(1) {
        if buf[i] == 0xFF && buf[i + 1] == 0xFF {
            match buf.get(i + 2) {
                Some(&0xFF) => continue, // padding; the real header starts later
                Some(_) => {
                    start = Some(i);
                    break;
                }
                None => return Ok(Parsed::NeedMore), // header, but nothing after it
            }
        }
    }

    let Some(start) = start else {
        // No header yet. If the tail could still be the start of one, wait.
        return if buf.len() < 2 || buf[buf.len() - 1] == 0xFF {
            Ok(Parsed::NeedMore)
        } else {
            Err(Error::BadHeader { scanned: buf.len() })
        };
    };

    let frame = &buf[start..];
    if frame.len() < 4 {
        return Ok(Parsed::NeedMore);
    }

    let id = frame[2];
    let declared = frame[3] as usize;
    if declared < 2 {
        return Err(Error::Truncated { declared, received: frame.len() });
    }

    let total = declared + 4;
    if total > MAX_PACKET_LEN {
        return Err(Error::PacketTooLong { len: total, max: MAX_PACKET_LEN });
    }
    if frame.len() < total {
        return Ok(Parsed::NeedMore);
    }

    let computed = checksum(&frame[2..total - 1]);
    let received = frame[total - 1];
    if computed != received {
        return Err(Error::Checksum { computed, received });
    }

    Ok(Parsed::Complete {
        packet: StatusPacket { id, error: frame[4], params: &frame[5..total - 1] },
        consumed: start + total,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registers::*;

    fn id(n: u8) -> ServoId {
        ServoId::new(n).unwrap()
    }

    // ---------------------------------------------------------------------
    // Golden vectors.
    //
    // Every expected byte string below was produced by running Feetech's own
    // SDK (`scservo_sdk`, vendored in .venv) against a stub port that records
    // writes instead of opening hardware. They are the reference
    // implementation's actual output, not hand-computed values.
    // ---------------------------------------------------------------------

    #[test]
    fn golden_ping() {
        assert_eq!(ping(id(1)).unwrap().as_bytes(), &[0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB]);
    }

    #[test]
    fn golden_read_present_position() {
        assert_eq!(
            read_reg(id(1), PRESENT_POSITION).unwrap().as_bytes(),
            &[0xFF, 0xFF, 0x01, 0x04, 0x02, 0x38, 0x02, 0xBE]
        );
    }

    #[test]
    fn golden_read_present_load() {
        assert_eq!(
            read_reg(id(3), PRESENT_LOAD).unwrap().as_bytes(),
            &[0xFF, 0xFF, 0x03, 0x04, 0x02, 0x3C, 0x02, 0xB8]
        );
    }

    #[test]
    fn golden_read_one_byte_register() {
        assert_eq!(
            read_reg(id(6), PRESENT_TEMPERATURE).unwrap().as_bytes(),
            &[0xFF, 0xFF, 0x06, 0x04, 0x02, 0x3F, 0x01, 0xB3]
        );
    }

    #[test]
    fn golden_write_torque_enable() {
        assert_eq!(
            write_reg(id(1), TORQUE_ENABLE, 1).unwrap().as_bytes(),
            &[0xFF, 0xFF, 0x01, 0x04, 0x03, 0x28, 0x01, 0xCE]
        );
        assert_eq!(
            write_reg(id(1), TORQUE_ENABLE, 0).unwrap().as_bytes(),
            &[0xFF, 0xFF, 0x01, 0x04, 0x03, 0x28, 0x00, 0xCF]
        );
    }

    #[test]
    fn golden_write_goal_position() {
        assert_eq!(
            write_reg(id(1), GOAL_POSITION, 2048).unwrap().as_bytes(),
            &[0xFF, 0xFF, 0x01, 0x05, 0x03, 0x2A, 0x00, 0x08, 0xC4]
        );
        assert_eq!(
            write_reg(id(1), GOAL_POSITION, 0).unwrap().as_bytes(),
            &[0xFF, 0xFF, 0x01, 0x05, 0x03, 0x2A, 0x00, 0x00, 0xCC]
        );
        assert_eq!(
            write_reg(id(1), GOAL_POSITION, 4095).unwrap().as_bytes(),
            &[0xFF, 0xFF, 0x01, 0x05, 0x03, 0x2A, 0xFF, 0x0F, 0xBE]
        );
        // spectre's calibrated shoulder_pan minimum
        assert_eq!(
            write_reg(id(1), GOAL_POSITION, 859).unwrap().as_bytes(),
            &[0xFF, 0xFF, 0x01, 0x05, 0x03, 0x2A, 0x5B, 0x03, 0x6E]
        );
    }

    #[test]
    fn golden_write_torque_limit() {
        assert_eq!(
            write_reg(id(2), TORQUE_LIMIT, 500).unwrap().as_bytes(),
            &[0xFF, 0xFF, 0x02, 0x05, 0x03, 0x30, 0xF4, 0x01, 0xD0]
        );
    }

    #[test]
    fn golden_sync_read_all_six_joints() {
        let ids: Vec<_> = (1..=6).map(id).collect();
        assert_eq!(
            sync_read_reg(&ids, PRESENT_POSITION).unwrap().as_bytes(),
            &[0xFF, 0xFF, 0xFE, 0x0A, 0x82, 0x38, 0x02, 1, 2, 3, 4, 5, 6, 0x26]
        );
        assert_eq!(
            sync_read_reg(&ids, PRESENT_LOAD).unwrap().as_bytes(),
            &[0xFF, 0xFF, 0xFE, 0x0A, 0x82, 0x3C, 0x02, 1, 2, 3, 4, 5, 6, 0x22]
        );
    }

    #[test]
    fn golden_sync_write_all_six_joints() {
        let entries: Vec<(ServoId, u16)> = vec![
            (id(1), 2048),
            (id(2), 1000),
            (id(3), 3000),
            (id(4), 512),
            (id(5), 2048),
            (id(6), 2500),
        ];
        assert_eq!(
            sync_write_reg(GOAL_POSITION, &entries).unwrap().as_bytes(),
            &[
                0xFF, 0xFF, 0xFE, 0x16, 0x83, 0x2A, 0x02, //
                0x01, 0x00, 0x08, //
                0x02, 0xE8, 0x03, //
                0x03, 0xB8, 0x0B, //
                0x04, 0x00, 0x02, //
                0x05, 0x00, 0x08, //
                0x06, 0xC4, 0x09, //
                0x9A,
            ]
        );
    }

    // ---- checksum ----

    #[test]
    fn checksum_matches_the_documented_formula() {
        // !(0x01 + 0x02 + 0x01) & 0xFF
        assert_eq!(checksum(&[0x01, 0x02, 0x01]), 0xFB);
    }

    #[test]
    fn checksum_wraps_rather_than_overflowing() {
        assert_eq!(checksum(&[0xFF, 0xFF]), !0xFEu8);
        assert_eq!(checksum(&[0x80, 0x80, 0x80]), !0x80u8);
    }

    #[test]
    fn every_built_packet_carries_a_valid_checksum() {
        let p = write_reg(id(4), GOAL_POSITION, 1234).unwrap();
        let b = p.as_bytes();
        assert_eq!(checksum(&b[2..b.len() - 1]), b[b.len() - 1]);
    }

    // ---- construction guards ----

    #[test]
    fn read_rejects_broadcast() {
        assert!(matches!(
            read(ServoId::BROADCAST, 56, 2),
            Err(Error::BroadcastNotAllowed("read"))
        ));
    }

    #[test]
    fn sync_ops_reject_an_empty_id_list() {
        assert!(matches!(sync_read(&[], 56, 2), Err(Error::EmptySync)));
        let none: [(ServoId, u16); 0] = [];
        assert!(matches!(sync_write_reg(GOAL_POSITION, &none), Err(Error::EmptySync)));
    }

    #[test]
    fn sync_write_rejects_a_packet_that_would_overrun() {
        let entries: Vec<(ServoId, u16)> = (1..=100).map(|i| (id(i), 0u16)).collect();
        assert!(matches!(
            sync_write_reg(GOAL_POSITION, &entries),
            Err(Error::PacketTooLong { .. })
        ));
    }

    // ---- parsing ----

    /// Frame a status packet the way a servo would, for parser tests.
    fn status(id: u8, error: u8, params: &[u8]) -> Vec<u8> {
        let mut p = vec![0xFF, 0xFF, id, (params.len() + 2) as u8, error];
        p.extend_from_slice(params);
        let ck = checksum(&p[2..]);
        p.push(ck);
        p
    }

    #[test]
    fn parses_a_position_reply() {
        // 2048 counts, little-endian.
        let raw = status(1, 0, &[0x00, 0x08]);
        let Parsed::Complete { packet, consumed } = parse_status(&raw).unwrap() else {
            panic!("expected a complete packet");
        };
        assert_eq!(packet.id, 1);
        assert_eq!(packet.error, 0);
        assert_eq!(consumed, raw.len());
        assert_eq!(packet.value::<u16>().unwrap(), 2048);
    }

    #[test]
    fn parses_a_negative_load_reply() {
        let raw = status(3, 0, &[0x64, 0x04]); // 0x0464 -> sign bit 10 set, mag 100
        let Parsed::Complete { packet, .. } = parse_status(&raw).unwrap() else {
            panic!("expected a complete packet");
        };
        assert_eq!(packet.value::<SignMag<10>>().unwrap(), SignMag(-100));
    }

    #[test]
    fn parses_a_ping_reply_with_no_params() {
        let raw = status(1, 0, &[]);
        let Parsed::Complete { packet, .. } = parse_status(&raw).unwrap() else {
            panic!("expected a complete packet");
        };
        assert_eq!(packet.id, 1);
        assert!(packet.params.is_empty());
    }

    #[test]
    fn surfaces_a_servo_error_byte() {
        let raw = status(2, 0b0010_0000, &[]);
        let Parsed::Complete { packet, .. } = parse_status(&raw).unwrap() else {
            panic!("expected a complete packet");
        };
        assert!(matches!(
            packet.check_status(),
            Err(Error::ServoStatus { id: 2, status: 0b0010_0000 })
        ));
    }

    #[test]
    fn rejects_a_corrupted_checksum() {
        let mut raw = status(1, 0, &[0x00, 0x08]);
        *raw.last_mut().unwrap() ^= 0xFF;
        assert!(matches!(parse_status(&raw), Err(Error::Checksum { .. })));
    }

    #[test]
    fn rejects_a_corrupted_payload() {
        // A flipped payload bit must fail the checksum, not silently decode.
        let mut raw = status(1, 0, &[0x00, 0x08]);
        raw[5] ^= 0x01;
        assert!(matches!(parse_status(&raw), Err(Error::Checksum { .. })));
    }

    #[test]
    fn waits_for_more_bytes_rather_than_erroring() {
        let raw = status(1, 0, &[0x00, 0x08]);
        for n in 0..raw.len() {
            assert!(
                matches!(parse_status(&raw[..n]), Ok(Parsed::NeedMore)),
                "prefix of {n} byte(s) should ask for more, not fail"
            );
        }
    }

    #[test]
    fn skips_leading_line_noise() {
        let mut raw = vec![0x00, 0x12, 0x34];
        let pkt = status(1, 0, &[0x00, 0x08]);
        raw.extend_from_slice(&pkt);

        let Parsed::Complete { packet, consumed } = parse_status(&raw).unwrap() else {
            panic!("expected a complete packet");
        };
        assert_eq!(packet.value::<u16>().unwrap(), 2048);
        assert_eq!(consumed, raw.len(), "consumed count must include skipped junk");
    }

    #[test]
    fn skips_run_in_padding_before_the_real_header() {
        // Extra 0xFF run-in: ids never reach 0xFF, so these are padding.
        let mut raw = vec![0xFF, 0xFF, 0xFF];
        raw.extend_from_slice(&status(1, 0, &[0x00, 0x08])[2..]);

        let Parsed::Complete { packet, .. } = parse_status(&raw).unwrap() else {
            panic!("expected a complete packet");
        };
        assert_eq!(packet.id, 1);
        assert_eq!(packet.value::<u16>().unwrap(), 2048);
    }

    #[test]
    fn consumes_exactly_one_packet_from_a_back_to_back_pair() {
        // Sync-read replies arrive as consecutive status packets.
        let mut raw = status(1, 0, &[0x00, 0x08]);
        let second = status(2, 0, &[0xE8, 0x03]);
        raw.extend_from_slice(&second);

        let Parsed::Complete { packet, consumed } = parse_status(&raw).unwrap() else {
            panic!("expected a complete packet");
        };
        assert_eq!(packet.id, 1);

        let Parsed::Complete { packet, .. } = parse_status(&raw[consumed..]).unwrap() else {
            panic!("expected a second complete packet");
        };
        assert_eq!(packet.id, 2);
        assert_eq!(packet.value::<u16>().unwrap(), 1000);
    }

    #[test]
    fn rejects_a_payload_of_the_wrong_width() {
        let raw = status(1, 0, &[0x00]); // one byte where a u16 is expected
        let Parsed::Complete { packet, .. } = parse_status(&raw).unwrap() else {
            panic!("expected a complete packet");
        };
        assert!(matches!(
            packet.value::<u16>(),
            Err(Error::WidthMismatch { expected: 2, actual: 1 })
        ));
    }

    #[test]
    fn rejects_an_impossible_declared_length() {
        assert!(matches!(
            parse_status(&[0xFF, 0xFF, 0x01, 0x00, 0x00, 0x00]),
            Err(Error::Truncated { .. })
        ));
    }

    #[test]
    fn reports_junk_with_no_header_at_all() {
        assert!(matches!(
            parse_status(&[0x01, 0x02, 0x03, 0x04]),
            Err(Error::BadHeader { scanned: 4 })
        ));
    }
}
