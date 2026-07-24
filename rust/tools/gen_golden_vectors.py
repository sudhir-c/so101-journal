"""Generate golden packet vectors by running Feetech's own SDK against a stub port.

Opens nothing. The SDK's packet builders write into `port.writePort()`; we
replace the port with a recorder, so the bytes captured are exactly what the
reference implementation would have put on the wire.
"""
import sys

sys.path.insert(
    0,
    "/Users/sudhirc/Desktop/Projects/so-101-arm/.venv/lib/python3.12/site-packages",
)

from scservo_sdk.packet_handler import PacketHandler


class StubPort:
    """Implements just enough of PortHandler to satisfy the packet builders."""

    def __init__(self):
        self.is_using = False
        self.written = []

    def clearPort(self):
        pass

    def writePort(self, packet):
        self.written.append(bytes(packet))
        return len(packet)

    def readPort(self, length):
        return []

    def getBytesAvailable(self):
        return 0

    def setPacketTimeout(self, n):
        pass

    def setPacketTimeoutMillis(self, n):
        pass

    def isPacketTimeout(self):
        return True  # force rxPacket to bail immediately


def capture(fn, *args):
    port = StubPort()
    h = PacketHandler(0)  # protocol_end 0 = little-endian = STS series
    try:
        fn(h, port, *args)
    except Exception as e:  # rx side may complain; the tx bytes are already recorded
        print(f"  (rx ignored: {type(e).__name__})", file=sys.stderr)
    assert len(port.written) == 1, f"expected 1 packet, got {len(port.written)}"
    return port.written[0]


def hexs(b):
    return ", ".join(f"0x{x:02X}" for x in b)


CASES = [
    ("ping id=1", lambda h, p: h.ping(p, 1)),
    ("read Present_Position(56,2) id=1", lambda h, p: h.readTx(p, 1, 56, 2)),
    ("read Present_Load(60,2) id=3", lambda h, p: h.readTx(p, 3, 60, 2)),
    ("read Present_Temperature(63,1) id=6", lambda h, p: h.readTx(p, 6, 63, 1)),
    ("write Torque_Enable(40)=1 id=1", lambda h, p: h.write1ByteTxOnly(p, 1, 40, 1)),
    ("write Torque_Enable(40)=0 id=1", lambda h, p: h.write1ByteTxOnly(p, 1, 40, 0)),
    ("write Goal_Position(42)=2048 id=1", lambda h, p: h.write2ByteTxOnly(p, 1, 42, 2048)),
    ("write Goal_Position(42)=0 id=1", lambda h, p: h.write2ByteTxOnly(p, 1, 42, 0)),
    ("write Goal_Position(42)=4095 id=1", lambda h, p: h.write2ByteTxOnly(p, 1, 42, 4095)),
    ("write Goal_Position(42)=859 id=1 (spectre pan min)",
     lambda h, p: h.write2ByteTxOnly(p, 1, 42, 859)),
    ("write Torque_Limit(48)=500 id=2", lambda h, p: h.write2ByteTxOnly(p, 2, 48, 500)),
    ("syncRead Present_Position(56,2) ids 1-6",
     lambda h, p: h.syncReadTx(p, 56, 2, [1, 2, 3, 4, 5, 6], 6)),
    ("syncRead Present_Load(60,2) ids 1-6",
     lambda h, p: h.syncReadTx(p, 60, 2, [1, 2, 3, 4, 5, 6], 6)),
]

# sync write: id, then little-endian value, per servo
SYNC_W = []
for sid, val in [(1, 2048), (2, 1000), (3, 3000), (4, 512), (5, 2048), (6, 2500)]:
    SYNC_W += [sid, val & 0xFF, (val >> 8) & 0xFF]
CASES.append((
    "syncWrite Goal_Position(42,2) ids 1-6",
    lambda h, p: h.syncWriteTxOnly(p, 42, 2, SYNC_W, len(SYNC_W)),
))

for name, fn in CASES:
    pkt = capture(fn)
    print(f"// {name}")
    print(f"&[{hexs(pkt)}],")
    print()
