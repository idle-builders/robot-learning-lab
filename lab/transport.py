"""Host encoder for the target's RLB1 command stream. No device opened implicitly."""

import math
import struct
import zlib


def encode_command(kind, sequence, session, values=()):
    types = {"command": 1, "stop": 2, "arm": 3}
    if kind not in types or not 0 <= sequence < 2**64 or not 0 < session < 2**64:
        raise ValueError("invalid frame type, sequence or session")
    if (kind == "command" and not 1 <= len(values) <= 32) or (
        kind != "command" and values
    ):
        raise ValueError("invalid joint vector")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite target")
    payload = struct.pack(f"<{len(values)}f", *values)
    frame = (
        struct.pack(
            "<4sBBHQQ",
            b"RLB1",
            types[kind],
            len(values),
            len(payload),
            sequence,
            session,
        )
        + payload
    )
    return frame + struct.pack("<I", zlib.crc32(frame))
