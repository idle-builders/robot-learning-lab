"""Bounded supported-rig host link; USB writes never imply target acceptance.

Not RobotIO: hardware has no simulated reset/advance or invented effort/pose.
No port discovery, implicit open, automatic arming, retries or reconnection.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import re
import time
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from .transport import encode_command

U64 = Annotated[int, Field(strict=True, ge=0, lt=2**64)]
U32 = Annotated[int, Field(strict=True, ge=0, lt=2**32)]
Byte = Annotated[int, Field(strict=True, ge=0, le=255)]
Flag = Annotated[int, Field(strict=True, ge=0, le=1)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Feedback(Strict):
    id: Annotated[int, Field(ge=1, le=253)]
    sample_ms: U64
    bytes: Annotated[list[Byte], Field(min_length=15, max_length=15)]


class ImuTelemetry(Strict):
    part: Literal["LSM6DSOTR"]
    config: Digest
    frame: Literal["sensor"]
    calibrated: Literal[False]
    status: Annotated[int, Field(ge=0, le=4)]
    error: Annotated[int, Field(ge=0, le=5)]
    valid: bool
    sequence: U32
    poll_start_us: U64
    earliest_update_us: U64
    read_end_us: U64
    max_poll_gap_us: U64
    saturated: Byte
    gyro_rad_s: Annotated[list[float], Field(min_length=3, max_length=3)]
    specific_force_m_s2: Annotated[list[float], Field(min_length=3, max_length=3)]


class Telemetry(Strict):
    schema_version: Annotated[int, Field(ge=1, le=1, alias="schema")]
    robot: str
    map: Digest
    session: Annotated[str, Field(pattern=r"^[0-9a-f]{16}$")]
    time_ms: U64
    status: Annotated[int, Field(ge=0, le=4)]
    armed: Flag
    calibrated: Flag
    voltage_v: Annotated[float, Field(ge=0)]
    rejected: U64
    rx_expired: U64 | None = None
    raw_feedback: Annotated[list[Feedback], Field(min_length=1, max_length=32)]
    imu: ImuTelemetry


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


class TelemetryDecoder:
    """Bounded partial-line storage; discard an oversized line through newline."""

    def __init__(self, maximum_line_bytes=16384):
        if (
            type(maximum_line_bytes) is not int
            or not 256 <= maximum_line_bytes <= 65536
        ):
            raise ValueError("invalid line bound")
        self.limit = maximum_line_bytes
        self.buffer = bytearray()
        self.discarding = False
        self.rejected_lines = 0

    def feed(self, data):
        if not isinstance(data, bytes) or len(data) > 65536:
            raise ValueError("bounded bytes chunk required")
        events = []
        for byte in data:
            if byte == 10:
                if self.discarding:
                    self.discarding = False
                    continue
                line = bytes(self.buffer)
                self.buffer.clear()
                try:
                    value = json.loads(line, object_pairs_hook=no_duplicate_keys)
                    frame = Telemetry.model_validate(value)
                    events.append(
                        dict(kind="telemetry", value=frame.model_dump(by_alias=True, exclude_none=True))
                    )
                except (ValueError, ValidationError, UnicodeError, RecursionError):
                    self.rejected_lines += 1
                    events.append(dict(kind="invalid_line", raw_hex=line.hex()))
            elif not self.discarding:
                if len(self.buffer) == self.limit:
                    self.buffer.clear()
                    self.discarding = True
                    self.rejected_lines += 1
                    events.append(dict(kind="oversized_line", limit_bytes=self.limit))
                else:
                    self.buffer.append(byte)
        return events


@dataclass(frozen=True)
class WriteReceipt:
    sequence: int
    session: str
    kind: str
    requested_bytes: int
    written_bytes: int | None
    host_issue_s: float
    host_return_s: float
    transport_status: str
    device_accepted: None = None
    first_apply_time: None = None


class HardwareLink:
    def __init__(
        self,
        stream,
        mapping,
        imu_hash,
        *,
        allow_commands=False,
        freshness_s=0.25,
        clock=time.monotonic,
    ):
        if (
            type(allow_commands) is not bool
            or type(mapping.get("calibrated")) is not bool
        ):
            raise ValueError("explicit boolean command/calibration flags required")
        canonical = deepcopy(mapping)
        digest = canonical.pop("map_hash", None)
        if (
            digest
            != hashlib.sha256(
                json.dumps(canonical, sort_keys=True).encode()
            ).hexdigest()
        ):
            raise ValueError("hardware map digest mismatch")
        if not re.fullmatch(r"[0-9a-f]{64}", imu_hash):
            raise ValueError("IMU configuration hash required")
        if not math.isfinite(freshness_s) or not 0 < freshness_s <= 1:
            raise ValueError("invalid host-arrival freshness interval")
        joints = canonical.get("joints", [])
        if not 1 <= len(joints) <= 32 or len({r["id"] for r in joints}) != len(joints):
            raise ValueError("unique global IDs required by target telemetry")
        if len({r["name"] for r in joints}) != len(joints):
            raise ValueError("unique joint names required")
        self.stream, self.mapping, self.imu_hash = stream, deepcopy(mapping), imu_hash
        self.allow_commands, self.freshness_s, self.clock = (
            allow_commands,
            freshness_s,
            clock,
        )
        self.decoder = TelemetryDecoder()
        self.last_chunk = b""
        self._latest = None
        self._session = None
        self._last_clock = None
        self._sequence = 0
        self.blocked_reason = None
        self.closed = False

    @property
    def latest(self):
        return deepcopy(self._latest)

    def _now(self):
        now = self.clock()
        if (
            not math.isfinite(now)
            or now < 0
            or (self._last_clock is not None and now < self._last_clock)
        ):
            self.blocked_reason = "host_clock_regression"
            raise RuntimeError(self.blocked_reason)
        self._last_clock = now
        return now

    def poll(self):
        if self.closed:
            raise RuntimeError("link closed")
        try:
            data = self.stream.read(4096)
        except Exception:
            self.blocked_reason = "read_failure"
            raise
        self.last_chunk = data
        arrival = self._now()
        try:
            events = self.decoder.feed(data)
        except Exception:
            self.blocked_reason = "invalid_read"
            raise
        for event in events:
            event["host_arrival_s"] = arrival
            if event["kind"] != "telemetry":
                continue
            frame = event["value"]
            if (
                frame["robot"] != self.mapping["robot"]
                or frame["map"] != self.mapping["map_hash"]
                or frame["imu"]["config"] != self.imu_hash
                or frame["calibrated"] != int(self.mapping["calibrated"])
                or [r["id"] for r in frame["raw_feedback"]]
                != [r["id"] for r in self.mapping["joints"]]
                or int(frame["session"], 16) == 0
            ):
                self.blocked_reason = "identity_mismatch"
                event["kind"] = "rejected_identity"
                continue
            if self._session is not None and self._session != frame["session"]:
                self.blocked_reason = "device_restarted"
            elif self._latest is not None and (
                frame["time_ms"] < self._latest["device"]["time_ms"]
                or frame["rejected"] < self._latest["device"]["rejected"]
                or (frame.get("rx_expired") is not None
                    and self._latest["device"].get("rx_expired") is not None
                    and frame["rx_expired"] < self._latest["device"]["rx_expired"])
            ):
                self.blocked_reason = "device_counter_regression"
            # Keep reboot observations, but never automatically restore motion authority.
            self._session = frame["session"]
            self._latest = dict(
                host_arrival_s=arrival,
                device=deepcopy(frame),
                clocks_synchronized=False,
                source="device_telemetry",
                feedback_availability={
                    row["name"]: "sampled_raw" if raw["sample_ms"] else "unavailable"
                    for row, raw in zip(
                        self.mapping["joints"], frame["raw_feedback"], strict=True
                    )
                },
                effort_nm=None,
                root_pose=None,
            )
            event["motion_blocked_reason"] = self.blocked_reason
        return events

    def _write(self, kind, values=()):
        if self.closed or not self.allow_commands:
            raise RuntimeError("command transport disabled")
        now = self._now()
        if self._latest is None:
            raise RuntimeError("no identified target session")
        # STOP can be requested despite stale telemetry or a local motion latch.
        # It still has unknown target acceptance and cannot prove torque removal.
        if kind != "stop":
            if self.blocked_reason:
                raise RuntimeError(self.blocked_reason)
            if now - self._latest["host_arrival_s"] >= self.freshness_s:
                self.blocked_reason = "telemetry_stale"
                raise RuntimeError(self.blocked_reason)
            frame = self._latest["device"]
            if not self.mapping["calibrated"] or not frame["calibrated"]:
                raise RuntimeError("target calibration unavailable")
            if frame["status"] in (3, 4):
                raise RuntimeError("target stop/fault is latched")
            if kind == "command" and not frame["armed"]:
                raise RuntimeError("no reported arm state")
        if self._sequence >= 2**64:
            raise RuntimeError("sequence exhausted")
        sequence = self._sequence
        packet = encode_command(kind, sequence, int(self._session, 16), values)
        self._sequence += 1  # Never retry an ambiguous write using the same sequence.
        written = None
        try:
            written = self.stream.write(packet)
            status = (
                "written"
                if type(written) is int and written == len(packet)
                else "partial_write"
            )
        except Exception:
            status = "write_exception"
        returned = self._now()
        if status != "written":
            self.blocked_reason = "ambiguous_write"
        if kind == "stop":
            self.blocked_reason = "stop_requested"
        return WriteReceipt(
            sequence, self._session, kind, len(packet), written, now, returned, status
        )

    def arm(self):
        return self._write("arm")

    def command(self, named_targets):
        joints = self.mapping["joints"]
        if not isinstance(named_targets, dict) or set(named_targets) != {
            j["name"] for j in joints
        }:
            raise ValueError("complete named mixed command required")
        values = []
        for row in joints:
            value = named_targets[row["name"]]
            lower, upper = (
                (-row["speed"], row["speed"])
                if row["mode"]
                else (row["lower"], row["upper"])
            )
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not lower <= value <= upper
            ):
                raise ValueError(f"invalid {row['command_unit']} target: {row['name']}")
            values.append(value)
        return self._write("command", values)

    def stop(self):
        return self._write("stop")

    def close(self):
        # Closing the host port does not acknowledge STOP or guarantee physical safety.
        self.closed = True
        self.blocked_reason = "link_closed"
        self.stream.close()


def open_serial(port, mapping, imu_hash, *, allow_commands=False):
    """Explicit port only. No discovery, network serial URLs, reset or auto ARM."""
    if not isinstance(port, str) or not port or "://" in port:
        raise ValueError("explicit native device path required")
    import serial  # Optional requirements-hardware.txt, not part of simulation runtime.

    # Validate identity configuration before opening a physical port.
    link = HardwareLink(None, mapping, imu_hash, allow_commands=allow_commands)
    device = serial.Serial(
        port=None, baudrate=115200, timeout=0, write_timeout=0.02, exclusive=True
    )
    # Pico SDK stdio_usb_connected checks DTR by default. 115200 avoids the
    # SDK 1200-baud bootloader convention; DTR is connectivity, not ARM.
    device.dtr = True
    device.rts = False
    device.port = port
    try:
        device.open()
    except Exception:
        device.close()
        raise
    link.stream = device
    return link
