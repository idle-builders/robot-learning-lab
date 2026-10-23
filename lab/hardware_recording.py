"""Project sealed read-only serial captures into canonical recordings, offline only.

Host arrival is the sole event clock. MCU timestamps stay in original raw records;
no acquisition-time mapping, calibration, joint decoding or physical pose is inferred.
"""

import argparse
import hashlib
import json
from pathlib import Path

from .artifact_paths import checked_path
from .hardware_link import HardwareLink
from .recording import EventWriter, Signal, encoded, ns, read, sha256, verify_recording


class ReplayStream:
    data = b""

    def read(self, size):
        if len(self.data) > size:
            raise ValueError("capture chunk exceeds original read size")
        return self.data


def checked_capture(folder):
    """Validate capture seals and reproduce every parsed event from original bytes."""
    folder = Path(folder)
    report = read(folder / "report.json")
    if (
        report.get("schema_version") != 1
        or report.get("source_kind") not in ("hardware_asserted", "synthetic_transport")
        or report.get("status") not in ("pass", "partial", "fail")
    ):
        raise ValueError("unsupported capture identity/status")
    if (
        report.get("command_writes") != 0
        or report.get("clocks_synchronized") is not False
    ):
        raise ValueError("only read-only unsynchronized captures are supported")
    inputs = {"capture-report.json": (folder / "report.json").read_bytes()}
    for name, expected in report["output_hashes"].items():
        path = checked_path(folder, name)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"capture output changed: {name}")
        key = "capture-" + name.replace("/", "__")
        if key in inputs:
            raise ValueError("duplicate capture archive name")
        inputs[key] = data
    for name, expected in report["source_hashes"].items():
        path = checked_path(folder / "source", name)
        if sha256(path) != expected:
            raise ValueError(f"capture source changed: {name}")
    required = {"source/hardware-map.json", "source/imu-config.json"}
    if not required.issubset(report["output_hashes"]):
        raise ValueError("capture did not seal hardware identity")
    mapping = read(folder / "source/hardware-map.json")
    imu = read(folder / "source/imu-config.json")
    imu_hash = hashlib.sha256(json.dumps(imu, sort_keys=True).encode()).hexdigest()
    stream, now = ReplayStream(), [0.0]
    link = HardwareLink(stream, mapping, imu_hash, clock=lambda: now[0])
    replayed, chunks = [], []
    for name in ("chunks.jsonl", "events.jsonl"):
        if (folder / name).exists() and name not in report["output_hashes"]:
            raise ValueError(f"unsealed capture stream: {name}")
    if "chunks.jsonl" in report["output_hashes"]:
        chunks = [
            json.loads(line)
            for line in (folder / "chunks.jsonl").read_text().splitlines()
        ]
    for chunk in chunks:
        now[0] = chunk["host_arrival_s"]
        stream.data = bytes.fromhex(chunk["bytes_hex"])
        replayed.extend(link.poll())
    parsed = (
        [
            json.loads(line)
            for line in (folder / "events.jsonl").read_text().splitlines()
        ]
        if "events.jsonl" in report["output_hashes"]
        else []
    )
    if encoded(parsed) != encoded(replayed):
        raise ValueError("parsed events differ from original-byte replay")
    telemetry = sum(e["kind"] == "telemetry" for e in replayed)
    if (
        telemetry != report["telemetry_frames"]
        or len(replayed) - telemetry != report["diagnostics"]
    ):
        raise ValueError("capture event counts disagree")
    if report["status"] != "fail":
        if (
            report.get("pending_line_bytes") != len(link.decoder.buffer)
            or report.get("discarding_line") != link.decoder.discarding
            or report.get("blocked_reason") != link.blocked_reason
        ):
            raise ValueError("capture final parser/identity state disagrees")
        clean = (
            telemetry > 0
            and not report["diagnostics"]
            and not link.blocked_reason
            and not link.decoder.buffer
            and not link.decoder.discarding
        )
        if (report["status"] == "pass") != bool(clean):
            raise ValueError("capture status disagrees with retained data")
    return report, mapping, replayed, chunks, inputs


def signal_catalog(mapping, origin):
    role = "measured" if origin == "hardware_asserted" else "diagnostic"
    signals = {}
    for name, description in (
        (
            "transport/event",
            "Complete original event, including exact MCU clocks, boot session, raw feedback and identity faults",
        ),
        (
            "transport/capture",
            "Original capture outcome and unresolved timing/identity state; not task success",
        ),
    ):
        signals[name] = Signal(
            payload="label", role="diagnostic", unit="JSON", description=description
        )
    for name, unit, description in (
        (
            "power/voltage_v",
            "V",
            "Reported ADC-derived bus voltage; physical ADC/divider calibration unqualified",
        ),
        (
            "supervisor/status",
            "code",
            "Reported target guard status; not per-command acceptance",
        ),
        ("supervisor/armed", "1", "Reported arm flag; not confirmed torque or motion"),
    ):
        signals[name] = Signal(
            role=role if name.startswith("power/") else "diagnostic",
            unit=unit,
            shape=[],
            description=description,
        )
    for field, unit in (("gyro_rad_s", "rad/s"), ("specific_force_m_s2", "m/s^2")):
        signals["imu/" + field] = Signal(
            role=role,
            unit=unit,
            shape=[3],
            labels=["x", "y", "z"],
            frame="imu_sensor",
            description="Uncalibrated sensor-frame driver output. Hardware origin is asserted; synthetic origins are diagnostic. Not transformed to torso or synchronized.",
        )
    for name, unit, shape, frame in (
        ("joint/position_rad", "rad", [len(mapping["joints"])], "joint_local"),
        ("joint/velocity_rad_s", "rad/s", [len(mapping["joints"])], "joint_local"),
        ("joint/effort_nm", "N*m", [len(mapping["joints"])], "joint_local"),
        ("base/position_m", "m", [3], "world"),
        ("base/quaternion_xyzw", "1", [4], "world"),
    ):
        signals[name] = Signal(
            role=role,
            unit=unit,
            shape=shape,
            frame=frame,
            labels=[j["name"] for j in mapping["joints"]]
            if name.startswith("joint/")
            else None,
            description="Unavailable: no calibrated joint decoder, torque measurement or base estimator in this adapter",
        )
    for name in ("camera/rgb", "command/acceptance", "command/application_time"):
        signals[name] = Signal(
            payload="unavailable",
            role="diagnostic",
            unit="unavailable",
            description="Not provided by read-only raw telemetry capture",
        )
    return signals


def export_hardware_recording(folder, output):
    folder, output = Path(folder), Path(output)
    report, mapping, events, chunks, inputs = checked_capture(folder)
    for module in (
        "hardware_recording.py",
        "hardware_link.py",
        "recording.py",
        "transport.py",
    ):
        inputs["adapter-" + module] = Path(__file__).with_name(module).read_bytes()
    output.mkdir(parents=True, exist_ok=False)
    (output / "source").mkdir()
    for name, data in inputs.items():
        (output / "source" / name).write_bytes(data)
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in inputs.items()}
    signals = signal_catalog(mapping, report["source_kind"])
    clock = "host_arrival_relative"
    start_ns = ns(chunks[0]["host_arrival_s"]) if chunks else 0
    writer = EventWriter(output / "events.jsonl", signals, clock=clock)
    note = "Host arrival only; device acquisition clock is unmapped. No sensor calibration or physical qualification inferred."

    def emit(name, value, t, validity="valid", detail=note):
        writer.append(
            dict(
                sequence=writer.count,
                signal=name,
                record_time_ns=t,
                capture_time_ns=None,
                clock=clock,
                validity=validity,
                value=value,
                note=detail,
            )
        )

    try:
        emit("transport/capture", encoded(report), 0)
        for event in events:
            t = ns(event["host_arrival_s"]) - start_ns
            emit("transport/event", encoded(event), t)
            if event["kind"] != "telemetry":
                continue
            frame = event["value"]
            for name, key in (
                ("power/voltage_v", "voltage_v"),
                ("supervisor/status", "status"),
                ("supervisor/armed", "armed"),
            ):
                emit(name, frame[key], t)
            imu = frame["imu"]
            for key in ("gyro_rad_s", "specific_force_m_s2"):
                validity = (
                    "unavailable"
                    if imu["sequence"] == 0 and imu["read_end_us"] == 0
                    else (
                        "valid"
                        if imu["valid"]
                        and not imu["saturated"]
                        and imu["status"] == 3
                        and imu["error"] == 0
                        else "invalid"
                    )
                )
                emit(
                    "imu/" + key,
                    None if validity == "unavailable" else imu[key],
                    t,
                    validity,
                )
            for name in signals:
                if name.startswith(("joint/", "base/", "camera/", "command/")):
                    emit(name, None, t, "unavailable", signals[name].description)
    finally:
        writer.close()
    identity = hashlib.sha256(
        encoded({"kind": "hardware_capture", "inputs": hashes}).encode()
    ).hexdigest()
    manifest = dict(
        schema_version=2,
        recording_id="recording-" + identity[:24],
        source_kind=report["source_kind"],
        adapter="hardware_capture",
        source_location=str(folder),
        backend={"name": "serial_capture", "version": 1},
        source_hashes=hashes,
        provenance_assurance="Capture hashes verified and original bytes replayed exactly; origin is asserted, not authenticated; no physical qualification",
        robot=mapping["robot"],
        joint_order=[j["name"] for j in mapping["joints"]],
        mass_model=None,
        scope="Offline projection of read-only capture; no robot task or physical qualification",
        seed=None,
        scenario=None,
        partition="unassigned",
        clock=clock,
        clock_note="Zero at first retained host chunk, not physical acquisition. MCU clocks retained in transport/event; no clock map or sensor ages inferred.",
        host_origin_ns=start_ns if chunks else None,
        outcome={
            "status": report["status"],
            "termination": report.get("error") or report.get("blocked_reason"),
            "duration_s": (ns(chunks[-1]["host_arrival_s"]) - start_ns) / 1e9
            if chunks
            else 0,
        },
        signals={k: v.model_dump() for k, v in signals.items()},
        rows=len(events),
        events=writer.count,
        events_sha256=sha256(output / "events.jsonl"),
        available_signals=sorted(writer.available_signals),
        limitations=[
            "Source capture status is preserved; import integrity is not a robot success result",
            "MCU timestamps can follow loop-start time; they are not mapped to host arrival or used as acquisition timestamps",
            "No calibrated joint decoder, motor torque, root pose, synchronized camera or acknowledged actions",
            "IMU is uncalibrated and sensor-frame only; synthetic transport is not a physical measurement",
            "No 3D robot pose or learning-ready dataset is generated",
            "Original raw capture, parser diagnostics and partial trailing bytes remain archived",
        ],
    )
    (output / "manifest.json").write_text(encoded(manifest) + "\n")
    (output / "manifest.sha256").write_text(sha256(output / "manifest.json") + "\n")
    return verify_recording(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_hardware_recording(args.source, args.output), indent=2))
