"""Bounded, read-only serial telemetry capture. A device path is always explicit."""

import argparse
import hashlib
import json
from pathlib import Path
import time

from .hardware_link import open_serial


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(
    port,
    mapping_path,
    imu_path,
    output,
    duration_s=10.0,
    *,
    source_kind="hardware_asserted",
):
    if not 0 < duration_s <= 3600:
        raise ValueError("capture duration must be finite and at most one hour")
    if source_kind not in ("hardware_asserted", "synthetic_transport"):
        raise ValueError("unknown capture origin")
    mapping = json.loads(mapping_path.read_text())
    imu = json.loads(imu_path.read_text())
    imu_hash = hashlib.sha256(json.dumps(imu, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        schema_version=1,
        status="fail",
        source_kind=source_kind,
        port=port,
        command_writes=0,
        physical_qualification="not_run",
        clocks_synchronized=False,
        telemetry_frames=0,
        diagnostics=0,
        source_hashes={},
    )
    for path in [
        mapping_path,
        imu_path,
        Path(__file__),
        Path("lab/hardware_link.py"),
        Path("lab/transport.py"),
    ]:
        relative = (
            Path(path.name)
            if path in (mapping_path, imu_path)
            else Path("lab") / path.name
        )
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        report["source_hashes"][str(relative)] = digest(destination)
    link = None
    try:
        link = open_serial(port, mapping, imu_hash)
        start = time.monotonic()
        with (
            (output / "chunks.jsonl").open("x") as raw,
            (output / "events.jsonl").open("x") as parsed,
        ):
            while time.monotonic() - start < duration_s:
                events = link.poll()
                if link.last_chunk:
                    raw.write(
                        json.dumps(
                            dict(
                                host_arrival_s=link._last_clock,
                                bytes_hex=link.last_chunk.hex(),
                            )
                        )
                        + "\n"
                    )
                for event in events:
                    parsed.write(json.dumps(event, allow_nan=False) + "\n")
                    report["telemetry_frames"] += event["kind"] == "telemetry"
                    report["diagnostics"] += event["kind"] != "telemetry"
                time.sleep(0.002)
        report.update(
            elapsed_s=time.monotonic() - start,
            blocked_reason=link.blocked_reason,
            pending_line_bytes=len(link.decoder.buffer),
            discarding_line=link.decoder.discarding,
            status="pass"
            if report["telemetry_frames"]
            and not report["diagnostics"]
            and not link.blocked_reason
            and not link.decoder.buffer
            and not link.decoder.discarding
            else "partial",
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if link is not None:
            try:
                link.close()
            except Exception as error:
                report["close_error"] = str(error)
                report["status"] = "fail"
    report["scope"] = (
        "Transport capture only; no clock synchronization, calibrated joint decoding, canonical recording/Rerun conversion or physical qualification"
    )
    report["output_hashes"] = {
        str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", required=True)
    p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--imu-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--duration-s", type=float, default=10.0)
    a = p.parse_args()
    report = capture(a.port, a.mapping, a.imu_config, a.output, a.duration_s)
    print(json.dumps(report, indent=2))
    if report["status"] == "fail":
        raise SystemExit(1)
