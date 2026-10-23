"""Local pseudo-terminal evidence for the host link; no physical devices."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import pty
import select
import threading
import time

from lab.hardware_link import open_serial
from lab.capture_hardware import capture


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def frame(mapping, imu_hash, **changes):
    value = dict(
        schema=1,
        robot=mapping["robot"],
        map=mapping["map_hash"],
        session="000000000000007b",
        time_ms=100,
        status=0,
        armed=0,
        calibrated=int(mapping["calibrated"]),
        voltage_v=11.1,
        rejected=0,
        raw_feedback=[
            dict(id=j["id"], sample_ms=0, bytes=[0] * 15) for j in mapping["joints"]
        ],
        imu=dict(
            part="LSM6DSOTR",
            config=imu_hash,
            frame="sensor",
            calibrated=False,
            status=0,
            error=0,
            valid=False,
            sequence=0,
            poll_start_us=0,
            earliest_update_us=0,
            read_end_us=0,
            max_poll_gap_us=0,
            saturated=0,
            gyro_rad_s=[0.0, 0.0, 0.0],
            specific_force_m_s2=[0.0, 0.0, 0.0],
        ),
    )
    value.update(changes)
    return value


def send_bounded(fd, data, pump=None):
    deadline = time.monotonic() + 1
    offset = 0
    while offset < len(data):
        if time.monotonic() >= deadline:
            raise TimeoutError("virtual sender deadline")
        try:
            offset += os.write(fd, data[offset : offset + 256])
        except BlockingIOError:
            pass
        time.sleep(0.001)
        if pump is not None:
            pump()


def run(output, map_root):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="fail",
        origin="synthetic_transport",
        physical_status="not_run",
        robots=[],
        source_hashes={},
    )
    for path in [
        Path(__file__),
        Path("lab/hardware_link.py"),
        Path("lab/transport.py"),
        Path("lab/capture_hardware.py"),
        Path("firmware/target_main.c"),
        Path("requirements-hardware-macos-arm64.lock"),
    ]:
        path = path.resolve().relative_to(Path.cwd())
        dest = output / "source" / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        report["source_hashes"][str(path)] = sha(dest)
    report["source_bindings"] = {}
    try:
        for name in ["pip", "scout"]:
            root = map_root / f"{name}-map"
            for filename in ["hardware-map.json", "imu-config.json"]:
                report["source_bindings"][str(root / filename)] = sha(root / filename)
            mapping = json.loads((root / "hardware-map.json").read_text())
            imu = json.loads((root / "imu-config.json").read_text())
            imu_hash = hashlib.sha256(
                json.dumps(imu, sort_keys=True).encode()
            ).hexdigest()
            master, slave = pty.openpty()
            os.set_blocking(master, False)
            port = os.ttyname(slave)
            link = open_serial(port, mapping, imu_hash, allow_commands=True)
            events = []
            checks = []
            wire = []

            def emit(data):
                send_bounded(master, data, lambda: events.extend(link.poll()))

            def send(value):
                packet = json.dumps(value).encode() + b"\n"
                emit(packet)
                time.sleep(0.005)
                received = link.poll()
                events.extend(received)
                return received

            try:
                packet = json.dumps(frame(mapping, imu_hash)).encode() + b"\n"
                os.write(master, packet[:17])
                time.sleep(0.005)
                assert link.poll() == []
                emit(packet[17:])
                time.sleep(0.005)
                events.extend(link.poll())
                assert link.latest["device"]["map"] == mapping["map_hash"]
                checks.append("fragmented_identity")
                os.write(master, b"{truncated\n")
                time.sleep(0.005)
                events.extend(link.poll())
                assert events[-1]["kind"] == "invalid_line"
                send(frame(mapping, imu_hash, time_ms=200))
                checks.append("malformed_recovery")
                for _ in range(18):
                    emit(b"x" * 1000)
                    time.sleep(0.002)
                    events.extend(link.poll())
                os.write(master, b"\n")
                time.sleep(0.005)
                events.extend(link.poll())
                assert link.decoder.rejected_lines == 2 and not link.decoder.buffer
                send(frame(mapping, imu_hash, time_ms=300))
                checks.append("bounded_oversize_recovery")
                stop = link.stop()
                assert stop.device_accepted is None and stop.first_apply_time is None
                ready, _, _ = select.select([master], [], [], 1)
                assert ready
                raw = os.read(master, 4096)
                assert raw[:4] == b"RLB1" and raw[4] == 2 and len(raw) == 28
                wire.append(dict(receipt=asdict(stop), bytes_hex=raw.hex()))
                checks.append("stop_write_acceptance_unknown")
                send(frame(mapping, imu_hash, session="000000000000007c", time_ms=1))
                assert link.blocked_reason == "device_restarted"
                checks.append("restart_latch")
                send(
                    frame(
                        mapping,
                        imu_hash,
                        session="000000000000007c",
                        time_ms=2,
                        map="b" * 64,
                    )
                )
                assert link.blocked_reason == "identity_mismatch"
                checks.append("map_mismatch_latch")
            finally:
                link.close()
                os.close(slave)
                os.close(master)
            (output / f"{name}-events.json").write_text(
                json.dumps(events, indent=2) + "\n"
            )
            (output / f"{name}-writes.json").write_text(
                json.dumps(wire, indent=2) + "\n"
            )
            # Exercise the real capture entry point with a separate virtual device.
            master, slave = pty.openpty()
            os.set_blocking(master, False)
            port = os.ttyname(slave)
            done = threading.Event()

            def produce():
                time.sleep(0.04)
                counter = 0
                while counter < 3 and not done.is_set():
                    send_bounded(
                        master,
                        json.dumps(
                            frame(mapping, imu_hash, time_ms=100 + 100 * counter)
                        ).encode()
                        + b"\n",
                    )
                    counter += 1
                    done.wait(0.04)

            worker = threading.Thread(target=produce)
            worker.start()
            try:
                captured = capture(
                    port,
                    root / "hardware-map.json",
                    root / "imu-config.json",
                    output / f"{name}-capture",
                    0.75,
                    source_kind="synthetic_transport",
                )
                assert captured["status"] == "pass" and captured["command_writes"] == 0
                checks.append("read_only_capture")
            finally:
                done.set()
                worker.join(timeout=1)
                os.close(slave)
                os.close(master)
            report["robots"].append(
                dict(
                    name=name,
                    status="pass",
                    checks=checks,
                    captured_frames=captured["telemetry_frames"],
                    events=len(events),
                    command_acknowledgements="unavailable",
                )
            )
        report["status"] = "pass"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["scope"] = (
            "Real pySerial over locally created pseudo-terminals; synthetic telemetry; no Pico/servo, USB timing or physical qualification"
        )
        report["output_hashes"] = {
            str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--map-root", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.output, a.map_root), indent=2))
