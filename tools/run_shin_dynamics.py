"""Repeat the retained control probes with the composed shin-cover inertia."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from tools.check_camera_support import sha, write
from tools.compose_forearm_enclosure import verify_report


def run(name, output):
    mass = Path(f"artifacts/{name}-shin-covered-mass-v1")
    verified = verify_report(mass)
    if verified["status"] != "pass":
        raise ValueError("Verified mass export required")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="running",
        source_bindings={str(mass / "report.json"): sha(mass / "report.json")},
        commands=[],
        scope=__doc__,
        physical_status="not_run",
        limits=[
            "Only mass/COM/full inertia changed; baseline contact and visual geometry remain. No cover collision qualification from dynamics."
        ],
    )
    (output / "launcher.py").write_bytes(Path(__file__).read_bytes())
    commands = []
    for kind in ("balance", "task"):
        baseline = Path(f"artifacts/forearm-accounted-{kind}-launch-v1/report.json")
        report["source_bindings"][str(baseline)] = sha(baseline)
        original = json.loads(baseline.read_text())
        for row in original["commands"]:
            command = row["command"]
            if f"variants/{name}-distal-grip.yaml" not in command:
                continue
            candidate = [sys.executable] + [
                x.replace(f"{name}-forearm-accounted-", f"{name}-shin-covered-")
                for x in command[1:]
            ]
            commands.append(candidate)
    if len(commands) != 7:
        raise ValueError("Expected three balance suites and four task runs")
    destinations = [Path(c[c.index("--output") + 1]) for c in commands]
    if any(p.exists() for p in destinations):
        raise FileExistsError("Refusing to overwrite any control experiment")
    for command, dest in zip(commands, destinations, strict=True):
        start = time.monotonic()
        with (output / f"{dest.name}.log").open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        result_path = dest / (
            "balance-report.json" if "lab.balance" in command else "report.json"
        )
        if not result_path.exists():
            report["commands"].append(
                dict(
                    command=command,
                    exit_code=result.returncode,
                    output=str(dest),
                    status="error",
                )
            )
            write(output / "progress.json", report)
            raise RuntimeError(f"Missing result: {result_path}")
        result_report = json.loads(result_path.read_text())
        row = dict(
            command=command,
            exit_code=result.returncode,
            elapsed_s=time.monotonic() - start,
            output=str(dest),
            status=result_report["status"],
            report_sha256=sha(result_path),
        )
        report["commands"].append(row)
        write(output / "progress.json", report)
        print(dest, result_report["status"], flush=True)
    report["status"] = "complete"
    report["output_hashes"] = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    write(output / "report.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name", choices=["pip", "scout"])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.name, a.output)
