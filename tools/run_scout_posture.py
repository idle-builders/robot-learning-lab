"""Replay fixed balance probes with one statically selected Scout rest posture."""

import argparse
from pathlib import Path
import subprocess
import sys
import time
import json
from tools.check_camera_support import sha, write
from tools.compose_forearm_enclosure import verify_report


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    selection = Path("artifacts/scout-posture-selection-v1")
    verify_report(selection)
    mass = Path("artifacts/scout-shin-covered-mass-v1")
    verify_report(mass)
    report = dict(
        status="running",
        scope=__doc__,
        commands=[],
        source_bindings={
            str(p): sha(p) for p in [selection / "report.json", mass / "report.json"]
        },
        physical_status="not_run",
    )
    (output / "launcher.py").write_bytes(Path(__file__).read_bytes())
    probes = [
        ("nominal", "normal", selection / "normal.yaml"),
        ("lower", "bounds", selection / "bounds.yaml"),
        ("upper", "bounds", selection / "bounds.yaml"),
        (
            "nominal",
            "baseline-stress",
            Path("experiments/balance-composed-observed-stress.yaml"),
        ),
        ("nominal", "stress", selection / "stress.yaml"),
    ]
    for profile, kind, suite in probes:
        report["source_bindings"][str(suite)] = sha(suite)
        model = mass / profile
        dest = Path(f"artifacts/scout-covered-rest-{profile}-{kind}-v1")
        if dest.exists():
            raise FileExistsError(dest)
        command = [
            sys.executable,
            "-m",
            "lab.balance",
            "variants/scout-distal-grip.yaml",
            "--model-run",
            str(model),
            "--mass-model",
            str(model / "mass-properties.json"),
            "--suite",
            str(suite),
            "--output",
            str(dest),
        ]
        start = time.monotonic()
        with (output / f"{dest.name}.log").open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        p = dest / "balance-report.json"
        row = dict(
            command=command,
            output=str(dest),
            exit_code=result.returncode,
            elapsed_s=time.monotonic() - start,
        )
        if p.exists():
            row.update(status=json.loads(p.read_text())["status"], report_sha256=sha(p))
        else:
            row.update(status="error")
        report["commands"].append(row)
        write(output / "progress.json", report)
        print(row["output"], row["status"], flush=True)
        if row["status"] == "error":
            raise RuntimeError(f"Missing producer result: {dest}")
    report["status"] = "complete"
    report["output_hashes"] = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    write(output / "report.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
