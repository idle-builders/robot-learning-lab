"""Run fixed control probes against each retained composed-assembly mass profile."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from tools.compose_forearm_enclosure import verify_report
from tools.check_camera_support import sha


def run(name, output):
    output.mkdir(parents=True, exist_ok=False)
    mass = Path(f"artifacts/{name}-integral-head-mass-v1")
    verified = verify_report(mass)
    if (
        verified["status"] != "pass"
        or verified["assembly_evidence"]["status"] != "fail"
    ):
        raise ValueError("Revisit experiment scope for changed source evidence")
    variant = f"variants/{name}-distal-grip.yaml"
    task = f"artifacts/distal-grip-park-tasks-v1/{name}.json"
    report = dict(
        status="running",
        scope=__doc__,
        source_mass_report=str(mass / "report.json"),
        source_mass_report_sha256=sha(mass / "report.json"),
        commands=[],
    )
    script = Path(__file__)
    (output / "launcher.py").write_bytes(script.read_bytes())
    commands = []
    for profile in ("nominal", "lower", "upper"):
        model = mass / profile
        common = [
            "--model-run",
            str(model),
            "--mass-model",
            str(model / "mass-properties.json"),
        ]
        suite = (
            "experiments/balance-integrated-trim.yaml"
            if profile == "nominal"
            else "experiments/balance-composed-mass.yaml"
        )
        commands.append(
            [
                sys.executable,
                "-m",
                "lab.balance",
                variant,
                *common,
                "--suite",
                suite,
                "--output",
                f"artifacts/{name}-composed-{profile}-balance-v1",
            ]
        )
        commands.append(
            [
                sys.executable,
                "-m",
                "lab.manipulation",
                variant,
                *common,
                "--task",
                task,
                "--controller",
                "feedback",
                "--release-clearance-m",
                "0.025",
                "--output",
                f"artifacts/{name}-composed-{profile}-task-v1",
            ]
        )
    model = mass / "nominal"
    commands.append(
        [
            sys.executable,
            "-m",
            "lab.manipulation",
            variant,
            "--model-run",
            str(model),
            "--mass-model",
            str(model / "mass-properties.json"),
            "--task",
            task,
            "--controller",
            "open",
            "--output",
            f"artifacts/{name}-composed-open-task-v1",
        ]
    )
    for command in commands:
        dest = Path(command[command.index("--output") + 1])
        if dest.exists():
            raise FileExistsError(dest)
        started = time.time()
        with (output / f"{dest.name}.log").open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        row = dict(
            command=command,
            exit_code=result.returncode,
            elapsed_s=time.time() - started,
            output=str(dest),
        )
        report["commands"].append(row)
        (output / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
        print(row, flush=True)
    report["status"] = "complete"
    report["scope"] = (
        "Fixed experiment launch completed; exit failures include deliberate negative controls and are not reclassified as passes"
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name", choices=["pip", "scout"])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(args.name, args.output)
