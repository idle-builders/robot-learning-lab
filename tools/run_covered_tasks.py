"""Replay the existing placement task with the complete covered-carrier mass models.

Controls, task thresholds and contact geometry are unchanged. Guards are absent.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from lab.artifact_store import retain_dependency
from lab.config import load_config, RobotConfig
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="running",
        scope=__doc__,
        commands=[],
        source_bindings={},
        dependency_manifests={},
    )
    paths = [
        Path(__file__).relative_to(Path.cwd()),
        Path("lab/config.py"),
        Path("lab/manipulation.py"),
    ]
    report["source_hashes"] = copy_source(output, {str(p): p for p in paths})
    try:
        for name in ("pip", "scout"):
            variant = Path(f"variants/{name}-carriers.yaml")
            model = Path(f"artifacts/{name}-covered-carriers-v2")
            verify_report(model)
            report["dependency_manifests"][name] = retain_dependency(model)
            report["source_bindings"][str(variant)] = sha(variant)
            task = Path(f"artifacts/distal-grip-park-tasks-v1/{name}.json")
            report["source_bindings"][str(task)] = sha(task)
            for label in ("lower", "nominal", "upper", "open"):
                profile = "nominal" if label == "open" else label
                if load_config(variant) != RobotConfig.model_validate_json(
                    (model / profile / "resolved-config.json").read_text()
                ):
                    raise ValueError("Configured robot differs from mass export")
                dest = output / f"{name}-{label}"
                command = [
                    sys.executable,
                    "-m",
                    "lab.manipulation",
                    str(variant),
                    "--model-run",
                    str(model / profile),
                    "--mass-model",
                    str(model / profile / "mass-properties.json"),
                    "--task",
                    str(task),
                    "--controller",
                    "open" if label == "open" else "feedback",
                    "--output",
                    str(dest),
                ]
                start = time.monotonic()
                with (output / f"{name}-{label}.log").open("x") as log:
                    proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                result = json.loads((dest / "report.json").read_text())
                report["commands"].append(
                    dict(
                        robot=name,
                        profile=profile,
                        controller="open" if label == "open" else "feedback",
                        command=command,
                        output=str(dest),
                        exit_code=proc.returncode,
                        status=result["status"],
                        report_sha256=sha(dest / "report.json"),
                        elapsed_s=time.monotonic() - start,
                    )
                )
                write(output / "progress.json", report)
                print(name, label, result["status"], flush=True)
        report["status"] = "complete"
    finally:
        write(output / "report.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
