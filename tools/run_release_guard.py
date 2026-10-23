"""Repeat the unchanged composed-mass task with release-aware reference control."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "launcher.py").write_bytes(Path(__file__).read_bytes())
    rows = []
    for name in ("pip", "scout"):
        for case in ("lower", "nominal", "upper", "open"):
            profile = "nominal" if case == "open" else case
            mass = Path(f"artifacts/{name}-integral-head-mass-v1/{profile}")
            dest = Path(f"artifacts/{name}-release-guard-{case}-v1")
            command = [
                sys.executable,
                "-m",
                "lab.manipulation",
                f"variants/{name}-distal-grip.yaml",
                "--model-run",
                str(mass),
                "--mass-model",
                str(mass / "mass-properties.json"),
                "--task",
                f"artifacts/distal-grip-park-tasks-v1/{name}.json",
                "--controller",
                "open" if case == "open" else "feedback",
                "--output",
                str(dest),
            ]
            if dest.exists():
                raise FileExistsError(dest)
            started = time.monotonic()
            with (output / f"{name}-{case}.log").open("x") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            row = dict(
                command=command,
                exit_code=result.returncode,
                elapsed_s=time.monotonic() - started,
                output=str(dest),
            )
            rows.append(row)
            (output / "progress.json").write_text(json.dumps(rows, indent=2))
            print(row, flush=True)
    (output / "report.json").write_text(
        json.dumps(
            dict(
                status="complete",
                commands=rows,
                scope="Completed launches; task failures retain their producer status",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
