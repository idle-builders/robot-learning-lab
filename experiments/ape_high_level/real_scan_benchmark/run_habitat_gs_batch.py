"""Try the same provisional moving-camera probe in visually screened captures."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from screen_habitat_gs import ASSETS


HERE = Path(__file__).resolve().parent
SCENES = [12, 14, 32, 36, 37, 38, 54, 55, 63, 64]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()
    rows = []
    for number in SCENES:
        scene = f"scene{number:02d}"
        split = "train" if number <= 55 else "val"
        folder = ASSETS / split / scene / (
            "gs_probe_extended" if args.extended else "gs_probe")
        command = [sys.executable, str(HERE / "run_habitat_gs_probe.py"),
                   "--scene-number", str(number), "--steps", "80"]
        if args.extended:
            command.append("--extended")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            row = {"scene_id": scene, "outcome": "probe_error",
                   "reason": result.stderr[-1200:]}
        else:
            data = json.loads((folder / "trace.json").read_text())
            row = {key: value for key, value in data.items() if key != "trace"}
        rows.append(row)
        print(scene, row["outcome"], "standoff", row.get("truth_standoff_within_band"),
              "black", row.get("mean_black_background_fraction"), flush=True)
        name = "habitat_gs_probe_extended_batch.json" if args.extended else "habitat_gs_probe_batch.json"
        (ASSETS / name).write_text(json.dumps({
            "schema": "ape.habitat-gs-real-capture-batch.v1",
            "warning": "Preselected visually screened scenes; per-scene probes are not photoreal or body qualification.",
            "scenes": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
