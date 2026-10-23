"""Repair derived velocity matrices using their retained per-world reports."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-cove50-recovery-20261008"


def main() -> None:
    updated = []
    for matrix_path in sorted(OUT.glob("*-matrix.json")):
        matrix = json.loads(matrix_path.read_text())
        changed = False
        for row in matrix.get("cases", []):
            if "mean_lateral_m" not in row or "report" not in row:
                continue
            report = json.loads((ROOT / row["report"]).read_text())
            lateral = [abs(world["lateral_m"]) for world in report["worlds"]]
            metrics = {
                "mean_abs_lateral_m": round(sum(lateral) / len(lateral), 4),
                "max_abs_lateral_m": round(max(lateral), 4),
            }
            if all("fraction_moving_frames_with_saturated_leg" in world
                   for world in report["worlds"]):
                metrics["mean_moving_leg_saturated_fraction"] = round(
                    sum(world["fraction_moving_frames_with_saturated_leg"]
                        for world in report["worlds"]) / len(lateral), 4)
            if any(row.get(key) != value for key, value in metrics.items()):
                row.update(metrics)
                changed = True
        if changed:
            matrix_path.write_text(json.dumps(matrix, indent=2) + "\n")
            updated.append(matrix_path.name)
    print(json.dumps({"updated": updated}))


if __name__ == "__main__":
    main()
