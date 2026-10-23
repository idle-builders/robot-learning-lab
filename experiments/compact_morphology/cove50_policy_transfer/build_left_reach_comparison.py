"""Combine frozen left-reach native matrices for the session inspector."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE = Path("artifacts/ape-cove50-side-reach-20261009")
INPUTS = (
    ("Mirror v1", "reach-left-mirror-v1", "development"),
    ("Four-joint v2", "reach-left-tuned-v2", "development"),
    ("Roll v3", "reach-left-roll-v3", "development"),
    ("Reference v4", "reach-left-reference-v4", "development"),
    ("Reference v4", "reach-left-reference-v4", "reserved"),
    ("Reference v4", "reach-left-reference-v4", "stress"),
)


def main() -> None:
    rows = []
    sources = []
    for version, directory, split in INPUTS:
        path = BASE / directory / split / "matrix.json"
        full_path = ROOT / path
        if not full_path.is_file():
            raise FileNotFoundError(full_path)
        matrix = json.loads(full_path.read_text())
        sources.append(str(path))
        for case in matrix["cases"]:
            rows.append(
                {
                    "version": version,
                    "case": case["case"],
                    "split": split,
                    "outcome": "Pass" if case["pass"] else (
                        "Fall" if not case["survived"] else "Contact miss"
                    ),
                    "pass": case["pass"],
                    "contact_s": case["contact_s"],
                    "both_feet_fraction": case["both_feet_fraction"],
                    "rated_leg_fraction": case["rated_leg_fraction"],
                    "max_hand_ground_force_n": case["max_hand_ground_force_n"],
                    "body": case["body"],
                    "rail_v": case["rail_v"],
                    "delay_ms": case["delay_ms"],
                    "result": case["result"],
                    "trace": case["trace"],
                    "featured": not case["pass"] or split == "reserved" or (
                        version in {"Roll v3", "Reference v4"}
                        and case["body"] == "nominal"
                    ),
                }
            )
    output = {
        "schema": "robot-learning-lab.cove50-left-reach-comparison.v1",
        "selected_design_pointer_sha256": "c0d7dc5c4e8b71d8003d42676e02a530c64abdec5536a6962bd97787514de219",
        "qualification": "Prepositioned left-hand fingertip contact only; neither grasp nor a learned left phase path.",
        "sources": sources,
        "cases": rows,
    }
    path = ROOT / BASE / "reach-left-comparison-v1.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
