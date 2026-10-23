"""Derive source-linked automatic versus specialist contact comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, choices=(102, 103, 104), default=102)
    args = parser.parse_args()
    base = ARTIFACTS / f"v{args.version}/native-opened-pairs"
    raw = (base / "matrix.json").read_bytes()
    source = json.loads(raw)
    rows = []
    for pair in source["pairs"]:
        automatic = pair["automatic_contact_s"]
        forced = pair["forced_contact_s"]
        delta = round(automatic - forced, 3)
        rows.append({
            "id": f"pair-{pair['seed']}",
            "seed": pair["seed"],
            "entry_error_xy_mm": [round(v * 1000, 3)
                                  for v in pair["entry"]["entry_error_xy_m"]],
            "camera_xy_mm": [round(v * 1000, 2)
                             for v in pair["entry"]["camera_world_xyz_m"][:2]],
            "orange_pixels": pair["entry"]["orange_pixels"],
            "entry_verified": True,
            "native_entry_qpos_max_abs": pair.get("native_entry_qpos_max_abs"),
            "automatic_contact_s": automatic,
            "forced_contact_s": forced,
            "automatic_minus_forced_s": delta,
            "automatic_full_gate": pair["automatic_full_gate"],
            "forced_full_gate": pair["forced_full_gate"],
            "longer_contact": ("automatic" if delta > 0 else
                               "specialist" if delta < 0 else "equal"),
        })
    report = {
        "schema": f"robot-learning-lab.cove50-v{args.version}-pair-comparison.v1",
        "role": "derived_opened_matched_route_diagnostic_not_score",
        "body_role": source["body_role"],
        "source_matrix_sha256": hashlib.sha256(raw).hexdigest(),
        "complete": source["complete"],
        "case_count": len(rows),
        "cases": rows,
    }
    output = base / "pair-comparison.json"
    temp = output.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temp, output)
    print(f"summarized {len(rows)} verified V{args.version} pairs")


if __name__ == "__main__":
    main()
