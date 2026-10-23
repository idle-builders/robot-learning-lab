"""Seal wider, still-unseen native conditions for the next reach candidate."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
SOURCE = BASE / "v41/fresh-challenge-plan.json"
OUT = BASE / "v44/fresh-starts-plan.json"
STRATA = ("positive_x", "high_y", "negative_corner", "varied_gap_neighborhood")
JITTER_MM = ((1, -1), (-1, 2), (2, 1), (-2, -1), (1, 2), (-1, -2),
             (2, -1), (-2, 1), (1, -2), (-1, 1), (2, 2), (-2, -2))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parent = json.loads(SOURCE.read_text())
    parent_groups = {s: [c for c in parent["cases"] if c["stratum"] == s]
                     for s in STRATA}
    if any(len(group) != 8 for group in parent_groups.values()):
        raise ValueError("Expected four balanced V41 strata")
    cases = []
    for stratum_index, stratum in enumerate(STRATA):
        for i in range(12):
            base = parent_groups[stratum][(i * 5) % 8]
            jx, jy = JITTER_MM[(i + 3 * stratum_index) % len(JITTER_MM)]
            cases.append({
                "seed": 10901 + 12 * stratum_index + i,
                "stratum": stratum,
                "rail_v": (10, 11, 12)[i % 3],
                "delay_ms": (0, 5, 10)[(i // 4) % 3],
                "toy_offset_x_m": round(base["toy_offset_x_m"] + jx / 1000, 4),
                "toy_offset_y_m": round(base["toy_offset_y_m"] + jy / 1000, 4),
                "gaze_head_yaw_rad": -1.5 if i % 2 == 0 else -1.3,
                "pitch_offset_rad": (.095, .105, .115)[(i // 2) % 3],
                "roll_clearance_rad": (-.24, -.25, -.26)[(i // 4) % 3],
                "parent_condition_seed_for_provenance_only": base["seed"],
            })
    if [row["seed"] for row in cases] != list(range(10901, 10949)):
        raise ValueError("Unexpected disjoint seed list")
    plan = {
        "schema": "robot-learning-lab.cove50-v44-still-sealed-fresh-starts.v1",
        "role": "new_disjoint_native_score_only_after_full_opened_retention",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_case_plan_sha256": sha(SOURCE),
        "generator_sha256": sha(Path(__file__)),
        "case_count": len(cases),
        "stratum_counts": dict(Counter(c["stratum"] for c in cases)),
        "voltage_counts": {str(v): sum(c["rail_v"] == v for c in cases) for v in (10, 11, 12)},
        "delay_counts": {str(d): sum(c["delay_ms"] == d for c in cases) for d in (0, 5, 10)},
        "gaze_counts": {str(y): sum(c["gaze_head_yaw_rad"] == y for c in cases)
                        for y in (-1.5, -1.3)},
        "pitch_counts": {str(p): sum(c["pitch_offset_rad"] == p for c in cases)
                         for p in (.095, .105, .115)},
        "roll_clearance_counts": {str(r): sum(c["roll_clearance_rad"] == r for c in cases)
                                  for r in (-.24, -.25, -.26)},
        "cases": cases,
        "opening_rule": "Keep these 48 starts sealed until a source-frozen automatic controller passes every unique older and V41-opened native retention start plus repeated thin-margin controls under the original full motor, support, contact and RGB gates. Then freeze the candidate runner and source hashes against this unchanged list before opening any 109xx seed.",
        "score_rule": "Require all 48 full native passes, at least four actual V10-height and scoped-Y selections and one calibrated-corner route, and report minimum contact duration and maximum rated-leg preposition fraction. Treat 1.0-1.2 s contacts as thin even when they technically pass; require at least 1.2 s for a robust-margin claim. This is only a historical rejected-body score, never selected-body qualification.",
        "design": "Twelve starts in each previously stressed toy-region stratum use new seeds, ±1–2 mm XY perturbations around still-physical V41 placements, balanced 10/11/12 V rails, 0/5/10 ms delays, both legal gaze commands, three arm pitch offsets and three clearance roll offsets. Stratum and parent provenance are never controller inputs.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if OUT.exists() and OUT.read_bytes() != encoded:
        raise ValueError("Sealed V44 start plan changed")
    OUT.write_bytes(encoded)
    print(f"sealed {OUT} sha256={sha(OUT)}")


if __name__ == "__main__":
    main()
