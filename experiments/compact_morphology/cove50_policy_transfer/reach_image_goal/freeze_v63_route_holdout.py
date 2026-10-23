"""Predeclare new nearby-route visits before seeing the V62 screen result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
V59_PLAN = BASE / "v59/combined-screen-plan.json"
V59_MATRIX = BASE / "v59/combined-opened-and-varied-screen/matrix.json"
OUT = BASE / "v63"
PLAN = OUT / "predeclared-route-holdout.json"
ANCHORS = ("large-01", "large-19", "large-23", "large-28", "large-32", "large-36",
           "edge-02", "edge-08", "edge-10", "edge-11",
           "positive-01", "positive-07", "positive-11", "positive-12")
ROUTE_PAIRS = ((("pitch-low", .095, -.25), ("roll-high", .105, -.24)),
               (("pitch-high", .115, -.25), ("roll-low", .105, -.26)))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = json.loads(V59_PLAN.read_text())
    outcome = json.loads(V59_MATRIX.read_text())
    if not outcome["complete"] or outcome["pass_count"] != 63:
        raise ValueError("Expected complete V59 opened anchor source")
    by_id = {row["id"]: row for row in source["cases"]}
    scored = {row["id"]: row for row in outcome["cases"]}
    if len(ANCHORS) != len(set(ANCHORS)) or any(anchor not in by_id for anchor in ANCHORS):
        raise ValueError("Holdout anchors missing or duplicated")
    cases = []
    for index, anchor in enumerate(ANCHORS):
        for label, pitch, roll in ROUTE_PAIRS[index % 2]:
            cases.append({**by_id[anchor], "anchor_id": anchor,
                          "id": f"{anchor}-{label}", "route_variant": label,
                          "pitch_offset_rad": pitch, "roll_clearance_rad": roll,
                          "baseline_v59_center_contact_s": scored[anchor]["contact_s"]})
    plan = {
        "schema": "robot-learning-lab.cove50-v63-predeclared-route-holdout.v1",
        "role": "opened_seed_new_route_variation_holdout_not_disjoint_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "v59_plan_sha256": sha(V59_PLAN), "v59_matrix_sha256": sha(V59_MATRIX),
        "manifest_source_sha256": sha(Path(__file__)),
        "case_count": len(cases), "anchor_count": len(ANCHORS),
        "cases": cases,
        "gate": "All 28 previously unseen anchor-by-route native visits pass full contact, support, rated effort, floor, self-contact, clipping and <=10 mm scored RGB gates. Compare each visit with its V59 authored-route anchor. Report contacts under 1.2 s as thin and retain every native result, trace and log. This is an opened-seed route holdout, not a sealed fresh score or selected-body qualification.",
        "design": "The fourteen anchors exclude all seven V60 tuning anchors. They span reopened fresh, large toy/motor/gaze variance, edge-entry and positive-entry neighbor families. Alternating anchor pairs cover low/high pitch and low/high roll at ±0.01 rad, while toy position, voltage, delay, gaze, seed and start are held from the independently frozen V59 plan. Freeze this manifest before seeing V62 native outcomes.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V63 route holdout differs from current source")
    PLAN.write_bytes(encoded)
    print(f"frozen {PLAN} sha256={sha(PLAN)} cases={len(cases)}")


if __name__ == "__main__":
    main()
