"""Predeclare scarce correction-view routes before V65 native outcomes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
V59_PLAN = BASE / "v59/combined-screen-plan.json"
V59_MATRIX = BASE / "v59/combined-opened-and-varied-screen/matrix.json"
OUT = BASE / "v67"
PLAN = OUT / "predeclared-branch-view-holdout.json"
FRONT_ANCHORS = ("large-34", "large-35", "large-39")
FRONT_ROUTES = (("pitch-high", .115, -.25), ("roll-high", .105, -.24))
EDGE_VARIANTS = (
    ("toy-x-low-pitch", 11203, 12, 10, .01225, .0055, -1.3, .095, -.25),
    ("toy-x-high-roll", 11203, 12, 10, .01275, .0055, -1.3, .105, -.24),
    ("toy-y-low-roll", 11203, 11, 5, .0125, .00525, -1.3, .105, -.26),
    ("toy-y-high-roll", 11203, 10, 0, .0125, .00575, -1.5, .105, -.24),
    ("new-seed-low-pitch", 11204, 11, 5, .0125, .0055, -1.3, .095, -.25),
    ("new-seed-low-roll", 11205, 12, 10, .0125, .0055, -1.5, .105, -.26),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = json.loads(V59_PLAN.read_text())
    outcome = json.loads(V59_MATRIX.read_text())
    if not outcome["complete"] or outcome["pass_count"] != 63:
        raise ValueError("Expected complete V59 opened branch source")
    by_id = {row["id"]: row for row in source["cases"]}
    scored = {row["id"]: row for row in outcome["cases"]}
    if any(anchor not in by_id for anchor in (*FRONT_ANCHORS, "edge-12")):
        raise ValueError("Branch holdout source anchor missing")
    cases = []
    for anchor in FRONT_ANCHORS:
        for route, pitch, roll in FRONT_ROUTES:
            cases.append({**by_id[anchor], "id": f"{anchor}-{route}",
                          "anchor_id": anchor, "route_variant": route,
                          "pitch_offset_rad": pitch, "roll_clearance_rad": roll,
                          "branch_view_family": "different_anchor_front_view",
                          "baseline_v59_center_contact_s": scored[anchor]["contact_s"]})
    for label, seed, rail_v, delay_ms, toy_x, toy_y, gaze, pitch, roll in EDGE_VARIANTS:
        case = {**by_id["edge-12"], "id": f"edge-view-{label}",
                "anchor_id": "edge-12", "seed": seed,
                "rail_v": rail_v, "delay_ms": delay_ms,
                "toy_offset_x_m": toy_x, "toy_offset_y_m": toy_y,
                "gaze_head_yaw_rad": gaze,
                "pitch_offset_rad": pitch, "roll_clearance_rad": roll,
                "route_variant": label,
                "development_role": "predeclared_branch_view_holdout",
                "branch_view_family": "varied_sparse_edge_view",
                "matched_prior_contact_s": None, "matched_prior_pass": None,
                "baseline_v41_contact_s": None,
                "baseline_v59_center_contact_s": None}
        cases.append(case)
    if len(cases) != 12 or len({case["id"] for case in cases}) != 12:
        raise ValueError("Expected twelve unique branch-view holdout visits")
    plan = {
        "schema": "robot-learning-lab.cove50-v67-predeclared-branch-view-holdout.v1",
        "role": "opened_branch_view_variation_holdout_not_disjoint_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "v59_plan_sha256": sha(V59_PLAN), "v59_matrix_sha256": sha(V59_MATRIX),
        "manifest_source_sha256": sha(Path(__file__)),
        "case_count": len(cases), "cases": cases,
        "gate": "All twelve native visits must pass support, rated effort, >=1 s continuous fingertip contact, floor, self-contact, clipping and <=10 mm scored RGB error. Require at least three observed stronger front-roll activations and three sparse-edge route activations. A view falling outside a rule's observable range is reported as missing exposure, not as a policy pass for that branch. The six front visits use different V59 entry anchors; six edge visits include new toy, motor, gaze and seed conditions around the scarce sparse-edge view.",
        "design": "Freeze before V65 native outcomes. Front anchors large-34, large-35 and large-39 were held out of the seven V60 tuning anchors and had V59 front-visible, locked-arm RGB entries. Each receives independent +0.01 rad pitch and roll routes. Edge variants perturb the V59 edge-12 view by ±0.25 mm toy X/Y, vary 10–12 V, 0–10 ms delay, use both legal gaze angles, and add unseen 11204/11205 seeds under pitch-low or roll changes. Image visibility and route selection are measured at replay, not assumed from seed or toy truth.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V67 branch-view manifest differs from current source")
    PLAN.write_bytes(encoded)
    print(f"frozen {PLAN} sha256={sha(PLAN)} cases={len(cases)}")


if __name__ == "__main__":
    main()
