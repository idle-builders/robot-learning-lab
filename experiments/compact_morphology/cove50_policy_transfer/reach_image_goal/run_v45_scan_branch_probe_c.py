"""Third native opened edge-view probe for V45's physical head-scan branch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v45_combined_screen as screen,
)

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v45"
PLAN = BASE / "scan-branch-probe-c-plan.json"
DEST = BASE / "scan-branch-probe-c"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    previous = json.loads(screen.PLAN.read_text())
    original = next(c for c in previous["cases"] if c["seed"] == 10806)
    if (original["toy_offset_x_m"] != .011 or original["toy_offset_y_m"] != .005 or
            original["gaze_head_yaw_rad"] != -1.3):
        raise ValueError("V45 edge-view source changed")
    first_matrix = BASE / "scan-branch-probe/matrix.json"
    second_matrix = BASE / "scan-branch-probe-b/matrix.json"
    first = json.loads(first_matrix.read_text())
    second = json.loads(second_matrix.read_text())
    if (not first["complete"] or first["pass_count"] != 0 or
            first["cases"][0]["reacquisition_probe_orange_pixels"] != 11 or
            not second["complete"] or second["pass_count"] != 0 or
            second["cases"][0]["reacquisition_probe_orange_pixels"] != 12):
        raise ValueError("Expected first two edge-view probes did not expose scan")
    case = {**original, "toy_offset_x_m": .012,
            "baseline_v41_contact_s": None,
            "development_role": "opened_edge_view_plus_1mm_toy_x_scan_exposure"}
    plan = {
        "schema": "robot-learning-lab.cove50-v45-scan-branch-probe-c-plan.v1",
        "role": "opened_native_branch_exposure_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": previous["source_sha256"],
        "v45_combined_screen_plan_sha256": sha(screen.PLAN),
        "first_scan_probe_matrix_sha256": sha(first_matrix),
        "second_scan_probe_matrix_sha256": sha(second_matrix),
        "runner_sha256": sha(Path(__file__)),
        "case_count": 1, "cases": [case],
        "selection": "The first native probe changed toy-frame Y by -1 mm, moving mainly world X and seeing 11 pixels. The second changed toy-frame X by -1 mm, moving world Y +1 mm and seeing 12 pixels. Saved qpos and a pose-only RGB sweep show the visibility edge lies in world Y -1 mm. This third opened probe changes toy-frame X by +1 mm. The sweep selects a case, not a policy outcome.",
        "gate": "The physical head scan must activate after a zero-pixel probe, use the alternate legal yaw, observe >=10 toy pixels at delayed reach entry, and satisfy the unchanged native support, motor, >=1 s continuous contact, floor, clipping and RGB gates. This is branch exposure, not disjoint score or new-body qualification.",
    }
    BASE.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V45 third scan probe plan differs from source")
    PLAN.write_bytes(encoded)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    screen.DEST = DEST
    DEST.mkdir(parents=True, exist_ok=True)
    row = screen.run_case(plan["cases"][0], plan)
    if row["reacquisition_probe_orange_pixels"] != 0 or row["reacquisition_trigger_s"] is None:
        row["faults"].append("physical head scan branch was not activated")
    row["pass"] = not row["faults"]
    matrix = {"schema": "robot-learning-lab.cove50-v45-scan-branch-probe-c.v1",
              "role": "opened_native_branch_exposure_not_fresh_score",
              "body_role": "rejected_historical_direct_frame_sandbox",
              "plan_sha256": sha(PLAN), "complete": True,
              "case_count": 1, "pass_count": int(row["pass"]), "cases": [row]}
    (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    print(json.dumps({key: row[key] for key in
                      ("seed", "pass", "reacquisition_trigger_s",
                       "reacquisition_probe_orange_pixels", "entry_orange_pixels",
                       "contact_s", "faults")}), flush=True)


if __name__ == "__main__":
    main()
