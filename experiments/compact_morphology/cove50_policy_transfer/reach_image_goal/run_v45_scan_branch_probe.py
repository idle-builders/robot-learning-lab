"""Native opened edge-view probe for V45's physical head-scan branch."""

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
PLAN = BASE / "scan-branch-probe-plan.json"
DEST = BASE / "scan-branch-probe"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    previous = json.loads(screen.PLAN.read_text())
    original = next(c for c in previous["cases"] if c["seed"] == 10806)
    if (original["toy_offset_x_m"] != .011 or original["toy_offset_y_m"] != .005 or
            original["gaze_head_yaw_rad"] != -1.3):
        raise ValueError("V45 edge-view source changed")
    case = {**original, "toy_offset_y_m": .004,
            "baseline_v41_contact_s": None,
            "development_role": "opened_edge_view_minus_1mm_y_scan_exposure"}
    plan = {
        "schema": "robot-learning-lab.cove50-v45-scan-branch-probe-plan.v1",
        "role": "opened_native_branch_exposure_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": previous["source_sha256"],
        "v45_combined_screen_plan_sha256": sha(screen.PLAN),
        "runner_sha256": sha(Path(__file__)),
        "case_count": 1, "cases": [case],
        "selection": "At the saved V45 10806 preposition pose, a pose-only virtual-RGB sweep saw 10 toy pixels at +5 mm Y and zero pixels after a -1 mm Y offset. This selects a native opened development probe; the pose-only observation is not a policy outcome.",
        "gate": "The physical head scan must activate after a zero-pixel probe, use the alternate legal yaw, observe >=10 toy pixels at delayed reach entry, and satisfy the unchanged native support, motor, >=1 s continuous contact, floor, clipping and RGB gates. This is branch exposure, not disjoint score or new-body qualification.",
    }
    BASE.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V45 scan probe plan differs from source")
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
    matrix = {"schema": "robot-learning-lab.cove50-v45-scan-branch-probe.v1",
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
