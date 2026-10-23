"""Opened paired static-gaze control for V45's scanned edge-view contact miss."""

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
PLAN = BASE / "scan-timing-control-plan.json"
DEST = BASE / "scan-timing-control"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    source = json.loads(screen.PLAN.read_text())
    edge_path = BASE / "scan-branch-probe-c/matrix.json"
    edge = json.loads(edge_path.read_text())
    if (not edge["complete"] or edge["pass_count"] != 0 or
            edge["cases"][0]["reacquisition_probe_orange_pixels"] != 0 or
            edge["cases"][0]["entry_orange_pixels"] != 40 or
            edge["cases"][0]["contact_s"] != .78):
        raise ValueError("Expected opened V45 physical scan visibility/contact failure")
    case = next(c for c in source["cases"] if c["seed"] == 10806).copy()
    case.update(toy_offset_x_m=.012, gaze_head_yaw_rad=-1.5,
                baseline_v41_contact_s=None,
                development_role="opened_scanned_edge_view_static_alt_yaw_control")
    plan = {
        "schema": "robot-learning-lab.cove50-v45-scan-timing-control-plan.v1",
        "role": "opened_paired_static_gaze_diagnostic_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": source["source_sha256"],
        "v45_combined_screen_plan_sha256": sha(screen.PLAN),
        "scanned_edge_probe_matrix_sha256": sha(edge_path),
        "runner_sha256": sha(Path(__file__)),
        "case_count": 1, "cases": [case],
        "hypothesis": "The V45 physical scan at 8.48 s recovers 40 pixels but delays reach to 9.0 s and holds contact only 0.78 s. Keep the same seed, toy, actor, timing and native route; command the alternate -1.5 rad yaw from posture start so reach begins at 8.5 s. A native pass would implicate scan timing/head transition, not prove an automatic recovery policy.",
        "gate": "Use the unchanged native support, motor, >=1 s continuous contact, floor, clipping and RGB <10 mm checks. This paired static-gaze visit is opened diagnosis, not a scored automatic scan.",
    }
    BASE.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V45 timing control plan differs from source")
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
    matrix = {"schema": "robot-learning-lab.cove50-v45-scan-timing-control.v1",
              "role": "opened_paired_static_gaze_diagnostic_not_fresh_score",
              "body_role": "rejected_historical_direct_frame_sandbox",
              "plan_sha256": sha(PLAN), "complete": True,
              "case_count": 1, "pass_count": int(row["pass"]), "cases": [row]}
    (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    print(json.dumps({key: row[key] for key in
                      ("seed", "pass", "reacquisition_trigger_s",
                       "entry_orange_pixels", "reach_start_s", "contact_s", "faults")}), flush=True)


if __name__ == "__main__":
    main()
