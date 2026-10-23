"""Compare automatic and prescribed specialist reach on opened camera starts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v95_native_case as native,
    run_v98_entry_exposure as v98,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v102"
PLAN = OUT / "opened-specialist-pairs-plan.json"
MATRIX = OUT / "native-opened-pairs/matrix.json"
FORCED_EVALUATOR = Path(__file__).with_name("evaluate_forced_specialist_v102.py")
V98_MATRIX = v98.DEST / "matrix.json"
V95_PILOT = BASE / "v95/two-region-pilot-plan.json"
SEEDS = (18005, 18018, 18023, 18031)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    pilot = json.loads(V95_PILOT.read_text())
    baseline_hashes = pilot["source_sha256"]
    pointer_path = ROOT / "robot/ape-current-design.json"
    if (baseline_hashes["evaluator"] != sha(native.EVALUATOR) or
            baseline_hashes["pointer"] != sha(pointer_path) or
            json.loads(pointer_path.read_text())["status"] !=
            "rejected_no_selected_mechanical_design"):
        raise ValueError("V102 source or rejected-body pointer changed")
    opened_bytes = V98_MATRIX.read_bytes()
    opened = json.loads(opened_bytes)
    by_seed = {row["seed"]: row for row in opened["cases"]}
    v98_cases = {case["seed"]: case for case in v98.freeze(persist=False)["cases"]}
    source_reports = {}
    cases = []
    for seed in SEEDS:
        row = by_seed.get(seed)
        if not row or not row["entry_available"]:
            raise ValueError(f"Seed {seed} has no opened V98 RGB entry")
        source_reports[str(seed)] = {"result": row["result"],
                                     "sha256": sha(ROOT / row["result"])}
        for forced in (False, True):
            cases.append({**v98_cases[seed],
                          "id": f"{'forced' if forced else 'automatic'}-{seed}",
                          "route_variant": "center",
                          "diagnostic_force_specialist": forced})
    plan = {
        "schema": "robot-learning-lab.cove50-v102-opened-specialist-pairs-plan.v1",
        "role": "opened_full_native_matched_route_diagnostic_not_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "case_count": len(cases), "cases": cases,
        "source_sha256": {
            "baseline": baseline_hashes,
            "forced": {**baseline_hashes, "evaluator": sha(FORCED_EVALUATOR)},
            "runner": sha(Path(__file__)),
            "native_case_runner": sha(Path(native.__file__)),
            "v95_pilot_plan": sha(V95_PILOT),
        },
        "opened_v98_source_reports": source_reports,
        "method": ("Four already opened V98 starts span distinct image-goal and camera "
                   "regions. Reuse their exact toy, motor, delay and gaze settings. "
                   "Run uninterrupted 9 s native contact evaluations in matched "
                   "automatic-V95 and prescribed-specialist pairs. The copied "
                   "forced evaluator changes only the arm route after the used "
                   "RGB entry; verify identical entry observation within each pair. "
                   "Score full support, rated effort, collision, RGB and fingertip "
                   "contact on both visits. These opened starts are development data, "
                   "not a disjoint score or a route-promotion gate."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V102 opened-pair plan or source changed")
    PLAN.write_bytes(encoded)
    return plan


def entry_features(row: dict) -> dict:
    report = json.loads((ROOT / row["result"]).read_text())
    entry = next(event for event in report["image_goal_capture_events"]
                 if event["used_for_control"])
    return {"entry_error_xy_m": report["selection_features"]["entry_error_xy_m"],
            "camera_world_xyz_m": entry["camera_world_xyz_m"],
            "orange_pixels": entry["orange_pixels"],
            "goal_world_xyz_m": entry["estimated_goal_world_xyz_m"]}


def write_matrix(plan: dict, rows: dict[str, dict]) -> None:
    ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
    paired = []
    for seed in SEEDS:
        baseline = rows.get(f"automatic-{seed}")
        forced = rows.get(f"forced-{seed}")
        if baseline and forced:
            first, second = entry_features(baseline), entry_features(forced)
            if (first["orange_pixels"] != second["orange_pixels"] or
                    any(abs(a - b) > 1e-6
                        for key in ("entry_error_xy_m", "camera_world_xyz_m",
                                    "goal_world_xyz_m")
                        for a, b in zip(first[key], second[key]))):
                raise ValueError(f"V102 pair {seed} changed the RGB entry before routing")
            paired.append({"seed": seed, "entry": first,
                           "automatic_contact_s": baseline["contact_s"],
                           "forced_contact_s": forced["contact_s"],
                           "automatic_full_gate": baseline["pass"],
                           "forced_full_gate": forced["pass"]})
    matrix = {"schema": "robot-learning-lab.cove50-v102-opened-specialist-pairs.v1",
              "role": plan["role"], "body_role": plan["body_role"],
              "plan_sha256": sha(PLAN),
              "complete": len(ordered) == len(plan["cases"]),
              "case_count": len(ordered), "pair_count": len(paired),
              "pass_count": sum(row["pass"] for row in ordered),
              "paired_entry_verified": len(paired) == len(SEEDS),
              "pairs": paired, "cases": ordered}
    MATRIX.parent.mkdir(parents=True, exist_ok=True)
    temp = MATRIX.with_suffix(".json.tmp")
    temp.write_text(json.dumps(matrix, indent=2) + "\n")
    os.replace(temp, MATRIX)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    rows = {}
    original = native.EVALUATOR
    for case in plan["cases"]:
        forced = case["diagnostic_force_specialist"]
        native.EVALUATOR = FORCED_EVALUATOR if forced else original
        source = plan["source_sha256"]["forced" if forced else "baseline"]
        dest = OUT / ("native-forced" if forced else "native-automatic")
        dest.mkdir(parents=True, exist_ok=True)
        try:
            row = native.run_case(case, {"source_sha256": source}, dest)
        finally:
            native.EVALUATOR = original
        report = json.loads((ROOT / row["result"]).read_text())
        if bool(report.get("diagnostic_force_specialist", False)) != forced:
            raise ValueError(f"V102 route-mode identity mismatch: {case['id']}")
        row["diagnostic_force_specialist"] = forced
        rows[case["id"]] = row
        write_matrix(plan, rows)
        print(json.dumps({key: row[key] for key in
                          ("id", "pass", "contact_s", "selection_mode", "faults")}),
              flush=True)


if __name__ == "__main__":
    main()
