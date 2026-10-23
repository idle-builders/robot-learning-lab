"""Finish opened matched reach pairs using saved native entry-state identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v95_native_case as native,
    run_v98_entry_exposure as v98,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v104"
PLAN = OUT / "opened-specialist-pairs-plan.json"
MATRIX = OUT / "native-opened-pairs/matrix.json"
FORCED_EVALUATOR = Path(__file__).with_name("evaluate_forced_specialist_v102.py")
V103_OUT = BASE / "v103"
V103_PLAN = V103_OUT / "opened-specialist-pairs-plan.json"
V98_MATRIX = v98.DEST / "matrix.json"
V95_PILOT = BASE / "v95/two-region-pilot-plan.json"
SEEDS = (18005, 18018, 18023, 18031)
REUSED_IDS = {f"{route}-{seed}" for seed in (18005, 18018, 18023)
              for route in ("automatic", "forced")}


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
        raise ValueError("V104 source or rejected-body pointer changed")
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
    reused = {}
    for case in cases:
        if case["id"] not in REUSED_IDS:
            continue
        route = "forced" if case["diagnostic_force_specialist"] else "automatic"
        stem = f"seed-{case['seed']}-{case['id']}"
        reused[case["id"]] = {}
        for suffix in ("json", "npz", "log"):
            path = V103_OUT / f"native-{route}" / f"{stem}.{suffix}"
            if not path.is_file():
                raise ValueError(f"Missing completed V103 visit: {path}")
            reused[case["id"]][suffix] = {"path": str(path.relative_to(ROOT)),
                                           "sha256": sha(path)}
    plan = {
        "schema": "robot-learning-lab.cove50-v104-opened-specialist-pairs-plan.v1",
        "role": "opened_full_native_matched_route_diagnostic_not_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "case_count": len(cases), "cases": cases,
        "source_sha256": {
            "baseline": baseline_hashes,
            "forced": {**baseline_hashes, "evaluator": sha(FORCED_EVALUATOR)},
            "runner": sha(Path(__file__)),
            "native_case_runner": sha(Path(native.__file__)),
            "v95_pilot_plan": sha(V95_PILOT),
            "v103_plan": sha(V103_PLAN),
        },
        "opened_v98_source_reports": source_reports,
        "reused_v103_visits": reused,
        "entry_tolerances": {"rounded_entry_error_xy_m": 0.000002,
                             "rounded_camera_world_xyz_m": 0.000015,
                             "estimated_goal_world_xyz_m": 0.000001,
                             "native_entry_qpos_max_abs": 0.000005},
        "supersedes": ("V103 stopped after six completed visits: the third pair's "
                       "rounded camera X differed by one 0.00001 m bin, while its "
                       "native entry qpos differed by only 0.0000013 in the saved "
                       "generalized coordinates, goal by 0.000000015 m, and toy "
                       "pixels and goal features matched. V104 retains all six exact "
                       "reports/traces and verifies saved native entry state before "
                       "running the last pair."),
        "method": ("Four already opened V98 starts span distinct image-goal and camera "
                   "regions. Reuse their exact toy, motor, delay and gaze settings. "
                   "Run uninterrupted 9 s native contact evaluations in matched "
                   "automatic-V95 and prescribed-specialist pairs. The copied "
                   "forced evaluator changes only the arm route after the used "
                   "RGB entry; verify equivalent saved native entry state within "
                   "the frozen pose, goal, camera and pixel tolerances. "
                   "Score full support, rated effort, collision, RGB and fingertip "
                   "contact on both visits. These opened starts are development data, "
                   "not a disjoint score or a route-promotion gate."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V104 opened-pair plan or source changed")
    PLAN.write_bytes(encoded)
    return plan


def entry_features(row: dict) -> dict:
    report = json.loads((ROOT / row["result"]).read_text())
    entry = next(event for event in report["image_goal_capture_events"]
                 if event["used_for_control"])
    return {"entry_error_xy_m": report["selection_features"]["entry_error_xy_m"],
            "camera_world_xyz_m": entry["camera_world_xyz_m"],
            "orange_pixels": entry["orange_pixels"],
            "goal_world_xyz_m": entry["estimated_goal_world_xyz_m"],
            "entry_time_s": entry["time_s"]}


def native_entry_qpos(row: dict, entry_time_s: float) -> np.ndarray:
    with np.load(ROOT / row["trace"]) as trace:
        index = round(entry_time_s / float(trace["step_dt_s"]))
        return trace["qpos"][index].copy()


def write_matrix(plan: dict, rows: dict[str, dict]) -> None:
    ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
    paired = []
    for seed in SEEDS:
        baseline = rows.get(f"automatic-{seed}")
        forced = rows.get(f"forced-{seed}")
        if baseline and forced:
            first, second = entry_features(baseline), entry_features(forced)
            tolerances = plan["entry_tolerances"]
            qpos_a = native_entry_qpos(baseline, first["entry_time_s"])
            qpos_b = native_entry_qpos(forced, second["entry_time_s"])
            if qpos_a.shape != qpos_b.shape:
                raise ValueError(f"V104 pair {seed} changed native state dimensions")
            qpos_diff = float(np.max(np.abs(qpos_a - qpos_b)))
            if (first["orange_pixels"] != second["orange_pixels"] or
                    first["entry_time_s"] != second["entry_time_s"] or
                    qpos_diff > tolerances["native_entry_qpos_max_abs"] or
                    any(abs(a - b) > tolerances[{
                            "entry_error_xy_m": "rounded_entry_error_xy_m",
                            "camera_world_xyz_m": "rounded_camera_world_xyz_m",
                            "goal_world_xyz_m": "estimated_goal_world_xyz_m"}[key]]
                        for key in ("entry_error_xy_m", "camera_world_xyz_m",
                                    "goal_world_xyz_m")
                        for a, b in zip(first[key], second[key]))):
                raise ValueError(f"V104 pair {seed} changed the native RGB entry")
            paired.append({"seed": seed, "entry": first,
                           "native_entry_qpos_max_abs": qpos_diff,
                           "automatic_contact_s": baseline["contact_s"],
                           "forced_contact_s": forced["contact_s"],
                           "automatic_full_gate": baseline["pass"],
                           "forced_full_gate": forced["pass"]})
    matrix = {"schema": "robot-learning-lab.cove50-v104-opened-specialist-pairs.v1",
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
        if case["id"] in plan["reused_v103_visits"]:
            stem = f"seed-{case['seed']}-{case['id']}"
            for suffix, saved in plan["reused_v103_visits"][case["id"]].items():
                source_path = ROOT / saved["path"]
                target_path = dest / f"{stem}.{suffix}"
                if sha(source_path) != saved["sha256"]:
                    raise ValueError(f"V103 reused visit changed: {source_path}")
                if target_path.exists():
                    if sha(target_path) != saved["sha256"]:
                        raise ValueError(f"V104 reused visit changed: {target_path}")
                else:
                    shutil.copy2(source_path, target_path)
        try:
            row = native.run_case(case, {"source_sha256": source}, dest)
        finally:
            native.EVALUATOR = original
        report = json.loads((ROOT / row["result"]).read_text())
        if bool(report.get("diagnostic_force_specialist", False)) != forced:
            raise ValueError(f"V104 route-mode identity mismatch: {case['id']}")
        row["diagnostic_force_specialist"] = forced
        rows[case["id"]] = row
        write_matrix(plan, rows)
        print(json.dumps({key: row[key] for key in
                          ("id", "pass", "contact_s", "selection_mode", "faults")}),
              flush=True)


if __name__ == "__main__":
    main()
