"""New-seed native variance around the observed specialist/scoped RGB boundary."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v88_native_case as native,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v88"
VARIATIONS = (
    # seed, toy x/y shift in metres, rail volts, delay ms, gaze yaw radians
    (15001, -.0010, 0.0, 12, 5, -1.3),
    (15002, -.0005, 0.0, 12, 5, -1.3),
    (15003, .0005, 0.0, 12, 5, -1.3),
    (15004, .0010, 0.0, 12, 5, -1.3),
    (15005, 0.0, -.0010, 12, 5, -1.3),
    (15006, 0.0, -.0005, 12, 5, -1.3),
    (15007, 0.0, .0005, 12, 5, -1.3),
    (15008, 0.0, .0010, 12, 5, -1.3),
    (15009, -.0005, .0005, 10, 0, -1.5),
    (15010, .0005, -.0005, 10, 10, -1.5),
    (15011, -.0010, .0010, 11, 10, -1.5),
    (15012, .0010, -.0010, 11, 0, -1.5),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> tuple[dict, Path, Path]:
    probe_path = OUT / "native-specialist-boundary-probe/matrix.json"
    legacy_path = OUT / "legacy-specialist-control-matrix.json"
    probe = json.loads(probe_path.read_text())
    legacy = json.loads(legacy_path.read_text())
    if not (probe["complete"] and probe["screen_gate_met"] and
            legacy["complete"] and legacy["preservation_gate_met"]):
        raise ValueError("Need passing paired V88 and legacy preservation probes")
    source_path = BASE / "v87/prior75-retention-plan.json"
    source = json.loads(source_path.read_text())
    anchor = next(case for case in source["cases"] if case["seed"] == 10514)
    cases = []
    for seed, dx, dy, rail, delay, gaze in VARIATIONS:
        cases.append({**anchor,
                      "id": f"boundary-neighbor-{seed}", "seed": seed,
                      "toy_offset_x_m": round(anchor["toy_offset_x_m"] + dx, 6),
                      "toy_offset_y_m": round(anchor["toy_offset_y_m"] + dy, 6),
                      "rail_v": rail, "delay_ms": delay,
                      "gaze_head_yaw_rad": gaze,
                      "baseline_contact_s": None,
                      "expected_center_live_lock": False,
                      "expected_thin_route_bias": False})
    hashes = {**source["source_sha256"], "evaluator": sha(native.EVALUATOR)}
    if hashes["pointer"] != sha(ROOT / "robot/ape-current-design.json"):
        raise ValueError("Mechanical design pointer changed")
    plan = {
        "schema": "robot-learning-lab.cove50-v88-boundary-variance-plan.v1",
        "role": "new_seed_opened_boundary_variance_not_disjoint_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": hashes,
        "source_v87_prior75_plan_sha256": sha(source_path),
        "v88_probe_matrix_sha256": sha(probe_path),
        "v88_legacy_matrix_sha256": sha(legacy_path),
        "native_case_runner_sha256": sha(Path(native.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(cases), "cases": cases,
        "variation": "Seed-disjoint ±0.5/1.0 mm toy X/Y shifts around older failure 10514, "
                     "10–12 V rail, 0–10 ms delay and −1.3/−1.5 rad gaze; actor, RGB "
                     "model, route timing and rejected body stay fixed.",
        "gate": "All twelve fresh native visits pass RGB, at least 1.2 s continuous contact, "
                "support, rated motor, floor and clipping checks. At least two visits "
                "must actually use each specialist and scoped-Y arm route; thin and "
                "center-lock rules stay off. This is opened neighborhood development, "
                "separate from the untouched 34-case challenge.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / "boundary-variance-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Frozen V88 boundary variance plan/source identity changed")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    return plan, plan_path, OUT / "native-boundary-variance"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one to four native evaluators")
    plan, path, dest = freeze()
    if args.freeze_only:
        print(f"frozen {path} sha256={sha(path)}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        iterator = iter(plan["cases"])
        pending = {}
        for _ in range(args.workers):
            case = next(iterator, None)
            if case is not None:
                pending[pool.submit(native.run_case, case, plan, dest)] = case
        while pending:
            future = next(as_completed(tuple(pending)))
            case = pending.pop(future)
            row = future.result()
            rows[row["id"]] = row
            ordered = [rows[item["id"]] for item in plan["cases"] if item["id"] in rows]
            complete = len(ordered) == len(plan["cases"])
            specialist = sum(item["selection_mode"] == "v16_specialist_locked"
                             for item in ordered)
            scoped = sum(item["selection_mode"] == "v21_scoped_y_locked"
                         for item in ordered)
            gate = (complete and all(item["pass"] and item["margin_1p2_s_met"]
                                     for item in ordered) and specialist >= 2 and scoped >= 2 and
                    not any(item["automatic_thin_route_bias_applied"] or
                            item["automatic_center_live_lock_applied"] for item in ordered))
            matrix = {
                "schema": "robot-learning-lab.cove50-v88-boundary-variance.v1",
                "role": plan["role"], "body_role": plan["body_role"],
                "plan_sha256": sha(path), "complete": complete,
                "case_count": len(ordered),
                "pass_count": sum(item["pass"] for item in ordered),
                "specialist_use_count": specialist, "scoped_y_use_count": scoped,
                "minimum_contact_s": min(item["contact_s"] for item in ordered),
                "variance_gate_met": gate, "cases": ordered,
            }
            temp = dest / "matrix.json.tmp"
            temp.write_text(json.dumps(matrix, indent=2) + "\n")
            os.replace(temp, dest / "matrix.json")
            print(json.dumps({key: row[key] for key in
                              ("id", "pass", "contact_s", "selection_mode", "faults")}),
                  flush=True)
            next_case = next(iterator, None)
            if next_case is not None:
                pending[pool.submit(native.run_case, next_case, plan, dest)] = next_case


if __name__ == "__main__":
    main()
