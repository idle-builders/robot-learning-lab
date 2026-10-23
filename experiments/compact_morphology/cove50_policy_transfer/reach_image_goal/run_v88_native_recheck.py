"""Source-frozen native retention ladder for the V88 RGB route boundary."""

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
SOURCES = {
    "variance16": ("boundary-variance-plan.json", 16),
    "controls4": ("boundary-controls-plan.json", 4),
    "routes63": ("routes63-retention-plan.json", 63),
    "prior75": ("prior75-retention-plan.json", 75),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ready(stage: str) -> dict[str, str]:
    probe_path = OUT / "native-specialist-boundary-probe/matrix.json"
    legacy_path = OUT / "legacy-specialist-control-matrix.json"
    boundary_path = OUT / "native-boundary-variance/matrix.json"
    probe = json.loads(probe_path.read_text())
    legacy = json.loads(legacy_path.read_text())
    boundary = json.loads(boundary_path.read_text())
    if not (probe["complete"] and probe["screen_gate_met"] and
            legacy["complete"] and legacy["preservation_gate_met"] and
            boundary["complete"] and boundary["variance_gate_met"]):
        raise ValueError("Need passing V88 paired, legacy and 12-start boundary variance")
    evidence = {"probe": sha(probe_path), "legacy": sha(legacy_path),
                "boundary_variance": sha(boundary_path)}
    for needed in ({"routes63": ("variance16", "controls4"),
                    "prior75": ("routes63",)}.get(stage, ())):
        path = OUT / f"native-{needed}-recheck/matrix.json"
        result = json.loads(path.read_text())
        if not result["complete"] or not result["screen_gate_met"]:
            raise ValueError(f"Need passing V88 {needed} before {stage}")
        evidence[needed] = sha(path)
    return evidence


def freeze(stage: str) -> tuple[dict, Path, Path]:
    prerequisite_hashes = ready(stage)
    source_name, count = SOURCES[stage]
    source_path = BASE / "v87" / source_name
    source = json.loads(source_path.read_text())
    if len(source["cases"]) != count:
        raise ValueError("Source case count changed")
    hashes = {**source["source_sha256"], "evaluator": sha(native.EVALUATOR)}
    if hashes["pointer"] != sha(ROOT / "robot/ape-current-design.json"):
        raise ValueError("Mechanical design pointer changed")
    plan = {
        "schema": f"robot-learning-lab.cove50-v88-{stage}-plan.v1",
        "role": f"opened_{stage}_native_recheck_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": hashes,
        "source_v87_plan_sha256": sha(source_path),
        "prerequisite_matrix_sha256": prerequisite_hashes,
        "native_case_runner_sha256": sha(Path(native.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "case_count": count,
        "cases": source["cases"],
        "change": "Only the observed specialist/scoped x boundary changes from V87: "
                  "the specialist interval closes at 1.5 mm. Actor, RGB, timing, "
                  "body and motor model are unchanged.",
        "gate": source["gate"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / f"{stage}-recheck-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Frozen V88 plan/source identity changed")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    return plan, plan_path, OUT / f"native-{stage}-recheck"


def rule_gate(stage: str, rows: list[dict]) -> bool:
    thin = [r for r in rows if r["automatic_thin_route_bias_applied"]]
    locks = [r for r in rows if r["automatic_center_live_lock_applied"]]
    if stage == "variance16":
        height = [r for r in rows if r["selection_mode"] == "v16_v10_height_locked"]
        return (len(height) >= 8 and not thin and not locks and
                all(r["margin_1p2_s_met"] for r in rows))
    if stage == "controls4":
        return not thin and not locks and all(r["margin_1p2_s_met"] for r in rows)
    if stage == "routes63":
        return (len(thin) == 6 and not locks and
                all(r["expected_thin_route_bias"] and r["margin_1p2_s_met"]
                    for r in thin))
    pilot = [r for r in rows if r["expected_center_live_lock"] is not None]
    boundary = [r for r in rows if r["seed"] in (10505, 10507)]
    rescue = [r for r in rows if r["seed"] == 10534]
    return (not thin and len(pilot) == 10 and
            all(r["automatic_center_live_lock_applied"] ==
                r["expected_center_live_lock"] for r in pilot) and
            len(boundary) == 2 and
            all(r["margin_1p2_s_met"] and
                r["selection_mode"] == "v16_v10_height_locked" for r in boundary) and
            len(rescue) == 1 and rescue[0]["margin_1p2_s_met"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=SOURCES)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    plan, plan_path, dest = freeze(args.stage)
    if args.freeze_only:
        print(f"frozen {plan_path} sha256={sha(plan_path)}")
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
            if (case.get("expected_selection_mode") is not None and
                    row["selection_mode"] != case["expected_selection_mode"]):
                row["faults"].append("route differs from frozen expectation")
                row["pass"] = False
            rows[row["id"]] = row
            ordered = [rows[item["id"]] for item in plan["cases"] if item["id"] in rows]
            complete = len(ordered) == len(plan["cases"])
            matrix = {
                "schema": f"robot-learning-lab.cove50-v88-{args.stage}-recheck.v1",
                "role": plan["role"], "body_role": plan["body_role"],
                "plan_sha256": sha(plan_path), "complete": complete,
                "case_count": len(ordered),
                "pass_count": sum(item["pass"] for item in ordered),
                "thin_rule_use_count": sum(item["automatic_thin_route_bias_applied"]
                                           for item in ordered),
                "lock_use_count": sum(item["automatic_center_live_lock_applied"]
                                      for item in ordered),
                "minimum_contact_s": min(item["contact_s"] for item in ordered),
                "rule_gate_met": rule_gate(args.stage, ordered),
                "screen_gate_met": (complete and all(item["pass"] for item in ordered)
                                    and rule_gate(args.stage, ordered)),
                "cases": ordered,
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
