"""Replay the ten fragile V41 starts with the frozen V84 observed lock."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v84_native_case as native,
    run_v83_prior75_retention as prior75,
    run_v84_center_live_lock_pilot as first,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v84"
PLAN = OUT / "prior75-pilot-plan.json"
DEST = OUT / "native-prior75-pilot"
SOURCE_MATRIX = BASE / "v83/prior75-pilot-matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    first_matrix = json.loads((first.DEST / "matrix.json").read_text())
    if (not first_matrix["complete"] or not first_matrix["pilot_gate_met"] or
            first_matrix["pass_count"] != 1):
        raise ValueError("Need passing V84 exact opened pilot")
    previous = json.loads(SOURCE_MATRIX.read_text())
    if not previous["complete"] or previous["case_count"] != 10 or previous["pass_count"] != 9:
        raise ValueError("Need completed V83 ten-start failure source")
    source_plan = json.loads(prior75.PLAN.read_text())
    by_seed = {case["seed"]: case for case in source_plan["cases"]}
    hits = set()
    cases = []
    for row in previous["cases"]:
        case = by_seed[row["seed"]]
        report = json.loads((ROOT / row["result"]).read_text())
        entry = next(event for event in report["image_goal_capture_events"]
                     if event["time_s"] == report["reach_start_s"])
        camera_x, camera_y = entry["camera_world_xyz_m"][:2]
        expected = (report["selection_mode"] == "v16_live" and
                    camera_x >= 0 and camera_y <= -.08 and
                    entry["orange_pixels"] >= 40 and
                    not report["off_authored_route_probe"])
        if expected:
            hits.add(case["seed"])
        cases.append({**case, "baseline_contact_s": row["contact_s"],
                      "expected_center_live_lock": expected})
    if (len(cases) != 10 or len({case["seed"] for case in cases}) != 10 or
            hits != {10534} or
            [case["seed"] for case in cases] != list(source_plan["pilot_seeds"])):
        raise ValueError("Ten-start observed exposure differs from prior V83 pilot")
    pilot_plan = json.loads(first.PLAN.read_text())
    sources = pilot_plan["source_sha256"]
    if sources["evaluator"] != sha(native.EVALUATOR):
        raise ValueError("V84 evaluator changed after one-start pilot")
    plan = {
        "schema": "robot-learning-lab.cove50-v84-prior75-pilot-plan.v1",
        "role": "older_opened_native_pilot_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "runner_sha256": sha(Path(__file__)),
        "native_case_runner_sha256": sha(Path(native.__file__)),
        "first_pilot_plan_sha256": sha(first.PLAN),
        "first_pilot_matrix_sha256": sha(first.DEST / "matrix.json"),
        "v83_prior75_plan_sha256": sha(prior75.PLAN),
        "v83_prior75_pilot_matrix_sha256": sha(SOURCE_MATRIX),
        "case_count": len(cases), "expected_lock_seeds": sorted(hits),
        "cases": cases,
        "gate": "All ten formerly passing V41 starts must pass fresh-process native contact, support, rated effort, floor, self-contact, clipping and RGB gates under unchanged toy, motor, gaze and authored-route inputs. The V84 observed center live-lock must activate exactly on the previously identified 10534 view and hold at least 1.2 s contact there. It must stay off in the nine other views; V83's off-author thin-route rule must stay off on all ten. Only a complete passing ten-start pilot permits wider older and route retention.",
        "design": "Replay the same ten V83 older-pilot seeds and parameters under the byte-frozen V84 evaluator, V57 arm and same body. Derive the single expected lock exposure before running from each saved V83 camera/pixel entry and prior selector mode, never from V84 outcomes. Do not reuse the V84 single-case native result as a pilot visit. This is opened retention on a rejected historical body.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V84 ten-start plan differs from current source")
    PLAN.write_bytes(encoded)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        iterator = iter(plan["cases"])
        pending = {}
        for _ in range(args.workers):
            case = next(iterator, None)
            if case is not None:
                pending[pool.submit(native.run_case, case, plan, DEST)] = case
        while pending:
            future = next(as_completed(tuple(pending)))
            case = pending.pop(future)
            row = future.result()
            rows[row["id"]] = row
            ordered = [rows[item["id"]] for item in plan["cases"] if item["id"] in rows]
            exposure = [item for item in ordered if item["automatic_center_live_lock_applied"]]
            matrix = {
                "schema": "robot-learning-lab.cove50-v84-prior75-pilot.v1",
                "role": plan["role"], "body_role": plan["body_role"],
                "plan_sha256": sha(PLAN),
                "complete": len(ordered) == len(plan["cases"]),
                "case_count": len(ordered),
                "pass_count": sum(item["pass"] for item in ordered),
                "lock_use_count": len(exposure),
                "thin_rule_use_count": sum(item["automatic_thin_route_bias_applied"]
                                           for item in ordered),
                "minimum_contact_s": min(item["contact_s"] for item in ordered),
                "margin_1p2_s_count": sum(item["margin_1p2_s_met"] for item in ordered),
                "pilot_gate_met": (len(ordered) == 10 and all(item["pass"] for item in ordered)
                                   and len(exposure) == 1 and exposure[0]["seed"] == 10534
                                   and exposure[0]["margin_1p2_s_met"] and
                                   not any(item["automatic_thin_route_bias_applied"]
                                           for item in ordered)),
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("id", "pass", "contact_s",
                               "automatic_center_live_lock_applied", "faults")}), flush=True)
            next_case = next(iterator, None)
            if next_case is not None:
                pending[pool.submit(native.run_case, next_case, plan, DEST)] = next_case


if __name__ == "__main__":
    main()
