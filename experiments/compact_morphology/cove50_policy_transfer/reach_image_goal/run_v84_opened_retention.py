"""Replay V84 on the frozen V70/V73 routes and earlier V41 starts."""

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
    run_v83_prior75_retention as prior75,
    run_v83_thin_route_retention as routes,
    run_v84_native_case as native,
    run_v84_prior75_pilot as pilot,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v84"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(stage: str) -> tuple[dict, Path, Path]:
    ten = json.loads((pilot.DEST / "matrix.json").read_text())
    if (not ten["complete"] or not ten["pilot_gate_met"] or
            ten["case_count"] != 10 or ten["pass_count"] != 10):
        raise ValueError("Need complete passing V84 ten-start pilot")
    ten_plan = json.loads(pilot.PLAN.read_text())
    if (ten["plan_sha256"] != sha(pilot.PLAN) or
            ten_plan["source_sha256"]["evaluator"] != sha(native.EVALUATOR) or
            ten_plan["native_case_runner_sha256"] != sha(Path(native.__file__))):
        raise ValueError("V84 pilot source identity changed")
    if stage == "routes63":
        source_plan = routes.PLAN
        source_matrix = routes.DEST / "matrix.json"
        earlier = json.loads(source_matrix.read_text())
        if (not earlier["complete"] or not earlier["screen_gate_met"] or
                earlier["case_count"] != 63 or earlier["pass_count"] != 63 or
                earlier["thin_rule_use_count"] != 6):
            raise ValueError("Need passing V83 63-route opened source")
        previous = {row["id"]: row for row in earlier["cases"]}
        source_cases = json.loads(source_plan.read_text())["cases"]
        cases = []
        for original in source_cases:
            row = previous[original["id"]]
            report = json.loads((ROOT / row["result"]).read_text())
            entry = next(event for event in report["image_goal_capture_events"]
                         if event["time_s"] == report["reach_start_s"])
            x, y = entry["camera_world_xyz_m"][:2]
            expected_lock = (not report["off_authored_route_probe"] and
                             report["selection_mode"] == "v16_live" and
                             x >= 0 and y <= -.08 and entry["orange_pixels"] >= 40)
            cases.append({**original, "baseline_contact_s": row["contact_s"],
                          "expected_center_live_lock": expected_lock})
        if (len(cases) != 63 or len({case["id"] for case in cases}) != 63 or
                any(case["expected_center_live_lock"] for case in cases)):
            raise ValueError("V83 route sources unexpectedly expose V84 center lock")
        gate = ("All 63 previously passing V70/V73 route visits must pass full fresh-process "
                "native contact, support, rated effort, floor, self-contact, clipping and "
                "RGB gates. Only the six predeclared thin-route visits may activate that "
                "rule, all six must hold at least 1.2 s fingertip contact, and the new "
                "authored-center live lock must remain off. Opened retention on a rejected "
                "historical body, not a selected-body qualification.")
    elif stage == "prior75":
        route_matrix = OUT / "native-routes63-retention/matrix.json"
        route_results = json.loads(route_matrix.read_text())
        if (not route_results["complete"] or not route_results["screen_gate_met"] or
                route_results["case_count"] != 63 or route_results["pass_count"] != 63):
            raise ValueError("Need passing V84 63-route retention before older starts")
        source_plan = prior75.PLAN
        source_matrix = prior75.SOURCE_MATRIX
        earlier = json.loads(source_matrix.read_text())
        if not earlier["complete"] or earlier["pass_count"] != 75:
            raise ValueError("Need V41 75/75 formerly passing source")
        pilot_by_seed = {row["seed"]: row for row in ten["cases"]}
        source_cases = json.loads(source_plan.read_text())["cases"]
        cases = []
        for original in source_cases:
            case = dict(original)
            case["baseline_contact_s"] = case.pop("v73_contact_s")
            if case["seed"] in pilot_by_seed:
                case["expected_center_live_lock"] = pilot_by_seed[case["seed"]][
                    "automatic_center_live_lock_applied"]
            cases.append(case)
        pilot_plan = json.loads(pilot.PLAN.read_text())
        if (len(cases) != 75 or len({case["seed"] for case in cases}) != 75 or
                set(pilot_by_seed) != {case["seed"] for case in pilot_plan["cases"]} or
                {seed for seed, row in pilot_by_seed.items()
                 if row["automatic_center_live_lock_applied"]} !=
                set(pilot_plan["expected_lock_seeds"])):
            raise ValueError("Older source or ten-start exposure differs")
        gate = ("All 75 formerly passing V41 starts must pass fresh-process native contact, "
                "support, rated effort, floor, self-contact, clipping and RGB gates under "
                "the frozen V84 evaluator. The off-author thin-route rule must stay off. "
                "The ten pre-opened pilot starts must repeat their observed center-lock "
                "exposure, including at least 1.2 s contact on 10534. Report all other "
                "center-lock activations and contact margins. Opened retention on the "
                "rejected historical body, not fresh score or selected-body qualification.")
    else:
        raise ValueError(stage)
    plan = {
        "schema": f"robot-learning-lab.cove50-v84-{stage}-retention-plan.v1",
        "role": f"opened_{stage}_native_retention_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": ten_plan["source_sha256"],
        "runner_sha256": sha(Path(__file__)),
        "native_case_runner_sha256": sha(Path(native.__file__)),
        "ten_pilot_plan_sha256": sha(pilot.PLAN),
        "ten_pilot_matrix_sha256": sha(pilot.DEST / "matrix.json"),
        "source_plan_sha256": sha(source_plan),
        "source_matrix_sha256": sha(source_matrix),
        "case_count": len(cases), "cases": cases,
        "gate": gate,
        "design": ("Reuse exact saved cases, motor, toy and gaze settings under the frozen "
                   "V84 evaluator and V57 arm in separate native processes. The six thin "
                   "route exposures come from the V83 predeclared plan. The center lock "
                   "is expected off on all 63 route visits from their prior camera and "
                   "selector evidence. The ten older pilot exposures are predeclared; "
                   "remaining older cases are observed and audited without selecting "
                   "policy changes from their outcomes."),
    }
    path = OUT / f"{stage}-retention-plan.json"
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if path.exists() and path.read_bytes() != encoded:
        raise ValueError(f"Frozen V84 {stage} plan differs from current source")
    path.write_bytes(encoded)
    return plan, path, OUT / f"native-{stage}-retention"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("routes63", "prior75"))
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    plan, path, dest = freeze(args.stage)
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
            thin = [item for item in ordered if item["automatic_thin_route_bias_applied"]]
            locks = [item for item in ordered if item["automatic_center_live_lock_applied"]]
            complete = len(ordered) == len(plan["cases"])
            if args.stage == "routes63":
                rule_gate = (len(thin) == 6 and
                             all(item["expected_thin_route_bias"] and
                                 item["margin_1p2_s_met"] for item in thin) and
                             not locks)
            else:
                pilot_seeds = {case["seed"] for case in
                               json.loads(pilot.PLAN.read_text())["cases"]}
                prior = [item for item in ordered if item["seed"] in pilot_seeds]
                rule_gate = (not thin and len(prior) == 10 and
                             all(item["automatic_center_live_lock_applied"] ==
                                 item["expected_center_live_lock"] for item in prior) and
                             next(item for item in prior if item["seed"] == 10534)[
                                 "margin_1p2_s_met"])
            matrix = {
                "schema": f"robot-learning-lab.cove50-v84-{args.stage}-retention.v1",
                "role": plan["role"], "body_role": plan["body_role"],
                "plan_sha256": sha(path), "complete": complete,
                "case_count": len(ordered), "pass_count": sum(item["pass"] for item in ordered),
                "thin_rule_use_count": len(thin), "lock_use_count": len(locks),
                "minimum_contact_s": min(item["contact_s"] for item in ordered),
                "margin_1p2_s_count": sum(item["margin_1p2_s_met"] for item in ordered),
                "rule_gate_met": rule_gate,
                "screen_gate_met": complete and all(item["pass"] for item in ordered) and rule_gate,
                "cases": ordered,
            }
            (dest / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("id", "pass", "contact_s", "automatic_center_live_lock_applied",
                               "automatic_thin_route_bias_applied", "faults")}), flush=True)
            next_case = next(iterator, None)
            if next_case is not None:
                pending[pool.submit(native.run_case, next_case, plan, dest)] = next_case


if __name__ == "__main__":
    main()
