"""Replay the 75 formerly passing V41 starts under frozen V83 control."""

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
    run_v83_thin_route_pilot as v83,
    run_v83_thin_route_retention as retention,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v83"
PLAN = OUT / "prior75-retention-plan.json"
DEST = OUT / "native-prior75-retention"
PILOT = OUT / "prior75-pilot-matrix.json"
PILOT_SEEDS = (10501, 10508, 10528, 10529, 10534, 10545, 10546, 10604,
               10608, 10615)
SOURCE_PLAN = BASE / "v41/opened-gate-plan.json"
SOURCE_MATRIX = BASE / "v41/opened-native/matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    earlier = json.loads((retention.DEST / "matrix.json").read_text())
    if (not earlier["complete"] or not earlier["screen_gate_met"] or
            earlier["case_count"] != 63 or earlier["pass_count"] != 63):
        raise ValueError("Need complete passing V83 63-route retention")
    v41_plan = json.loads(SOURCE_PLAN.read_text())
    v41_matrix = json.loads(SOURCE_MATRIX.read_text())
    if (not v41_matrix["complete"] or v41_matrix["case_count"] != 75 or
            v41_matrix["pass_count"] != 75 or len(v41_plan["cases"]) != 75 or
            v41_matrix["plan_sha256"] != sha(SOURCE_PLAN)):
        raise ValueError("Need complete 75/75 V41 source gate")
    previous = {row["seed"]: row for row in v41_matrix["cases"]}
    if len(previous) != 75 or not set(PILOT_SEEDS).issubset(previous):
        raise ValueError("V41 source seeds or pilot cases differ")
    cases = []
    for original in v41_plan["cases"]:
        seed = original["seed"]
        row = previous[seed]
        if not row["pass"]:
            raise ValueError(f"V41 previously passing seed {seed} now differs")
        cases.append({
            "id": f"prior-{seed}", "seed": seed,
            "route_variant": "center", "pitch_offset_rad": .105,
            "roll_clearance_rad": -.25,
            "rail_v": original["rail_v"], "delay_ms": original["delay_ms"],
            "toy_offset_x_m": original["toy_offset_x_m"],
            "toy_offset_y_m": original["toy_offset_y_m"],
            "gaze_head_yaw_rad": original["gaze_head_yaw_rad"],
            "v73_contact_s": row["contact_s"],
            "prior_role": original["role"],
            "expected_thin_route_bias": False,
        })
    if len({case["seed"] for case in cases}) != 75:
        raise ValueError("V41 source starts are not unique")
    pilot_plan = json.loads(v83.PLAN.read_text())
    if pilot_plan["source_sha256"]["evaluator"] != sha(v83.EVALUATOR):
        raise ValueError("Frozen V83 evaluator changed")
    plan = {
        "schema": "robot-learning-lab.cove50-v83-prior75-retention-plan.v1",
        "role": "earlier_opened_native_retention_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": pilot_plan["source_sha256"],
        "runner_sha256": sha(Path(__file__)),
        "center_validator_sha256": sha(Path(retention.__file__)),
        "v83_retention_plan_sha256": sha(retention.PLAN),
        "v83_retention_matrix_sha256": sha(retention.DEST / "matrix.json"),
        "v41_plan_sha256": sha(SOURCE_PLAN),
        "v41_matrix_sha256": sha(SOURCE_MATRIX),
        "case_count": 75, "cases": cases,
        "pilot_seeds": PILOT_SEEDS,
        "gate": "All 75 earlier V41 opened starts must pass fresh-process native contact, support, rated effort, hand-floor, self-contact, clipping and scored RGB with frozen V83 control, and the new thin-route rule must never activate on their unchanged authored route. Separately report whether every contact lasts at least 1.2 s. This is earlier opened retention on the rejected Direct Frame, not a fresh or selected-body score.",
        "design": "V41's 75/75 formerly passing opened starts retain their exact seed, toy, motor and gaze settings. First replay ten prior failures and fragile image/contact controls, including the two weakest earlier passing contacts; if all pass with zero new-rule activations, replay all 75. Use the frozen V83 evaluator and V57 arm at the unchanged authored .105/-.25 route. The V83 center-route validator corrects the pilot-only off-author guard but keeps its native outcome, source, image and branch checks. Do not reuse V41 result files as V83 outcomes.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V83 prior75 retention differs from current source")
    PLAN.write_bytes(encoded)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    row = retention.run_retention_case(case, plan)
    row["prior_role"] = case["prior_role"]
    row["baseline_contact_s"] = row.pop("v73_contact_s")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--stage", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    if args.stage == "full":
        pilot = json.loads(PILOT.read_text())
        if (not pilot["complete"] or not pilot["screen_gate_met"] or
                pilot["case_count"] != len(PILOT_SEEDS)):
            raise ValueError("Need passing ten-case V83 prior75 pilot")
    selected = ([case for case in plan["cases"] if case["seed"] in PILOT_SEEDS]
                if args.stage == "pilot" else plan["cases"])
    DEST.mkdir(parents=True, exist_ok=True)
    retention.DEST = DEST
    v83.DEST = DEST
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        iterator = iter(selected)
        pending = {}
        for _ in range(args.workers):
            case = next(iterator, None)
            if case is not None:
                pending[pool.submit(run_case, case, plan)] = case
        while pending:
            future = next(as_completed(tuple(pending)))
            case = pending.pop(future)
            row = future.result()
            rows[row["id"]] = row
            ordered = [rows[item["id"]] for item in selected if item["id"] in rows]
            screen_pass = (len(ordered) == len(selected) and all(item["pass"] and
                           not item["automatic_thin_route_bias_applied"] for item in ordered))
            matrix = {
                "schema": f"robot-learning-lab.cove50-v83-prior75-{args.stage}-retention.v1",
                "role": plan["role"], "body_role": plan["body_role"],
                "stage": args.stage, "plan_sha256": sha(PLAN),
                "complete": len(ordered) == len(selected),
                "case_count": len(ordered),
                "pass_count": sum(item["pass"] for item in ordered),
                "thin_rule_use_count": sum(item["automatic_thin_route_bias_applied"]
                                           for item in ordered),
                "minimum_contact_s": min(item["contact_s"] for item in ordered),
                "margin_1p2_s_count": sum(item["margin_1p2_s_met"] for item in ordered),
                "screen_gate_met": screen_pass,
                "all_margin_1p2_s_met": (screen_pass and
                                          all(item["margin_1p2_s_met"] for item in ordered)),
                "cases": ordered,
            }
            (PILOT if args.stage == "pilot" else DEST / "matrix.json").write_text(
                json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("id", "pass", "contact_s", "faults")}), flush=True)
            next_case = next(iterator, None)
            if next_case is not None:
                pending[pool.submit(run_case, next_case, plan)] = next_case


if __name__ == "__main__":
    main()
