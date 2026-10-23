"""Check V83's unpiloted sibling exposures and thin-contact repeatability."""

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
    run_v73_route_holdout as v73,
    run_v83_thin_route_pilot as v83,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v83"
PLAN = OUT / "sibling-and-repeatability-plan.json"
DEST = OUT / "native-sibling-and-repeats"
SIBLINGS = ("positive-07-roll-low", "edge-11-pitch-high")
REPEAT_SOURCE = "positive-01-pitch-low"
REPEAT_COUNT = 3


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    pilot = json.loads((v83.DEST / "matrix.json").read_text())
    if (not pilot["complete"] or not pilot["pilot_gate_met"] or
            pilot["pass_count"] != 8 or pilot["thin_rule_use_count"] != 4):
        raise ValueError("Need complete passing V83 eight-route pilot")
    previous = json.loads((v73.DEST / "matrix.json").read_text())
    originals = {case["id"]: case for case in json.loads(v73.PLAN.read_text())["cases"]}
    baseline = {row["id"]: row for row in previous["cases"]}
    cases = [{**originals[ident], "v73_contact_s": baseline[ident]["contact_s"],
              "expected_thin_route_bias": True,
              "stage_role": "previously_unpiloted_sibling"} for ident in SIBLINGS]
    cases.extend({**originals[REPEAT_SOURCE],
                  "id": f"{REPEAT_SOURCE}-repeat-{index}",
                  "v73_contact_s": baseline[REPEAT_SOURCE]["contact_s"],
                  "expected_thin_route_bias": True,
                  "stage_role": "exact_process_repeat"}
                 for index in range(1, REPEAT_COUNT + 1))
    if not all(baseline[ident]["pass"] for ident in SIBLINGS):
        raise ValueError("Sibling V73 baselines must be full-gate passes")
    pilot_plan = json.loads(v83.PLAN.read_text())
    plan = {
        "schema": "robot-learning-lab.cove50-v83-sibling-repeatability-plan.v1",
        "role": "postpilot_opened_sibling_and_repeatability_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": pilot_plan["source_sha256"],
        "runner_sha256": sha(Path(__file__)),
        "pilot_plan_sha256": sha(v83.PLAN),
        "pilot_matrix_sha256": sha(v83.DEST / "matrix.json"),
        "v73_plan_sha256": sha(v73.PLAN),
        "v73_matrix_sha256": sha(v73.DEST / "matrix.json"),
        "case_count": len(cases), "cases": cases,
        "gate": "Both V73 sibling routes that would activate the observed rule, plus three separate-process exact repeats of the earlier 0.70 s pitch-low miss, must each pass the full native gate, activate the expected observed region and hold at least 1.2 s continuous fingertip contact. All five outcomes count; failure stops before full retention. This is opened post-holdout stability evidence, not new-seed variation or qualification.",
        "design": "A source audit of V70, V73 and V74 reports found six V73 route visits inside the V83 rule regions, but the eight-route pilot contained only four. Test the remaining positive-07 roll-low and edge-11 pitch-high sibling routes before wider retention. Repeat the narrowest pilot success three times in separate native processes to expose contact variability without changing seed, toy, motor, gaze or route.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V83 sibling and repeat plan differs from current source")
    PLAN.write_bytes(encoded)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    DEST.mkdir(parents=True, exist_ok=True)
    v83.DEST = DEST
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(v83.run_case, case, plan): case["id"] for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            row["stage_role"] = next(case["stage_role"] for case in plan["cases"]
                                      if case["id"] == row["id"])
            rows[row["id"]] = row
            ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
            gate_met = len(ordered) == len(plan["cases"]) and all(
                item["pass"] and item["automatic_thin_route_bias_applied"] and
                item["margin_1p2_s_met"] for item in ordered)
            matrix = {
                "schema": "robot-learning-lab.cove50-v83-sibling-repeatability.v1",
                "role": plan["role"], "body_role": plan["body_role"],
                "plan_sha256": sha(PLAN), "complete": len(ordered) == len(plan["cases"]),
                "case_count": len(ordered), "pass_count": sum(item["pass"] for item in ordered),
                "margin_count": sum(item["margin_1p2_s_met"] for item in ordered),
                "gate_met": gate_met, "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("id", "stage_role", "pass", "contact_s", "automatic_thin_route_region", "faults")}),
                  flush=True)


if __name__ == "__main__":
    main()
