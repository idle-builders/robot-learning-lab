"""Score V83 on new-seed route variants across all V73 anchors."""

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
    run_v83_thin_route_retention as retention,
    run_v83_prior75_retention as prior75,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v83"
PLAN = OUT / "fresh-neighbor-challenge-plan.json"
DEST = OUT / "native-fresh-neighbors"

# The three thin-contact anchors contribute both route variants twice. Eleven
# other anchors supply no-activation controls, so every V73 anchor is covered.
# Toy shifts are body-frame metres relative to the V73 base values.
VARIATIONS = (
    ("positive-01-pitch-low", 12001, 10, 0, -1.3, .00025, .00025, True),
    ("positive-01-roll-high", 12002, 12, 10, -1.3, -.00025, .00050, True),
    ("positive-01-pitch-low", 12003, 11, 5, -1.5, .00050, 0.0, True),
    ("positive-01-roll-high", 12004, 10, 10, -1.3, -.00050, -.00025, True),
    ("positive-07-pitch-high", 12005, 11, 0, -1.5, .00025, .00025, True),
    ("positive-07-roll-low", 12006, 12, 10, -1.5, -.00025, .00050, True),
    ("positive-07-pitch-high", 12007, 10, 5, -1.3, .00050, 0.0, True),
    ("positive-07-roll-low", 12008, 11, 10, -1.5, -.00050, -.00025, True),
    ("edge-11-pitch-high", 12009, 10, 0, -1.3, .00025, .00025, True),
    ("edge-11-roll-low", 12010, 12, 10, -1.3, -.00025, .00050, True),
    ("edge-11-pitch-high", 12011, 11, 5, -1.5, .00050, 0.0, True),
    ("edge-11-roll-low", 12012, 10, 10, -1.3, -.00050, -.00025, True),
    ("large-32-pitch-low", 12013, 10, 0, -1.3, .00025, .00025, False),
    ("large-32-roll-high", 12014, 12, 10, -1.5, -.00025, .00050, False),
    ("edge-02-pitch-low", 12015, 11, 5, -1.3, .00050, 0.0, False),
    ("edge-02-roll-high", 12016, 10, 10, -1.5, -.00050, -.00025, False),
    ("positive-11-pitch-low", 12017, 12, 0, -1.3, .00025, .00025, False),
    ("positive-11-roll-high", 12018, 11, 10, -1.5, -.00025, -.00050, False),
    ("large-01-roll-high", 12019, 11, 5, -1.3, .00075, -.00050, False),
    ("large-19-pitch-high", 12020, 12, 10, -1.5, -.00075, .00100, False),
    ("large-23-roll-high", 12021, 10, 0, -1.3, .00100, .00050, False),
    ("large-28-pitch-high", 12022, 11, 10, -1.5, -.00100, -.00075, False),
    ("large-36-roll-low", 12023, 12, 5, -1.3, .00050, .00100, False),
    ("edge-08-pitch-high", 12024, 10, 10, -1.5, -.00050, -.00100, False),
    ("edge-10-roll-high", 12025, 11, 0, -1.3, .00100, -.00050, False),
    ("positive-12-pitch-high", 12026, 12, 10, -1.5, -.00100, .00075, False),
)

# Bounded changes to the already varied V73 commanded route. They must still
# clear the V83 off-author-route entry condition; failed activation counts.
ROUTE_JITTER = {
    12003: (-.003, 0.0), 12004: (0.0, .003),
    12007: (.003, 0.0), 12008: (0.0, -.003),
    12011: (-.003, 0.0), 12012: (0.0, .003),
    12020: (.002, 0.0), 12022: (-.002, 0.0),
    12024: (.002, 0.0), 12026: (-.002, 0.0),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    prior = json.loads((retention.DEST / "matrix.json").read_text())
    if (not prior["complete"] or not prior["screen_gate_met"] or
            prior["pass_count"] != 63 or prior["thin_rule_use_count"] != 6):
        raise ValueError("Need complete passing V83 63-route opened retention")
    older = json.loads((prior75.DEST / "matrix.json").read_text())
    if (not older["complete"] or not older["screen_gate_met"] or
            older["case_count"] != 75 or older["pass_count"] != 75 or
            older["thin_rule_use_count"]):
        raise ValueError("Need complete passing V83 75-start older retention")
    pilot_plan = json.loads(v83.PLAN.read_text())
    sources = pilot_plan["source_sha256"]
    if (sources["evaluator"] != sha(v83.EVALUATOR) or
            sources["runner"] != sha(Path(v83.__file__))):
        raise ValueError("V83 policy source changed after pilot freeze")
    originals = {case["id"]: case for case in json.loads(v73.PLAN.read_text())["cases"]}
    baseline = {row["id"]: row for row in
                json.loads((v73.DEST / "matrix.json").read_text())["cases"]}
    cases = []
    for base_id, seed, rail_v, delay_ms, gaze, shift_x, shift_y, expected in VARIATIONS:
        original = originals[base_id]
        pitch_jitter, roll_jitter = ROUTE_JITTER.get(seed, (0.0, 0.0))
        cases.append({**original,
                      "id": f"newseed-{seed}-{base_id}",
                      "seed": seed, "rail_v": rail_v, "delay_ms": delay_ms,
                      "gaze_head_yaw_rad": gaze,
                      "toy_offset_x_m": round(original["toy_offset_x_m"] + shift_x, 6),
                      "toy_offset_y_m": round(original["toy_offset_y_m"] + shift_y, 6),
                      "pitch_offset_rad": round(original["pitch_offset_rad"] + pitch_jitter, 6),
                      "roll_clearance_rad": round(original["roll_clearance_rad"] + roll_jitter, 6),
                      "v73_contact_s": baseline[base_id]["contact_s"],
                      "base_route_id": base_id,
                      "expected_thin_route_bias": expected})
    historical_seeds = {row["seed"] for row in
                        json.loads((retention.DEST / "matrix.json").read_text())["cases"]}
    anchors = {case["base_route_id"].rsplit("-", 2)[0] for case in cases}
    if (len(cases) != 26 or len({case["id"] for case in cases}) != 26 or
            len({case["seed"] for case in cases}) != 26 or
            any(case["seed"] in historical_seeds for case in cases) or
            sum(case["expected_thin_route_bias"] for case in cases) != 12 or
            len(anchors) != 14):
        raise ValueError("Fresh challenge must cover fourteen anchors with 26 new seeds and twelve target views")
    plan = {
        "schema": "robot-learning-lab.cove50-v83-fresh-neighbor-challenge-plan.v1",
        "role": "new_seed_varied_route_challenge_not_new_body_qualification",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "runner_sha256": sha(Path(__file__)),
        "retention_plan_sha256": sha(retention.PLAN),
        "retention_matrix_sha256": sha(retention.DEST / "matrix.json"),
        "prior75_plan_sha256": sha(prior75.PLAN),
        "prior75_matrix_sha256": sha(prior75.DEST / "matrix.json"),
        "v73_plan_sha256": sha(v73.PLAN),
        "v73_matrix_sha256": sha(v73.DEST / "matrix.json"),
        "case_count": len(cases), "target_count": 12, "control_count": 14,
        "cases": cases,
        "gate": "All 26 new-seed native starts must pass contact, support, rated effort, hand-floor, self-contact, clipping and scored RGB. All twelve target views must activate the V83 observed rule and hold at least 1.2 s continuous fingertip contact; all fourteen control views must keep the new rule off. Any missing RGB entry or branch activation counts as failure. This is varied scoring around previously opened anchors on a rejected body, not a new-anchor or selected-body qualification.",
        "design": "Require passing 63-route V70/V73 retention and 75-start earlier V41 retention before freezing 26 unused seeds. The three thin-contact anchors contribute both prior route variants at two new-seed conditions each; eleven other anchors provide fourteen no-activation controls, covering all fourteen V73 anchors. Perturb toy X/Y by at most 0.5 mm on targets and 1 mm on controls, pitch or roll by up to 0.003 rad on selected routes, rail voltage over 10–12 V, control delay over 0–10 ms, and gaze over the two supported angles. The actor, image model, body and automatic rule remain exactly frozen from V83 opened retention.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V83 fresh neighbor plan differs from current source")
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
        iterator = iter(plan["cases"])
        pending = {}
        for _ in range(args.workers):
            case = next(iterator, None)
            if case is not None:
                pending[pool.submit(v83.run_case, case, plan)] = case
        while pending:
            future = next(as_completed(tuple(pending)))
            original = pending.pop(future)
            row = future.result()
            row["base_route_id"] = original["base_route_id"]
            rows[row["id"]] = row
            ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
            target = [item for item in ordered if item["expected_thin_route_bias"]]
            controls = [item for item in ordered if not item["expected_thin_route_bias"]]
            gate_met = (len(ordered) == len(plan["cases"]) and
                        all(item["pass"] for item in ordered) and
                        len(target) == plan["target_count"] and
                        len(controls) == plan["control_count"] and
                        all(item["automatic_thin_route_bias_applied"] and
                            item["margin_1p2_s_met"] for item in target) and
                        all(not item["automatic_thin_route_bias_applied"] for item in controls))
            matrix = {
                "schema": "robot-learning-lab.cove50-v83-fresh-neighbor-challenge.v1",
                "role": plan["role"], "body_role": plan["body_role"],
                "plan_sha256": sha(PLAN), "complete": len(ordered) == len(plan["cases"]),
                "case_count": len(ordered), "pass_count": sum(item["pass"] for item in ordered),
                "target_rule_use_count": sum(item["automatic_thin_route_bias_applied"]
                                             for item in target),
                "control_rule_use_count": sum(item["automatic_thin_route_bias_applied"]
                                              for item in controls),
                "score_gate_met": gate_met, "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("id", "pass", "contact_s", "automatic_thin_route_region", "faults")}),
                  flush=True)
            next_case = next(iterator, None)
            if next_case is not None:
                pending[pool.submit(v83.run_case, next_case, plan)] = next_case


if __name__ == "__main__":
    main()
