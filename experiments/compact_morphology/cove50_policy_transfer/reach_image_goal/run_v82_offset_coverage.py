"""Check the V79 offset direction on ten more opened route anchors."""

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
    run_v79_neighbor_offset_transfer as v79,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v82"
PLAN = OUT / "broad-offset-coverage-plan.json"
DEST = OUT / "native-broad-offsets"
V79_MATRIX = v79.DEST / "matrix.json"
ANCHORS = ("large-01-pitch-low", "large-19-roll-low",
           "large-23-pitch-low", "large-28-roll-low",
           "large-32-pitch-low", "edge-02-roll-high",
           "edge-08-pitch-high", "edge-10-pitch-low",
           "positive-11-roll-high", "positive-12-pitch-high")
BIASES_M = (-.004, 0.0)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    previous = json.loads(V79_MATRIX.read_text())
    source_matrix = json.loads(v73.DEST.joinpath("matrix.json").read_text())
    if (not previous["complete"] or previous["pass_count"] != 5 or
            not source_matrix["complete"] or source_matrix["pass_count"] != 27):
        raise ValueError("Need complete V73 holdout and V79 offset direction")
    originals = {case["id"]: case for case in json.loads(v73.PLAN.read_text())["cases"]}
    baseline = {row["id"]: row for row in source_matrix["cases"]}
    if any(not baseline[ident]["pass"] or
           baseline[ident]["automatic_front_route_bias_y_m"] or
           baseline[ident]["automatic_front_center_bias_applied"] or
           baseline[ident]["automatic_edge_route_bias_applied"]
           for ident in ANCHORS):
        raise ValueError("Coverage routes must be passing V73 visits without V70 route correction")
    cases = [{**originals[ident], "diagnostic_y_bias_m": bias,
              "baseline_v73_contact_s": baseline[ident]["contact_s"]}
             for ident in ANCHORS for bias in BIASES_M]
    plan = {
        "schema": "robot-learning-lab.cove50-v82-broad-offset-coverage-plan.v1",
        "role": "post_holdout_prescribed_offset_retention_diagnostic_not_controller",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {**json.loads(v79.PLAN.read_text())["source_sha256"],
                          "v79_runner": sha(Path(v79.__file__))},
        "runner_sha256": sha(Path(__file__)),
        "v73_plan_sha256": sha(v73.PLAN),
        "v73_matrix_sha256": sha(v73.DEST / "matrix.json"),
        "v79_matrix_sha256": sha(V79_MATRIX),
        "case_count": len(cases), "anchor_count": len(ANCHORS), "cases": cases,
        "gate": "Ten additional anchors spanning large, edge and positive entries each receive -4 and 0 mm experimenter-prescribed body-Y image-goal offsets. Each native visit must meet contact, support, rated-effort, floor, self-contact, clipping and RGB gates. A broad -4 mm candidate needs 10/10 passes and no case more than 0.5 s below its same-process zero-offset contact. This is post-holdout exploration, not learned inference or fresh scoring.",
        "design": "Use V73 passing anchors with no V70 route correction and distinct toy, voltage, delay, gaze and pitch/roll conditions. Pixels range from 10 to 113 in the V73 reports. Keep the V57 actor, camera pipeline, motors and physics fixed. Pair each -4 mm probe with a zero-offset repeat to expose contact variability.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V82 coverage plan differs from current source")
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
    v79.DEST = DEST
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(v79.run_case, case, plan):
                   (case["id"], case["diagnostic_y_bias_m"]) for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            original = next(case for case in plan["cases"] if
                            case["id"] == row["id"] and
                            case["diagnostic_y_bias_m"] == row["diagnostic_y_bias_m"])
            row["baseline_v73_contact_s"] = original["baseline_v73_contact_s"]
            rows[(row["id"], row["diagnostic_y_bias_m"])] = row
            ordered = [rows[(case["id"], case["diagnostic_y_bias_m"])]
                       for case in plan["cases"]
                       if (case["id"], case["diagnostic_y_bias_m"]) in rows]
            pairs = [(rows.get((ident, -.004)), rows.get((ident, 0.0))) for ident in ANCHORS]
            gate_met = (len(ordered) == len(plan["cases"]) and
                        all(negative["pass"] and zero["pass"] and
                            negative["contact_s"] >= zero["contact_s"] - .5
                            for negative, zero in pairs))
            matrix = {
                "schema": "robot-learning-lab.cove50-v82-broad-offset-coverage.v1",
                "role": plan["role"], "body_role": plan["body_role"],
                "plan_sha256": sha(PLAN), "complete": len(ordered) == len(plan["cases"]),
                "case_count": len(ordered), "pass_count": sum(row["pass"] for row in ordered),
                "broad_minus4_gate_met": gate_met,
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("id", "diagnostic_y_bias_m", "pass", "contact_s", "faults")}),
                  flush=True)


if __name__ == "__main__":
    main()
