"""Compare automatic and prescribed specialist contact on varied opened entries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v95_native_case as native,
    run_v104_opened_specialist_pairs as v104,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v122"
PLAN = OUT / "near-route-pairs-plan.json"
DEST = OUT / "native-near-route-pairs"
MATRIX = DEST / "matrix.json"
FORCED_EVALUATOR = Path(__file__).with_name("evaluate_forced_specialist_v102.py")
V95_PILOT = BASE / "v95/two-region-pilot-plan.json"
V121_PLAN = BASE / "v121/route-neighborhood-plan.json"
V121_MATRIX = BASE / "v121/native-route-neighborhood-entry/matrix.json"
V105_PLAN = BASE / "v105/varied-entry-exposure-plan.json"
V105_MATRIX = BASE / "v105/native-varied-entry-exposure/matrix.json"
SELECTED = (
    ("near-image-x", "v121", "opened-19026-toy-x1-y0"),
    ("near-image-y", "v121", "opened-19026-toy-x0-y1"),
    ("near-both", "v121", "opened-19026-toy-x0-y0"),
    ("roll-high-control", "v105", "entry-19003"),
    ("pitch-low-control", "v105", "entry-19005"),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    pointer = ROOT / "robot/ape-current-design.json"
    if json.loads(pointer.read_text())["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("V122 is scoped to the rejected historical sandbox")
    baseline = json.loads(V95_PILOT.read_text())["source_sha256"]
    if baseline["evaluator"] != sha(native.EVALUATOR) or baseline["pointer"] != sha(pointer):
        raise ValueError("V95 source or rejected-body pointer changed")
    v121_plan = json.loads(V121_PLAN.read_text())
    v121 = json.loads(V121_MATRIX.read_text())
    v105_plan = json.loads(V105_PLAN.read_text())
    v105 = json.loads(V105_MATRIX.read_text())
    if (not v121["complete"] or v121["case_count"] != 9 or
            v121["plan_sha256"] != sha(V121_PLAN) or
            v121["new_specialist_count"] != 0 or
            not v105["complete"] or v105["case_count"] != 64):
        raise ValueError("Opened entry evidence is incomplete or changed")
    sources = {
        "v121": (v121_plan, v121), "v105": (v105_plan, v105),
    }
    cases = []
    opened_reports = {}
    for label, cohort, ident in SELECTED:
        source_plan, source_matrix = sources[cohort]
        source_case = next(case for case in source_plan["cases"] if case["id"] == ident)
        source_row = next(row for row in source_matrix["cases"] if row["id"] == ident)
        if not source_row["entry_available"] or source_row["new_specialist_applied"]:
            raise ValueError(f"Missing unopened-route RGB entry: {ident}")
        opened_reports[label] = {"result": source_row["result"],
                                 "sha256": sha(ROOT / source_row["result"])}
        for forced in (False, True):
            cases.append({**source_case, "id": f"{label}-{'forced' if forced else 'automatic'}",
                          "comparison_label": label, "opened_cohort": cohort,
                          "diagnostic_force_specialist": forced,
                          "route_variant": source_case.get("route_variant", "center")})
    plan = {
        "schema": "robot-learning-lab.cove50-v122-near-route-pairs-plan.v1",
        "role": "opened_full_native_matched_route_comparison_not_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {
            "baseline": baseline,
            "forced": {**baseline, "evaluator": sha(FORCED_EVALUATOR)},
            "runner": sha(Path(__file__)),
            "native_runner": sha(Path(native.__file__)),
            "entry_comparison": sha(Path(v104.__file__)),
            "v95_pilot_plan": sha(V95_PILOT),
            "v121_plan": sha(V121_PLAN), "v121_matrix": sha(V121_MATRIX),
            "v105_plan": sha(V105_PLAN), "v105_matrix": sha(V105_MATRIX),
        },
        "opened_source_reports": opened_reports,
        "entry_tolerances": {"rounded_entry_error_xy_m": .000002,
                             "rounded_camera_world_xyz_m": .000015,
                             "estimated_goal_world_xyz_m": .000001,
                             "native_entry_qpos_max_abs": .000005},
        "pair_count": len(SELECTED), "case_count": len(cases), "cases": cases,
        "hypothesis": ("The rare specialist branch is absent in V105 and the opened "
                       "V121 3×3 neighborhood. Compare its actual arm actor with "
                       "the unchanged automatic route on three near-boundary toy "
                       "positions and two V105 starts with different route and motor "
                       "conditions. Each pair has the same native RGB entry; only "
                       "the post-entry arm route changes."),
        "decision_gate": ("Consider collecting broader observed-route training data "
                          "only if at least two of three near-boundary forced visits "
                          "gain at least 0.5 s continuous contact, no forced visit "
                          "loses a full gate or 1.2 s margin, and all five paired "
                          "entries match. Otherwise retain the automatic route and "
                          "seek a different controller or physical contact teacher. "
                          "This opened diagnostic cannot promote a policy."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V122 plan or sources changed")
    PLAN.write_bytes(encoded)
    return plan


def source_entry_features(report_path: str) -> dict:
    report = json.loads((ROOT / report_path).read_text())
    entry = next(event for event in report["image_goal_capture_events"]
                 if event["used_for_control"])
    return {"entry_error_xy_m": report["selection_features"]["entry_error_xy_m"],
            "camera_world_xyz_m": entry["camera_world_xyz_m"],
            "orange_pixels": entry["orange_pixels"]}


def close_entry(first: dict, second: dict, tolerances: dict) -> bool:
    return (first["orange_pixels"] == second["orange_pixels"] and
            all(abs(a-b) <= tolerances[{
                "entry_error_xy_m": "rounded_entry_error_xy_m",
                "camera_world_xyz_m": "rounded_camera_world_xyz_m",
                "goal_world_xyz_m": "estimated_goal_world_xyz_m"}[key]]
                for key in ("entry_error_xy_m", "camera_world_xyz_m", "goal_world_xyz_m")
                for a, b in zip(first[key], second[key])))


def write_matrix(plan: dict, rows: dict[str, dict]) -> dict:
    ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
    pairs = []
    for label, _, _ in SELECTED:
        base = rows.get(f"{label}-automatic")
        forced = rows.get(f"{label}-forced")
        if base is None or forced is None:
            continue
        first, second = v104.entry_features(base), v104.entry_features(forced)
        qpos_a = v104.native_entry_qpos(base, first["entry_time_s"])
        qpos_b = v104.native_entry_qpos(forced, second["entry_time_s"])
        if qpos_a.shape != qpos_b.shape:
            raise ValueError(f"V122 native state shape differs for {label}")
        qpos_gap = float(np.max(np.abs(qpos_a - qpos_b)))
        source = source_entry_features(plan["opened_source_reports"][label]["result"])
        tol = plan["entry_tolerances"]
        same_pair = (first["entry_time_s"] == second["entry_time_s"] and
                     qpos_gap <= tol["native_entry_qpos_max_abs"] and
                     close_entry(first, second, tol))
        same_preflight = (first["orange_pixels"] == source["orange_pixels"] and
                          all(abs(a-b) <= tol[{
                              "entry_error_xy_m": "rounded_entry_error_xy_m",
                              "camera_world_xyz_m": "rounded_camera_world_xyz_m"}[key]]
                              for key in ("entry_error_xy_m", "camera_world_xyz_m")
                              for a, b in zip(first[key], source[key])))
        if not same_pair or not same_preflight:
            raise ValueError(f"V122 paired native RGB entry differs: {label}")
        pairs.append({"id": label, "seed": base["seed"],
                      "opened_cohort": next(case["opened_cohort"] for case in plan["cases"]
                                            if case["comparison_label"] == label),
                      "native_entry_qpos_max_abs": qpos_gap,
                      "entry_matched": True,
                      "automatic_contact_s": base["contact_s"],
                      "forced_contact_s": forced["contact_s"],
                      "contact_delta_s": round(forced["contact_s"]-base["contact_s"], 3),
                      "automatic_pass": base["pass"], "forced_pass": forced["pass"],
                      "automatic_margin_1p2_s_met": base["margin_1p2_s_met"],
                      "forced_margin_1p2_s_met": forced["margin_1p2_s_met"],
                      "automatic_result": base["result"], "forced_result": forced["result"],
                      "automatic_faults": base["faults"], "forced_faults": forced["faults"]})
    near = [pair for pair in pairs if pair["id"].startswith("near-")]
    result = {"schema": "robot-learning-lab.cove50-v122-near-route-pairs.v1",
              "role": plan["role"], "body_role": plan["body_role"],
              "plan_sha256": sha(PLAN), "complete": len(ordered) == plan["case_count"],
              "case_count": len(ordered), "pair_count": len(pairs),
              "paired_entry_verified": len(pairs) == plan["pair_count"],
              "forced_full_pass_count": sum(row["pass"] for row in ordered
                                            if row["id"].endswith("-forced")),
              "near_gain_half_second_count": sum(pair["contact_delta_s"] >= .5
                                                 for pair in near),
              "full_gate_regression_count": sum(pair["automatic_pass"] and
                                                 not pair["forced_pass"] for pair in pairs),
              "margin_regression_count": sum(pair["automatic_margin_1p2_s_met"] and
                                             not pair["forced_margin_1p2_s_met"]
                                             for pair in pairs),
              "decision_gate_met": (len(pairs) == plan["pair_count"] and
                                    sum(pair["contact_delta_s"] >= .5
                                        for pair in near) >= 2 and
                                    not any(pair["automatic_pass"] and
                                            not pair["forced_pass"] or
                                            pair["automatic_margin_1p2_s_met"] and
                                            not pair["forced_margin_1p2_s_met"]
                                            for pair in pairs)),
              "pairs": pairs, "cases": ordered}
    DEST.mkdir(parents=True, exist_ok=True)
    temporary = MATRIX.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, MATRIX)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    rows = ({row["id"]: row for row in json.loads(MATRIX.read_text())["cases"]}
            if MATRIX.exists() else {})
    original = native.EVALUATOR
    for case in plan["cases"]:
        if case["id"] in rows:
            continue
        forced = case["diagnostic_force_specialist"]
        native.EVALUATOR = FORCED_EVALUATOR if forced else original
        try:
            row = native.run_case(case, {"source_sha256": plan["source_sha256"][
                "forced" if forced else "baseline"]}, DEST)
        finally:
            native.EVALUATOR = original
        report = json.loads((ROOT / row["result"]).read_text())
        if bool(report.get("diagnostic_force_specialist", False)) != forced:
            raise ValueError(f"V122 forced-route identity mismatch: {case['id']}")
        row["diagnostic_force_specialist"] = forced
        rows[case["id"]] = row
        result = write_matrix(plan, rows)
        print(f"V122 {case['id']}: {row['contact_s']:.2f} s, "
              f"{'pass' if row['pass'] else 'fail'}; {result['pair_count']} pairs", flush=True)
    result = write_matrix(plan, rows)
    print(f"V122 paired route comparison: {result['pair_count']}/{plan['pair_count']} "
          f"pairs, gate {'met' if result['decision_gate_met'] else 'failed'}", flush=True)


if __name__ == "__main__":
    main()
