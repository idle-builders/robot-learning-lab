"""Freeze and collect independent-seed RGB entry families for route training.

The three strata come from V150 automatic contact failures. Every family has a
new simulator seed and several correlated toy placements; these are camera-only
entries. Contact selection and any later validation need separate freezes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v98_entry_exposure as entry_runner,
    v118_observable_selector as observations,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v156"
PLAN = OUT / "independent-seed-entry-family-plan.json"
DEST = OUT / "native-entry-only"
MATRIX = DEST / "matrix.json"
V149_PLAN = BASE / "v149/new-seed-route-exposure-plan.json"
V149_MATRIX = BASE / "v149/entry-exposure-inspector-matrix.json"
V149_SUMMARY = BASE / "v149/entry-exposure-summary.json"
V150_MATRIX = BASE / "v150/native-paired-contact/matrix.json"
V150_SUMMARY = BASE / "v150/source-checked-paired-summary.json"
V155_PLAN = BASE / "v155/opened-live-entry-map-plan.json"
V147_RESERVE = BASE / "v147/untouched-score-reserve-plan.json"
V147_TRAIN = BASE / "v147/new-seed-direction-training-plan.json"
V139_SCORE = BASE / "v139/reserved-contact-score-plan.json"
POINTER = ROOT / "robot/ape-current-design.json"
STRATA = (("roll_live", 24591, 12),
          ("roll_locked", 24583, 6),
          ("pitch_locked", 24536, 6))
LIVE_GRID_M = tuple((x, y) for x in (-.001, 0.0, .001)
                    for y in (-.001, 0.0, .001))
LOCKED_CROSS_M = ((0.0, 0.0), (-.0005, 0.0), (.0005, 0.0),
                  (0.0, -.0005), (0.0, .0005))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict, *, frozen: bool = False) -> None:
    encoded = json.dumps(value, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if frozen and path.exists() and path.read_text() != encoded:
        raise ValueError(f"Frozen V156 plan changed: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    os.replace(temporary, path)


def cases(anchors: dict[int, dict]) -> tuple[list[dict], list[dict]]:
    rng = random.Random(156)
    families = []
    result = []
    seed = 24601
    for stratum, anchor_seed, family_count in STRATA:
        anchor = anchors[anchor_seed]
        for index in range(family_count):
            same = index == 0
            center_x = round(anchor["toy_offset_x_m"] +
                             (0 if same else rng.uniform(-.0002, .0002)), 6)
            center_y = round(anchor["toy_offset_y_m"] +
                             (0 if same else rng.uniform(-.0002, .0002)), 6)
            roll_high = stratum.startswith("roll_")
            family = {
                "id": f"family-{seed}", "seed": seed,
                "stratum": stratum, "opened_anchor_seed_audit_only": anchor_seed,
                "cohort": anchor["cohort"],
                "target_route_exposure": ("v16_live" if stratum == "roll_live"
                                          else "v16_locked"),
                "toy_center_x_m": center_x, "toy_center_y_m": center_y,
                "rail_v": (anchor["rail_v"] if same else
                           (10, 11, 12)[index % 3]),
                "delay_ms": (anchor["delay_ms"] if same else
                             (0, 5, 10)[(index // 2) % 3]),
                "gaze_head_yaw_rad": (anchor["gaze_head_yaw_rad"] if same else
                                      (-1.3, -1.5)[index % 2]),
                "pitch_offset_rad": (anchor["pitch_offset_rad"] if same else
                                     ((.105,) if roll_high else
                                      (.095, .1))[index % (1 if roll_high else 2)]),
                "roll_clearance_rad": (anchor["roll_clearance_rad"] if same else
                                       ((-.24, -.245) if roll_high else
                                        (-.25,))[(index // 3) % (2 if roll_high else 1)]),
            }
            families.append(family)
            grid = LIVE_GRID_M if stratum == "roll_live" else LOCKED_CROSS_M
            for place, (dx, dy) in enumerate(grid):
                result.append({
                    "id": f"entry-{seed}-{place:02d}",
                    "family_id": family["id"], "seed": seed,
                    "stratum": stratum, "cohort": family["cohort"],
                    "opened_anchor_seed_audit_only": anchor_seed,
                    "target_route_exposure": family["target_route_exposure"],
                    "toy_delta_x_m": dx, "toy_delta_y_m": dy,
                    "toy_offset_x_m": round(center_x + dx, 6),
                    "toy_offset_y_m": round(center_y + dy, 6),
                    "rail_v": family["rail_v"],
                    "delay_ms": family["delay_ms"],
                    "gaze_head_yaw_rad": family["gaze_head_yaw_rad"],
                    "pitch_offset_rad": family["pitch_offset_rad"],
                    "roll_clearance_rad": family["roll_clearance_rad"],
                })
            seed += 1
    return families, result


def freeze() -> dict:
    pointer = json.loads(POINTER.read_text())
    prior = json.loads(V149_PLAN.read_text())
    entry_matrix = json.loads(V149_MATRIX.read_text())
    entry_summary = json.loads(V149_SUMMARY.read_text())
    baseline = json.loads(V150_MATRIX.read_text())
    baseline_summary = json.loads(V150_SUMMARY.read_text())
    v155 = json.loads(V155_PLAN.read_text())
    reserve = json.loads(V147_RESERVE.read_text())
    entry_source = entry_runner.freeze(persist=False)["source_sha256"]
    source_keys = ("evaluator", "original_v95_evaluator", "body",
                   "pointer", "squat", "actor", "alternate_actor",
                   "specialist_actor", "reference", "geometry", "calibration")
    if (pointer["selected_design_id"] is not None or
            pointer["status"] != "rejected_no_selected_mechanical_design" or
            reserve["status"] != "sealed_no_contact_visits" or
            any(entry_source[key] != prior["source_sha256"][key]
                for key in source_keys) or
            entry_source["runner"] != prior["source_sha256"]["entry_runner"] or
            entry_matrix["plan_sha256"] != sha(V149_PLAN) or
            entry_matrix["case_count"] != 96 or
            entry_summary["inspector_matrix_sha256"] != sha(V149_MATRIX) or
            not entry_summary["complete"] or
            entry_summary["source_checked_entry_count"] != 96 or
            not baseline["complete"] or baseline["case_count"] != 48 or
            baseline_summary["matrix_sha256"] != sha(V150_MATRIX) or
            not baseline_summary["complete"] or
            v155["case_count"] != 25 or
            v155["correlated_anchor_seed_count"] != 1 or
            v155["contains_contact_visits"]):
        raise ValueError("V156 source, selected-body status or reserve changed")
    anchors = {case["seed"]: case for case in prior["cases"]}
    checked = {row["seed"]: row for row in entry_matrix["cases"]}
    base_rows = {row["seed"]: row for row in baseline["cases"]}
    for stratum, anchor_seed, _ in STRATA:
        if (not checked[anchor_seed]["source_checked"] or
                checked[anchor_seed]["selection_mode"] !=
                ("v16_live" if stratum == "roll_live" else "v16_locked") or
                base_rows[anchor_seed]["baseline_pass"] or
                base_rows[anchor_seed]["baseline_margin_1p2_s_met"]):
            raise ValueError(f"V156 opened anchor is not a saved baseline miss: {anchor_seed}")
    families, cases_list = cases(anchors)
    known_seeds = {case["seed"] for path in (V139_SCORE, V147_TRAIN,
                                             V147_RESERVE, V149_PLAN)
                   for case in json.loads(path.read_text())["cases"]}
    fresh_seeds = {family["seed"] for family in families}
    if (len(families) != 24 or len(cases_list) != 168 or
            len(fresh_seeds) != 24 or fresh_seeds & known_seeds or
            len({case["id"] for case in cases_list}) != len(cases_list)):
        raise ValueError("V156 new-seed families overlap or count changed")
    source = {**entry_source,
              "entry_runner": entry_source["runner"],
              "runner": sha(Path(__file__)),
              "v149_plan": sha(V149_PLAN),
              "v149_matrix": sha(V149_MATRIX),
              "v149_summary": sha(V149_SUMMARY),
              "v150_matrix": sha(V150_MATRIX),
              "v150_summary": sha(V150_SUMMARY),
              "v155_plan": sha(V155_PLAN),
              "untouched_score_reserve": sha(V147_RESERVE)}
    plan = {
        "schema": "robot-learning-lab.cove50-v156-seed-family-entry-plan.v1",
        "role": "new_seed_correlated_family_camera_only_training_search_not_contact_or_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": source,
        "family_count": len(families), "case_count": len(cases_list),
        "contains_contact_visits": False,
        "prior_visibility": (
            "Frozen after complete V150 baseline, opened V151 route diagnosis, "
            "and partial V152 prescribed outcomes through 32/48 were visible. "
            "The three physical anchor choices use only already source-checked "
            "V150 automatic baseline misses; V152 outcomes and V155 camera "
            "entries do not determine these cases. The independent V147 score "
            "reserve remains sealed."),
        "method": (
            "Use 24 genuinely new simulator seeds: 12 families near the opened "
            "roll-high live-route failure, six near the roll-high locked-route "
            "failure, and six near the pitch-low locked-route failure. Within "
            "each seed family hold motor, delay, gaze, pitch and roll fixed "
            "and vary toy X/Y on a nine-point ±1 mm grid for the live stratum "
            "or a five-point ±0.5 mm cross for locked strata. Add bounded "
            "independent center jitter and vary motor, delay, gaze, pitch and "
            "roll across families. Stop each run at the first used controlled "
            "RGB frame, record the actual route and camera features, and "
            "perform no contact. All entries sharing one simulator seed are "
            "one correlated family; any later contact choice is at most one "
            "predeclared entry per family. This entire plan is independent of "
            "V155's not-yet-run one-seed map result."),
        "families": families, "cases": cases_list,
    }
    write_json(PLAN, plan, frozen=True)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    row = entry_runner.run_case(case, plan)
    result = ROOT / row["result"]
    trace = result.with_suffix(".npz")
    report = json.loads(result.read_text())
    observed = observations.observed_features(report) if row["entry_available"] else None
    if row["entry_available"] and observed["selection_mode"] != row["selection_mode"]:
        raise ValueError(f"V156 observed route differs: {case['id']}")
    return {**row, "family_id": case["family_id"],
            "stratum": case["stratum"], "cohort": case["cohort"],
            "target_route_exposure": case["target_route_exposure"],
            "toy_delta_x_m": case["toy_delta_x_m"],
            "toy_delta_y_m": case["toy_delta_y_m"],
            "observed_features": observed,
            "result_sha256": sha(result), "trace_sha256": sha(trace)}


def write_matrix(plan: dict, rows: dict[str, dict]) -> dict:
    ordered = [rows[case["id"]] for case in plan["cases"]
               if case["id"] in rows]
    matrix = {
        "schema": "robot-learning-lab.cove50-v156-seed-family-entry-matrix.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN),
        "case_count": len(ordered), "planned_count": plan["case_count"],
        "family_count": len({row["family_id"] for row in ordered}),
        "planned_family_count": plan["family_count"],
        "complete": len(ordered) == plan["case_count"],
        "entry_available_count": sum(row["entry_available"] for row in ordered),
        "observed_route_counts": dict(sorted(Counter(
            row["selection_mode"] for row in ordered).items())),
        "contains_contact_outcomes": False,
        "cases": ordered,
    }
    write_json(MATRIX, matrix)
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one to four CUDA entry evaluators")
    plan = freeze()
    if args.freeze_only:
        print(f"V156 froze {plan['family_count']} independent seed families, "
              f"{plan['case_count']} camera entries; sha256={sha(PLAN)}")
        return
    refs = plan["source_sha256"]
    if (refs["runner"] != sha(Path(__file__)) or
            refs["entry_runner"] != sha(Path(entry_runner.__file__)) or
            refs["v149_matrix"] != sha(V149_MATRIX) or
            refs["v155_plan"] != sha(V155_PLAN) or
            refs["untouched_score_reserve"] != sha(V147_RESERVE)):
        raise ValueError("Frozen V156 source changed")
    DEST.mkdir(parents=True, exist_ok=True)
    old = json.loads(MATRIX.read_text()) if MATRIX.exists() else None
    if old and old["plan_sha256"] != sha(PLAN):
        raise ValueError("V156 existing matrix belongs to another plan")
    rows = {row["id"]: row for row in old["cases"]} if old else {}
    entry_runner.DEST = DEST
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan): case
                   for case in plan["cases"] if case["id"] not in rows}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            matrix = write_matrix(plan, rows)
            print(f"V156 {row['id']}: {row['selection_mode']} "
                  f"{matrix['case_count']}/{plan['case_count']} entries", flush=True)


if __name__ == "__main__":
    main()
