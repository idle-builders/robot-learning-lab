"""Probe the observable rare-route boundary around one opened RGB entry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v98_entry_exposure as v98,
    run_v100_opened_entry_probe as v100,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = v98.BASE
OUT = BASE / "v121"
PLAN = OUT / "route-neighborhood-plan.json"
DEST = OUT / "native-route-neighborhood-entry"
MATRIX = DEST / "matrix.json"
V105_PLAN = BASE / "v105/varied-entry-exposure-plan.json"
V105_MATRIX = BASE / "v105/native-varied-entry-exposure/matrix.json"
SEED = 19026
DELTAS_M = (-.0004, 0.0, .0004)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("V121 is scoped to the rejected historical sandbox")
    source_plan = json.loads(V105_PLAN.read_text())
    source_matrix = json.loads(V105_MATRIX.read_text())
    source_case = next(case for case in source_plan["cases"] if case["seed"] == SEED)
    source_row = next(row for row in source_matrix["cases"] if row["seed"] == SEED)
    if not source_row["entry_available"] or source_row["new_specialist_applied"]:
        raise ValueError("V121 source entry is not the opened route miss")
    source_report = ROOT / source_row["result"]
    cases = []
    for ix, dx in enumerate(DELTAS_M):
        for iy, dy in enumerate(DELTAS_M):
            cases.append({**source_case,
                          "id": f"opened-{SEED}-toy-x{ix}-y{iy}",
                          "toy_offset_x_m": round(source_case["toy_offset_x_m"] + dx, 7),
                          "toy_offset_y_m": round(source_case["toy_offset_y_m"] + dy, 7),
                          "toy_delta_x_m": dx, "toy_delta_y_m": dy})
    v98_sources = v98.freeze(persist=False)["source_sha256"]
    plan = {
        "schema": "robot-learning-lab.cove50-v121-route-neighborhood-plan.v1",
        "role": "opened_rgb_entry_only_rare_route_coverage_diagnostic",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {**v98_sources,
                          "v100_predicate_runner": sha(Path(v100.__file__)),
                          "v105_plan": sha(V105_PLAN),
                          "v105_matrix": sha(V105_MATRIX),
                          "v105_entry_report": sha(source_report),
                          "runner": sha(Path(__file__))},
        "source_entry_result": source_row["result"],
        "source_entry_error_xy_m": json.loads(source_report.read_text())["selection_features"]["entry_error_xy_m"],
        "case_count": len(cases), "cases": cases,
        "hypothesis": ("On opened seed 19026 the camera region and 94 toy pixels already "
                       "satisfy the rare-route predicates, while observed image-goal "
                       "X=2.304 mm and Y=5.536 mm sit near the 1.5–2.0 and "
                       "4.5–5.5 mm windows. A frozen 3×3 toy X/Y grid at ±0.4 mm "
                       "tests whether those observable conditions can coincide."),
        "decision_gate": ("Run only through the first used-for-control RGB frame. "
                          "If at least one visit activates the new route, retain its "
                          "entry report and freeze a separate full-native contact "
                          "comparison before any router training. If none activates, "
                          "measure which observed predicates remain disjoint and do "
                          "not count this as contact or held-out evidence."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V121 plan or source changed")
    PLAN.write_bytes(encoded)
    return plan


def write_matrix(plan: dict, rows: dict[str, dict]) -> dict:
    ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
    result = {"schema": "robot-learning-lab.cove50-v121-route-neighborhood.v1",
              "role": plan["role"], "body_role": plan["body_role"],
              "plan_sha256": sha(PLAN), "complete": len(ordered) == plan["case_count"],
              "case_count": len(ordered),
              "new_specialist_count": sum(row["new_specialist_applied"] for row in ordered),
              "all_predicates_met_count": sum(row["all_predicates_met"] for row in ordered),
              "cases": ordered}
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
    v98.DEST = DEST
    rows = ({row["id"]: row for row in json.loads(MATRIX.read_text())["cases"]}
            if MATRIX.exists() else {})
    for case in plan["cases"]:
        if case["id"] in rows:
            continue
        row = v100.predicate_row(v98.run_case(case, plan))
        row.update({key: case[key] for key in
                    ("toy_offset_x_m", "toy_offset_y_m", "toy_delta_x_m", "toy_delta_y_m")})
        rows[case["id"]] = row
        write_matrix(plan, rows)
        print(json.dumps({key: row[key] for key in
                          ("id", "entry_error_xy_mm", "camera_xy_mm", "orange_pixels",
                           "route_blockers", "new_specialist_applied")}), flush=True)
    result = write_matrix(plan, rows)
    print(f"V121 opened entry route coverage: {result['case_count']}/{plan['case_count']}; "
          f"new route {result['new_specialist_count']}", flush=True)


if __name__ == "__main__":
    main()
