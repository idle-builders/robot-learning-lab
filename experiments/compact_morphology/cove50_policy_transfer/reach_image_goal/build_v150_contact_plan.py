"""Freeze varied paired contact training from V149 observed RGB entries only."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
ENTRY_PLAN = BASE / "v149/new-seed-route-exposure-plan.json"
ENTRY_MATRIX = BASE / "v149/entry-exposure-inspector-matrix.json"
ENTRY_SUMMARY = BASE / "v149/entry-exposure-summary.json"
METHOD = BASE / "v150/selection-method-freeze.json"
OUT = BASE / "v150/paired-contact-training-plan.json"
V147 = BASE / "v147/new-seed-direction-training-plan.json"
RESERVE = BASE / "v147/untouched-score-reserve-plan.json"
POINTER = ROOT / "robot/ape-current-design.json"

COHORTS = {
    "opened_roll_high_neighborhood": (24501, "v21_scoped_y_locked"),
    "opened_pitch_low_neighborhood": (24502, "v16_specialist_locked"),
}
PHYSICAL = ("toy_offset_x_m", "toy_offset_y_m", "rail_v", "delay_ms",
            "gaze_head_yaw_rad", "pitch_offset_rad", "roll_clearance_rad")
OBSERVED = ("entry_error_x_mm", "entry_error_y_mm", "camera_x_mm",
            "camera_y_mm", "orange_pixels")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coordinates(rows: list[dict]) -> dict[int, tuple[float, ...]]:
    names = PHYSICAL + OBSERVED
    raw = {row["seed"]: tuple(float(row[key] if key in PHYSICAL else
                                     row["observed_features"][key])
                              for key in names) for row in rows}
    limits = [(min(vector[i] for vector in raw.values()),
               max(vector[i] for vector in raw.values()))
              for i in range(len(names))]
    return {seed: tuple((value - low) / (high - low) if high > low else 0.0
                        for value, (low, high) in zip(vector, limits))
            for seed, vector in raw.items()}


def choose_diverse(pool: list[dict], chosen: list[dict], count: int,
                   vectors: dict[int, tuple[float, ...]]) -> None:
    remaining = {row["seed"]: row for row in pool if row not in chosen}
    while count and remaining:
        def distance(seed: int) -> float:
            return min((math.dist(vectors[seed], vectors[row["seed"]])
                        for row in chosen), default=float("inf"))

        seed = max(remaining, key=lambda candidate: (distance(candidate),
                                                      -candidate))
        chosen.append(remaining.pop(seed))
        count -= 1
    if count:
        raise ValueError("Requested more entries than the route pool contains")


def main() -> None:
    entry_plan = json.loads(ENTRY_PLAN.read_text())
    matrix = json.loads(ENTRY_MATRIX.read_text())
    summary = json.loads(ENTRY_SUMMARY.read_text())
    method = json.loads(METHOD.read_text())
    v147 = json.loads(V147.read_text())
    reserve = json.loads(RESERVE.read_text())
    runner = Path(__file__).with_name("run_v150_paired_contact_training.py")
    if (method["selection_source_sha256"] != sha(Path(__file__)) or
            method["runner_source_sha256"] != sha(runner) or
            method["v149_plan_sha256"] != sha(ENTRY_PLAN) or
            method["score_reserve_sha256"] != sha(RESERVE) or
            summary["source_checked_entry_count"] != 96 or
            summary["plan_sha256"] != sha(ENTRY_PLAN) or
            summary["inspector_matrix_sha256"] != sha(ENTRY_MATRIX) or
            matrix["plan_sha256"] != sha(ENTRY_PLAN) or
            matrix["source_matrix_sha256"] != summary["matrix_sha256"] or
            len(matrix["cases"]) != 96 or
            reserve["status"] != "sealed_no_contact_visits" or
            v147["source_sha256"]["untouched_score_reserve"] != sha(RESERVE) or
            entry_plan["source_sha256"]["pointer"] != sha(POINTER) or
            json.loads(POINTER.read_text())["selected_design_id"] is not None):
        raise ValueError("Complete source-checked entries and sealed reserve required")
    by_seed = {row["seed"]: row for row in matrix["cases"]}
    if (len(by_seed) != 96 or
            [row["seed"] for row in matrix["cases"]] !=
            [row["seed"] for row in entry_plan["cases"]] or
            any(not row["source_checked"] or
                (row["entry_available"] and
                 (row["observed_features"] is None or
                  row["observed_features"]["selection_mode"] !=
                  row["selection_mode"])) or
                (not row["entry_available"] and
                 row["observed_features"] is not None)
                for row in matrix["cases"])):
        raise ValueError("V149 entry coverage or observed route differs")
    selected = []
    for cohort, (anchor_seed, target_route) in COHORTS.items():
        all_rows = [row for row in matrix["cases"] if row["cohort"] == cohort]
        rows = [row for row in all_rows if row["entry_available"]]
        if len(all_rows) != 48 or len(rows) < 24 or not by_seed[anchor_seed]["entry_available"]:
            raise ValueError(f"Expected 48 checked entries in {cohort}")
        target = [row for row in rows if row["selection_mode"] == target_route]
        if len(target) < 4:
            raise ValueError(f"Failure route too rare for paired training: {cohort}")
        vectors = coordinates(rows)
        chosen = [by_seed[anchor_seed]]
        target_count = min(12, len(target))
        choose_diverse(target, chosen,
                       target_count - int(chosen[0]["selection_mode"] == target_route),
                       vectors)
        contrast = [row for row in rows if row["selection_mode"] != target_route]
        choose_diverse(contrast, chosen,
                       min(24 - len(chosen), len(contrast) -
                           sum(row["selection_mode"] != target_route for row in chosen)),
                       vectors)
        choose_diverse(rows, chosen, 24 - len(chosen), vectors)
        if len(chosen) != 24 or len({row["seed"] for row in chosen}) != 24:
            raise ValueError(f"Invalid paired training selection: {cohort}")
        selected.extend(chosen)
    if ({row["seed"] for row in selected} &
            {row["seed"] for row in reserve["cases"]} or
            len({row["seed"] for row in selected}) != 48):
        raise ValueError("Training selection overlaps sealed score reserve")
    plan_cases = []
    for row in selected:
        source = next(case for case in entry_plan["cases"]
                      if case["seed"] == row["seed"])
        plan_cases.append({**source, "id": f"paired-training-{row['seed']}",
                           "entry_selection_mode": row["selection_mode"],
                           "entry_observed_features": row["observed_features"],
                           "entry_report": row["result"],
                           "entry_report_sha256": row["result_sha256"],
                           "entry_trace_sha256": row["trace_sha256"]})
    source = {**v147["source_sha256"],
              "v149_entry_plan": sha(ENTRY_PLAN),
              "v149_entry_matrix": sha(ENTRY_MATRIX),
              "v149_entry_summary": sha(ENTRY_SUMMARY),
              "v150_selection_method": sha(METHOD),
              "v150_selection_builder": sha(Path(__file__)),
              "v150_runner": sha(runner)}
    plan = {
        "schema": "robot-learning-lab.cove50-v150-paired-contact-training-plan.v1",
        "role": "observed_entry_selected_paired_training_not_independent_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": source,
        "case_count": 48, "visit_count": 144,
        "target_shifts_body_y_m": [0, -.002, .002],
        "selection_method": method["method"],
        "selection_visibility": method["prior_visibility"],
        "cases": plan_cases,
    }
    encoded = json.dumps(plan, indent=2) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and OUT.read_text() != encoded:
        raise ValueError("Frozen V150 paired contact plan changed")
    OUT.write_text(encoded)
    print(f"V150 froze {len(plan_cases)} paired starts, 144 native contact visits; "
          f"plan sha256={sha(OUT)}")


if __name__ == "__main__":
    main()
