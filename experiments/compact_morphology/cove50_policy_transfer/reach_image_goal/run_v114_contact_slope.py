"""Opened full-native target sweep on two V113 weak-contact starts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v108_native_case as shifted,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
V113_PLAN = BASE / "v113/varied-paired-contact-data-plan.json"
V113_DATA = BASE / "v113/native-varied-paired-contact-data"
OUT = BASE / "v114"
PLAN = OUT / "opened-contact-slope-plan.json"
DEST = OUT / "native-opened-contact-slope"
SEEDS = (19036, 19057)
OFFSETS_MM = (-4, -3, -1, 1, 2)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temp, path)


def freeze() -> dict:
    source = json.loads(V113_PLAN.read_text())
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("V114 applies only to the rejected historical control sandbox")
    if source["source_sha256"]["shift_evaluator"] != sha(shifted.EVALUATOR):
        raise ValueError("V113 target evaluator source has changed")
    selected = {case["seed"]: case for case in source["cases"]
                if case["seed"] in SEEDS}
    if set(selected) != set(SEEDS):
        raise ValueError("V113 source cases missing")
    reference = []
    for seed in SEEDS:
        for suffix in ("zero", "neg2"):
            path = V113_DATA / f"seed-{seed}-paired-{seed}-{suffix}.json"
            report = json.loads(path.read_text())
            if (report["seed"] != seed or
                    report["selection_mode"] != selected[seed]["expected_selection_mode"] or
                    report["body_sha256"] != source["source_sha256"]["body"] or
                    report["selected_pointer_sha256"] != source["source_sha256"]["pointer"]):
                raise ValueError(f"V113 paired source mismatch: {path}")
            reference.append({"seed": seed, "variant": suffix,
                              "path": str(path.relative_to(ROOT)),
                              "sha256": sha(path),
                              "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"]})
    observed = {(r["seed"], r["variant"]): r["contact_s"] for r in reference}
    if not (observed[(19036, "zero")] < 1 and
            observed[(19036, "neg2")] < 1 and
            observed[(19057, "zero")] >= 1.2 and
            1 <= observed[(19057, "neg2")] < 1.2):
        raise ValueError("V114 weak-pair selection premise no longer holds")
    sources = {**source["source_sha256"],
               "evaluator": sha(shifted.EVALUATOR),
               "runner": sha(Path(__file__))}
    cases = [
        {**selected[seed], "id": f"slope-{seed}-{offset:+d}mm",
         "diagnostic_goal_shift_body_y_m": offset / 1000}
        for seed in SEEDS for offset in OFFSETS_MM
    ]
    plan = {
        "schema": "robot-learning-lab.cove50-v114-contact-slope-plan.v1",
        "role": "opened_contact_slope_training_diagnostic_not_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "v113_plan_sha256": sha(V113_PLAN),
        "v113_reference_reports": reference,
        "selection": ("After the first three V113 varied full-native pairs, select "
                      "19036 because unchanged and −2 mm both failed and 19057 "
                      "because −2 mm lost the 1.2 s contact margin. These "
                      "opened outcomes choose diagnostic starts, never a score."),
        "method": ("Hold each physical start, observed-RGB route, motor, gaze "
                   "and toy placement fixed. Replay five additional prescribed "
                   "post-route body-Y target offsets, −4, −3, −1, +1, +2 mm. "
                   "Compare with the already retained zero and −2 mm V113 "
                   "visits. Retain all contact, RGB, support, rated-effort, "
                   "action and collision outcomes, including failures."),
        "case_count": len(cases),
        "cases": cases,
        "qualification": ("A slope or rescue on two opened starts is training "
                          "evidence only. It cannot qualify a general target "
                          "rule, learned controller or new mechanical body."),
    }
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    OUT.mkdir(parents=True, exist_ok=True)
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V114 plan or source changed")
    PLAN.write_bytes(encoded)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    plan = freeze()
    print(f"frozen {PLAN} sha256={sha(PLAN)}", flush=True)
    if args.freeze_only:
        return
    DEST.mkdir(parents=True, exist_ok=True)
    matrix_path = DEST / "matrix.json"
    existing = json.loads(matrix_path.read_text()) if matrix_path.exists() else None
    if existing and existing["plan_sha256"] != sha(PLAN):
        raise ValueError("Existing V114 matrix uses a different plan")
    rows = {row["id"]: row for row in existing["cases"]} if existing else {}
    run_plan = {"source_sha256": plan["source_sha256"]}
    for case in plan["cases"]:
        if case["id"] in rows:
            continue
        rows[case["id"]] = shifted.run_case(case, run_plan, DEST)
        ordered = [rows[c["id"]] for c in plan["cases"] if c["id"] in rows]
        matrix = {
            "schema": "robot-learning-lab.cove50-v114-contact-slope.v1",
            "role": plan["role"], "body_role": plan["body_role"],
            "plan_sha256": sha(PLAN), "case_count": len(ordered),
            "complete": len(ordered) == len(plan["cases"]),
            "pass_count": sum(row["pass"] for row in ordered),
            "margin_1p2_s_count": sum(row["margin_1p2_s_met"] for row in ordered),
            "cases": ordered,
        }
        write_atomic(matrix_path, matrix)
        row = rows[case["id"]]
        print(json.dumps({"id": row["id"], "contact_s": row["contact_s"],
                          "pass": row["pass"], "margin_1p2_s_met": row["margin_1p2_s_met"]}),
              flush=True)


if __name__ == "__main__":
    main()
