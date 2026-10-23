"""Paired native screen of image refinement with stable calibrated-RGB routing."""

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
    run_v16_opened_matrix as v16,
    run_v35_wider_rgb_screen as core,
    run_v39_corner_image_screen as screen,
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v40"
PLAN = OUT / "route-stability-screen-plan.json"
DEST = OUT / "route-stability-screen"
EVALUATOR = Path(__file__).with_name("evaluate_selective_goal_v40.py")
TARGETS = (10528, 10508, 10501, 10513, 10538, 10534, 10539, 10609, 10615)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    paths = ((BASE / "v21/opened-gate-plan.json", BASE / "v21/opened-native/matrix.json"),
             (BASE / "v32/opened-gate-plan.json", BASE / "v32/opened-native/matrix.json"))
    inputs, baseline, identities = {}, {}, {}
    for plan_path, matrix_path in paths:
        inputs.update({c["seed"]: c for c in json.loads(plan_path.read_text())["cases"]})
        baseline.update({c["seed"]: c for c in json.loads(matrix_path.read_text())["cases"]})
        identities[str(plan_path.relative_to(ROOT))] = sha(plan_path)
        identities[str(matrix_path.relative_to(ROOT))] = sha(matrix_path)
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(actor=sha(screen.ACTOR), evaluator=sha(EVALUATOR),
                   selector=sha(screen.SELECTOR), template=sha(core.TEMPLATE))
    cases = []
    for seed in TARGETS:
        source, old = inputs[seed], baseline[seed]
        cases.append({"seed": seed,
                      "role": "regressed_prior_pass" if seed == 10528 else
                              "corner_image_rescue" if seed == 10508 else "retention_control",
                      "baseline_pass": old["pass"], "baseline_contact_s": old["contact_s"],
                      "baseline_trace": old["trace"],
                      **{key: source[key] for key in
                         ("rail_v", "delay_ms", "toy_offset_x_m",
                          "toy_offset_y_m", "gaze_head_yaw_rad")}})
    plan = {"schema": "robot-learning-lab.cove50-v40-route-stability-screen-plan.v1",
            "role": "opened_route_and_image_regression_diagnostic_not_fresh",
            "body_role": "rejected_historical_direct_frame_sandbox",
            "source_sha256": sources,
            "source_plan_and_matrix_sha256": identities,
            "v39_gate_stop_sha256": sha(BASE / "v39/opened-native/gate-stop.json"),
            "v39_screen_plan_sha256": sha(screen.PLAN),
            "core_runner_sha256": sha(Path(core.__file__)),
            "screen_runner_sha256": sha(Path(screen.__file__)),
            "runner_sha256": sha(Path(__file__)),
            "case_count": len(cases), "cases": cases,
            "rule": "For the V39 non-sparse template branch only, choose actor/height/live route from the pre-template calibrated RGB goal; continue controlling the arm with the observed template-refined goal. No toy truth or seed enters inference.",
            "advance_gate": "All nine opened native pairs pass, including restoration of prior 10528 and retention of 10508 image rescue; then re-run all 75 earlier/recent starts before any disjoint fresh score."}
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V40 plan differs from source")
    PLAN.write_bytes(encoded)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    row = screen.run_case(case, plan)
    report = json.loads((ROOT / row["result"]).read_text())
    entry = next(e for e in report["image_goal_capture_events"] if e["time_s"] == 7.5)
    row["route_goal_source"] = report["selection_features"]["route_goal_source"]
    row["selected_image_error_xy_m"] = report["selection_features"]["selected_image_error_xy_m"]
    if entry["template_used_for_control"] and entry["orange_pixels"] >= 64:
        if row["route_goal_source"] != "pre_template_calibrated_rgb":
            row["faults"].append("corner route not based on calibrated RGB")
    elif row["route_goal_source"] != "selected_image_goal":
        row["faults"].append("ordinary route source changed")
    row["pass"] = not row["faults"]
    return row


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
    v16.SOURCES["actor"] = screen.ACTOR
    core.EVALUATOR = EVALUATOR
    core.SELECTOR = screen.SELECTOR
    core.DEST = DEST
    core.EXPECTED_VARIANT = "v40_calibrated_route_with_observed_corner_refinement_candidate"
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan): case["seed"] for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["seed"]] = row
            ordered = [rows[c["seed"]] for c in plan["cases"] if c["seed"] in rows]
            matrix = {"schema": "robot-learning-lab.cove50-v40-route-stability-screen.v1",
                      "role": "opened_route_and_image_regression_diagnostic_not_fresh",
                      "body_role": "rejected_historical_direct_frame_sandbox",
                      "plan_sha256": sha(PLAN), "complete": len(ordered) == len(plan["cases"]),
                      "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
                      "cases": ordered}
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("seed", "selection_mode", "route_goal_source",
                               "pass", "contact_s", "faults")}), flush=True)


if __name__ == "__main__":
    main()
