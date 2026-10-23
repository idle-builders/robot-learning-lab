"""Flatten retained RGB diagnostics for the inspector's case-table view."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    fit_path = BASE / "calibration-v3/fit-report.json"
    fit = json.loads(fit_path.read_text())
    fit_cases = []
    for trial in fit["trials"]:
        all_error = trial["validation_error"]["all"]
        old = trial["validation_error"]["historical"]
        opened = trial["validation_error"]["opened_v21"]
        fit_cases.append({"name": trial["name"],
                          "median_mm": all_error["median_mm"], "p95_mm": all_error["p95_mm"],
                          "max_mm": all_error["max_mm"],
                          "historical_max_mm": old["max_mm"],
                          "opened_max_mm": opened["max_mm"],
                          "failure_10604_nominal_error_mm": trial["opened_10604_nominal_error_mm"],
                          "pass": all_error["median_mm"] <= 1.5 and all_error["max_mm"] <= 3 and
                                  trial["opened_10604_nominal_error_mm"] <= 2})
    fit_matrix = {"schema": "robot-learning-lab.cove50-v3-calibration-inspector-table.v1",
                  "role": "offline_state_disjoint_fit_screen_not_native_control",
                  "source_report_sha256": sha(fit_path),
                  "case_count": len(fit_cases), "pass_count": sum(row["pass"] for row in fit_cases),
                  "cases": fit_cases}
    (fit_path.parent / "fit-matrix.json").write_text(json.dumps(fit_matrix, indent=2) + "\n")

    offline_path = BASE / "v30/offline-report.json"
    offline = json.loads(offline_path.read_text())
    cases = [{**row, "featured": ((row["seed"], row["sample"]) in
             {(10604, 0), (10602, 3), (10608, 0)})}
             for row in offline["rows"]]
    offline_matrix = {"schema": "robot-learning-lab.cove50-v30-template-inspector-table.v1",
                      "role": "opened_varied_image_accuracy_not_native_contact",
                      "source_report_sha256": sha(offline_path),
                      "case_count": len(cases),
                      "accepted_count": sum(row["template_applied"] for row in cases),
                      "offline_gate_pass": offline["offline_gate_pass"],
                      "cases": cases}
    (offline_path.parent / "offline-matrix.json").write_text(json.dumps(offline_matrix, indent=2) + "\n")

    gaze_path = BASE / "v31/offline-report.json"
    gaze = json.loads(gaze_path.read_text())
    gaze_cases = [{**row, "featured": ((row["seed"], row["sample"]) in
                  {(10604, 0), (10602, 3), (10608, 0)})}
                  for row in gaze["rows"]]
    gaze_matrix = {"schema": "robot-learning-lab.cove50-v31-template-gaze-inspector-table.v1",
                   "role": "opened_varied_image_accuracy_before_native_contact",
                   "source_report_sha256": sha(gaze_path),
                   "case_count": len(gaze_cases),
                   "accepted_count": sum(row["template_applied"] for row in gaze_cases),
                   "offline_gate_pass": gaze["offline_gate_pass"],
                   "cases": gaze_cases}
    (gaze_path.parent / "offline-matrix.json").write_text(json.dumps(gaze_matrix, indent=2) + "\n")

    prior_path = BASE / "v31/prior-offline-report.json"
    prior = json.loads(prior_path.read_text())
    prior_cases = [{**row, "featured": ((row["seed"], row["sample"]) in
                   {(10433, 0), (10501, 0), (10528, 1)})}
                   for row in prior["rows"]]
    prior_matrix = {"schema": "robot-learning-lab.cove50-v31-prior-image-inspector-table.v1",
                    "role": "earlier_opened_varied_image_accuracy_before_native_contact",
                    "source_report_sha256": sha(prior_path),
                    "case_count": len(prior_cases),
                    "accepted_count": sum(row["template_applied"] for row in prior_cases),
                    "offline_gate_pass": prior["offline_gate_pass"],
                    "cases": prior_cases}
    (prior_path.parent / "prior-offline-matrix.json").write_text(json.dumps(prior_matrix, indent=2) + "\n")
    print(f"V3 {len(fit_cases)} fits, V30 {len(cases)} views, V31 {len(gaze_cases)} recent + {len(prior_cases)} prior views")


if __name__ == "__main__":
    main()
