"""Audit V25's saved shape-model ablation and full gap/media evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_online_corridor as baseline_audit
import online_corridor_trial as baseline
import plan_bounded_trial as plan
import shape_surface_trial as shape
from kitchen_trial import sha


def main() -> None:
    old_out = baseline.OUT
    baseline.OUT = shape.OUT
    try:
        baseline_audit.main()
    finally:
        baseline.OUT = old_out
    report_path = shape.OUT / "report.json"
    report = json.loads(report_path.read_text())
    audit_path = shape.OUT / "audit.json"
    audit = json.loads(audit_path.read_text())
    audit["schema"] = "robot-learning-lab.ape-robocasa-shape-surface-audit.v1"
    audit["checks"]["shape_source"] = (
        report["shape_surface_source_sha256"] == sha(Path(shape.__file__)))
    audit["checks"]["v24_baseline"] = (
        report["v24_report_sha256"] == sha(plan.OUT / "report.json"))
    audit["checks"]["shape_rule"] = (
        report["algorithm"]["circle_radius_m"] == list(shape.CIRCLE_RADIUS_M)
        and report["algorithm"]["circle_max_residual_m"] ==
        shape.CIRCLE_MAX_RESIDUAL_M)
    models = {"circle": 0, "median": 0}
    for row in report["results"]:
        key = f'{row["world"]}:{row["case"]}'
        with np.load(row["trace"]["path"], allow_pickle=False) as saved:
            packets = [json.loads(raw) for raw in saved["packet_json"]]
        audit["checks"][f"bounded_and_shape:{key}"] = all(
            abs(p["route_guard"]["forecast_horizon_s"] -
                min(shape.guard.HORIZON_S,
                    max(0., plan.SKILL_END_S-p["captured_at_s"]) +
                    plan.STOP_EQUIVALENT_S)) < 1e-4 and
            all(t["surface_model"] in models for t in
                p["route_guard"]["tracks"])
            for p in packets)
        for packet in packets:
            for track in packet["route_guard"]["tracks"]:
                models[track["surface_model"]] += 1
    audit["surface_models_reported"] = models
    audit["pass"] = all(audit["checks"].values())
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"pass": audit["pass"],
                      "summary": audit["summary"],
                      "surface_models_reported": models}, indent=2))
    if not audit["pass"]:
        raise SystemExit("V25 audit failed")


if __name__ == "__main__":
    main()
