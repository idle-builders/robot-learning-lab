"""Exact scene/horizon replay with direction-neutral held-hazard closing speed."""

from __future__ import annotations

import json
from pathlib import Path

import run_robocasa_layout2_person_route_v697 as full
import run_robocasa_layout2_route_v407 as route
import run_robocasa_layout2_side_prop_long_hold_v703 as control
import run_robocasa_layout2_side_prop_v701 as side
import run_robocasa_memory_return_v599 as memory


SOURCE = Path(__file__)
OUT = side.ROOT / "artifacts/ape-robocasa-layout2-side-prop-closing-gate-v704"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {"schema": "robot-learning-lab.layout2-side-prop-closing-gate.v1",
                   "same_scene_and_hold_horizon_as": str(control.OUT / "report.json"),
                   "sole_policy_change": "Held moving-track admission uses radial closing speed toward the localized ape >=0.1m/s instead of a fixed negative-x velocity test. The existing route/held-pose candidate screen and no-best-effort setting are unchanged.",
                   "source_sha256": route.sha256(SOURCE),
                   "memory_controller_source_sha256": route.sha256(memory.SOURCE),
                   "control_report_sha256": route.sha256(control.OUT / "report.json")}
    (OUT / "predeclared.json").write_text(json.dumps(declaration, indent=2) + "\n")
    old_out, old_horizon = side.OUT, full.POST_STOP_UNTIL_S
    try:
        side.OUT = OUT
        full.POST_STOP_UNTIL_S = control.HORIZON_S
        side.main()
    finally:
        side.OUT, full.POST_STOP_UNTIL_S = old_out, old_horizon
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report["schema"] = "robot-learning-lab.layout2-side-prop-closing-gate-result.v1"
    report["closing_gate_source_sha256"] = route.sha256(SOURCE)
    report["memory_controller_source_sha256"] = route.sha256(memory.SOURCE)
    report["predeclaration_sha256"] = route.sha256(OUT / "predeclared.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    summary = report["episode"]["memory_return"] or {}
    print(summary.get("route_status"), summary.get("held_reactive_escape_count"),
          summary.get("post_stop_minimum_proxy_gap_m_evaluator_only"),
          summary.get("post_stop_margin_breach_evaluator_only"), flush=True)


if __name__ == "__main__":
    main()
