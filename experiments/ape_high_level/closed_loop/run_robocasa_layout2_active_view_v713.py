"""Retry one bounded person view when the mapped station arrives short."""

from __future__ import annotations

import json
from pathlib import Path

import run_robocasa_layout2_joint_station_v712 as station
import run_robocasa_layout2_person_route_v697 as full
import run_robocasa_layout2_route_v407 as route


SOURCE = Path(__file__)
OUT = station.side.ROOT / "artifacts/ape-robocasa-layout2-active-view-v713"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {"schema": "robot-learning-lab.layout2-active-view-declaration.v1",
                   "matched_station_and_side_prop_as": str(station.OUT / "report.json"),
                   "sole_policy_change": "When fresh person depth is unhealthy at the 15cm arrival tolerance, allow one <=12cm known-free, fresh-range, 20ms-guarded move toward the already selected RGB-D person station, then recapture before the owner request.",
                   "no_truth_for_actor": True,
                   "source_sha256": route.sha256(SOURCE),
                   "person_runner_source_sha256": route.sha256(full.SOURCE),
                   "control_report_sha256": route.sha256(station.OUT / "report.json")}
    (OUT / "active-view-declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    old_out, old_active = station.OUT, full.ACTIVE_VIEW_CORRECTION
    try:
        station.OUT = OUT
        full.ACTIVE_VIEW_CORRECTION = True
        station.main()
    finally:
        station.OUT, full.ACTIVE_VIEW_CORRECTION = old_out, old_active
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report["schema"] = "robot-learning-lab.layout2-active-view-result.v1"
    report["active_view_source_sha256"] = route.sha256(SOURCE)
    report["active_view_declaration_sha256"] = route.sha256(OUT / "active-view-declaration.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    ep = report["episode"]
    memory = ep["memory_return"] or {}
    print(ep["route_state"], ep["outcome"], memory.get("route_status"),
          memory.get("item_status"), memory.get("post_stop_minimum_proxy_gap_m_evaluator_only"),
          flush=True)


if __name__ == "__main__":
    main()
