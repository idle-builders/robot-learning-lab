"""Matched east-side owner station using live range mapping for the route."""

from __future__ import annotations

import json
from pathlib import Path

import run_robocasa_layout2_person_route_v697 as full
import run_robocasa_layout2_route_v407 as route
import run_robocasa_layout2_side_prop_v701 as side
import run_robocasa_layout2_survey_station_v709 as survey


SOURCE = Path(__file__)
OUT = side.ROOT / "artifacts/ape-robocasa-layout2-live-station-v710"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {"schema": "robot-learning-lab.layout2-live-station-declaration.v1",
                   "matched_to": str(survey.OUT / "report.json"),
                   "sole_change": "Do not seed route grid from the previously surveyed map; keep same RGB-D person-derived 0.96m/0.15m east station, side-prop path and 23s horizon.",
                   "source_sha256": route.sha256(SOURCE),
                   "survey_report_sha256": route.sha256(survey.OUT / "report.json")}
    (OUT / "predeclared.json").write_text(json.dumps(declaration, indent=2) + "\n")
    previous = (side.OUT, full.POST_STOP_UNTIL_S, full.STANDOFF_M,
                full.STATION_LATERAL_M, full.SEED_ROUTE_FROM_SURVEY)
    try:
        side.OUT = OUT
        full.POST_STOP_UNTIL_S = 23.
        full.STANDOFF_M = survey.STANDOFF_M
        full.STATION_LATERAL_M = survey.LATERAL_M
        full.SEED_ROUTE_FROM_SURVEY = False
        side.main()
    finally:
        (side.OUT, full.POST_STOP_UNTIL_S, full.STANDOFF_M,
         full.STATION_LATERAL_M, full.SEED_ROUTE_FROM_SURVEY) = previous
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report["schema"] = "robot-learning-lab.layout2-live-station-result.v1"
    report["live_station_source_sha256"] = route.sha256(SOURCE)
    report["predeclaration_sha256"] = route.sha256(OUT / "predeclared.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    ep = report["episode"]
    memory = ep["memory_return"] or {}
    print(ep["route_state"], ep["outcome"], memory.get("route_status"),
          memory.get("item_status"), memory.get("post_stop_minimum_proxy_gap_m_evaluator_only"),
          flush=True)


if __name__ == "__main__":
    main()
