"""Use a surveyed known-free owner station east of the side-prop corridor."""

from __future__ import annotations

import json
from pathlib import Path

import run_robocasa_layout2_person_route_v697 as full
import run_robocasa_layout2_route_v407 as route
import run_robocasa_layout2_side_prop_v701 as side


SOURCE = Path(__file__)
OUT = side.ROOT / "artifacts/ape-robocasa-layout2-survey-station-v709"
STANDOFF_M = .96
LATERAL_M = .15
HORIZON_S = 23.


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {"schema": "robot-learning-lab.layout2-survey-station-declaration.v1",
                   "scene_and_prop": "same layout2/style1 and 17.8–21.8s side path as V701–V706",
                   "person_station_rule": "0.96m toward ape and 0.15m lateral from first RGB-D person surface; candidate x≈1.10,y≈−3.45m",
                   "route_map": "seed actor grid with the existing layout2 sensor-derived survey map, then update it with live range; unknown cells remain unknown",
                   "reason_for_candidate": "Development preflight: this map cell and a south/east refuge are surveyed free; aimed show view contains person and item; simulated 50cm proxy has at least 10cm capped room clearance. This is not a held-out selection or a future-hazard certificate.",
                   "standoff_m": STANDOFF_M,
                   "lateral_m": LATERAL_M,
                   "post_stop_until_s": HORIZON_S,
                   "source_sha256": route.sha256(SOURCE),
                   "person_runner_source_sha256": route.sha256(full.SOURCE),
                   "prior_map_sha256": route.sha256(side.transfer.PRIOR)}
    (OUT / "station-declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    previous = (side.OUT, full.POST_STOP_UNTIL_S, full.STANDOFF_M,
                full.STATION_LATERAL_M, full.SEED_ROUTE_FROM_SURVEY)
    try:
        side.OUT = OUT
        full.POST_STOP_UNTIL_S = HORIZON_S
        full.STANDOFF_M = STANDOFF_M
        full.STATION_LATERAL_M = LATERAL_M
        full.SEED_ROUTE_FROM_SURVEY = True
        side.main()
    finally:
        (side.OUT, full.POST_STOP_UNTIL_S, full.STANDOFF_M,
         full.STATION_LATERAL_M, full.SEED_ROUTE_FROM_SURVEY) = previous
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report["schema"] = "robot-learning-lab.layout2-survey-station-result.v1"
    report["survey_station_source_sha256"] = route.sha256(SOURCE)
    report["station_declaration_sha256"] = route.sha256(OUT / "station-declaration.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    ep = report["episode"]
    memory = ep["memory_return"] or {}
    print(ep["route_state"], ep["outcome"], memory.get("route_status"),
          memory.get("item_status"), memory.get("post_stop_minimum_proxy_gap_m_evaluator_only"),
          flush=True)


if __name__ == "__main__":
    main()
