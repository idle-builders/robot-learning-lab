"""Map-planned lateral RGB-D owner station against the same side prop."""

from __future__ import annotations

import json
from pathlib import Path

import run_robocasa_layout2_person_route_v697 as full
import run_robocasa_layout2_route_v407 as route
import run_robocasa_layout2_side_prop_v701 as side


SOURCE = Path(__file__)
OUT = side.ROOT / "artifacts/ape-robocasa-layout2-side-prop-lateral-station-v706"
LATERAL_M = .15
HORIZON_S = 23.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {"schema": "robot-learning-lab.layout2-side-prop-lateral-station.v1",
                   "matched_prop_and_room_as": str(side.OUT / "report.json"),
                   "station_rule": "0.8 m toward initial ape from person RGB-D surface, then 0.15 m lateral toward image-right/east in the person-ape frame; map route must admit it",
                   "standoff_m": .8, "lateral_m": LATERAL_M,
                   "post_stop_until_s": HORIZON_S,
                   "success_checks": "Fresh person and item evidence plus useful return, or explicit help, with >=5 cm body margin through 23 s; compare V704/V705 without selecting from evaluator path",
                   "source_sha256": route.sha256(SOURCE)}
    (OUT / "station-declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    previous = side.OUT, full.POST_STOP_UNTIL_S, full.STANDOFF_M, full.STATION_LATERAL_M
    try:
        side.OUT = OUT
        full.POST_STOP_UNTIL_S = HORIZON_S
        full.STANDOFF_M = .8
        full.STATION_LATERAL_M = LATERAL_M
        side.main()
    finally:
        side.OUT, full.POST_STOP_UNTIL_S, full.STANDOFF_M, full.STATION_LATERAL_M = previous
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report["schema"] = "robot-learning-lab.layout2-side-prop-lateral-station-result.v1"
    report["lateral_station_source_sha256"] = route.sha256(SOURCE)
    report["station_declaration_sha256"] = route.sha256(OUT / "station-declaration.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    ep = report["episode"]
    summary = ep["memory_return"] or {}
    print(ep["route_state"], ep["outcome"], summary.get("route_status"),
          summary.get("item_status"),
          summary.get("post_stop_minimum_proxy_gap_m_evaluator_only"), flush=True)


if __name__ == "__main__":
    main()
