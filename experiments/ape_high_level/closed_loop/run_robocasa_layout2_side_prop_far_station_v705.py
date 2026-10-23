"""Earlier owner station test against the same layout-2 side approach."""

from __future__ import annotations

import json
from pathlib import Path

import run_robocasa_layout2_person_route_v697 as full
import run_robocasa_layout2_route_v407 as route
import run_robocasa_layout2_side_prop_v701 as side
import run_robocasa_memory_return_v599 as memory


SOURCE = Path(__file__)
OUT = side.ROOT / "artifacts/ape-robocasa-layout2-side-prop-far-station-v705"
STANDOFF_M = 1.2
HORIZON_S = 23.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {"schema": "robot-learning-lab.layout2-side-prop-far-station.v1",
                   "matched_prop_and_room_as": str(side.OUT / "report.json"),
                   "station_rule": "1.2 m from RGB-D person surface toward initial localized ape pose, versus 0.8 m in V701/V704",
                   "station_standoff_m": STANDOFF_M,
                   "post_stop_until_s": HORIZON_S,
                   "held_closing_speed_gate": True,
                   "success_checks": "Fresh person confirmation, visible shown item, useful return/refind or explicit task pause, and >=5 cm independent full-body proxy margin through 23 s",
                   "source_sha256": route.sha256(SOURCE),
                   "memory_source_sha256": route.sha256(memory.SOURCE)}
    (OUT / "station-declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    previous = side.OUT, full.POST_STOP_UNTIL_S, full.STANDOFF_M
    try:
        side.OUT = OUT
        full.POST_STOP_UNTIL_S = HORIZON_S
        full.STANDOFF_M = STANDOFF_M
        side.main()
    finally:
        side.OUT, full.POST_STOP_UNTIL_S, full.STANDOFF_M = previous
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report["schema"] = "robot-learning-lab.layout2-side-prop-far-station-result.v1"
    report["far_station_source_sha256"] = route.sha256(SOURCE)
    report["station_declaration_sha256"] = route.sha256(OUT / "station-declaration.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    ep = report["episode"]
    memory_result = ep["memory_return"] or {}
    print(ep["route_state"], ep["outcome"], memory_result.get("route_status"),
          memory_result.get("item_status"),
          memory_result.get("post_stop_minimum_proxy_gap_m_evaluator_only"), flush=True)


if __name__ == "__main__":
    main()
