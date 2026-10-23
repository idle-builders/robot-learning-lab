"""Control: extend the V701 live held monitor beyond the prop's arrival."""

from __future__ import annotations

import json
from pathlib import Path

import run_robocasa_layout2_person_route_v697 as full
import run_robocasa_layout2_route_v407 as route
import run_robocasa_layout2_side_prop_v701 as side


SOURCE = Path(__file__)
OUT = side.ROOT / "artifacts/ape-robocasa-layout2-side-prop-long-hold-v703"
HORIZON_S = 23.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {"schema": "robot-learning-lab.layout2-side-prop-long-hold.v1",
                   "same_scene_path_and_actor_as": str(side.OUT / "report.json"),
                   "change": "Extend continuous post-stop range/20ms near-body monitor from 20.8 s to 23.0 s, beyond the prop's 21.8 s arrival; no change to velocity-direction gate or escape policy.",
                   "post_stop_until_s": HORIZON_S,
                   "source_sha256": route.sha256(SOURCE),
                   "side_source_sha256": route.sha256(side.SOURCE),
                   "full_source_sha256": route.sha256(full.SOURCE)}
    (OUT / "horizon-declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    old_out, old_horizon = side.OUT, full.POST_STOP_UNTIL_S
    try:
        side.OUT = OUT
        full.POST_STOP_UNTIL_S = HORIZON_S
        side.main()
    finally:
        side.OUT, full.POST_STOP_UNTIL_S = old_out, old_horizon
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report["schema"] = "robot-learning-lab.layout2-side-prop-long-hold-result.v1"
    report["long_hold_source_sha256"] = route.sha256(SOURCE)
    report["horizon_declaration_sha256"] = route.sha256(OUT / "horizon-declaration.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    memory = report["episode"]["memory_return"] or {}
    print(report["episode"]["route_state"], memory.get("route_status"),
          memory.get("post_stop_minimum_proxy_gap_m_evaluator_only"),
          memory.get("post_stop_margin_breach_evaluator_only"), flush=True)


if __name__ == "__main__":
    main()
