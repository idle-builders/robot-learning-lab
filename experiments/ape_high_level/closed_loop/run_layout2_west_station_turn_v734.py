"""Changed-bearing challenge of V732's frozen west-station threat rule."""

from __future__ import annotations

import json
from pathlib import Path

import run_layout2_west_station_threat_gate_v732 as trial
import run_robocasa_layout2_inference_turn_v723 as turn
import run_robocasa_layout2_route_v407 as route


SOURCE = Path(__file__)
OUT = trial.side.ROOT / "artifacts/ape-robocasa-layout2-west-station-turn-v734"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {"schema": "robot-learning-lab.layout2-west-station-turn-challenge.v1",
                   "frozen_station_and_gate_source_sha256": route.sha256(Path(trial.__file__)),
                   "turn_prop_source_sha256": route.sha256(Path(turn.__file__)),
                   "turn_after_capture_s": turn.TURN_AFTER_CAPTURE_S,
                   "turn_end_xy_m": turn.END.tolist(),
                   "profile": "same initially straight path, then fixture-clear turn toward V721's old exit",
                   "source_sha256": route.sha256(SOURCE)}
    (OUT / "turn-declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    trial.main(prop_factory=turn.TurnProp, out=OUT,
               profile=declaration["profile"])
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text())
    report["turn_declaration_sha256"] = route.sha256(OUT / "turn-declaration.json")
    report["turn_source_sha256"] = route.sha256(Path(turn.__file__))
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
