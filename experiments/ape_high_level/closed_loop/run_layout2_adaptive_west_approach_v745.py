"""First full owner run of frozen V744 selector against the direct approach."""

from __future__ import annotations

import json
from pathlib import Path

import run_layout2_adaptive_refuge_v744 as adaptive
import run_layout2_west_station_approach_v736 as approach
import run_robocasa_layout2_route_v407 as route


SOURCE = Path(__file__)
OUT = adaptive.west.side.ROOT / "artifacts/ape-robocasa-layout2-adaptive-west-approach-v745"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {"schema": "robot-learning-lab.layout2-adaptive-west-approach.v1",
                   "selector_source_sha256": route.sha256(Path(adaptive.selector.__file__)),
                   "prop_source_sha256": route.sha256(Path(approach.__file__)),
                   "profile": "V736 fixture-clear prop directly approaches west station after item capture",
                   "source_sha256": route.sha256(SOURCE)}
    (OUT / "approach-declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    adaptive.main(prop_factory=approach.WestApproachProp, out=OUT,
                  profile=declaration["profile"])
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report["approach_declaration_sha256"] = route.sha256(OUT / "approach-declaration.json")
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
