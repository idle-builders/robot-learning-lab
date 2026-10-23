"""Recheck the complete static layout-2 owner task after held-track gate change."""

from __future__ import annotations

import json
from pathlib import Path

import run_robocasa_layout2_person_route_v697 as full
import run_robocasa_layout2_route_v407 as route
import run_robocasa_memory_return_v599 as memory


SOURCE = Path(__file__)
OUT = full.ROOT / "artifacts/ape-robocasa-layout2-static-regression-v707"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old_out = full.OUT
    try:
        full.OUT = OUT
        full.main()
    finally:
        full.OUT = old_out
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report["schema"] = "robot-learning-lab.layout2-static-regression.v1"
    report["regression_source_sha256"] = route.sha256(SOURCE)
    report["memory_controller_source_sha256"] = route.sha256(memory.SOURCE)
    path.write_text(json.dumps(report, indent=2) + "\n")
    ep = report["episode"]
    print(ep["route_state"], ep["outcome"],
          (ep["memory_return"] or {}).get("item_status"), flush=True)


if __name__ == "__main__":
    main()
