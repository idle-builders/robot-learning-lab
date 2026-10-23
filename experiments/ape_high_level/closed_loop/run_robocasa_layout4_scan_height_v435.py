"""Transfer the revealed low/mid scan-height configurations to layout 4.

Layout 4 was used in earlier range/owner work, so this is a cross-layout
development check, not a held-out final estimate. No prop is added here.
"""

from __future__ import annotations

import json
from pathlib import Path

import run_robocasa_layout2_localized_route_v408 as actor
import run_robocasa_layout2_route_v407 as route
from run_robocasa_layout4_transfer_v412 import frame_layout4, layout4_scene
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-layout4-scan-height-v435"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    actor.model_with_layout2 = layout4_scene
    actor.frame_with_map = frame_layout4
    route.MARGIN_M = .15
    cases = []
    for label, heights in (("mid_30cm", (.30,)),
                           ("low_05cm", (.05,)),
                           ("dual_05_30cm", (.05, .30))):
        actor.planar_scan = height_scan(heights)
        actor.OUT = OUT / label
        row = actor.episode(5., .30, "range_growing")
        cases.append({"case": label, "scan_heights_m": heights,
                      "map_margin_m": .15, "episode": row})
        print(label, row["outcome"], row["steps"],
              round(row["end_pose_error_m"], 3), flush=True)
    route.LIDAR_Z_M = .30
    route.MARGIN_M = .10
    report = {"source_sha256": route.sha256(Path(__file__)),
              "frozen_actor_source_sha256": route.sha256(Path(actor.__file__)),
              "layout4_source_sha256": route.sha256(
                  Path(__file__).with_name("run_robocasa_layout4_transfer_v412.py")),
              "height_adapter_source_sha256": route.sha256(
                  Path(__file__).with_name("run_robocasa_scan_height_v433.py")),
              "scope": "layout4/style1 cross-layout development check of 5cm, 30cm and fused scan planes; 15cm map margin, 5m range, 30cm slip",
              "cases": cases}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
