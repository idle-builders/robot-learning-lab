"""Compare low, 30 cm and fused low+30 cm range planes on one moving route.

The actor receives only the planar ranges. Fusion projects the nearest valid
return per bearing onto one 2-D grid; it is a software ablation, not a mount
design or a model of walking leg occlusion.
"""

from __future__ import annotations

import json
from pathlib import Path

import run_robocasa_layout2_route_v407 as route
import run_robocasa_moving_prop_v423 as trial
from audit_robocasa_moving_prop_v424 import audit_episode
from audit_robocasa_post_stop_v427 import replay


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-scan-height-v433"
BASE_SCAN = route.planar_scan


def height_scan(heights_m: tuple[float, ...]):
    def scan(model, data, xy, max_range_m, rng):
        planes = []
        for height in heights_m:
            route.LIDAR_Z_M = height
            planes.append(BASE_SCAN(model, data, xy, max_range_m, rng))
        return [min((plane[beam] for plane in planes if plane[beam] is not None),
                    default=None)
                for beam in range(len(planes[0]))]
    return scan


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    route.MARGIN_M = .15
    cases = []
    for label, heights in (("mid_30cm", (.30,)),
                           ("low_05cm", (.05,)),
                           ("low_10cm", (.10,)),
                           ("dual_05_30cm", (.05, .30)),
                           ("dual_10_30cm", (.10, .30))):
        trial.planar_scan = height_scan(heights)
        trial.OUT = OUT / label
        episode = trial.episode(True)
        prop_audit = audit_episode(episode)
        held = replay(label, episode) if episode["end_time_s"] <= trial.MOVE_END_S else None
        cases.append({"case": label, "scan_heights_m": heights,
                      "map_margin_m": .15, "episode": episode,
                      "prop_audit": {key: value for key, value in prop_audit.items()
                                     if key != "closest"},
                      "post_stop": ({key: value for key, value in held.items()
                                     if key != "trace"} if held is not None else None)})
        print(label, episode["outcome"], episode["steps"],
              round(episode["end_time_s"], 2), flush=True)
    route.LIDAR_Z_M = .30
    route.MARGIN_M = .10
    trial.planar_scan = BASE_SCAN
    report = {"source_sha256": route.sha256(Path(__file__)),
              "frozen_trial_source_sha256": route.sha256(Path(trial.__file__)),
              "frozen_grid_source_sha256": route.sha256(Path(route.__file__)),
              "scope": "matched route/prop at 15cm grid margin, noisy 72-beam low/mid/fused planar scans, no hardware occlusion",
              "cases": cases}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
