"""Test whether wider generic map clearance avoids a late moving-prop pinch.

The actor sees only its range-updated grid, not the prop's future trajectory.
This ablates one planner margin, preserving the same script/scene/guard.
"""

from __future__ import annotations

import json
from pathlib import Path

import run_robocasa_layout2_route_v407 as route
import run_robocasa_moving_prop_v423 as trial
from audit_robocasa_moving_prop_v424 import audit_episode
from audit_robocasa_post_stop_v427 import replay


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-margin-ablation-v431"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for margin in (.10, .15, .20, .25, .30, .35):
        route.MARGIN_M = margin
        label = f"margin_{int(margin * 100):02d}cm"
        trial.OUT = OUT / label
        episode = trial.episode(True)
        immediate = audit_episode(episode)
        post_stop = replay(label, episode) if episode["end_time_s"] <= trial.MOVE_END_S else None
        cases.append({"margin_m": margin, "episode": episode,
                      "immediate_audit": {key: value for key, value in immediate.items()
                                          if key != "closest"},
                      "post_stop": ({key: value for key, value in post_stop.items()
                                     if key != "trace"} if post_stop else None)})
        print(label, episode["outcome"], "steps", episode["steps"],
              "end", round(episode["end_time_s"], 2),
              "moving_gap", round(episode["min_true_prop_gap_m"], 3),
              "post_stop", round(post_stop["minimum_post_stop_gap_m"], 3)
              if post_stop else None, flush=True)
    route.MARGIN_M = .10
    report = {"source_sha256": route.sha256(Path(__file__)),
              "frozen_trial_source_sha256": route.sha256(Path(trial.__file__)),
              "frozen_grid_source_sha256": route.sha256(Path(route.__file__)),
              "scope": "generic range-grid margin 10-35cm; no prop future/class input to actor",
              "cases": cases}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
