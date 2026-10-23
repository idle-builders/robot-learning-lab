"""Freeze V222p on a later, nearer crossing that may reach a held ape."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from run import ROOT, sha
import run_low_prop_sensor_join_v206p as v206
import run_low_prop_role_fault_v212p as v212
import run_low_prop_coverage_recover_v216p as v216
import run_low_prop_track_validity_v222p as v222


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-low-prop-held-pose-v224p"
FAULT_WINDOW_S = (11.6, 13.2)
CASES = (
    {"id": "near_fast_returning", "from_xy_m": [2.85, -1.05],
     "to_xy_m": [2.85, -1.05], "begin_s": 11.2, "end_s": 12.8,
     "waypoints": [[11.2, 2.85, -1.05], [12.0, 2.85, -1.65],
                   [12.8, 2.85, -1.05]]},
    {"id": "near_fast_one_way", "from_xy_m": [2.85, -1.05],
     "to_xy_m": [2.85, -1.65], "begin_s": 11.2, "end_s": 12.0,
     "waypoints": [[11.2, 2.85, -1.05], [12.0, 2.85, -1.65]]},
    {"id": "near_stationary_control", "from_xy_m": [2.85, -1.05],
     "to_xy_m": [2.85, -1.05], "begin_s": 11.2, "end_s": 12.8,
     "waypoints": [[11.2, 2.85, -1.05], [12.8, 2.85, -1.05]]},
)


def main():
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Reference-body status changed")
    prior = np.load(v206.dynamic.PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    old_out, old_window, old_margin = v222.OUT, v212.WINDOW_S, v206.route.MARGIN_M
    v222.OUT = OUT
    v212.WINDOW_S = FAULT_WINDOW_S
    v206.route.MARGIN_M = .15
    vision = v206.dynamic.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([
        Image.open(v206.dynamic.owner.TEACH_CROP).convert("RGB")])[0]
    episodes = []
    try:
        for case in CASES:
            result = v222.one(case, "rear_occluded_control", prior, vision, exemplar)
            episodes.append(result)
            print(case["id"], result["route_state"], result["outcome"],
                  "track_holds", sum(d["event"] in (
                      "track_validity_lost", "track_validity_wait")
                      for d in result["track_validity_decisions"]),
                  "min_gap", round(result["route_episode"][
                      "min_true_prop_gap_m"], 3), flush=True)
    finally:
        v222.OUT = old_out
        v212.WINDOW_S = old_window
        v206.route.MARGIN_M = old_margin
    report = {
        "schema": "robot-learning-lab.ape-low-prop-held-pose.v1",
        "source_sha256": sha(SOURCE),
        "v222_source_sha256": sha(Path(v222.__file__)),
        "v216_source_sha256": sha(Path(v216.__file__)),
        "current_design_status": pointer["status"],
        "fault_window_s": FAULT_WINDOW_S,
        "changed_factor": "Freeze V222p actor; shift initial prop15cm "
                          "toward route, start11.2s, traverse60cm by12.0s "
                          "at0.75m/s, then return by12.8s or remain. "
                          "Rear-sector control loss11.6–13.2s.",
        "scope": "Same layout4 owner route and RGB-D find; three matched "
                 "paths, ideal simultaneous low ray scan and kinematic "
                 "historical 50cm reference body. The full moving and "
                 "held pose is scored; physical stopping is not tested.",
        "episodes": episodes,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
