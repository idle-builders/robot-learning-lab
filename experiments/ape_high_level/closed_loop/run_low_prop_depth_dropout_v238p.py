"""Pair the RGB-D cross-check with a complete missing-depth fault."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from run import ROOT, sha
import run_low_prop_depth_crosscheck_v236p as depth
import run_low_prop_sensor_join_v206p as v206
import run_low_prop_role_fault_v212p as v212
import run_low_prop_held_pose_v224p as v224


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-low-prop-depth-dropout-v238p"


def main():
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Reference-body status changed")
    baseline = json.loads((depth.OUT / "report.json").read_text())
    if not baseline["episodes"] or baseline["depth_view"][
            "random_hole_fraction"] != .15:
        raise AssertionError("RGB-D candidate baseline changed")
    prior = np.load(v206.dynamic.PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    old_window, old_margin = v212.WINDOW_S, v206.route.MARGIN_M
    old_out, old_holes = depth.OUT, depth.DEPTH_HOLE_FRACTION
    v212.WINDOW_S = (999., 1000.)
    v206.route.MARGIN_M = .15
    depth.OUT = OUT
    depth.DEPTH_HOLE_FRACTION = 1.
    vision = v206.dynamic.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([
        Image.open(v206.dynamic.owner.TEACH_CROP).convert("RGB")])[0]
    episodes = []
    try:
        for case in v224.CASES:
            result = depth.one(case, prior, vision, exemplar)
            episodes.append(result)
            print(case["id"], result["route_state"], result["outcome"],
                  "depth_release", sum(d["decision"] == "depth_clears_low_veto"
                                       for d in result["low_depth_decisions"]),
                  "gap", round(result["route_episode"][
                      "min_true_prop_gap_m"], 3), flush=True)
    finally:
        v212.WINDOW_S = old_window
        v206.route.MARGIN_M = old_margin
        depth.OUT = old_out
        depth.DEPTH_HOLE_FRACTION = old_holes
    report = {
        "schema": "robot-learning-lab.ape-low-prop-depth-dropout.v1",
        "source_sha256": sha(SOURCE),
        "depth_crosscheck_source_sha256": sha(Path(depth.__file__)),
        "depth_crosscheck_report_sha256": sha(old_out / "report.json"),
        "current_design_status": pointer["status"],
        "changed_factor": "Set only current low RGB-D measured depth to "
                          "100% missing pixels; retain its RGB capture, "
                          "low range projection bias, packet delay, pose "
                          "estimate and owner turn.",
        "scope": "Matched three complete near-fast owner episodes. This "
                 "tests missing-depth fallback, not optical common-mode "
                 "failure or a selected-body camera mount.",
        "episodes": episodes,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
