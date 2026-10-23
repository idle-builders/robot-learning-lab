"""Stress the aligned low tracker with a fixed 3 cm lateral pose error.

The actor still stores its unmodified estimated capture pose. Only the pose
passed to low-object point projection is biased toward or away from the
route, modeling an uncalibrated low-source extrinsic/pose error. All actual
ray packets, mid localization, world motion and owner RGB-D task are intact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import novel_range_object as tracking
from run import ROOT, sha
import run_low_prop_sensor_join_v206p as v206
import run_low_prop_role_fault_v212p as v212
import run_low_prop_aligned_capture_v232p as v232
import run_low_prop_held_pose_v224p as v224


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-low-prop-capture-pose-bias-v234p"
BIASES = (("toward_route_3cm", np.array([0., -.03])),
          ("away_from_route_3cm", np.array([0., .03])))


def one(case, name, bias, prior, vision, exemplar):
    original_update, original_out = tracking.NovelRangeObject.update, v232.OUT
    projections = []

    def biased_update(tracker, ranges, actor_capture_pose, capture_time_s):
        injected = np.asarray(actor_capture_pose) + bias
        result = original_update(tracker, ranges, injected, capture_time_s)
        projections.append({"capture_time_s": capture_time_s,
                            "actor_capture_pose_xy_m": np.asarray(
                                actor_capture_pose).tolist(),
                            "low_projection_pose_xy_m": injected.tolist(),
                            "track_status": result["status"]})
        return result

    tracking.NovelRangeObject.update = biased_update
    v232.OUT = OUT / name
    try:
        result = v232.one(case, "aligned_age_local", prior, vision, exemplar)
    finally:
        tracking.NovelRangeObject.update = original_update
        v232.OUT = original_out
    result["low_projection_bias_name"] = name
    result["low_projection_bias_xy_m"] = bias.tolist()
    result["low_projection_events"] = projections
    return result


def main():
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Reference-body status changed")
    prior = np.load(v206.dynamic.PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    original_window, original_margin = v212.WINDOW_S, v206.route.MARGIN_M
    v212.WINDOW_S = (999., 1000.)
    v206.route.MARGIN_M = .15
    vision = v206.dynamic.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([
        Image.open(v206.dynamic.owner.TEACH_CROP).convert("RGB")])[0]
    episodes = []
    try:
        for case in v224.CASES:
            for name, bias in BIASES:
                result = one(case, name, bias, prior, vision, exemplar)
                episodes.append(result)
                print(case["id"], name, result["route_state"],
                      result["outcome"], "low_vetoes", len(result[
                          "low_corridor_vetoes"]),
                      "min_gap", round(result["route_episode"][
                          "min_true_prop_gap_m"], 3), flush=True)
    finally:
        v212.WINDOW_S = original_window
        v206.route.MARGIN_M = original_margin
    report = {
        "schema": "robot-learning-lab.ape-low-prop-capture-pose-bias.v1",
        "source_sha256": sha(SOURCE),
        "v232_source_sha256": sha(Path(v232.__file__)),
        "baseline_report_sha256": sha(v232.OUT / "report.json"),
        "current_design_status": pointer["status"],
        "projection_biases_xy_m": {name: bias.tolist()
                                   for name, bias in BIASES},
        "changed_factor": "Apply fixed +/−3cm world-Y bias only to "
                          "the low ray tracker's estimated capture-pose "
                          "point projection. Aligned scan timing, age "
                          "hold, local corridor veto, mid scan and owner "
                          "task stay frozen.",
        "scope": "Matched layout4 near-fast return, one-way and "
                 "stationary owner episodes. Pose bias is an imposed "
                 "calibration stress, not a measured error distribution. "
                 "The low source and body remain ideal reference proxies.",
        "episodes": episodes,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
