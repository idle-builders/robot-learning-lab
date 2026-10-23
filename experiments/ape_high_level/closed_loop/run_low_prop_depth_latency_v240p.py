"""Delay the current low RGB-D view by one 400 ms route decision."""

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
OUT = ROOT / "artifacts/ape-robocasa-low-prop-depth-latency-v240p"
PROFILES = (("delayed_unguarded", None),
            ("delayed_age_refusal", .2))


def main():
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Reference-body status changed")
    baseline_path = depth.OUT / "report.json"
    baseline = json.loads(baseline_path.read_text())
    if not baseline["episodes"] or baseline["depth_view"][
            "random_hole_fraction"] != .15:
        raise AssertionError("Current-depth candidate baseline changed")
    prior = np.load(v206.dynamic.PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    old_window, old_margin = v212.WINDOW_S, v206.route.MARGIN_M
    old_out = depth.OUT
    v212.WINDOW_S = (999., 1000.)
    v206.route.MARGIN_M = .15
    vision = v206.dynamic.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([
        Image.open(v206.dynamic.owner.TEACH_CROP).convert("RGB")])[0]
    episodes = []
    try:
        for case in v224.CASES:
            for profile, age_limit in PROFILES:
                depth.OUT = OUT / profile
                result = depth.one(case, prior, vision, exemplar,
                                   depth_delay_steps=1,
                                   depth_age_limit_s=age_limit)
                result["depth_latency_profile"] = profile
                episodes.append(result)
                print(case["id"], profile, result["route_state"],
                      result["outcome"], "depth_release", sum(
                          d["decision"] == "depth_clears_low_veto" for d in
                          result["low_depth_decisions"]), "gap", round(
                          result["route_episode"]["min_true_prop_gap_m"], 3),
                      flush=True)
    finally:
        v212.WINDOW_S = old_window
        v206.route.MARGIN_M = old_margin
        depth.OUT = old_out
    report = {
        "schema": "robot-learning-lab.ape-low-prop-depth-latency.v1",
        "source_sha256": sha(SOURCE),
        "depth_crosscheck_source_sha256": sha(Path(depth.__file__)),
        "current_depth_report_sha256": sha(baseline_path),
        "current_design_status": pointer["status"],
        "depth_delivery_delay_steps": 1,
        "profiles": {name: {"depth_age_limit_s": limit} for name, limit in
                     PROFILES},
        "changed_factor": "Deliver the previous decision's rendered low "
                          "RGB-D packet 400ms late. Project it at its "
                          "saved actor-estimated capture pose. Compare "
                          "unguarded use with a 200ms maximum depth age "
                          "that falls back to the low-range veto.",
        "scope": "Matched three full near-fast owner episodes against "
                 "V236p current depth. Track-aimed camera yaw remains "
                 "instantaneous; this tests capture/delivery age rather "
                 "than head actuation, exposure smear or a selected body.",
        "episodes": episodes,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
