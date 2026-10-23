"""Screen manufacturer ideal depth ranges in the full low-prop owner loop.

This is deliberately a one-factor range-window screen, not a device model.
Every profile retains the same rendered 70-degree camera, 1 cm noise, 15%
holes, instant aim, pose and policy. The windows are sourced from current
manufacturer ideal-range claims; real stereo/ToF failure is not emulated.
"""

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
OUT = ROOT / "artifacts/ape-robocasa-low-prop-depth-range-window-v242p"
PROFILES = (
    ("d405_ideal_7to50cm", (.07, .50),
     "https://www.realsenseai.com/products/stereo-depth-camera-d405/"),
    ("oak_d_sr_ideal_30to100cm", (.30, 1.00),
     "https://docs.luxonis.com/hardware/products/OAK-D%20SR"),
    ("d435i_ideal_30to300cm", (.30, 3.00),
     "https://www.realsenseai.com/products/depth-camera-d435i/"),
)


def main():
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Reference-body status changed")
    baseline_path = depth.OUT / "report.json"
    baseline = json.loads(baseline_path.read_text())
    if baseline["depth_view"]["valid_range_m"] != [.2, 1.5]:
        raise AssertionError("Current-depth baseline range changed")
    prior = np.load(v206.dynamic.PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    old_window, old_margin = v212.WINDOW_S, v206.route.MARGIN_M
    old_out, old_range = depth.OUT, depth.DEPTH_RANGE_M
    v212.WINDOW_S = (999., 1000.)
    v206.route.MARGIN_M = .15
    vision = v206.dynamic.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([
        Image.open(v206.dynamic.owner.TEACH_CROP).convert("RGB")])[0]
    episodes = []
    try:
        for case in v224.CASES:
            for profile, valid_range, _ in PROFILES:
                depth.OUT = OUT / profile
                depth.DEPTH_RANGE_M = valid_range
                result = depth.one(case, prior, vision, exemplar)
                result["depth_range_profile"] = profile
                result["depth_valid_range_m"] = valid_range
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
        depth.DEPTH_RANGE_M = old_range
    report = {
        "schema": "robot-learning-lab.ape-low-prop-depth-range-window.v1",
        "source_sha256": sha(SOURCE),
        "crosscheck_source_sha256": sha(Path(depth.__file__)),
        "current_depth_report_sha256": sha(baseline_path),
        "current_design_status": pointer["status"],
        "profiles": {name: {"ideal_range_m": valid_range,
                            "manufacturer_url": source}
                     for name, valid_range, source in PROFILES},
        "changed_factor": "Only the actor's valid rendered-depth range "
                          "window changes to each published ideal range. "
                          "Keep the 70-degree virtual camera, 1cm noise, "
                          "15% holes, current delivery, full low/mid range "
                          "loop and owner task unchanged.",
        "scope": "Matched three near-fast full owner episodes. This is "
                 "an optimistic range screen; it does not model each "
                 "device's actual FOV, optics, stereo texture/reflectance, "
                 "exposure, latency, power, package or selected-body mount.",
        "episodes": episodes,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
