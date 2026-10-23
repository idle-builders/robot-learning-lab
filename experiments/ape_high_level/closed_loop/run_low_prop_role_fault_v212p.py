"""Drop a predeclared low-scan sector during a moving owner route.

The V210p mid-map/low-track actor is frozen. This injects low-source optical
occlusion only, preserving the mid scan, route code and owner response.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from run import ROOT, sha
import run_low_prop_sensor_join_v206p as v206
import run_low_prop_source_track_v208p as v208
import run_low_prop_role_split_v210p as v210


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-low-prop-role-fault-v212p"
WINDOW_S = (7.2, 14.0)
SECTOR_HALF_WIDTH_RAD = math.radians(50.)
MATRIX = (("returning_crossing", "front_occluded"),
          ("returning_crossing", "rear_occluded_control"),
          ("one_way_crossing", "front_occluded"))


def one(case, fault, prior, vision, exemplar):
    original_plane = v206.plane
    original_fuse = v206.fuse
    original_out = v208.OUT
    index = {"low": 0}
    fault_log = []

    def fault_plane(model, data, xy, height_m, beams, minimum_m, rng):
        values, status = original_plane(
            model, data, xy, height_m, beams, minimum_m, rng)
        if height_m == .10 and beams == 72:
            sample = index["low"]
            index["low"] += 1
            time_s = sample * .4
            angles = -math.pi + 2 * math.pi * np.arange(beams) / beams
            if fault == "front_occluded":
                sector = np.abs(angles) <= SECTOR_HALF_WIDTH_RAD
            else:
                sector = (math.pi - np.abs(angles)) <= SECTOR_HALF_WIDTH_RAD
            active = WINDOW_S[0] <= time_s <= WINDOW_S[1]
            masked = sector if active else np.zeros(beams, dtype=bool)
            values[masked] = np.nan
            status[masked] = 3  # optical occlusion: unmeasured, not free
            fault_log.append({"sample_index": sample,
                              "decision_time_s": round(time_s, 4),
                              "occluded_rays": int(np.count_nonzero(masked))})
        return values, status

    v206.plane = fault_plane
    v206.fuse = v210.mid_only_for_map
    v208.OUT = OUT / fault
    try:
        result = v208.one(case, prior, vision, exemplar)
    finally:
        v206.plane = original_plane
        v206.fuse = original_fuse
        v208.OUT = original_out
    result["profile"] = "mid_map_low_track_" + fault
    result["fault"] = fault
    result["fault_log"] = fault_log
    if len(fault_log) != len(result["route_episode"]["trace"]):
        raise AssertionError("Fault schedule lost packet alignment")
    return result


def main():
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Reference-body status changed")
    cases = {case["id"]: case for case in json.loads(v206.dynamic.CONFIG.read_text())["cases"]}
    prior = np.load(v206.dynamic.PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    old_margin = v206.route.MARGIN_M
    v206.route.MARGIN_M = .15
    vision = v206.dynamic.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([
        Image.open(v206.dynamic.owner.TEACH_CROP).convert("RGB")])[0]
    episodes = []
    try:
        for case_id, fault in MATRIX:
            result = one(cases[case_id], fault, prior, vision, exemplar)
            episodes.append(result)
            print(case_id, fault, result["route_state"], result["outcome"],
                  "holds", sum(d["decision"] ==
                               "hold_and_reobserve_moving_object"
                               for d in result["decision_history"]),
                  "min_gap", round(result["route_episode"][
                      "min_true_prop_gap_m"], 3), flush=True)
    finally:
        v206.route.MARGIN_M = old_margin
    report = {
        "schema": "robot-learning-lab.ape-low-prop-role-fault.v1",
        "source_sha256": sha(SOURCE),
        "v210_source_sha256": sha(Path(v210.__file__)),
        "v208_source_sha256": sha(Path(v208.__file__)),
        "v206_source_sha256": sha(Path(v206.__file__)),
        "current_design_status": pointer["status"],
        "fault_window_s": WINDOW_S,
        "fault_sector_half_width_deg": 50.,
        "fault_status_code": 3,
        "scope": "Same V210p role-split low-prop owner loop. 10cm "
                 "virtual 72-ray source has a fixed front or matched rear "
                 "100-degree no-measurement sector only from7.2 to14.0s. "
                 "The 30cm map scan remains healthy. The sector is a "
                 "predeclared correlated occlusion proxy, not an actual "
                 "self-mask or device failure rate. The actor still lacks "
                 "an explicit low-source coverage/age gate.",
        "episodes": episodes,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
