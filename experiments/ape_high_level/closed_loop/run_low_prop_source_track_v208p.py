"""Track low returns by source rather than pooling them with a mid scan.

Only V206p's NovelRangeObject input changes. Navigation mapping/localization,
owner interaction, world, prop path and the independent 30 cm guard remain.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import novel_range_object as tracking
from run import ROOT, sha
import run_low_prop_sensor_join_v206p as v206


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-low-prop-source-track-v208p"
PROFILE = "mid_a1_low72_virtual"


def one(case, prior, vision, exemplar):
    low_packets = []
    state = {"low": None, "scan_index": -1}
    original_plane = v206.plane
    original_update = tracking.NovelRangeObject.update
    original_out = v206.OUT

    def saved_plane(model, data, xy, height_m, beams, minimum_m, rng):
        values, statuses = original_plane(
            model, data, xy, height_m, beams, minimum_m, rng)
        if height_m == .10 and beams == 72:
            index = len(low_packets)
            path = OUT / "raw-low72" / case["id"] / f"scan-{index:03d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, ranges_m=values, status=statuses,
                                sample_index=index)
            low_packets.append({"path": str(path.relative_to(ROOT)),
                                "sha256": sha(path)})
            state["low"] = [float(value) if status == 0 else None
                            for value, status in zip(values, statuses)]
            state["scan_index"] = index
        return values, statuses

    def separate_track(tracker, pooled_ranges, pose_xy, capture_time_s):
        if state["low"] is None or len(pooled_ranges) != 360:
            raise AssertionError("Raw low scan missing at tracker update")
        if abs(capture_time_s - state["scan_index"] * .4) > 1e-5:
            raise AssertionError("Tracker capture lost low source alignment")
        return original_update(tracker, state["low"], pose_xy,
                               capture_time_s)

    v206.plane = saved_plane
    v206.OUT = OUT
    tracking.NovelRangeObject.update = separate_track
    try:
        result = v206.one(PROFILE, case, prior, vision, exemplar)
    finally:
        v206.plane = original_plane
        v206.OUT = original_out
        tracking.NovelRangeObject.update = original_update
    if len(low_packets) != len(result["route_episode"]["trace"]):
        raise AssertionError("Raw low source missing from some route packets")
    result["profile"] = "mid_a1_low72_source_track"
    result["sensor_profile"] = PROFILE
    result["raw_low_source_packets"] = low_packets
    return result


def main():
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Reference-body status changed")
    cases = {case["id"]: case for case in json.loads(v206.dynamic.CONFIG.read_text())["cases"]}
    prior = np.load(v206.dynamic.PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    original_margin = v206.route.MARGIN_M
    v206.route.MARGIN_M = .15
    vision = v206.dynamic.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([
        Image.open(v206.dynamic.owner.TEACH_CROP).convert("RGB")])[0]
    episodes = []
    try:
        for case_id in v206.CASES:
            result = one(cases[case_id], prior, vision, exemplar)
            episodes.append(result)
            print(case_id, result["route_state"], result["outcome"],
                  "holds", sum(d["decision"] ==
                               "hold_and_reobserve_moving_object"
                               for d in result["decision_history"]),
                  "min_gap", round(result["route_episode"][
                      "min_true_prop_gap_m"], 3), flush=True)
    finally:
        v206.route.MARGIN_M = original_margin
    report = {
        "schema": "robot-learning-lab.ape-low-prop-source-track.v1",
        "source_sha256": sha(SOURCE),
        "v206_source_sha256": sha(Path(v206.__file__)),
        "config_sha256": sha(v206.dynamic.CONFIG),
        "prior_sha256": sha(v206.dynamic.PRIOR),
        "current_design_status": pointer["status"],
        "changed_factor": "NovelRangeObject receives raw 10cm 72-ray "
                          "virtual low scan instead of minimum-range fusion "
                          "with the 30cm A1-style plane. Route map and "
                          "localization still consume the fused 360 rays.",
        "scope": "Three paired V206p layout4 low-prop episodes, same seeded "
                 "stationary/returning/one-way paths and compliant scripted "
                 "owner. This is source-separated algorithm diagnosis, not "
                 "physical sensor, gait or stop qualification.",
        "episodes": episodes,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
