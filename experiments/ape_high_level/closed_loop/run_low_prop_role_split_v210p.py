"""Use 30 cm returns for map/pose and a separate 10 cm stream for motion.

This changes one role assignment from V208p. The route, owner response,
prop, sensor rays and 20 ms mid-height proximity guard are otherwise shared.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from run import ROOT, sha
import run_low_prop_sensor_join_v206p as v206
import run_low_prop_source_track_v208p as v208


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-low-prop-role-split-v210p"


def mid_only_for_map(scans, beams):
    if len(scans) != 2 or beams != 360 or len(scans[0][0]) != 360:
        raise AssertionError("Expected paired mid A1 / virtual low72 sources")
    return scans[0]


def main():
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Reference-body status changed")
    cases = {case["id"]: case for case in json.loads(v206.dynamic.CONFIG.read_text())["cases"]}
    prior = np.load(v206.dynamic.PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    old_fuse, old_out = v206.fuse, v208.OUT
    old_margin = v206.route.MARGIN_M
    v206.fuse = mid_only_for_map
    v208.OUT = OUT
    v206.route.MARGIN_M = .15
    vision = v206.dynamic.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([
        Image.open(v206.dynamic.owner.TEACH_CROP).convert("RGB")])[0]
    episodes = []
    try:
        for case_id in v206.CASES:
            result = v208.one(cases[case_id], prior, vision, exemplar)
            result["profile"] = "mid_map_low_track"
            episodes.append(result)
            print(case_id, result["route_state"], result["outcome"],
                  "holds", sum(d["decision"] ==
                               "hold_and_reobserve_moving_object"
                               for d in result["decision_history"]),
                  "min_gap", round(result["route_episode"][
                      "min_true_prop_gap_m"], 3), flush=True)
    finally:
        v206.fuse = old_fuse
        v208.OUT = old_out
        v206.route.MARGIN_M = old_margin
    report = {
        "schema": "robot-learning-lab.ape-low-prop-role-split.v1",
        "source_sha256": sha(SOURCE),
        "v206_source_sha256": sha(Path(v206.__file__)),
        "v208_source_sha256": sha(Path(v208.__file__)),
        "config_sha256": sha(v206.dynamic.CONFIG),
        "prior_sha256": sha(v206.dynamic.PRIOR),
        "current_design_status": pointer["status"],
        "changed_factor": "30cm A1-style valid hits only feed planar "
                          "mapping/scan localization; raw 10cm virtual "
                          "72-ray packets only feed the novel low-object "
                          "motion tracker. A missing source cannot be "
                          "inferred from the other height.",
        "scope": "Same three paired V206p low-prop layout4 owner episodes. "
                 "Ideal synchronous scans, kinematic 50cm proxy, scripted "
                 "person/can and unqualified 30cm independent guard; "
                 "source-role ablation, not device qualification.",
        "episodes": episodes,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
