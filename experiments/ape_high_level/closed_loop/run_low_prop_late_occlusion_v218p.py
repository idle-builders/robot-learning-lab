"""Freeze V216p and shift the low-view fault toward the prop crossing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from run import ROOT, sha
import run_low_prop_sensor_join_v206p as v206
import run_low_prop_role_fault_v212p as v212
import run_low_prop_coverage_recover_v216p as v216


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-low-prop-late-occlusion-v218p"
LATE_WINDOW_S = (10.8, 14.0)
MATRIX = (("returning_crossing", "front_occluded"),
          ("returning_crossing", "rear_occluded_control"),
          ("one_way_crossing", "front_occluded"))


def main():
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Reference-body status changed")
    cases = {case["id"]: case for case in json.loads(v206.dynamic.CONFIG.read_text())["cases"]}
    prior = np.load(v206.dynamic.PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    old_out, old_window, old_margin = v216.OUT, v212.WINDOW_S, v206.route.MARGIN_M
    v212.WINDOW_S = LATE_WINDOW_S
    v206.route.MARGIN_M = .15
    vision = v206.dynamic.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([
        Image.open(v206.dynamic.owner.TEACH_CROP).convert("RGB")])[0]
    episodes = []
    try:
        for case_id, fault in MATRIX:
            v216.OUT = OUT / f"{case_id}-{fault}"
            result = v216.one(cases[case_id], fault, prior, vision, exemplar)
            result["fault_window_s"] = LATE_WINDOW_S
            episodes.append(result)
            print(case_id, fault, result["route_state"], result["outcome"],
                  "holds", sum(d["event"] in (
                      "first_occlusion", "occluded_wait", "fresh_wait")
                      for d in result["coverage_recovery_decisions"]),
                  "min_gap", round(result["route_episode"][
                      "min_true_prop_gap_m"], 3), flush=True)
    finally:
        v216.OUT = old_out
        v212.WINDOW_S = old_window
        v206.route.MARGIN_M = old_margin
    report = {
        "schema": "robot-learning-lab.ape-low-prop-late-occlusion.v1",
        "source_sha256": sha(SOURCE),
        "v216_source_sha256": sha(Path(v216.__file__)),
        "v212_source_sha256": sha(Path(v212.__file__)),
        "current_design_status": pointer["status"],
        "fault_window_s": LATE_WINDOW_S,
        "fault_sector_half_width_deg": 50.,
        "changed_factor": "Only V212p's predeclared low-source sector "
                          "occlusion shifts from7.2–14.0s to10.8–14.0s; "
                          "V216p's 60cm route coverage and three-packet "
                          "recovery rule stays frozen.",
        "scope": "Same layout4 low-prop single-process owner turn and "
                 "historical kinematic 50cm proxy. This is a held-out fault "
                 "timing stress, not physical safety or scan-timing proof.",
        "episodes": episodes,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
