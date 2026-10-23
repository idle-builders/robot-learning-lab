"""Challenge the frozen V216p owner loop with a late, fast low crossing.

The altered prop path is evaluator/world data. Actor map, tracker, coverage
gate, recovery and owner recognition remain unchanged. Front and rear losses
use the same duration, sector width and initial scene.
"""

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
OUT = ROOT / "artifacts/ape-robocasa-low-prop-fast-crossing-v220p"
FAULT_WINDOW_S = (10.8, 12.8)
CASES = (
    {
        "id": "fast_returning_crossing",
        "from_xy_m": [3.0, -1.05],
        "to_xy_m": [3.0, -1.05],
        "begin_s": 10.4,
        "end_s": 13.6,
        "waypoints": [[10.4, 3.0, -1.05], [12.0, 3.0, -1.65],
                      [13.6, 3.0, -1.05]],
    },
    {
        "id": "fast_one_way_crossing",
        "from_xy_m": [3.0, -1.05],
        "to_xy_m": [3.0, -1.65],
        "begin_s": 10.4,
        "end_s": 12.0,
        "waypoints": [[10.4, 3.0, -1.05], [12.0, 3.0, -1.65]],
    },
)
MATRIX = ((case, fault) for case in CASES
          for fault in ("front_occluded", "rear_occluded_control"))


def main():
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Reference-body status changed")
    prior = np.load(v206.dynamic.PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    old_out, old_window, old_margin = v216.OUT, v212.WINDOW_S, v206.route.MARGIN_M
    v212.WINDOW_S = FAULT_WINDOW_S
    v206.route.MARGIN_M = .15
    vision = v206.dynamic.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([
        Image.open(v206.dynamic.owner.TEACH_CROP).convert("RGB")])[0]
    episodes = []
    try:
        for case, fault in MATRIX:
            v216.OUT = OUT / f"{case['id']}-{fault}"
            result = v216.one(case, fault, prior, vision, exemplar)
            result["fault_window_s"] = FAULT_WINDOW_S
            episodes.append(result)
            print(case["id"], fault, result["route_state"], result["outcome"],
                  "coverage_holds", sum(d["event"] in (
                      "first_occlusion", "occluded_wait", "fresh_wait")
                      for d in result["coverage_recovery_decisions"]),
                  "min_gap", round(result["route_episode"][
                      "min_true_prop_gap_m"], 3), flush=True)
    finally:
        v216.OUT = old_out
        v212.WINDOW_S = old_window
        v206.route.MARGIN_M = old_margin
    report = {
        "schema": "robot-learning-lab.ape-low-prop-fast-crossing.v1",
        "source_sha256": sha(SOURCE),
        "v216_source_sha256": sha(Path(v216.__file__)),
        "v212_source_sha256": sha(Path(v212.__file__)),
        "current_design_status": pointer["status"],
        "fault_window_s": FAULT_WINDOW_S,
        "fault_sector_half_width_deg": 50.,
        "changed_factor": "World prop starts at10.4s, traverses60cm by12.0s "
                          "(0.375m/s), then returns by13.6s or remains. "
                          "The front/rear 100-degree low-source occlusion "
                          "lasts10.8–12.8s. V216p actor rules are frozen.",
        "scope": "Same layout4 low-prop route and fresh RGB-D owner turn. "
                 "The scan is ideal and synchronous; the ape is kinematic "
                 "historical reference geometry with no physical stop, "
                 "self-mask, actual sensor mount or selected body.",
        "episodes": episodes,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
