"""Stop a route edge whose low-source corridor rays are occluded.

The V212p faulted role-split actor is frozen except for this actor-side
pre-motion coverage gate. The gate reads only its saved low status packet and
the next route cells; it receives no prop truth or evaluator clearance.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from run import ROOT, sha
from run_spec_planar_route_v202p import StatusGrid
import run_low_prop_sensor_join_v206p as v206
import run_low_prop_role_fault_v212p as v212


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-low-prop-coverage-gate-v214p"
CORRIDOR_LOOKAHEAD_M = .6
RAY_NEIGHBORS = (-1, 0, 1)


def one(case, fault, prior, vision, exemplar):
    original_plan = StatusGrid.plan
    original_out = v212.OUT
    gate_log = []

    def covered_plan(grid, origin_xy, goal_xy):
        requested, status, path = original_plan(grid, origin_xy, goal_xy)
        if requested is None:
            return requested, status, path
        scan_index = len(grid.track_history) - 1
        packet = (OUT / fault / "raw-low72" / case["id"] /
                  f"scan-{scan_index:03d}.npz")
        with np.load(packet, allow_pickle=False) as data:
            beam_status = data["status"]
        blocked = []
        for cell in path[1:]:
            point = grid.xy(cell)
            delta = point - origin_xy
            distance = float(np.linalg.norm(delta))
            if distance > CORRIDOR_LOOKAHEAD_M:
                break
            angle = math.atan2(delta[1], delta[0])
            center = int(round((angle + math.pi) / (2 * math.pi) * 72)) % 72
            neighbors = [int(beam_status[(center + offset) % 72])
                         for offset in RAY_NEIGHBORS]
            if all(code == 3 for code in neighbors):
                blocked.append({"cell": [int(cell[0]), int(cell[1])],
                                "bearing_rad": round(angle, 5),
                                "distance_m": round(distance, 5),
                                "statuses": neighbors})
        if blocked:
            if not gate_log or gate_log[-1]["scan_index"] != scan_index:
                gate_log.append({
                    "scan_index": scan_index,
                    "time_s": round(scan_index * .4, 4),
                    "estimated_origin_xy_m": np.asarray(origin_xy).tolist(),
                    "proposed_xy_m": np.asarray(requested).tolist(),
                    "first_occluded_cell": blocked[0],
                    "decision": "low_source_occluded_stop",
                })
            return None, "low_source_occluded_stop", []
        return requested, status, path

    StatusGrid.plan = covered_plan
    v212.OUT = OUT
    try:
        result = v212.one(case, fault, prior, vision, exemplar)
    finally:
        StatusGrid.plan = original_plan
        v212.OUT = original_out
    result["coverage_gate_log"] = gate_log
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
        for case_id, fault in v212.MATRIX:
            result = one(cases[case_id], fault, prior, vision, exemplar)
            episodes.append(result)
            print(case_id, fault, result["route_state"], result["outcome"],
                  "gate", len(result["coverage_gate_log"]),
                  "min_gap", round(result["route_episode"][
                      "min_true_prop_gap_m"], 3), flush=True)
    finally:
        v206.route.MARGIN_M = old_margin
    report = {
        "schema": "robot-learning-lab.ape-low-prop-coverage-gate.v1",
        "source_sha256": sha(SOURCE),
        "v212_source_sha256": sha(Path(v212.__file__)),
        "v206_source_sha256": sha(Path(v206.__file__)),
        "current_design_status": pointer["status"],
        "corridor_lookahead_m": CORRIDOR_LOOKAHEAD_M,
        "neighbor_beam_offsets": RAY_NEIGHBORS,
        "changed_factor": "Before the next map-derived step, inspect the "
                          "raw low-source status for bearings of route "
                          "cells within60cm. If all three adjacent beams "
                          "are explicitly optically occluded, refuse the "
                          "step with low_source_occluded_stop.",
        "scope": "Same V212p fixed 100-degree front/rear low-source "
                 "occlusion, owner response and kinematic 50cm proxy. This "
                 "gate only tests explicit status3 optical loss, not "
                 "ordinary no-return, pose uncertainty, all body-height "
                 "coverage or a physical stop response.",
        "episodes": episodes,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
