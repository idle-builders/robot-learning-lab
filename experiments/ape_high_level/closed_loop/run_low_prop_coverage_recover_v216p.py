"""Hold, reobserve low coverage, then resume the same owner route.

No evaluator prop location enters the actor. A fixed source-health loss marks
the route's raw low-ray sector unknown; three later fresh unoccluded packets
are required before a new movement request. The 20 ms guard remains live.
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
OUT = ROOT / "artifacts/ape-robocasa-low-prop-coverage-recover-v216p"
LOOKAHEAD_M = .6
FRESH_CLEAR_PACKETS = 3
MAX_WAIT_S = 10.


def one(case, fault, prior, vision, exemplar):
    original_plan = StatusGrid.plan
    original_out = v212.OUT
    state = {"pending_since_s": None, "blocked_beams": None,
             "fresh_clear": 0, "cached_scan": None, "cached_result": None}
    decisions = []

    def route_plan(grid, origin_xy, goal_xy):
        index = len(grid.track_history) - 1
        if state["cached_scan"] == index:
            return state["cached_result"]
        result = original_plan(grid, origin_xy, goal_xy)
        requested, plan_status, path = result
        raw = (OUT / fault / "raw-low72" / case["id"] /
               f"scan-{index:03d}.npz")
        with np.load(raw, allow_pickle=False) as packet:
            status = packet["status"]
        time_s = index * .4
        event = None
        if state["pending_since_s"] is not None:
            healthy = all(int(status[beam]) != 3
                          for beam in state["blocked_beams"])
            state["fresh_clear"] = (state["fresh_clear"] + 1
                                    if healthy else 0)
            if time_s - state["pending_since_s"] > MAX_WAIT_S:
                result = (None, "low_coverage_timeout_stop", [])
                event = "timeout"
            elif state["fresh_clear"] < FRESH_CLEAR_PACKETS:
                result = (np.asarray(origin_xy).copy(),
                          "hold_reobserve_low_coverage", [])
                event = "fresh_wait" if healthy else "occluded_wait"
            else:
                event = "fresh_release"
                state["pending_since_s"] = None
                state["blocked_beams"] = None
                state["fresh_clear"] = 0
        elif requested is not None:
            for cell in path[1:]:
                point = grid.xy(cell)
                delta = point - origin_xy
                distance = float(np.linalg.norm(delta))
                if distance > LOOKAHEAD_M:
                    break
                bearing = math.atan2(delta[1], delta[0])
                center = int(round((bearing + math.pi) /
                                   (2 * math.pi) * 72)) % 72
                neighbors = tuple((center + offset) % 72
                                  for offset in (-1, 0, 1))
                if all(int(status[beam]) == 3 for beam in neighbors):
                    state["pending_since_s"] = time_s
                    state["blocked_beams"] = neighbors
                    state["fresh_clear"] = 0
                    result = (np.asarray(origin_xy).copy(),
                              "hold_reobserve_low_coverage", [])
                    event = "first_occlusion"
                    break
        if event is not None:
            decisions.append({
                "scan_index": index, "time_s": round(time_s, 4),
                "event": event,
                "fresh_clear_count": state["fresh_clear"],
                "blocked_beams": list(state["blocked_beams"])
                if state["blocked_beams"] is not None else None,
                "result": result[1],
            })
        state["cached_scan"] = index
        state["cached_result"] = result
        return result

    StatusGrid.plan = route_plan
    v212.OUT = OUT
    try:
        result = v212.one(case, fault, prior, vision, exemplar)
    finally:
        StatusGrid.plan = original_plan
        v212.OUT = original_out
    result["coverage_recovery_decisions"] = decisions
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
                  "holds", sum(d["event"] in (
                      "first_occlusion", "occluded_wait", "fresh_wait")
                      for d in result["coverage_recovery_decisions"]),
                  "min_gap", round(result["route_episode"][
                      "min_true_prop_gap_m"], 3), flush=True)
    finally:
        v206.route.MARGIN_M = old_margin
    report = {
        "schema": "robot-learning-lab.ape-low-prop-coverage-recover.v1",
        "source_sha256": sha(SOURCE),
        "v212_source_sha256": sha(Path(v212.__file__)),
        "v206_source_sha256": sha(Path(v206.__file__)),
        "current_design_status": pointer["status"],
        "lookahead_m": LOOKAHEAD_M,
        "fresh_clear_packets": FRESH_CLEAR_PACKETS,
        "maximum_wait_s": MAX_WAIT_S,
        "scope": "Same V212p front/rear low-sector fault and V210p "
                 "mid-map/low-track owner loop. An actor-only zero-translation "
                 "hold reobserves the originally blocked low rays each "
                 "400ms; three fresh status-restored packets release route "
                 "planning. The script has an ideal simultaneous scan and "
                 "20ms mid-height guard, no body stopping dynamics, physical "
                 "mount, articulated owner or voiced request.",
        "episodes": episodes,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
