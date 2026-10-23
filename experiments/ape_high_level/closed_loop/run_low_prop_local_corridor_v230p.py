"""Check the next 60 cm of route against a fresh low object and stop tail."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from run import ROOT, sha
from run_spec_planar_route_v202p import StatusGrid
import run_low_prop_sensor_join_v206p as v206
import run_low_prop_role_fault_v212p as v212
import run_low_prop_capture_age_v227p as v227
import run_low_prop_capture_corridor_v229p as v229
import run_low_prop_held_pose_v224p as v224


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-low-prop-local-corridor-v230p"
LOOKAHEAD_M = .6


def one(case, prior, vision, exemplar):
    base_plan, old_out = StatusGrid.plan, v227.OUT
    vetoes = []

    def local_corridor_plan(grid, origin_xy, goal_xy):
        result = base_plan(grid, origin_xy, goal_xy)
        requested, _, path = result
        track = grid.track_history[-1]
        center, radius = track["center_xy_m"], track["radius_m"]
        if (requested is not None and path and center is not None and
                radius is not None and track["status"] in
                ("tracked", "shape_confirmed")):
            cells = [grid.xy(cell) for cell in path
                     if np.linalg.norm(grid.xy(cell) - origin_xy) <= LOOKAHEAD_M]
            if cells:
                minimum = min(float(np.linalg.norm(point - np.asarray(center)))
                              for point in cells)
                required = (v229.BODY_RADIUS_M + float(radius) +
                            v229.REQUIRED_GAP_M + v229.HISTORICAL_STOP_PATH_M)
                if minimum < required:
                    result = (None, "ask_owner_clear_corridor", [])
                    vetoes.append({
                        "scan_index": len(grid.track_history) - 1,
                        "time_s": round(float(track["capture_time_s"]), 4),
                        "track_status": track["status"],
                        "track_center_xy_m": center,
                        "track_radius_m": radius,
                        "minimum_lookahead_center_separation_m": minimum,
                        "required_separation_m": required,
                        "lookahead_m": LOOKAHEAD_M,
                        "decision": result[1],
                    })
        return result

    StatusGrid.plan = local_corridor_plan
    v227.OUT = OUT
    try:
        result = v227.one(case, "capture_age_gate", prior, vision, exemplar)
    finally:
        StatusGrid.plan = base_plan
        v227.OUT = old_out
    result["low_corridor_vetoes"] = vetoes
    result["capture_age_profile"] = "capture_age_and_local_corridor"
    return result


def main():
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Reference-body status changed")
    native = json.loads(v229.NATIVE_STOP_REPORT.read_text())
    if (native["body_status"] != "historical_rejected_direct_frame" or
            native["summary"]["path_to_8s_m_range"][1] >
            v229.HISTORICAL_STOP_PATH_M):
        raise AssertionError("Historical stop-path source changed")
    prior = np.load(v206.dynamic.PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    old_window, old_margin = v212.WINDOW_S, v206.route.MARGIN_M
    v212.WINDOW_S = (999., 1000.)
    v206.route.MARGIN_M = .15
    vision = v206.dynamic.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([
        Image.open(v206.dynamic.owner.TEACH_CROP).convert("RGB")])[0]
    episodes = []
    try:
        for case in v224.CASES:
            result = one(case, prior, vision, exemplar)
            episodes.append(result)
            print(case["id"], result["route_state"], result["outcome"],
                  "low_vetoes", len(result["low_corridor_vetoes"]),
                  "min_gap", round(result["route_episode"][
                      "min_true_prop_gap_m"], 3), flush=True)
    finally:
        v212.WINDOW_S = old_window
        v206.route.MARGIN_M = old_margin
    report = {
        "schema": "robot-learning-lab.ape-low-prop-local-corridor.v1",
        "source_sha256": sha(SOURCE),
        "v227_source_sha256": sha(Path(v227.__file__)),
        "v229_source_sha256": sha(Path(v229.__file__)),
        "historical_stop_report_sha256": sha(v229.NATIVE_STOP_REPORT),
        "current_design_status": pointer["status"],
        "delay_window_s": v227.DELAY_WINDOW_S,
        "age_limit_s": v227.AGE_LIMIT_S,
        "fresh_release_packets": v227.FRESH_RELEASE_PACKETS,
        "low_route_reserve": {"lookahead_m": LOOKAHEAD_M,
                              "body_radius_m": v229.BODY_RADIUS_M,
                              "required_gap_m": v229.REQUIRED_GAP_M,
                              "historical_stop_path_m": v229.HISTORICAL_STOP_PATH_M},
        "changed_factor": "V229p's entire-route low-object veto is "
                          "restricted to the next60cm of planned travel. "
                          "Age-gate, source roles, native stop proxy and "
                          "owner turn are unchanged.",
        "scope": "Complete paired near-fast return, one-way and "
                 "stationary owner episodes with actually delayed "
                 "400ms low source. Ideal rays, kinematic historical "
                 "body and unqualified stopping remain.",
        "episodes": episodes,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
