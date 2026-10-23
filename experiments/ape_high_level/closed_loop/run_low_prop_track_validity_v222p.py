"""Hold a route when a previously coherent low track loses motion validity.

The fast-crossing V220p scene exposed that valid low rays can still yield no
usable object velocity. This actor revision keeps the V216p coverage rule and
adds a bounded, route-proximate track-validity hold. Evaluator prop position is
never read by the rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from run import ROOT, sha
from run_spec_planar_route_v202p import StatusGrid
import run_low_prop_sensor_join_v206p as v206
import run_low_prop_role_fault_v212p as v212
import run_low_prop_coverage_recover_v216p as v216
import run_low_prop_fast_crossing_v220p as v220


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-low-prop-track-validity-v222p"
SPEED_BOUND_M_S = .45
REACH_HORIZON_S = 1.2
MAX_REOBSERVE_S = 5.2
FRESH_STABLE_PACKETS = 3
QUIET_SPEED_M_S = .05
PATH_CLEAR_GAP_M = .08
BODY_RADIUS_M = .17
PROP_RADIUS_M = .24
MATRIX = (
    (v220.CASES[0], "rear_occluded_control"),
    (v220.CASES[1], "rear_occluded_control"),
    ({"id": "stationary_control", "from_xy_m": [3.0, -1.05],
      "to_xy_m": [3.0, -1.05], "begin_s": 10.4, "end_s": 13.6,
      "waypoints": [[10.4, 3.0, -1.05], [13.6, 3.0, -1.05]]},
     "rear_occluded_control"),
)


def one(case, fault, prior, vision, exemplar):
    original_plan, old_out = StatusGrid.plan, v216.OUT
    state = {"last_track": None, "pending_since_s": None, "fresh_stable": 0}
    decisions = []

    def validity_plan(grid, origin_xy, goal_xy):
        result = original_plan(grid, origin_xy, goal_xy)
        requested, _, path = result
        track = grid.track_history[-1]
        time_s = float(track["capture_time_s"])
        velocity = track["velocity_xy_m_s"]
        center = track["center_xy_m"]
        path_points = [grid.xy(cell) for cell in path]
        def nearest_path(point):
            return (min(float(np.linalg.norm(np.asarray(point) - xy))
                        for xy in path_points) if path_points else float("inf"))

        last_track = state["last_track"]
        reachable_radius = (BODY_RADIUS_M + PROP_RADIUS_M + PATH_CLEAR_GAP_M +
                            SPEED_BOUND_M_S * REACH_HORIZON_S)
        if (state["pending_since_s"] is None and requested is not None and
                last_track is not None and time_s - last_track["time_s"] <= 1.2 and
                track["status"] in ("shape_uncertain", "shape_confirmed") and
                velocity is None and
                nearest_path(last_track["center_xy_m"]) < reachable_radius):
            state["pending_since_s"] = time_s
            state["fresh_stable"] = 0
            event = "track_validity_lost"
        elif state["pending_since_s"] is not None:
            event = "track_validity_wait"
        else:
            event = None

        if state["pending_since_s"] is not None:
            speed = (float(np.linalg.norm(velocity))
                     if velocity is not None else float("inf"))
            static_gap = (nearest_path(center) - BODY_RADIUS_M -
                          float(track["radius_m"])
                          if center is not None and track["radius_m"] is not None
                          else float("-inf"))
            stable = (track["status"] == "tracked" and
                      speed < QUIET_SPEED_M_S and
                      static_gap > PATH_CLEAR_GAP_M)
            state["fresh_stable"] = state["fresh_stable"] + 1 if stable else 0
            if time_s - state["pending_since_s"] > MAX_REOBSERVE_S:
                result = (None, "ask_owner_clear_corridor", [])
                event = "track_validity_timeout_help"
            elif state["fresh_stable"] >= FRESH_STABLE_PACKETS:
                event = "track_validity_release"
                state["pending_since_s"] = None
                state["fresh_stable"] = 0
            else:
                result = (np.asarray(origin_xy).copy(),
                          "hold_reobserve_track_validity", [])
            decisions.append({
                "scan_index": len(grid.track_history) - 1,
                "time_s": round(time_s, 4),
                "event": event,
                "track_status": track["status"],
                "track_speed_m_s": speed if np.isfinite(speed) else None,
                "static_route_gap_m": static_gap if np.isfinite(static_gap) else None,
                "last_tracked_path_distance_m": (
                    nearest_path(last_track["center_xy_m"])
                    if last_track is not None else None),
                "fresh_stable_count": state["fresh_stable"],
                "result": result[1],
            })
        if track["status"] == "tracked" and center is not None:
            state["last_track"] = {"time_s": time_s,
                                   "center_xy_m": center}
        return result

    StatusGrid.plan = validity_plan
    v216.OUT = OUT / f"{case['id']}-{fault}"
    try:
        result = v216.one(case, fault, prior, vision, exemplar)
    finally:
        StatusGrid.plan = original_plan
        v216.OUT = old_out
    result["track_validity_decisions"] = decisions
    return result


def main():
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Reference-body status changed")
    prior = np.load(v206.dynamic.PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    old_window, old_margin = v212.WINDOW_S, v206.route.MARGIN_M
    v212.WINDOW_S = v220.FAULT_WINDOW_S
    v206.route.MARGIN_M = .15
    vision = v206.dynamic.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([
        Image.open(v206.dynamic.owner.TEACH_CROP).convert("RGB")])[0]
    episodes = []
    try:
        for case, fault in MATRIX:
            result = one(case, fault, prior, vision, exemplar)
            episodes.append(result)
            print(case["id"], result["route_state"], result["outcome"],
                  "track_holds", sum(d["event"] in (
                      "track_validity_lost", "track_validity_wait")
                      for d in result["track_validity_decisions"]),
                  "min_gap", round(result["route_episode"][
                      "min_true_prop_gap_m"], 3), flush=True)
    finally:
        v212.WINDOW_S = old_window
        v206.route.MARGIN_M = old_margin
    report = {
        "schema": "robot-learning-lab.ape-low-prop-track-validity.v1",
        "source_sha256": sha(SOURCE),
        "v216_source_sha256": sha(Path(v216.__file__)),
        "v220_source_sha256": sha(Path(v220.__file__)),
        "v220_report_sha256": sha(v220.OUT / "report.json"),
        "current_design_status": pointer["status"],
        "fault_window_s": v220.FAULT_WINDOW_S,
        "actor_rule": {
            "speed_bound_m_s": SPEED_BOUND_M_S,
            "reach_horizon_s": REACH_HORIZON_S,
            "max_reobserve_s": MAX_REOBSERVE_S,
            "fresh_stable_packets": FRESH_STABLE_PACKETS,
            "quiet_speed_m_s": QUIET_SPEED_M_S,
            "path_clear_gap_m": PATH_CLEAR_GAP_M,
        },
        "scope": "Revealed-case actor revision on V220p fast low-prop "
                 "scene. Low-track loss near the planned path requests a "
                 "bounded zero-translation reobserve. Stable recovered "
                 "velocity and route gap release; timeout asks for help. "
                 "Ideal synchronous rays and kinematic 50cm reference body "
                 "remain; no physical stopping or mounted-sensor claim.",
        "episodes": episodes,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
