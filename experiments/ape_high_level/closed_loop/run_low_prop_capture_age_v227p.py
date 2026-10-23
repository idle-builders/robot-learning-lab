"""Deliver an actually earlier low scan in the live fast-prop owner task.

The scan rendered at each decision is saved with its true capture pose/time.
From 10.4 to 12.4 s the actor instead receives the preceding 400 ms scan.
The frozen V222p actor is compared with and without a delivery-age hold; the
30 cm map/localizer and owner RGB-D path remain unchanged.
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
import run_low_prop_track_validity_v222p as v222
import run_low_prop_held_pose_v224p as v224


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-low-prop-capture-age-v227p"
DELAY_WINDOW_S = (10.4, 12.4)
LOW_DELIVERY_DELAY_STEPS = 1
AGE_LIMIT_S = .2
FRESH_RELEASE_PACKETS = 3
MAX_AGE_WAIT_S = 5.2
PROFILES = ("naive_delivery", "capture_age_gate")


def one(case, profile, prior, vision, exemplar):
    base_plane, base_plan, old_out = v206.plane, StatusGrid.plan, v222.OUT
    captured, deliveries, gate_events = [], [], []
    age_state = {"age_s": 0., "capture_index": None,
                 "pending_since_s": None, "fresh_count": 0}
    branch = OUT / profile

    def delayed_plane(model, data, xy, height_m, beams, minimum_m, rng):
        values, status = base_plane(model, data, xy, height_m, beams, minimum_m, rng)
        if height_m != .10 or beams != 72:
            return values, status
        index = len(captured)
        now_s = round(index * .4, 4)
        path = branch / "captured-low72" / case["id"] / f"scan-{index:03d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, ranges_m=values, status=status,
                            sample_index=index, capture_time_s=now_s,
                            actual_capture_xy_m=np.asarray(xy))
        captured.append({"path": str(path.relative_to(ROOT)),
                         "sha256": sha(path), "capture_time_s": now_s,
                         "actual_capture_xy_m": np.asarray(xy).tolist()})
        source_index = (index - LOW_DELIVERY_DELAY_STEPS
                        if DELAY_WINDOW_S[0] <= now_s <= DELAY_WINDOW_S[1]
                        and index >= LOW_DELIVERY_DELAY_STEPS else index)
        if source_index == index:
            delivered_values, delivered_status = values, status
        else:
            with np.load(ROOT / captured[source_index]["path"],
                         allow_pickle=False) as packet:
                delivered_values = packet["ranges_m"].copy()
                delivered_status = packet["status"].copy()
        age_s = round(now_s - captured[source_index]["capture_time_s"], 4)
        age_state["age_s"] = age_s
        age_state["capture_index"] = source_index
        deliveries.append({"decision_index": index,
                           "decision_time_s": now_s,
                           "capture_index": source_index,
                           "capture_time_s": captured[source_index]["capture_time_s"],
                           "capture_pose_xy_m": captured[source_index][
                               "actual_capture_xy_m"],
                           "age_s": age_s})
        return delivered_values, delivered_status

    def age_plan(grid, origin_xy, goal_xy):
        result = base_plan(grid, origin_xy, goal_xy)
        if profile != "capture_age_gate":
            return result
        index = len(grid.track_history) - 1
        now_s = round(index * .4, 4)
        stale = age_state["age_s"] > AGE_LIMIT_S
        event = None
        if stale and age_state["pending_since_s"] is None:
            age_state["pending_since_s"] = now_s
            age_state["fresh_count"] = 0
            event = "first_stale"
        elif age_state["pending_since_s"] is not None:
            event = "stale_wait" if stale else "fresh_wait"
        if age_state["pending_since_s"] is not None:
            age_state["fresh_count"] = (0 if stale else
                                        age_state["fresh_count"] + 1)
            if now_s - age_state["pending_since_s"] > MAX_AGE_WAIT_S:
                result = (None, "ask_owner_clear_corridor", [])
                event = "age_timeout_help"
            elif age_state["fresh_count"] >= FRESH_RELEASE_PACKETS:
                event = "fresh_release"
                age_state["pending_since_s"] = None
                age_state["fresh_count"] = 0
            else:
                result = (np.asarray(origin_xy).copy(),
                          "hold_reobserve_stale_low_scan", [])
            gate_events.append({"scan_index": index, "time_s": now_s,
                                "event": event,
                                "capture_index": age_state["capture_index"],
                                "age_s": age_state["age_s"],
                                "fresh_count": age_state["fresh_count"],
                                "result": result[1]})
        return result

    v206.plane = delayed_plane
    StatusGrid.plan = age_plan
    v222.OUT = branch
    try:
        result = v222.one(case, "rear_occluded_control", prior, vision, exemplar)
    finally:
        v206.plane = base_plane
        StatusGrid.plan = base_plan
        v222.OUT = old_out
    if len(captured) != len(result["route_episode"]["trace"]):
        raise AssertionError("Capture log lost route decision alignment")
    result["capture_age_profile"] = profile
    result["captured_low_packets"] = captured
    result["low_deliveries"] = deliveries
    result["capture_age_gate_events"] = gate_events
    return result


def main():
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Reference-body status changed")
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
            for profile in PROFILES:
                result = one(case, profile, prior, vision, exemplar)
                episodes.append(result)
                print(case["id"], profile, result["route_state"],
                      result["outcome"], "age_holds", sum(
                          event["event"] in ("first_stale", "stale_wait",
                                             "fresh_wait")
                          for event in result["capture_age_gate_events"]),
                      "min_gap", round(result["route_episode"][
                          "min_true_prop_gap_m"], 3), flush=True)
    finally:
        v212.WINDOW_S = old_window
        v206.route.MARGIN_M = old_margin
    report = {
        "schema": "robot-learning-lab.ape-low-prop-capture-age.v1",
        "source_sha256": sha(SOURCE),
        "v222_source_sha256": sha(Path(v222.__file__)),
        "v224_source_sha256": sha(Path(v224.__file__)),
        "current_design_status": pointer["status"],
        "delay_window_s": DELAY_WINDOW_S,
        "low_delivery_delay_steps": LOW_DELIVERY_DELAY_STEPS,
        "age_limit_s": AGE_LIMIT_S,
        "fresh_release_packets": FRESH_RELEASE_PACKETS,
        "maximum_age_wait_s": MAX_AGE_WAIT_S,
        "changed_factor": "During10.4–12.4s return the previous actual "
                          "low scan, captured400ms earlier from its own "
                          "actual body pose; preserve mid scan and owner "
                          "images. Compare frozen V222p to an actor-side "
                          "age hold with three fresh packets to release.",
        "scope": "Paired delayed-delivery stress in the same three V224p "
                 "near low-prop owner episodes. Capture-time rays are real "
                 "MuJoCo raycasts at the earlier prop/body pose but device "
                 "latency, optics and kinematic stop remain proxies.",
        "episodes": episodes,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
