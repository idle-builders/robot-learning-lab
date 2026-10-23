"""Continue multi-object range sensing while the ape holds through motion.

This is a packet replay, not the original V485 episode continued in place.
It rebuilds the seeded scene, replays saved actual and estimated route poses
to reconstruct track state, then holds those poses and samples every 0.4 s.
Evaluator object positions score identities but never enter the tracker.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from multi_novel_range_objects import MultiNovelRangeObjects
import run_robocasa_layout2_two_props_v482 as two
from run_robocasa_layout2_route_v407 import sha256
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-robocasa-layout2-multitrack-v485/report.json"
PRIOR = ROOT / "artifacts/ape-robocasa-layout2-survey-v477/sensor-derived-map.npz"
OUT = ROOT / "artifacts/ape-robocasa-layout2-multitrack-v485"
SCAN = height_scan((.05, .30))


def replay(case, prior):
    scenario, episode = case["scenario"], case["episode"]
    two.ACTIVE_CASE = scenario
    model = two.scene_with_two_props(obstacle_xy=scenario["primary_xy_m"])
    data = mujoco.MjData(model)
    primary_body = model.body("route_obstacle").id
    secondary_body = model.body("second_obstacle").id
    rng = np.random.default_rng(423)
    tracker = MultiNovelRangeObjects(prior["xs"], prior["ys"],
                                     prior["free"], prior["occupied"])
    rows = []

    def observe(time_s, actual, estimate, phase):
        model.body_pos[primary_body, :2] = scenario["primary_xy_m"]
        model.body_pos[secondary_body, :2] = two.secondary_xy(scenario, time_s)
        mujoco.mj_forward(model, data)
        ranges = SCAN(model, data, np.asarray(actual), 5., rng)
        result = tracker.update(ranges, np.asarray(estimate), time_s)
        truth = {"primary": np.asarray(scenario["primary_xy_m"]),
                 "secondary": two.secondary_xy(scenario, time_s)}
        matches = {}
        for label, center in truth.items():
            candidates = [track for track in result["tracks"] if track["hits"] >= 2]
            if candidates:
                nearest = min(candidates,
                              key=lambda track: np.linalg.norm(np.asarray(
                                  track["center_xy_m"]) - center))
                matches[label] = {"id": nearest["id"],
                                  "error_m": float(np.linalg.norm(
                                      np.asarray(nearest["center_xy_m"]) - center)),
                                  "status": nearest["status"],
                                  "age_s": nearest["age_s"]}
        rows.append({"time_s": round(time_s, 3), "phase": phase,
                     "actual_xy_m": list(actual), "estimated_xy_m": list(estimate),
                     "range_returns": sum(v is not None for v in ranges),
                     "belief": result, "evaluator_matches": matches})

    for route_row in episode["trace"]:
        observe(float(route_row["time_s"]), route_row["actual_xy_m"],
                route_row["estimated_xy_m"], "route_capture")
    actual = episode["end_true_xy_m"]
    estimate = episode["end_estimated_xy_m"]
    for time_s in np.arange(episode["end_time_s"] + .4,
                            scenario["end_s"] + .401, .4):
        observe(float(time_s), actual, estimate, "post_stop_hold")
    return {"scenario_id": scenario["id"], "route_end_s": episode["end_time_s"],
            "hold_end_s": rows[-1]["time_s"], "rows": rows}


def main():
    source = json.loads(SOURCE.read_text())
    prior = np.load(PRIOR)
    cases = [replay(case, prior) for case in source["cases"][1:]]

    def same_packet_belief(expected, replayed):
        if abs(expected["capture_time_s"] - replayed["capture_time_s"]) > 1e-9:
            return False
        if len(expected["tracks"]) != len(replayed["tracks"]):
            return False
        for left, right in zip(expected["tracks"], replayed["tracks"]):
            if left["id"] != right["id"] or left["status"] != right["status"]:
                return False
            if np.linalg.norm(np.asarray(left["center_xy_m"]) -
                              np.asarray(right["center_xy_m"])) > 1e-9:
                return False
            if abs(left["radius_m"] - right["radius_m"]) > 1e-9:
                return False
        return True

    route_matches = []
    for original, result in zip(source["cases"][1:], cases):
        packet_rows = result["rows"][:len(original["multi_history"])]
        route_matches.append(all(same_packet_belief(left, right["belief"])
                                 for left, right in zip(original["multi_history"],
                                                        packet_rows)))
    by_id = {case["scenario_id"]: case for case in cases}
    south = by_id["second_south_toward_route"]["rows"][-1]
    north = by_id["second_north_away"]["rows"][-1]
    south_match = south["evaluator_matches"]["secondary"]
    north_match = north["evaluator_matches"]["secondary"]
    output = {"source_sha256": sha256(Path(__file__)),
              "trial_report_sha256": sha256(SOURCE),
              "tracker_source_sha256": sha256(Path(__file__).with_name(
                  "multi_novel_range_objects.py")),
              "scope": "new seeded range-packet replay of V485 saved route poses followed by fixed-pose 0.4s sensing through second-prop motion; evaluator labels withheld from tracker",
              "route_packet_replay_pass": all(route_matches),
              "south_mover_current_within_10cm_pass": (
                  south_match["id"] == 2 and south_match["status"] == "confirmed_now"
                  and south_match["error_m"] <= .10),
              "north_away_marked_last_seen_pass": (
                  north_match["id"] == 2 and north_match["status"] == "last_seen"
                  and north_match["age_s"] >= 1.),
              "cases": cases}
    (OUT / "hold-sensing-v487.json").write_text(json.dumps(output, indent=2) + "\n")
    for case in cases:
        late = case["rows"][-1]
        tracks = [(track["id"], track["status"], round(track["age_s"], 1),
                   [round(x, 2) for x in track["center_xy_m"]])
                  for track in late["belief"]["tracks"]]
        print(case["scenario_id"], "end", late["time_s"],
              "tracks", tracks, "match", late["evaluator_matches"], flush=True)


if __name__ == "__main__":
    main()
