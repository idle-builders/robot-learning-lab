"""Independent actor-choice, sensor-clock, geometry and media audit for V59."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import audit_owner_fixed_clock_pair as matched
import audit_owner_low_prop_scan_planes as sensor_audit
import owner_approach_aware_meeting as trial


def choice_from_rows(rows):
    hold = [row for row in rows if row["stage"] == "post_alarm_range_watch"]
    steps = [row for row in rows if row["stage"] == "pre_show_restage_step"]
    assert hold and steps and len(steps) == 24
    initial = np.asarray(steps[0]["ape_xy_m"])+np.array([.025, 0.])
    first, last = hold[0]["actor_packet"], hold[-1]["actor_packet"]
    moving = []
    for ray in range(45, 62):
        if first[ray] is None or last[ray] is None or \
                last[ray] <= 1.5 or first[ray]-last[ray] < .5:
            continue
        angle = math.radians(-180.+5.*ray)
        moving.append((ray, initial+last[ray]*np.array([
            math.cos(angle), math.sin(angle)])))
    assert len(moving) == 1 and moving[0][0] == 53
    recorded = steps[0]["actor_decision"]
    assert len(recorded["dynamic_surfaces_from_hold"]) == 1
    assert np.linalg.norm(np.asarray(recorded["dynamic_surfaces_from_hold"][0]
                                     ["surface_xy_m"])-moving[0][1]) < 1e-8
    endpoints = trial.v51.v41.JointHazardResponse.endpoints(last, initial)
    independent = []
    for name, direction in trial.v56.DIRECTIONS.items():
        for distance in trial.v56.DISTANCES_M:
            target = initial+distance*direction
            path_gap = min(float(np.min(np.linalg.norm(
                endpoints-(initial+u*direction), axis=1)))
                for u in np.linspace(0., distance, 15))
            reserve = float(np.min(np.linalg.norm(endpoints-target, axis=1)))
            owner_range = float(np.linalg.norm(
                trial.v56.OWNER_SHOW_STATION_XY_M-target))
            eligible = (path_gap >= .27 and reserve >= .27 and
                        owner_range <= 2.6)
            risk = []
            for _, surface in moving:
                for elapsed in np.arange(0., distance/.25+3.5+.05, .1):
                    pose = initial+min(elapsed*.25/distance, 1.)*(target-initial)
                    lateral = max(abs(pose[0]-surface[0])-.4, 0.)
                    longitudinal = max(surface[1]-pose[1]-1.2*elapsed,
                                       pose[1]-surface[1], 0.)
                    risk.append(math.hypot(lateral, longitudinal)-.17)
            score = min(reserve-.17, min(risk))
            independent.append((name, distance, score, eligible))
    eligible = [row for row in independent if row[-1]]
    expected = max(eligible, key=lambda row: (row[2], -row[1]))
    assert expected[:2] == ("west", .60)
    assert recorded["choice"] == "west_60cm"
    assert abs(recorded["time_indexed_choice_margin_m"]-expected[2]) < 1e-8
    candidates = recorded["candidate_scores"]
    assert candidates is not None and len(candidates) == len(independent)
    for row, (name, distance, score, eligible) in zip(candidates, independent):
        assert row["name"] == name and row["distance_m"] == distance
        assert row["eligible"] == eligible
        assert abs(row["actor_combined_margin_m"]-score) < 1e-8
    last_t = hold[-1]["time_s"]
    for index, step in enumerate(steps):
        assert abs(step["time_s"]-(last_t+(index+1)*.1)) < 1e-8
        assert np.linalg.norm(np.asarray(step["ape_xy_m"])-
                              (initial+np.array([-.025*(index+1), 0.]))) < 1e-8
        assert step["actor_decision"]["packet_time_s"] == \
            round(step["time_s"]-.1, 5)
    return {"moving_surface_ray": moving[0][0],
            "choice": recorded["choice"],
            "actor_combined_margin_m": round(expected[2], 5),
            "restage_steps": len(steps)}


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    assert report["schema"] == "robot-learning-lab.ape-approach-aware-meeting.v1"
    for key, source in (("source_sha256", trial.__file__),
                        ("runner_source_sha256", trial.runner.__file__),
                        ("v56_report_sha256", trial.v56.OUT / "report.json"),
                        ("scene_sha256", trial.runner.v35.moving.SCENE_XML),
                        ("current_design_pointer_sha256",
                         trial.ROOT / "robot/ape-current-design.json")):
        assert report[key] == trial.v52.sha(source)
    source = trial.runner.original()
    policy = json.loads(trial.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    fixture = json.loads((trial.v56.OUT / "audit.json").read_text())
    assert fixture["passed"]
    result_rows = []
    original_paths = dict(trial.v50.PATHS)
    try:
        trial.v50.PATHS.update(trial.v56.PATHS)
        for item in report["results"]:
            run = item["run"]
            rows = json.loads(Path(run["trace"]["path"]).read_text())
            scores = choice_from_rows(rows)
            alarm = next((row for row in rows if row["stage"] ==
                          "near_body_escape_plan"), None)
            commanded = sum(row["near_body_command"]["commanded_step_m"]
                            for row in rows if row["stage"] ==
                            "second_inference_range_watch" and
                            row["near_body_command"] is not None)
            derived = {**item, "policy": "surface_track",
                       "case": trial.v54.CASE,
                       "near_body_alarm_s": (alarm["time_s"] if alarm else None),
                       "near_body_commanded_m": commanded}
            checked = matched.score_item(derived, {
                "checkpoint_s": report["checkpoint_s"],
                "monitor_ticks": trial.runner.FIXED_MONITOR_TICKS,
                "monitor_period_s": trial.runner.SHOW_MONITOR_PERIOD_S},
                source, scenario)
            assert checked["minimum_person_gap_m"] >= .05
            assert checked["minimum_prop_gap_m"] >= .05
            assert run["outcome"] in (
                "new_show_motion_unresolved_again",
                "near_body_escape_owner_turn_unresolved")
            result_rows.append({**scores, **checked})
    finally:
        trial.v50.PATHS.clear()
        trial.v50.PATHS.update(original_paths)
    traces = [json.loads(Path(item["run"]["trace"]["path"]).read_text())
              for item in report["results"]]
    preturn_end = report["checkpoint_s"]+trial.v50.WAYPOINTS[-1][0]
    selected = [[row for row in rows if row["stage"] in (
        "post_alarm_range_watch", "pre_show_restage_step",
        "second_inference_range_watch") and row["time_s"] <= preturn_end+.000005]
        for rows in traces]
    assert len(set(map(len, selected))) == 1
    for aligned in zip(*selected):
        assert len(set(row["stage"] for row in aligned)) == 1
        for row in aligned[1:]:
            sensor_audit.same_packet(row["actor_packet"], aligned[0]["actor_packet"])
            assert row.get("ape_xy_m") == aligned[0].get("ape_xy_m")
            assert row.get("actor_action") == aligned[0].get("actor_action")
    result = {"passed": True, "rows": result_rows,
              "matching_pre_turn_actor_rows_per_path": len(selected[0]),
              "scope": "Actor-side new-return tube scoring, prior-packet restage steps, matched pre-turn history, independently regenerated fixed-clock scans and 20 ms MuJoCo gaps, decoded videos. Tube speed/lateral extent are uncalibrated development assumptions; historical kinematic proxy only."}
    (trial.OUT / "audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(trial.OUT / "audit.json", len(result_rows))


if __name__ == "__main__":
    main()
