"""V76: screen range-eligible directions omitted from V59's wait menu.

Candidate eligibility uses only the actual V60 pre-move range packet and
remembered owner station. Future six-path gaps are evaluator-only feasibility
evidence, not information for the actor selector.
"""

from __future__ import annotations

import json
import math

import mujoco
import numpy as np

import owner_deferred_wait_pose_feasibility as v75


v72, runner = v75.v72, v75.runner
ROOT = v75.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v76-missing-wait-directions"
DIRECTIONS = {"northwest": np.array([-1., 1.])/math.sqrt(2),
              "north": np.array([0., 1.]),
              "westnorth": np.array([-2., 1.])/math.sqrt(5)}
DISTANCES_M = (.35, .45, .55, .70, .85)


def first_below(times, gaps):
    indexes = np.flatnonzero(gaps < v75.PROVISIONAL_MARGIN_M)
    return float(times[indexes[0]]) if len(indexes) else None


def main():
    source = runner.original()
    summary = source["summary"]
    policy = json.loads(runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] ==
                    "late_crossing")
    base = json.loads(v75.V60_REPORT.read_text())
    run = base["results"][0]["run"]
    trace = json.loads(open(run["trace"]["path"]).read())
    last_wait = [row for row in trace if row["stage"] ==
                 "post_alarm_range_watch"][-1]
    packet = last_wait["actor_packet"]
    checkpoint_s = base["checkpoint_s"]
    move_start_s = run["resume_time_s"]
    end_s = checkpoint_s+v72.v64.v60.WATCH_END_AFTER_CHECKPOINT_S
    times = np.arange(move_start_s, end_s+.00001, v75.PERIOD_S)
    model = mujoco.MjModel.from_xml_path(str(
        runner.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    start, _, geometry_t0, _, _ = v72.v64.v51.v46.low_prop_geometry(
        runner.geometry_at)(summary,
            source["result"]["result"]["episode"], scenario,
            model, data)
    assert abs(geometry_t0-checkpoint_s) < 1e-8
    endpoints = v72.v64.v51.v41.JointHazardResponse.endpoints(
        packet, start)
    healthy = len([x for x in packet if x is not None]) >= 36
    candidates = []
    for name, direction in DIRECTIONS.items():
        for distance in DISTANCES_M:
            target = start+distance*direction
            gaps = [float(np.min(np.linalg.norm(
                endpoints-(start+u*direction), axis=1)))
                for u in np.linspace(0., distance, 15)]
            owner_distance = float(np.linalg.norm(
                v72.v64.v60.v59.v56.OWNER_SHOW_STATION_XY_M-target))
            eligible = bool(healthy and min(gaps) >=
                            v72.v64.v60.v59.MIN_ENDPOINT_GAP_M and
                            gaps[-1] >=
                            v72.v64.v60.v59.MIN_ENDPOINT_GAP_M and
                            owner_distance <=
                            v72.v64.v60.v59.MAX_OWNER_STATION_RANGE_M)
            candidates.append({"name": name,
                               "distance_m": distance,
                               "target_xy_m": target.tolist(),
                               "actor_minimum_path_endpoint_gap_m": min(gaps),
                               "actor_held_endpoint_reserve_m": gaps[-1],
                               "actor_owner_station_range_m": owner_distance,
                               "actor_eligible": eligible})
    eligible = [x for x in candidates if x["actor_eligible"]]
    prop = np.empty((len(eligible), len(v75.VARIANTS), len(times)))
    person = np.empty_like(prop)
    results = []
    for ci, row in enumerate(eligible):
        target = np.asarray(row["target_xy_m"])
        distance = row["distance_m"]
        move_end_s = move_start_s+distance/v75.SPEED_MPS
        scripted_return_s = (move_end_s+
            v72.v64.v60.SCRIPTED_RETURN_DELAY_AFTER_RESTAGE_S)
        paths = []
        for vi, (variant, path, finish_s) in enumerate(v75.VARIANTS):
            endpoint = v72.v64.PATHS[path]
            for ti, time_s in enumerate(times):
                fraction = np.clip((time_s-move_start_s)*
                                   v75.SPEED_MPS/distance, 0., 1.)
                pose = start+fraction*(target-start)
                other = runner.human_xy(
                    v72.v64.v54.CASE, summary, float(time_s),
                    checkpoint_s, scripted_return_s)
                runner.v35.two.set_distractor(model, other)
                model.body_pos[model.body("audit_ape").id, :2] = pose
                model.body_pos[model.body("route_obstacle").id, :2] = \
                    v75.v74.v65.prop_at(float(time_s), checkpoint_s,
                                        endpoint, finish_s)
                mujoco.mj_forward(model, data)
                prop[ci, vi, ti] = runner.prop_gap(model, data, pose)
                person[ci, vi, ti] = runner.geometry.gap(
                    model, data, pose, other)
            p, h = prop[ci, vi], person[ci, vi]
            paths.append({"variant": variant,
                          "minimum_prop_gap_m": float(p.min()),
                          "minimum_person_gap_m": float(h.min()),
                          "first_prop_below_5cm_s": first_below(times, p),
                          "first_person_below_5cm_s": first_below(times, h)})
        worst_prop = min(x["minimum_prop_gap_m"] for x in paths)
        worst_person = min(x["minimum_person_gap_m"] for x in paths)
        results.append({**row, "move_end_s": move_end_s,
                        "scripted_person_return_s": scripted_return_s,
                        "worst_prop_gap_m": worst_prop,
                        "worst_person_gap_m": worst_person,
                        "all_six_above_5cm": bool(
                            worst_prop >= .05 and worst_person >= .05),
                        "paths": paths})
        print(row["name"], distance, round(worst_prop, 3),
              round(worst_person, 3), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / "eligible-candidate-20ms-gaps.npz"
    np.savez_compressed(raw, time_s=times,
                        prop_gap_m=prop, person_gap_m=person)
    report = {"schema": "robot-learning-lab.ape-missing-wait-direction-screen.v1",
              "source_sha256": v72.v52.sha(__file__),
              "v60_report_sha256": v72.v52.sha(v75.V60_REPORT),
              "v75_report_sha256": v72.v52.sha(v75.OUT / "report.json"),
              "scene_sha256": v72.v52.sha(
                  runner.v35.moving.SCENE_XML),
              "current_design_pointer_sha256": v72.v52.sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "move_start_s": move_start_s,
              "start_xy_m": start.tolist(),
              "period_s": v75.PERIOD_S,
              "variant_names": [x[0] for x in v75.VARIANTS],
              "candidate_menu": candidates,
              "eligible_results": results,
              "supported_candidate_count": sum(
                  x["all_six_above_5cm"] for x in results),
              "raw": {"path": str(raw), "sha256": v72.v52.sha(raw)},
              "scope": "Evaluator-only six-path future screen of previously omitted northward candidate directions. Actor eligibility uses the actual pre-move range packet and task-known owner station; future gaps cannot be used by an actor selector. Historical kinematic proxy and 20ms geometry are not physical qualification."}
    target = OUT / "report.json"
    target.write_text(json.dumps(report, indent=2)+"\n")
    print(target, report["supported_candidate_count"],
          "supported candidates")


if __name__ == "__main__":
    main()
