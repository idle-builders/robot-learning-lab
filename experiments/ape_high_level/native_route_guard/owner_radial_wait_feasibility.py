"""V81: uniform wait-pose feasibility screen from the V60 actor checkpoint.

The menu is constructed only from the retained pre-move low/mid packet and
owner station. Future eight scripted low-prop paths and person gaps are
evaluator-only; this screen does not implement a chooser or authorize motion.
"""

from __future__ import annotations

import json
import math

import mujoco
import numpy as np

import owner_northwest_new_turn_stress as v79


v77 = v79.v77
ROOT = v79.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v81-radial-wait-feasibility"
ANGLES_DEG = tuple(22.5*i for i in range(16))
DISTANCES_M = (.35, .45, .55, .70, .85)
PERIOD_S = .02
PROVISIONAL_GAP_M = .05
PATHS = tuple((name, v77.v72.v64.PATHS[path], finish_s)
              for name, path, finish_s in v77.VARIANTS) + tuple(
    (name, endpoint, v79.FINISH_AFTER_CHECKPOINT_S)
    for name, endpoint in v79.ENDPOINTS.items())


def static_fixture_gap(model, data, start, target, static_geoms):
    ape_geom = model.geom("audit_ape_geom").id
    minimum = (float("inf"), None)
    for fraction in np.linspace(0., 1., 19):
        model.body_pos[model.body("audit_ape").id, :2] = (
            start+fraction*(target-start))
        mujoco.mj_forward(model, data)
        for geom, name in static_geoms:
            distance = float(mujoco.mj_geomDistance(
                model, data, ape_geom, geom, 1., None))
            if distance < minimum[0]:
                minimum = (distance, name)
    return minimum


def main():
    source = v77.runner.original()
    summary = source["summary"]
    policy = json.loads(v77.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    v60 = json.loads((ROOT / "artifacts/ape-native-route-guard/"
                       "v60-deferred-owner-show/report.json").read_text())
    trace = json.loads(open(v60["results"][0]["run"]["trace"]["path"]).read())
    packet = [row for row in trace if row["stage"] ==
              "post_alarm_range_watch"][-1]["actor_packet"]
    start = np.asarray(json.loads((v77.v76.OUT / "report.json").read_text())[
        "start_xy_m"])
    checkpoint_s = float(v60["checkpoint_s"])
    move_start_s = float(v60["results"][0]["run"]["resume_time_s"])
    end_s = checkpoint_s+v77.v72.v64.v60.WATCH_END_AFTER_CHECKPOINT_S
    times = np.arange(move_start_s, end_s+.00001, PERIOD_S)
    model = mujoco.MjModel.from_xml_path(str(v77.runner.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    actual, _, t0, _, _ = v77.v72.v64.v51.v46.low_prop_geometry(
        v77.runner.geometry_at)(
            summary, source["result"]["result"]["episode"],
            scenario, model, data)
    assert np.allclose(actual, start) and abs(t0-checkpoint_s) < 1e-8
    endpoints = v77.v59.v51.v41.JointHazardResponse.endpoints(packet, start)
    ape_geom = model.geom("audit_ape_geom").id
    static_geoms = []
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if (geom == ape_geom or name.startswith("audit_ape") or
                name == "route_obstacle_geom" or "person" in name or
                "distractor" in name or "floor" in name):
            continue
        static_geoms.append((geom, name))
    candidates = []
    for angle_deg in ANGLES_DEG:
        direction = np.asarray([
            math.cos(math.radians(angle_deg)),
            math.sin(math.radians(angle_deg))])
        for distance in DISTANCES_M:
            target = start+distance*direction
            path_gap = min(float(np.min(np.linalg.norm(
                endpoints-(start+fraction*distance*direction), axis=1)))
                for fraction in np.linspace(0., 1., 15))
            station = float(np.linalg.norm(
                v77.v59.v56.OWNER_SHOW_STATION_XY_M-target))
            range_eligible = bool(
                len([p for p in packet if p is not None]) >= 36 and
                path_gap >= v77.v59.MIN_ENDPOINT_GAP_M and
                station <= v77.v59.MAX_OWNER_STATION_RANGE_M)
            row = {"angle_deg": angle_deg, "distance_m": distance,
                   "target_xy_m": target.tolist(),
                   "actor_minimum_endpoint_gap_m": path_gap,
                   "actor_owner_station_range_m": station,
                   "actor_range_eligible": range_eligible}
            if not range_eligible:
                candidates.append(row)
                continue
            fixture_gap, fixture_name = static_fixture_gap(
                model, data, start, target, static_geoms)
            row["evaluator_static_fixture_gap_m"] = fixture_gap
            row["evaluator_nearest_fixture"] = fixture_name
            row["evaluator_fixture_clear"] = fixture_gap >= PROVISIONAL_GAP_M
            if not row["evaluator_fixture_clear"]:
                candidates.append(row)
                continue
            move_end_s = move_start_s+distance/v79.v77.v75.SPEED_MPS
            return_s = (move_end_s+
                        v77.v72.v64.v60.SCRIPTED_RETURN_DELAY_AFTER_RESTAGE_S)
            path_results = []
            for name, endpoint, finish_s in PATHS:
                prop_min = person_min = float("inf")
                first_prop_breach = first_person_breach = None
                for t in times:
                    fraction = float(np.clip(
                        (t-move_start_s)*v77.v75.SPEED_MPS/distance,
                        0., 1.))
                    pose = start+fraction*(target-start)
                    person = v77.runner.human_xy(
                        v77.v72.v64.v54.CASE, summary, float(t),
                        checkpoint_s, return_s)
                    v77.runner.v35.two.set_distractor(model, person)
                    model.body_pos[model.body("audit_ape").id, :2] = pose
                    model.body_pos[model.body("route_obstacle").id, :2] = \
                        v77.v65.prop_at(float(t), checkpoint_s,
                                        endpoint, finish_s)
                    mujoco.mj_forward(model, data)
                    prop_gap = v77.runner.prop_gap(model, data, pose)
                    person_gap = v77.runner.geometry.gap(
                        model, data, pose, person)
                    prop_min = min(prop_min, prop_gap)
                    person_min = min(person_min, person_gap)
                    if prop_gap < PROVISIONAL_GAP_M and first_prop_breach is None:
                        first_prop_breach = float(t)
                    if person_gap < PROVISIONAL_GAP_M and first_person_breach is None:
                        first_person_breach = float(t)
                path_results.append({"path": name,
                                     "minimum_prop_gap_m": prop_min,
                                     "minimum_person_gap_m": person_min,
                                     "first_prop_below_5cm_s": first_prop_breach,
                                     "first_person_below_5cm_s": first_person_breach})
            row["evaluator_paths"] = path_results
            row["evaluator_worst_prop_gap_m"] = min(
                path["minimum_prop_gap_m"] for path in path_results)
            row["evaluator_worst_person_gap_m"] = min(
                path["minimum_person_gap_m"] for path in path_results)
            row["evaluator_all_eight_clear"] = all(
                path["minimum_prop_gap_m"] >= PROVISIONAL_GAP_M and
                path["minimum_person_gap_m"] >= PROVISIONAL_GAP_M
                for path in path_results)
            candidates.append(row)
            print(angle_deg, distance, round(row[
                "evaluator_worst_prop_gap_m"], 3),
                  round(row["evaluator_worst_person_gap_m"], 3),
                  row["evaluator_all_eight_clear"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    passed = [row for row in candidates if row.get("evaluator_all_eight_clear")]
    report = {"schema": "robot-learning-lab.ape-radial-wait-screen.v1",
              "source_sha256": v77.v72.v52.sha(__file__),
              "v60_report_sha256": v77.v72.v52.sha(
                  ROOT / "artifacts/ape-native-route-guard/"
                  "v60-deferred-owner-show/report.json"),
              "v79_report_sha256": v77.v72.v52.sha(v79.OUT / "report.json"),
              "scene_sha256": v77.v72.v52.sha(
                  v77.runner.v35.moving.SCENE_XML),
              "current_design_pointer_sha256": v77.v72.v52.sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "move_start_s": move_start_s,
              "watch_end_s": end_s,
              "start_xy_m": start.tolist(),
              "angles_deg": ANGLES_DEG,
              "distances_m": DISTANCES_M,
              "paths": [name for name, _, _ in PATHS],
              "candidate_count": len(candidates),
              "range_eligible_count": sum(r["actor_range_eligible"]
                                          for r in candidates),
              "fixture_clear_count": sum(r.get("evaluator_fixture_clear", False)
                                         for r in candidates),
              "all_eight_clear_count": len(passed),
              "all_eight_clear_candidates": [
                  {k: r[k] for k in ("angle_deg", "distance_m",
                                    "target_xy_m", "evaluator_worst_prop_gap_m",
                                    "evaluator_worst_person_gap_m")}
                  for r in passed],
              "candidates": candidates,
              "scope": "Uniform actor-range-eligible radial search followed by evaluator-only static-fixture and eight-script 20ms moving geometry. No actor policy, owner-camera visibility qualification, selected body or physical margin."}
    target = OUT / "report.json"
    target.write_text(json.dumps(report, indent=2)+"\n")
    print(target, len(passed), "all-eight candidates")


if __name__ == "__main__":
    main()
