"""Independent actor-menu, 20ms raw-gap and geometry audit for V76."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import owner_deferred_missing_pose_screen as v76


def check(ok, message, checks):
    if not ok:
        raise AssertionError(message)
    checks.append(message)


def first_below(times, values):
    ids = np.flatnonzero(values < .05)
    return float(times[ids[0]]) if len(ids) else None


def main():
    report = json.loads((v76.OUT / "report.json").read_text())
    checks, sha = [], v76.v72.v52.sha
    for key, source in (
        ("source_sha256", v76.__file__),
        ("v60_report_sha256", v76.v75.V60_REPORT),
        ("v75_report_sha256", v76.v75.OUT / "report.json"),
        ("scene_sha256", v76.runner.v35.moving.SCENE_XML),
        ("current_design_pointer_sha256",
         v76.ROOT / "robot/ape-current-design.json")):
        check(report[key] == sha(source), key, checks)
    base = json.loads(v76.v75.V60_REPORT.read_text())
    run = base["results"][0]["run"]
    trace = json.loads(Path(run["trace"]["path"]).read_text())
    packet = [row for row in trace if row["stage"] ==
              "post_alarm_range_watch"][-1]["actor_packet"]
    start = np.asarray(report["start_xy_m"])
    endpoints = v76.v72.v64.v51.v41.JointHazardResponse.endpoints(
        packet, start)
    expected_menu = []
    for name, direction in v76.DIRECTIONS.items():
        for distance in v76.DISTANCES_M:
            target = start+distance*direction
            path_gaps = [float(np.min(np.linalg.norm(
                endpoints-(start+u*direction), axis=1)))
                for u in np.linspace(0., distance, 15)]
            owner_range = float(np.linalg.norm(
                v76.v72.v64.v60.v59.v56.OWNER_SHOW_STATION_XY_M-target))
            eligible = bool(len([p for p in packet if p is not None]) >= 36
                and min(path_gaps) >=
                    v76.v72.v64.v60.v59.MIN_ENDPOINT_GAP_M
                and path_gaps[-1] >=
                    v76.v72.v64.v60.v59.MIN_ENDPOINT_GAP_M
                and owner_range <=
                    v76.v72.v64.v60.v59.MAX_OWNER_STATION_RANGE_M)
            expected_menu.append((name, distance, target,
                                  min(path_gaps), path_gaps[-1],
                                  owner_range, eligible))
    menu = report["candidate_menu"]
    check(len(menu) == len(expected_menu) == 15,
          "complete added direction menu", checks)
    for index, (actual, expected) in enumerate(zip(menu, expected_menu)):
        name, distance, target, path_gap, held_gap, owner_range, eligible = expected
        check(actual["name"] == name and
              actual["distance_m"] == distance and
              np.allclose(actual["target_xy_m"], target) and
              abs(actual["actor_minimum_path_endpoint_gap_m"]-path_gap) < 1e-8
              and abs(actual["actor_held_endpoint_reserve_m"]-held_gap) < 1e-8
              and abs(actual["actor_owner_station_range_m"]-owner_range) < 1e-8
              and actual["actor_eligible"] == eligible,
              f"actor eligibility {index}", checks)
    eligible = [row for row in menu if row["actor_eligible"]]
    results = report["eligible_results"]
    check(len(eligible) == len(results) == 10,
          "ten range-eligible omitted poses", checks)
    raw = Path(report["raw"]["path"])
    check(sha(raw) == report["raw"]["sha256"],
          "raw gap matrix hash", checks)
    with np.load(raw) as npz:
        times, props, people = (npz["time_s"],
                                npz["prop_gap_m"],
                                npz["person_gap_m"])
    check(props.shape == people.shape == (10, 6, len(times)) and
          np.allclose(np.diff(times), .02, atol=1e-8),
          "fixed 20ms matrix", checks)
    source = v76.runner.original()
    summary = source["summary"]
    policy = json.loads(v76.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] ==
                    "late_crossing")
    model = mujoco.MjModel.from_xml_path(str(
        v76.runner.v35.moving.SCENE_XML))
    sim = mujoco.MjData(model)
    geometry_start, _, t0, _, _ = \
        v76.v72.v64.v51.v46.low_prop_geometry(
            v76.runner.geometry_at)(
                summary, source["result"]["result"]["episode"],
                scenario, model, sim)
    check(np.allclose(geometry_start, start) and
          abs(t0-report["checkpoint_s"]) < 1e-8,
          "source checkpoint pose/time", checks)
    for ci, (elig, row) in enumerate(zip(eligible, results)):
        check(row["name"] == elig["name"] and
              row["distance_m"] == elig["distance_m"] and
              np.allclose(row["target_xy_m"], elig["target_xy_m"]),
              f"eligible candidate {ci}", checks)
        target = np.asarray(row["target_xy_m"])
        for vi, (variant, path, finish_s) in enumerate(v76.v75.VARIANTS):
            cell = row["paths"][vi]
            prop, human = props[ci, vi], people[ci, vi]
            check(cell["variant"] == variant and
                  abs(prop.min()-cell["minimum_prop_gap_m"]) < 1e-8 and
                  abs(human.min()-cell["minimum_person_gap_m"]) < 1e-8,
                  f"{ci}/{vi} raw minima", checks)
            check(first_below(times, prop) == cell[
                "first_prop_below_5cm_s"] and
                  first_below(times, human) == cell[
                      "first_person_below_5cm_s"],
                  f"{ci}/{vi} first breach", checks)
            endpoint = v76.v72.v64.PATHS[path]
            for metric, values in (("prop", prop),
                                   ("person", human)):
                ti = int(values.argmin())
                time_s = float(times[ti])
                fraction = np.clip((time_s-report["move_start_s"])*
                    v76.v75.SPEED_MPS/row["distance_m"], 0., 1.)
                pose = start+fraction*(target-start)
                person_xy = v76.runner.human_xy(
                    v76.v72.v64.v54.CASE, summary, time_s, t0,
                    row["scripted_person_return_s"])
                v76.runner.v35.two.set_distractor(model, person_xy)
                model.body_pos[model.body("audit_ape").id, :2] = pose
                model.body_pos[model.body("route_obstacle").id, :2] = \
                    v76.v75.v74.v65.prop_at(time_s, t0,
                                            endpoint, finish_s)
                mujoco.mj_forward(model, sim)
                measured = (v76.runner.prop_gap(model, sim, pose)
                            if metric == "prop" else
                            v76.runner.geometry.gap(
                                model, sim, pose, person_xy))
                check(abs(measured-float(values[ti])) < 1e-5,
                      f"{ci}/{vi} independent {metric} minimum", checks)
        check(abs(min(x["minimum_prop_gap_m"] for x in row["paths"])-
                  row["worst_prop_gap_m"]) < 1e-8 and
              abs(min(x["minimum_person_gap_m"] for x in row["paths"])-
                  row["worst_person_gap_m"]) < 1e-8,
              f"{ci} worst gaps", checks)
    supported = [row for row in results if row["all_six_above_5cm"]]
    check(report["supported_candidate_count"] == len(supported) == 1
          and supported[0]["name"] == "northwest" and
          supported[0]["distance_m"] == .55 and
          supported[0]["worst_prop_gap_m"] >= .10 and
          supported[0]["worst_person_gap_m"] >= .20,
          "one newly feasible northwest 55cm pose", checks)
    target = np.asarray(supported[0]["target_xy_m"])
    ape_geom = model.geom("audit_ape_geom").id
    fixture_min = (float("inf"), None)
    for fraction in np.linspace(0., 1., 23):
        pose = start+fraction*(target-start)
        model.body_pos[model.body("audit_ape").id, :2] = pose
        mujoco.mj_forward(model, sim)
        for geom in range(model.ngeom):
            name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
            if (geom == ape_geom or name.startswith("audit_ape") or
                    name == "route_obstacle_geom" or "person" in name or
                    "distractor" in name or "floor" in name):
                continue
            gap = float(mujoco.mj_geomDistance(
                model, sim, ape_geom, geom, 1., None))
            if gap < fixture_min[0]:
                fixture_min = (gap, name)
    check(fixture_min[0] > .05,
          "northwest move clears sampled static proxy fixtures", checks)
    output = {"schema": "robot-learning-lab.ape-missing-wait-audit.v1",
              "source_sha256": sha(__file__),
              "status": "pass", "checks": len(checks),
              "eligible_candidates": len(results),
              "supported_candidates": [
                  {"name": row["name"], "distance_m": row["distance_m"],
                   "worst_prop_gap_m": row["worst_prop_gap_m"],
                   "worst_person_gap_m": row["worst_person_gap_m"]}
                  for row in supported],
              "independent_minimum_replays": 2*len(results)*
                  len(v76.v75.VARIANTS),
              "northwest_55cm_minimum_static_proxy_gap_m": fixture_min[0],
              "nearest_static_geom": fixture_min[1]}
    target = v76.OUT / "audit.json"
    target.write_text(json.dumps(output, indent=2)+"\n")
    print(target, len(checks), "checks passed")


if __name__ == "__main__":
    main()
