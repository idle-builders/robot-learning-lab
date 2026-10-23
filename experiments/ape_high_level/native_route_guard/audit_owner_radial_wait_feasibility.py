"""Independent actor-menu and rejecting-path geometry audit for V81."""

from __future__ import annotations

import json
import math

import mujoco
import numpy as np

import owner_radial_wait_feasibility as trial


def check(ok, label, checks):
    if not ok:
        raise AssertionError(label)
    checks.append(label)


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    checks = []
    sha = trial.v77.v72.v52.sha
    check(report["schema"] == "robot-learning-lab.ape-radial-wait-screen.v1",
          "schema", checks)
    for key, path in (
        ("source_sha256", trial.__file__),
        ("v60_report_sha256", trial.ROOT /
         "artifacts/ape-native-route-guard/v60-deferred-owner-show/report.json"),
        ("v79_report_sha256", trial.v79.OUT / "report.json"),
        ("scene_sha256", trial.v77.runner.v35.moving.SCENE_XML),
        ("current_design_pointer_sha256",
         trial.ROOT / "robot/ape-current-design.json"),
    ):
        check(report[key] == sha(path), f"source {key}", checks)
    source = trial.v77.runner.original()
    summary = source["summary"]
    policy = json.loads(trial.v77.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    v60 = json.loads((trial.ROOT / "artifacts/ape-native-route-guard/"
                      "v60-deferred-owner-show/report.json").read_text())
    rows = json.loads(open(v60["results"][0]["run"]["trace"]["path"]).read())
    packet = [row for row in rows if row["stage"] ==
              "post_alarm_range_watch"][-1]["actor_packet"]
    start = np.asarray(report["start_xy_m"])
    model = mujoco.MjModel.from_xml_path(str(
        trial.v77.runner.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    actual, _, t0, _, _ = trial.v77.v72.v64.v51.v46.low_prop_geometry(
        trial.v77.runner.geometry_at)(
            summary, source["result"]["result"]["episode"],
            scenario, model, data)
    check(np.allclose(actual, start) and
          abs(t0-report["checkpoint_s"]) < 1e-8,
          "source pose and checkpoint", checks)
    endpoints = trial.v77.v59.v51.v41.JointHazardResponse.endpoints(
        packet, start)
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
    candidates = report["candidates"]
    check(len(candidates) == report["candidate_count"] == 80 and
          report["paths"] == [name for name, _, _ in trial.PATHS],
          "complete radial grid and path list", checks)
    times = np.arange(report["move_start_s"],
                      report["watch_end_s"]+.00001, trial.PERIOD_S)
    assessed = 0
    failures = []
    for index, row in enumerate(candidates):
        angle = trial.ANGLES_DEG[index // len(trial.DISTANCES_M)]
        distance = trial.DISTANCES_M[index % len(trial.DISTANCES_M)]
        direction = np.asarray([math.cos(math.radians(angle)),
                                math.sin(math.radians(angle))])
        target = start+distance*direction
        gap = min(float(np.min(np.linalg.norm(
            endpoints-(start+fraction*distance*direction), axis=1)))
            for fraction in np.linspace(0., 1., 15))
        owner = float(np.linalg.norm(
            trial.v77.v59.v56.OWNER_SHOW_STATION_XY_M-target))
        eligible = (len([p for p in packet if p is not None]) >= 36 and
                    gap >= trial.v77.v59.MIN_ENDPOINT_GAP_M and
                    owner <= trial.v77.v59.MAX_OWNER_STATION_RANGE_M)
        check(row["angle_deg"] == angle and row["distance_m"] == distance
              and np.allclose(row["target_xy_m"], target) and
              abs(row["actor_minimum_endpoint_gap_m"]-gap) < 1e-8 and
              abs(row["actor_owner_station_range_m"]-owner) < 1e-8 and
              row["actor_range_eligible"] == eligible,
              f"actor grid candidate {index}", checks)
        if not row.get("evaluator_fixture_clear"):
            continue
        if row.get("evaluator_all_eight_clear"):
            fixture_min = (float("inf"), None)
            for fraction in np.linspace(0., 1., 19):
                model.body_pos[model.body("audit_ape").id, :2] = \
                    start+fraction*(target-start)
                mujoco.mj_forward(model, data)
                for geom, name in static_geoms:
                    distance_to_fixture = float(mujoco.mj_geomDistance(
                        model, data, ape_geom, geom, 1., None))
                    if distance_to_fixture < fixture_min[0]:
                        fixture_min = (distance_to_fixture, name)
            check(abs(fixture_min[0]-row["evaluator_static_fixture_gap_m"])
                  < 1e-8 and fixture_min[1] ==
                  row["evaluator_nearest_fixture"] and
                  fixture_min[0] >= trial.PROVISIONAL_GAP_M,
                  f"supported candidate {index} static path", checks)
        path_rows = row["evaluator_paths"]
        check(len(path_rows) == len(trial.PATHS) == 8 and
              [p["path"] for p in path_rows] == report["paths"],
              f"candidate {index} eight scripts", checks)
        failing = next((i for i, p in enumerate(path_rows) if
                        p["minimum_prop_gap_m"] < trial.PROVISIONAL_GAP_M or
                        p["minimum_person_gap_m"] < trial.PROVISIONAL_GAP_M),
                       None)
        paths_to_replay = range(len(trial.PATHS)) if failing is None else (failing,)
        move_end = report["move_start_s"]+distance/trial.v77.v75.SPEED_MPS
        return_s = (move_end+trial.v77.v72.v64.v60.
                    SCRIPTED_RETURN_DELAY_AFTER_RESTAGE_S)
        for path_index in paths_to_replay:
            name, endpoint, finish_s = trial.PATHS[path_index]
            prop_min = person_min = float("inf")
            first_prop = first_person = None
            for t in times:
                fraction = float(np.clip((t-report["move_start_s"])*
                                         trial.v77.v75.SPEED_MPS/distance,
                                         0., 1.))
                pose = start+fraction*(target-start)
                person = trial.v77.runner.human_xy(
                    trial.v77.v72.v64.v54.CASE, summary, float(t),
                    report["checkpoint_s"], return_s)
                trial.v77.runner.v35.two.set_distractor(model, person)
                model.body_pos[model.body("audit_ape").id, :2] = pose
                model.body_pos[model.body("route_obstacle").id, :2] = \
                    trial.v77.v65.prop_at(float(t), report["checkpoint_s"],
                                          endpoint, finish_s)
                mujoco.mj_forward(model, data)
                pg = trial.v77.runner.prop_gap(model, data, pose)
                hg = trial.v77.runner.geometry.gap(model, data, pose, person)
                prop_min = min(prop_min, pg)
                person_min = min(person_min, hg)
                if pg < trial.PROVISIONAL_GAP_M and first_prop is None:
                    first_prop = float(t)
                if hg < trial.PROVISIONAL_GAP_M and first_person is None:
                    first_person = float(t)
                if failing is not None and (first_prop is not None or
                                            first_person is not None):
                    break
            recorded = path_rows[path_index]
            if failing is None:
                check(abs(prop_min-recorded["minimum_prop_gap_m"]) < 1e-8 and
                      abs(person_min-recorded["minimum_person_gap_m"]) < 1e-8,
                      f"supported candidate {index} full path {name}", checks)
            else:
                observed_first = min(
                    t for t in (first_prop, first_person) if t is not None)
                recorded_first = min(
                    t for t in (recorded["first_prop_below_5cm_s"],
                                recorded["first_person_below_5cm_s"])
                    if t is not None)
                check(observed_first == recorded_first,
                      f"candidate {index} independently rejected on {name}",
                      checks)
                failures.append({"candidate_index": index,
                                 "path": name,
                                 "first_prop_below_5cm_s": first_prop,
                                 "first_person_below_5cm_s": first_person})
            assessed += 1
    claimed = [row for row in candidates if row.get("evaluator_all_eight_clear")]
    check(len(claimed) == report["all_eight_clear_count"],
          "all-eight count", checks)
    check(report["range_eligible_count"] == sum(
        row["actor_range_eligible"] for row in candidates) and
        report["fixture_clear_count"] == sum(
            row.get("evaluator_fixture_clear", False) for row in candidates),
          "eligibility totals", checks)
    v77 = json.loads((trial.v77.OUT / "report.json").read_text())
    v79 = json.loads((trial.v79.OUT / "report.json").read_text())
    prior_runs = [row["run"] for row in v77["results"]+v79["results"]]
    preturn = []
    for run in prior_runs:
        saved = json.loads(open(run["trace"]["path"]).read())
        preturn.append([row for row in saved if row["stage"] ==
                        "post_alarm_range_watch"])
    check(len(preturn) == 8 and all(len(rows) == 14 for rows in preturn),
          "eight matched actor histories", checks)
    for tick in range(14):
        reference = preturn[0][tick]
        for path_index, history in enumerate(preturn[1:], start=1):
            check(history[tick]["time_s"] == reference["time_s"] and
                  history[tick]["actor_packet"] ==
                  reference["actor_packet"],
                  f"path {path_index} identical pre-move packet {tick}", checks)
    output = {"schema": "robot-learning-lab.ape-radial-wait-audit.v1",
              "passed": True, "checks": len(checks),
              "independent_path_replays": assessed,
              "matched_pre_move_packets_per_path": 14,
              "independently_rejected_candidates": failures,
              "all_eight_clear_count": len(claimed),
              "scope": "Actor packet radial menu and independent rejecting-path 20ms MuJoCo replay, all eight paths for any supported candidate. Future routes are evaluator-only; sampled fixtures/kinematic historical proxy, no camera visibility or physical stop qualification."}
    (trial.OUT / "audit.json").write_text(json.dumps(output, indent=2)+"\n")
    print(trial.OUT / "audit.json", len(checks), "checks passed")


if __name__ == "__main__":
    main()
