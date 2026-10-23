"""Independent full-episode audit of V87 owner-view and V88 changed turns."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import audit_owner_joint_hazard_response as media
import owner_radial_social_view_episode as v87
import owner_radial_social_view_new_turns as v88


ROOT = v87.v86.ROOT
OUT = v88.OUT


def check(ok, label, checks):
    if not ok:
        raise AssertionError(label)
    checks.append(label)


def raw_view_choice(packet, start):
    """Recompute the range eligibility and owner-distance objective."""
    v59 = v87.v86.v85.v59
    start = np.asarray(start)
    endpoints = v59.v51.v41.JointHazardResponse.endpoints(packet, start)
    ranked = []
    for angle_deg in v87.v86.v85.v81.ANGLES_DEG:
        direction = np.asarray([math.cos(math.radians(angle_deg)),
                                math.sin(math.radians(angle_deg))])
        for distance in v87.v86.v85.v81.DISTANCES_M:
            target = start+distance*direction
            gaps = [float(np.min(np.linalg.norm(
                endpoints-(start+f*distance*direction), axis=1)))
                for f in np.linspace(0., 1., 15)]
            station = float(np.linalg.norm(
                v59.v56.OWNER_SHOW_STATION_XY_M-target))
            if (sum(value is not None for value in packet) >= 36 and
                    min(gaps) >= v59.MIN_ENDPOINT_GAP_M and
                    gaps[-1] >= v59.MIN_ENDPOINT_GAP_M and
                    station <= v59.MAX_OWNER_STATION_RANGE_M):
                ranked.append((station, -min(gaps), angle_deg, distance,
                               target))
    return min(ranked, key=lambda row: row[:2]), len(ranked)


def main():
    sha = v87.v86.v82.v77.v72.v52.sha
    reports = (("v87", json.loads((v87.OUT / "report.json").read_text()),
                v87.v86.v85.v81.PATHS),
               ("v88", json.loads((v88.OUT / "report.json").read_text()),
                v88.PATHS))
    checks = []
    for label, report, _ in reports:
        check(report["source_sha256"] == sha(
              v87.__file__ if label == "v87" else v88.__file__) and
              report["v86_source_sha256"] == sha(v87.v86.__file__) and
              report["runner_source_sha256"] == sha(v87.v86.runner.__file__)
              and report["scene_sha256"] == sha(
                  v87.v86.runner.v35.moving.SCENE_XML) and
              report["current_design_pointer_sha256"] == sha(
                  ROOT / "robot/ape-current-design.json"),
              f"{label} source chain", checks)
    check(reports[1][1]["v87_source_sha256"] == sha(v87.__file__) and
          reports[1][1]["v87_report_sha256"] == sha(v87.OUT / "report.json")
          and len(reports[0][1]["results"]) == 8 and
          len(reports[1][1]["results"]) == 9,
          "frozen V87 policy and complete grid", checks)
    source = v87.v86.runner.original()
    summary = source["summary"]
    policy = json.loads(v87.v86.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    model = mujoco.MjModel.from_xml_path(str(
        v87.v86.runner.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    prop_body = model.body("route_obstacle").id
    model.geom_size[model.geom("route_obstacle_geom").id, 1] = \
        v87.v86.v82.v77.v72.v64.v51.v46.PROP_HALF_HEIGHT_M
    model.body_pos[prop_body, 2] = \
        v87.v86.v82.v77.v72.v64.v51.v46.PROP_CENTER_Z_M
    start, _, checkpoint_s, _, _ = \
        v87.v86.v82.v77.v72.v64.v51.v46.low_prop_geometry(
            v87.v86.runner.geometry_at)(
                summary, source["result"]["result"]["episode"],
                scenario, model, data)
    for fixture in reports[1][1]["fixture_sweeps"]:
        result = v87.v86.v85.v81.v79.static_fixture_min(
            model, data, checkpoint_s,
            np.asarray(fixture["endpoint_xy_m"]))
        check(abs(result["minimum_gap_m"]-fixture[
              "minimum_gap_m"]) < 1e-9 and
              result["minimum_gap_m"] > .05,
              f"{fixture['path']} fixture-clear prop sweep", checks)
    reference_hold = None
    findings = []
    for version, report, path_table in reports:
        check(abs(report["checkpoint_s"]-checkpoint_s) < 1e-9,
              f"{version} exact checkpoint", checks)
        for item in report["results"]:
            label = f"{version}/{item['path']}"
            run = item["run"]
            rows = json.loads(Path(run["trace"]["path"]).read_text())
            check(sha(run["trace"]["path"]) == run["trace"]["sha256"] and
                  sha(run["video"]["path"]) == run["video"]["sha256"] and
                  media.video_frames(run["video"]["path"],
                                     (528, 1040, 3)) == run["video"]["frames"],
                  f"{label} decoded full owner video", checks)
            flank = next(row for row in rows if row["stage"] ==
                         "flank_sensor_video")
            check(sha(flank["path"]) == flank["sha256"] and
                  media.video_frames(flank["path"],
                                     (480, 640, 3)) == flank["frames"],
                  f"{label} decoded flank RGB-D video", checks)
            hold = [row for row in rows if row["stage"] ==
                    "post_alarm_range_watch"]
            steps = [row for row in rows if row["stage"] ==
                     "pre_show_restage_step"]
            waits = [row for row in rows if row["stage"] ==
                     "pre_show_restage_wait"]
            watch = [row for row in rows if row["stage"] ==
                     "second_inference_range_watch"]
            check(len(hold) == item["actor_history_rows"] == 14 and
                  len(steps) == item["restage_steps"] == 34 and
                  len(waits) == item["restage_waits"] == 2 and
                  len(watch) == v87.v86.runner.FIXED_MONITOR_TICKS and
                  item["fresh_captures"] == 1,
                  f"{label} full task and scan clock", checks)
            hold_packets = [row["actor_packet"] for row in hold]
            if reference_hold is None:
                reference_hold = hold_packets
            check(hold_packets == reference_hold,
                  f"{label} observation-equivalent pre-move history", checks)
            choice, eligible = raw_view_choice(hold[-1]["actor_packet"], start)
            selected = item["actor_chosen"]
            check(eligible == 58 and choice[2:4] == (
                  selected["angle_deg"], selected["distance_m"]) and
                  np.allclose(choice[4], selected["target_xy_m"], atol=1e-9)
                  and abs(choice[0]-selected[
                      "owner_station_range_m"]) < 1e-9 and
                  selected["angle_deg"] == 112.5 and
                  selected["distance_m"] == .85,
                  f"{label} raw-packet social-view choice", checks)
            first = steps[0]["actor_decision"]
            check(first["candidate_source"] ==
                  "actor range-eligible radial menu; nearest remembered owner station"
                  and abs(first["owner_view_choice_range_m"]-
                          choice[0]) < 1e-9,
                  f"{label} logged actor choice", checks)
            direction = np.asarray([
                math.cos(math.radians(choice[2])),
                math.sin(math.radians(choice[2]))])
            ordered = [row for row in rows if row["stage"] in
                       ("pre_show_restage_step", "pre_show_restage_wait")]
            prior_packet = hold[-1]["actor_packet"]
            prior_pose = np.asarray(start)
            for index, row in enumerate(ordered):
                proposed = prior_pose+.025*direction
                endpoints = v87.v86.v85.v59.v51.v41.JointHazardResponse.endpoints(
                    prior_packet, prior_pose)
                gap = float(np.min(np.linalg.norm(
                    endpoints-proposed, axis=1)))
                decision = row["actor_decision"]
                check(abs(gap-decision["minimum_endpoint_gap_m"]) < 1e-8 and
                      ((gap < v87.v86.v85.v59.v56.MIN_PATH_ENDPOINT_GAP_M)
                       == (row["stage"] == "pre_show_restage_wait")),
                      f"{label} preceding packet gate {index}", checks)
                if row["stage"] == "pre_show_restage_step":
                    check(np.allclose(row["ape_xy_m"], proposed, atol=1e-8),
                          f"{label} executed step {index}", checks)
                    prior_pose = proposed
                else:
                    check(np.allclose(row["ape_xy_m"], prior_pose, atol=1e-8),
                          f"{label} actual held pose {index}", checks)
                prior_packet = row["actor_packet"]
            check(np.allclose(steps[-1]["ape_xy_m"], choice[4], atol=1e-8),
                  f"{label} reached selected view", checks)
            timeline = [(checkpoint_s, np.asarray(start)),
                        (run["resume_time_s"], np.asarray(start))]
            timeline.extend((row["time_s"], np.asarray(row["ape_xy_m"]))
                            for row in ordered+watch)
            timeline.sort(key=lambda pair: pair[0])
            times = np.asarray([t for t, _ in timeline])
            poses = np.asarray([xy for _, xy in timeline])
            _, endpoint, finish_s = next(row for row in path_table
                                         if row[0] == item["path"])
            prop_min = person_min = float("inf")
            first_breach = None
            for sample in run["evaluator_gap_samples"]:
                t = sample["time_s"]
                pose = np.asarray([np.interp(t, times, poses[:, axis])
                                   for axis in range(2)])
                check(np.linalg.norm(pose-sample["ape_xy_m"]) < .00011,
                      f"{label} 20ms commanded pose {t:.2f}", checks)
                person = v87.v86.runner.human_xy(
                    v87.v86.v82.v77.v72.v64.v54.CASE, summary, t,
                    run["request_time_s"],
                    run["second_inference_start_s"])
                v87.v86.runner.v35.two.set_distractor(model, person)
                model.body_pos[model.body("audit_ape").id, :2] = pose
                model.body_pos[prop_body, :2] = \
                    v87.v86.v82.v77.v65.prop_at(
                        t, checkpoint_s, endpoint, finish_s)
                mujoco.mj_forward(model, data)
                pg = v87.v86.runner.prop_gap(model, data, pose)
                hg = v87.v86.runner.geometry.gap(model, data, pose, person)
                check(abs(pg-sample["prop_gap_m"]) < .00011 and
                      abs(hg-sample["gap_m"]) < .00011,
                      f"{label} 20ms MuJoCo geometry {t:.2f}", checks)
                prop_min, person_min = min(prop_min, pg), min(person_min, hg)
                if pg < .05 and first_breach is None:
                    first_breach = t
            check(abs(prop_min-run["minimum_prop_gap_m"]) < .00011 and
                  abs(person_min-run["minimum_proxy_gap_m"]) < .00011,
                  f"{label} independent minimum gaps", checks)
            alarm = next((row["time_s"] for row in rows if row["stage"] ==
                          "near_body_escape_plan"), None)
            if version == "v87":
                check(prop_min >= .05 and run["outcome"] ==
                      "new_show_metric_item_seen_unattributed" and
                      item["item_valid_depth_pixels"] > 0,
                      f"{label} development task with proxy margin", checks)
            else:
                check(prop_min < 0. and first_breach is not None and
                      (alarm is None or alarm > first_breach),
                      f"{label} changed-turn overlap before any alarm", checks)
            findings.append({"version": version, "path": item["path"],
                             "outcome": run["outcome"],
                             "minimum_prop_gap_m": round(prop_min, 5),
                             "minimum_person_gap_m": round(person_min, 5),
                             "first_prop_below_5cm_s": first_breach,
                             "near_body_alarm_s": alarm,
                             "item_valid_depth_pixels":
                                 item["item_valid_depth_pixels"],
                             "video": run["video"]["path"]})
    audit = {"schema": "robot-learning-lab.ape-radial-social-view-audit.v1",
             "passed": True, "checks": len(checks), "results": findings,
             "scope": "Source/fixture provenance, identical pre-move packets, independently recomputed owner-view choice and preceding-scan steps/waits, 20ms MuJoCo person/prop gaps and decoded full/flank videos for 17 full episodes. One stylized kitchen and historical kinematic body only."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(OUT / "audit.json", len(checks), "checks passed")


if __name__ == "__main__":
    main()
