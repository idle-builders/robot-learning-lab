"""Audit V91's frozen actor, speed-only change, geometry and videos."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import audit_owner_joint_hazard_response as media
import audit_owner_radial_social_view as social_audit
import owner_social_view_turn_speed_sweep as trial


def check(ok, label, checks):
    if not ok:
        raise AssertionError(label)
    checks.append(label)


def main():
    v86 = trial.v86
    sha = v86.v82.v77.v72.v52.sha
    report = json.loads((trial.OUT / "report.json").read_text())
    control = json.loads((trial.v87.OUT / "report.json").read_text())
    checks = []
    check(report["source_sha256"] == sha(trial.__file__) and
          report["v86_source_sha256"] == sha(v86.__file__) and
          report["v87_report_sha256"] == sha(trial.v87.OUT / "report.json")
          and report["v88_report_sha256"] == sha(trial.v88.OUT / "report.json")
          and report["runner_source_sha256"] == sha(v86.runner.__file__) and
          report["scene_sha256"] == sha(v86.runner.v35.moving.SCENE_XML) and
          report["current_design_pointer_sha256"] == sha(
              v86.ROOT / "robot/ape-current-design.json") and
          len(report["results"]) == 7,
          "frozen actor and source provenance", checks)
    source = v86.runner.original()
    summary = source["summary"]
    policy = json.loads(v86.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    model = mujoco.MjModel.from_xml_path(str(v86.runner.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    prop_body = model.body("route_obstacle").id
    model.geom_size[model.geom("route_obstacle_geom").id, 1] = \
        v86.v82.v77.v72.v64.v51.v46.PROP_HALF_HEIGHT_M
    model.body_pos[prop_body, 2] = \
        v86.v82.v77.v72.v64.v51.v46.PROP_CENTER_Z_M
    start, _, checkpoint_s, _, _ = \
        v86.v82.v77.v72.v64.v51.v46.low_prop_geometry(
            v86.runner.geometry_at)(
                summary, source["result"]["result"]["episode"],
                scenario, model, data)
    check(abs(report["checkpoint_s"]-checkpoint_s) < 1e-8 and
          abs(report["turn_start_after_checkpoint_s"]-
              v86.v82.v77.v65.TURN_START_S) < 1e-8,
          "actual checkpoint and fixed turn start", checks)
    reference_hold = None
    findings = []
    for item in report["results"]:
        name = item["path"]
        run = item["run"]
        rows = json.loads(Path(run["trace"]["path"]).read_text())
        check(sha(run["trace"]["path"]) == run["trace"]["sha256"] and
              sha(run["video"]["path"]) == run["video"]["sha256"] and
              media.video_frames(run["video"]["path"],
                                 (528, 1040, 3)) == run["video"]["frames"],
              f"{name} decoded full video", checks)
        flank = next(row for row in rows if row["stage"] ==
                     "flank_sensor_video")
        check(sha(flank["path"]) == flank["sha256"] and
              media.video_frames(flank["path"],
                                 (480, 640, 3)) == flank["frames"],
              f"{name} decoded flank video", checks)
        hold = [row for row in rows if row["stage"] ==
                "post_alarm_range_watch"]
        steps = [row for row in rows if row["stage"] ==
                 "pre_show_restage_step"]
        waits = [row for row in rows if row["stage"] ==
                 "pre_show_restage_wait"]
        watch = [row for row in rows if row["stage"] ==
                 "second_inference_range_watch"]
        packets = [row["actor_packet"] for row in hold]
        if reference_hold is None:
            reference_hold = packets
        check(packets == reference_hold and len(hold) == 14 and
              len(steps) == 34 and len(waits) == 2 and
              len(watch) == v86.runner.FIXED_MONITOR_TICKS and
              item["fresh_captures"] == 1,
              f"{name} identical pre-move history and full task clock", checks)
        choice, eligible = social_audit.raw_view_choice(
            hold[-1]["actor_packet"], start)
        check(eligible == 58 and choice[2:4] == (112.5, .85) and
              item["actor_chosen"]["angle_deg"] == 112.5 and
              item["actor_chosen"]["distance_m"] == .85,
              f"{name} frozen actor-only pose", checks)
        ordered = [row for row in rows if row["stage"] in
                   ("pre_show_restage_step", "pre_show_restage_wait")]
        timeline = [(checkpoint_s, np.asarray(start)),
                    (run["resume_time_s"], np.asarray(start))]
        timeline.extend((row["time_s"], np.asarray(row["ape_xy_m"]))
                        for row in ordered+watch)
        timeline.sort(key=lambda pair: pair[0])
        times = np.asarray([t for t, _ in timeline])
        poses = np.asarray([xy for _, xy in timeline])
        _, endpoint, finish_s = next(row for row in trial.PATHS
                                     if row[0] == name)
        if name != "outward_normal":
            variant = next(row for row in report["variants"] if row[
                "name"] == name)
            check(abs(variant["finish_after_checkpoint_s"]-finish_s) < 1e-8 and
                  abs(variant["turn_duration_s"]-(finish_s-
                    v86.v82.v77.v65.TURN_START_S)) < 1e-8,
                  f"{name} declared speed-only trajectory", checks)
        prop_min = person_min = float("inf")
        first_breach = None
        for sample in run["evaluator_gap_samples"]:
            t = sample["time_s"]
            pose = np.asarray([np.interp(t, times, poses[:, axis])
                               for axis in range(2)])
            check(np.linalg.norm(pose-sample["ape_xy_m"]) < .00011,
                  f"{name} 20ms commanded pose {t:.2f}", checks)
            person = v86.runner.human_xy(
                v86.v82.v77.v72.v64.v54.CASE, summary, t,
                run["request_time_s"], run["second_inference_start_s"])
            v86.runner.v35.two.set_distractor(model, person)
            model.body_pos[model.body("audit_ape").id, :2] = pose
            model.body_pos[prop_body, :2] = v86.v82.v77.v65.prop_at(
                t, checkpoint_s, endpoint, finish_s)
            mujoco.mj_forward(model, data)
            pg = v86.runner.prop_gap(model, data, pose)
            hg = v86.runner.geometry.gap(model, data, pose, person)
            check(abs(pg-sample["prop_gap_m"]) < .00011 and
                  abs(hg-sample["gap_m"]) < .00011,
                  f"{name} 20ms MuJoCo geometry {t:.2f}", checks)
            prop_min, person_min = min(prop_min, pg), min(person_min, hg)
            if pg < .05 and first_breach is None:
                first_breach = t
        check(abs(prop_min-run["minimum_prop_gap_m"]) < .00011 and
              abs(person_min-run["minimum_proxy_gap_m"]) < .00011 and
              (name == "outward_normal" or prop_min < 0.),
              f"{name} independent minimum gap", checks)
        alarm = next((row["time_s"] for row in rows if row["stage"] ==
                      "near_body_escape_plan"), None)
        response = sum(row["near_body_command"]["commanded_step_m"]
                       for row in watch if row["near_body_command"])
        findings.append({"path": name, "outcome": run["outcome"],
                         "minimum_prop_gap_m": round(prop_min, 5),
                         "first_prop_below_5cm_s": first_breach,
                         "near_body_alarm_s": alarm,
                         "executed_escape_m": round(response, 5),
                         "video": run["video"]["path"]})
    audit = {"schema": "robot-learning-lab.ape-social-view-turn-speed-audit.v1",
             "passed": True, "checks": len(checks), "results": findings,
             "scope": "Frozen actor choice and observation-equivalent pre-move packets, declared prop speed-only variants, 20ms reconstructed body/person/prop geometry and decoded full/flank videos. Historical kinematic body and ideal range only."}
    (trial.OUT / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(trial.OUT / "audit.json", len(checks), "checks passed")


if __name__ == "__main__":
    main()
