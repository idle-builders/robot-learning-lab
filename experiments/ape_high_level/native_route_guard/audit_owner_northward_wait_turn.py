"""Independent trace, sensor, geometry and media checks for the V77 owner episodes."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import audit_owner_joint_hazard_response as media
import audit_owner_low_prop_scan_planes as sensor
import owner_northward_wait_turn as trial


def check(ok, label, checks):
    if not ok:
        raise AssertionError(label)
    checks.append(label)


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    assert report["schema"] == "robot-learning-lab.ape-northward-owner-turn.v1"
    checks = []
    sha = trial.v72.v52.sha
    for key, path in (
        ("source_sha256", trial.__file__),
        ("v76_report_sha256", trial.v76.OUT / "report.json"),
        ("v59_source_sha256", trial.v59.__file__),
        ("runner_source_sha256", trial.runner.__file__),
        ("scene_sha256", trial.runner.v35.moving.SCENE_XML),
        ("current_design_pointer_sha256",
         trial.ROOT / "robot/ape-current-design.json"),
    ):
        check(report[key] == sha(path), f"source {key}", checks)
    base = trial.runner.original()
    summary = base["summary"]
    policy = json.loads(trial.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    v76 = json.loads((trial.v76.OUT / "report.json").read_text())
    start = np.asarray(v76["start_xy_m"])
    direction = trial.v76.DIRECTIONS["northwest"]
    target = start + .55*direction
    model = mujoco.MjModel.from_xml_path(str(trial.runner.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    prop_geom = model.geom("route_obstacle_geom").id
    prop_body = model.body("route_obstacle").id
    model.geom_size[prop_geom, 1] = trial.v72.v64.v51.v46.PROP_HALF_HEIGHT_M
    model.body_pos[prop_body, 2] = trial.v72.v64.v51.v46.PROP_CENTER_Z_M
    all_rows = []
    findings = []
    for item in report["results"]:
        run = item["run"]
        rows = json.loads(Path(run["trace"]["path"]).read_text())
        all_rows.append(rows)
        check(sha(run["trace"]["path"]) == run["trace"]["sha256"],
              f"{item['variant']} trace hash", checks)
        check(sha(run["video"]["path"]) == run["video"]["sha256"],
              f"{item['variant']} video hash", checks)
        check(media.video_frames(run["video"]["path"], (528, 1040, 3)) ==
              run["video"]["frames"], f"{item['variant']} decoded video", checks)
        flank = next(row for row in rows if row["stage"] == "flank_sensor_video")
        check(sha(flank["path"]) == flank["sha256"] and
              media.video_frames(flank["path"], (480, 640, 3)) == flank["frames"],
              f"{item['variant']} flank video", checks)
        hold = [row for row in rows if row["stage"] == "post_alarm_range_watch"]
        steps = [row for row in rows if row["stage"] == "pre_show_restage_step"]
        watch = [row for row in rows if row["stage"] ==
                 "second_inference_range_watch"]
        captures = [row for row in rows if row["stage"] ==
                    "fresh_new_show_capture"]
        decision = next(row for row in rows if row["stage"] ==
                        "pre_show_restage_decision")
        final = next(row for row in rows if row["stage"] ==
                     "second_inference_result")
        check(len(hold) == 14 and len(steps) == item["restage_steps"] == 22
              and len(watch) == 36 and len(captures) ==
              item["fresh_item_captures"] == 1,
              f"{item['variant']} complete owner clock", checks)
        check(item["actor_choice"] == decision["actor_decision"]["choice"] ==
              "northwest_55cm" and item["actor_pose_decision"] ==
              decision["actor_decision"],
              f"{item['variant']} actor choice", checks)
        moving = decision["actor_decision"]["dynamic_surfaces_from_hold"]
        check(len(moving) == 1 and moving[0]["ray_index"] == 53,
              f"{item['variant']} retained moving ray", checks)
        old, now = (hold[0]["actor_packet"][53],
                    hold[-1]["actor_packet"][53])
        angle = math.radians(-180.+5.*53)
        surface = start+now*np.array([math.cos(angle), math.sin(angle)])
        check(old-now >= .5 and now > 1.5 and
              np.allclose(surface, moving[0]["surface_xy_m"], atol=1e-8),
              f"{item['variant']} moving surface from sensor", checks)
        last_hold_t = hold[-1]["time_s"]
        for index, row in enumerate(steps):
            expected = start+(index+1)*.025*direction
            check(np.allclose(row["ape_xy_m"], expected, atol=1e-8) and
                  abs(row["time_s"]-(last_hold_t+(index+1)*.1)) < 1e-8,
                  f"{item['variant']} step {index}", checks)
        check(np.allclose(steps[-1]["ape_xy_m"], target, atol=1e-8) and
              abs(captures[0]["time_s"]-run["second_inference_start_s"]) < 1e-8,
              f"{item['variant']} pose and fresh capture", checks)
        # The task's scalar alarm precedes any moving-return alarm. Preserve
        # that distinction: clearance alone is not a successful item result.
        first_risk = next((row for row in watch if row["risk_streak"] >= 2), None)
        check(first_risk is not None and abs(first_risk["time_s"]-
              run["second_inference_start_s"]-.1) < 1e-5 and
              first_risk["near_body_observation"]["alarm"] is False and
              first_risk["nearest_side_range_m"] < .70 and
              abs(final["alarm_s"]-first_risk["time_s"]) < 1e-5,
              f"{item['variant']} static-range false task alarm", checks)
        check(run["outcome"] in (
              "new_show_motion_unresolved_again",
              "near_body_escape_owner_turn_unresolved"),
              f"{item['variant']} unresolved outcome", checks)
        endpoint = trial.v72.v64.PATHS[item["path"]]
        finish_s = item["finish_after_checkpoint_s"]
        samples = run["evaluator_gap_samples"]
        person_min = prop_min = float("inf")
        for sample in samples:
            t = sample["time_s"]
            pose = np.asarray(sample["ape_xy_m"])
            person = trial.runner.human_xy(
                trial.v72.v64.v54.CASE, summary, t,
                run["request_time_s"], run["second_inference_start_s"])
            trial.runner.v35.two.set_distractor(model, person)
            model.body_pos[model.body("audit_ape").id, :2] = pose
            model.body_pos[prop_body, :2] = trial.v65.prop_at(
                t, report["checkpoint_s"], endpoint, finish_s)
            mujoco.mj_forward(model, data)
            person_gap = trial.runner.geometry.gap(model, data, pose, person)
            prop_gap = trial.runner.prop_gap(model, data, pose)
            check(abs(person_gap-sample["gap_m"]) < .00011 and
                  abs(prop_gap-sample["prop_gap_m"]) < .00011,
                  f"{item['variant']} geometry {t:.2f}", checks)
            person_min = min(person_min, person_gap)
            prop_min = min(prop_min, prop_gap)
        check(abs(person_min-run["minimum_proxy_gap_m"]) < .00011 and
              abs(prop_min-run["minimum_prop_gap_m"]) < .00011 and
              person_min >= .05 and prop_min >= .05,
              f"{item['variant']} minimum gaps", checks)
        findings.append({"variant": item["variant"],
                         "choice": item["actor_choice"],
                         "outcome": run["outcome"],
                         "minimum_person_gap_m": round(person_min, 5),
                         "minimum_prop_gap_m": round(prop_min, 5),
                         "first_scalar_alarm_s": final["alarm_s"],
                         "first_near_body_alarm_s": final["near_body_alarm_s"],
                         "video": run["video"]["path"]})
    # Each script branches only after the same held pre-turn sensing history.
    reference = all_rows[0]
    common = [row for row in reference if row["stage"] in
              ("post_alarm_range_watch", "pre_show_restage_step")]
    for other in all_rows[1:]:
        paired = [row for row in other if row["stage"] in
                  ("post_alarm_range_watch", "pre_show_restage_step")]
        check(len(paired) == len(common), "matched pre-turn row count", checks)
        for index, (left, right) in enumerate(zip(common, paired)):
            sensor.same_packet(left["actor_packet"], right["actor_packet"])
            check(left["stage"] == right["stage"] and
                  left["time_s"] == right["time_s"],
                  f"matched pre-turn row {index}", checks)
    output = {"schema": "robot-learning-lab.ape-northward-owner-audit.v1",
              "passed": True, "checks": len(checks),
              "results": findings,
              "matched_pre_turn_rows_per_path": len(common),
              "scope": "Six development owner episodes; independent 20ms MuJoCo proxy gaps, actor choice/step and capture clocks, matched pre-turn packets, decoded videos and trace hashes. The planner was selected after a six-script evaluator screen. Historical kinematic proxy and ideal range; no physical stop or held-out generalization."}
    (trial.OUT / "audit.json").write_text(json.dumps(output, indent=2)+"\n")
    print(trial.OUT / "audit.json", len(checks), "checks passed")


if __name__ == "__main__":
    main()
