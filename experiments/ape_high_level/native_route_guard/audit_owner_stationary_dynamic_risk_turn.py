"""Independent paired episode and geometry audit of the V78 risk gate."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import audit_owner_joint_hazard_response as media
import audit_owner_low_prop_scan_planes as sensor
import owner_stationary_dynamic_risk_turn as trial


def check(ok, label, checks):
    if not ok:
        raise AssertionError(label)
    checks.append(label)


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    prior = json.loads((trial.v77.OUT / "report.json").read_text())
    sha = trial.v77.v72.v52.sha
    checks = []
    check(report["schema"] ==
          "robot-learning-lab.ape-stationary-dynamic-risk.v1",
          "schema", checks)
    for key, source in (
        ("source_sha256", trial.__file__),
        ("v77_source_sha256", trial.v77.__file__),
        ("runner_source_sha256", trial.v77.runner.__file__),
        ("v76_report_sha256", trial.v77.v76.OUT / "report.json"),
        ("scene_sha256", trial.v77.runner.v35.moving.SCENE_XML),
        ("current_design_pointer_sha256",
         trial.v77.ROOT / "robot/ape-current-design.json"),
    ):
        check(report[key] == sha(source), f"source {key}", checks)
    model = mujoco.MjModel.from_xml_path(str(
        trial.v77.runner.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    prop_body = model.body("route_obstacle").id
    prop_geom = model.geom("route_obstacle_geom").id
    model.geom_size[prop_geom, 1] = trial.v77.v72.v64.v51.v46.PROP_HALF_HEIGHT_M
    model.body_pos[prop_body, 2] = trial.v77.v72.v64.v51.v46.PROP_CENTER_Z_M
    source = trial.v77.runner.original()
    summary = source["summary"]
    scored = []
    for item, before in zip(report["results"], prior["results"]):
        label = item["variant"]
        check(label == before["variant"] and
              item["actor_choice"] == before["actor_choice"] ==
              "northwest_55cm", f"{label} matched variant and pose", checks)
        run = item["run"]
        old = before["run"]
        rows = json.loads(Path(run["trace"]["path"]).read_text())
        baseline = json.loads(Path(old["trace"]["path"]).read_text())
        check(sha(run["trace"]["path"]) == run["trace"]["sha256"] and
              sha(run["video"]["path"]) == run["video"]["sha256"],
              f"{label} media hashes", checks)
        check(media.video_frames(run["video"]["path"], (528, 1040, 3)) ==
              run["video"]["frames"], f"{label} decoded full video", checks)
        hold = [r for r in rows if r["stage"] == "post_alarm_range_watch"]
        old_hold = [r for r in baseline if r["stage"] ==
                    "post_alarm_range_watch"]
        watch = [r for r in rows if r["stage"] ==
                 "second_inference_range_watch"]
        old_watch = [r for r in baseline if r["stage"] ==
                     "second_inference_range_watch"]
        check(len(hold) == len(old_hold) == 14 and
              len(watch) == len(old_watch) == 36,
              f"{label} paired sensor clocks", checks)
        for index, (a, b) in enumerate(zip(watch, old_watch)):
            sensor.same_packet(a["actor_packet"], b["actor_packet"])
            check(a["time_s"] == b["time_s"] and
                  a["ape_xy_m"] == b["ape_xy_m"] and
                  a["near_body_observation"] == b["near_body_observation"] and
                  a["evaluator_prop_gap_m"] == b["evaluator_prop_gap_m"],
                  f"{label} matched physical watch {index}", checks)
            decision = a["task_risk_decision"]
            observation = a["near_body_observation"]
            check(decision["risk"] == observation["alarm"] and
                  decision["capture_time_s"] ==
                  observation["capture_time_s"] and
                  a["risk_streak"] == (0 if not decision["risk"] else
                  (watch[index-1]["risk_streak"]+1 if index else 1)),
                  f"{label} dynamic task gate {index}", checks)
        for a, b in zip(hold, old_hold):
            sensor.same_packet(a["actor_packet"], b["actor_packet"])
        final = next(r for r in rows if r["stage"] ==
                     "second_inference_result")
        check(watch[1]["task_risk_decision"]["risk"] is False and
              watch[1]["nearest_side_range_m"] < .70 and
              (final["alarm_s"] is None or
               final["alarm_s"] > watch[1]["time_s"]),
              f"{label} fixed wall retained without task alarm", checks)
        is_inward = label == "inward_normal"
        check((final["near_body_alarm_s"] is not None) == is_inward and
              (run["outcome"] ==
               "near_body_escape_owner_turn_unresolved") == is_inward,
              f"{label} near-body outcome", checks)
        if not is_inward:
            check(run["outcome"] ==
                  "new_show_metric_item_seen_unattributed" and
                  any(r["stage"] == "fresh_item_candidate" for r in rows) and
                  any(r["stage"] == "new_show_metric_item_seen" for r in rows),
                  f"{label} item seen without owner attribution", checks)
        endpoint = trial.v77.v72.v64.PATHS[item["path"]]
        person_min = prop_min = float("inf")
        for sample in run["evaluator_gap_samples"]:
            t = sample["time_s"]
            pose = np.asarray(sample["ape_xy_m"])
            person = trial.v77.runner.human_xy(
                trial.v77.v72.v64.v54.CASE, summary, t,
                run["request_time_s"], run["second_inference_start_s"])
            trial.v77.runner.v35.two.set_distractor(model, person)
            model.body_pos[model.body("audit_ape").id, :2] = pose
            model.body_pos[prop_body, :2] = trial.v77.v65.prop_at(
                t, report["checkpoint_s"], endpoint,
                item["finish_after_checkpoint_s"])
            mujoco.mj_forward(model, data)
            person_gap = trial.v77.runner.geometry.gap(
                model, data, pose, person)
            prop_gap = trial.v77.runner.prop_gap(model, data, pose)
            check(abs(person_gap-sample["gap_m"]) < .00011 and
                  abs(prop_gap-sample["prop_gap_m"]) < .00011,
                  f"{label} independent geometry {t:.2f}", checks)
            person_min = min(person_min, person_gap)
            prop_min = min(prop_min, prop_gap)
        check(abs(person_min-run["minimum_proxy_gap_m"]) < .00011 and
              abs(prop_min-run["minimum_prop_gap_m"]) < .00011 and
              person_min >= .05 and prop_min >= .05,
              f"{label} conservative minimum gap", checks)
        scored.append({"variant": label, "outcome": run["outcome"],
                       "person_gap_m": round(person_min, 5),
                       "prop_gap_m": round(prop_min, 5),
                       "near_body_alarm_s": final["near_body_alarm_s"],
                       "video": run["video"]["path"]})
    audit = {"schema": "robot-learning-lab.ape-stationary-risk-audit.v1",
             "passed": True, "checks": len(checks), "results": scored,
             "scope": "Matched V77/V78 sensor clocks, raw scans and movement, independent 20ms MuJoCo gaps, decoded full videos and task-risk decision replay. Six development scripts only; no held-out, physical stop or selected-body claim."}
    (trial.OUT / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(trial.OUT / "audit.json", len(checks), "checks passed")


if __name__ == "__main__":
    main()
