"""Independent paired packet, geometry and media audit of V80."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import audit_owner_joint_hazard_response as media
import audit_owner_low_prop_scan_planes as sensor
import owner_northwest_range_height_ablation as trial


def check(ok, label, checks):
    if not ok:
        raise AssertionError(label)
    checks.append(label)


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    baseline = json.loads((trial.v79.OUT / "report.json").read_text())
    checks = []
    sha = trial.v77.v72.v52.sha
    check(report["schema"] ==
          "robot-learning-lab.ape-northwest-range-heights.v1",
          "schema", checks)
    for key, path in (
        ("source_sha256", trial.__file__),
        ("v79_source_sha256", trial.v79.__file__),
        ("v79_report_sha256", trial.v79.OUT / "report.json"),
        ("runner_source_sha256", trial.v77.runner.__file__),
        ("scene_sha256", trial.v77.runner.v35.moving.SCENE_XML),
        ("current_design_pointer_sha256",
         trial.ROOT / "robot/ape-current-design.json"),
    ):
        check(report[key] == sha(path), f"source {key}", checks)
    source = trial.v77.runner.original()
    summary = source["summary"]
    policy = json.loads(trial.v77.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    model = mujoco.MjModel.from_xml_path(str(
        trial.v77.runner.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    prop_geom = model.geom("route_obstacle_geom").id
    prop_body = model.body("route_obstacle").id
    model.geom_size[prop_geom, 1] = \
        trial.v77.v72.v64.v51.v46.PROP_HALF_HEIGHT_M
    model.body_pos[prop_body, 2] = \
        trial.v77.v72.v64.v51.v46.PROP_CENTER_Z_M
    _, _, checkpoint_s, _, _ = trial.v77.v72.v64.v51.v46.low_prop_geometry(
        trial.v77.runner.geometry_at)(
            summary, source["result"]["result"]["episode"],
            scenario, model, data)
    check(abs(checkpoint_s-report["checkpoint_s"]) < 1e-8,
          "checkpoint reproduction", checks)
    rows_out = []
    watch_packets = {}
    for item in report["results"]:
        label = item["profile"]+"/"+item["path"]
        heights = tuple(item["heights_m"])
        check(heights == tuple(trial.PROFILES[item["profile"]]),
              f"{label} profile declared", checks)
        base = next(x for x in baseline["results"] if
                    x["variant"] == item["path"])
        run = item["run"]
        trace = json.loads(Path(run["trace"]["path"]).read_text())
        old = json.loads(Path(base["run"]["trace"]["path"]).read_text())
        check(sha(run["trace"]["path"]) == run["trace"]["sha256"] and
              sha(run["video"]["path"]) == run["video"]["sha256"] and
              media.video_frames(run["video"]["path"], (528, 1040, 3)) ==
              run["video"]["frames"], f"{label} media", checks)
        old_hold = [r for r in old if r["stage"] ==
                    "post_alarm_range_watch"]
        hold = [r for r in trace if r["stage"] ==
                "post_alarm_range_watch"]
        old_steps = [r for r in old if r["stage"] ==
                     "pre_show_restage_step"]
        steps = [r for r in trace if r["stage"] ==
                 "pre_show_restage_step"]
        check(len(hold) == len(old_hold) == 14 and
              len(steps) == len(old_steps) == 22 and
              item["actor_choice"] == base["actor_choice"] ==
              "northwest_55cm", f"{label} fixed approach", checks)
        for index, (a, b) in enumerate(zip(hold+steps,
                                           old_hold+old_steps)):
            sensor.same_packet(a["actor_packet"], b["actor_packet"])
            check(a["time_s"] == b["time_s"] and
                  a.get("ape_xy_m") == b.get("ape_xy_m"),
                  f"{label} matched approach {index}", checks)
        scanner = trial.v77.v72.v64.v51.v46.plane_packet(
            trial.v77.runner.observed_packet, heights)
        watch = [r for r in trace if r["stage"] ==
                 "second_inference_range_watch"]
        check(len(watch) == 36, f"{label} full held watch", checks)
        watch_packets[(item["profile"], item["path"])] = [
            row["actor_packet"] for row in watch]
        endpoint = np.asarray(item["endpoint_xy_m"])
        for index, row in enumerate(watch):
            t = row["sensor_time_s_raw"]
            pose = np.asarray(row["ape_xy_m"])
            person = trial.v77.runner.human_xy(
                trial.v77.v72.v64.v54.CASE, summary, t,
                run["request_time_s"], run["second_inference_start_s"])
            trial.v77.runner.v35.two.set_distractor(model, person)
            model.body_pos[model.body("audit_ape").id, :2] = pose
            model.body_pos[prop_body, :2] = trial.v77.v65.prop_at(
                t, report["checkpoint_s"], endpoint,
                trial.v79.FINISH_AFTER_CHECKPOINT_S)
            mujoco.mj_forward(model, data)
            actual = scanner(model, data, pose,
                             trial.v77.v72.v64.v54.CASE, 1000+index,
                             person_xy=person, hidden_return=False)
            sensor.same_packet(actual, row["actor_packet"])
            check(row["task_risk_decision"]["risk"] ==
                  row["near_body_observation"]["alarm"],
                  f"{label} task decision {index}", checks)
        minimum = float("inf")
        for sample in run["evaluator_gap_samples"]:
            t = sample["time_s"]
            pose = np.asarray(sample["ape_xy_m"])
            model.body_pos[model.body("audit_ape").id, :2] = pose
            model.body_pos[prop_body, :2] = trial.v77.v65.prop_at(
                t, report["checkpoint_s"], endpoint,
                trial.v79.FINISH_AFTER_CHECKPOINT_S)
            mujoco.mj_forward(model, data)
            gap = trial.v77.runner.prop_gap(model, data, pose)
            check(abs(gap-sample["prop_gap_m"]) < .00011,
                  f"{label} 20ms prop gap {t:.2f}", checks)
            minimum = min(minimum, gap)
        check(abs(minimum-run["minimum_prop_gap_m"]) < .00011,
              f"{label} independent minimum", checks)
        final = next(r for r in trace if r["stage"] ==
                     "second_inference_result")
        if item["profile"] == "mid_30cm":
            check(final["near_body_alarm_s"] is None and
                  run["outcome"] ==
                  "new_show_metric_item_seen_unattributed",
                  f"{label} mid-only item outcome", checks)
        else:
            check(final["near_body_alarm_s"] is not None and
                  run["outcome"] ==
                  "near_body_escape_owner_turn_unresolved",
                  f"{label} low-only escape outcome", checks)
        rows_out.append({"profile": item["profile"], "path": item["path"],
                         "outcome": run["outcome"],
                         "minimum_prop_gap_m": round(minimum, 5),
                         "near_body_alarm_s": final["near_body_alarm_s"],
                         "video": run["video"]["path"]})
    for profile in trial.PROFILES:
        threat = watch_packets[(profile, "near_northwest")]
        clear = watch_packets[(profile, "east_clear_control")]
        equal = sum(a == b for a, b in zip(threat, clear))
        check(equal == (4 if profile == "low_05cm" else 36),
              f"{profile} paired route visibility", checks)
    audit = {"schema": "robot-learning-lab.ape-northwest-height-audit.v1",
             "passed": True, "checks": len(checks), "results": rows_out,
             "paired_equal_inference_packets": {
                 profile: sum(a == b for a, b in zip(
                     watch_packets[(profile, "near_northwest")],
                     watch_packets[(profile, "east_clear_control")]))
                 for profile in trial.PROFILES},
             "scope": "Same fused approach, independently reconstructed low/mid held packets and 20ms prop gaps, full video decoding on two adversarial development routes. Ideal rays and historical kinematic proxy only."}
    (trial.OUT / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(trial.OUT / "audit.json", len(checks), "checks passed")


if __name__ == "__main__":
    main()
