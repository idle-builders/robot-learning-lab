"""Independent provenance, packet-causality, geometry and media audit for V82–V84."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import audit_owner_joint_hazard_response as media
import owner_radial_wait_resume_probe as trial


def check(ok, label, checks):
    if not ok:
        raise AssertionError(label)
    checks.append(label)


def main():
    current = json.loads((trial.OUT / "report.json").read_text())
    prior = json.loads((trial.v82.OUT / "report.json").read_text())
    probe = json.loads((trial.v83.OUT / "report.json").read_text())
    screen = json.loads((trial.v82.v81.OUT / "report.json").read_text())
    sha = trial.v82.v77.v72.v52.sha
    checks = []
    check(current["schema"] == "robot-learning-lab.ape-radial-wait-resume.v1"
          and current["restage_wait_on_block"] and
          current["restage_max_ticks"] == trial.MAX_TICKS,
          "schema and bounded wait contract", checks)
    for key, path in (
        ("source_sha256", trial.__file__),
        ("v82_source_sha256", trial.v82.__file__),
        ("v83_report_sha256", trial.v83.OUT / "report.json"),
        ("runner_source_sha256", trial.wait_runner.__file__),
        ("v81_report_sha256", trial.v82.v81.OUT / "report.json"),
        ("scene_sha256", trial.v82.v77.runner.v35.moving.SCENE_XML),
        ("current_design_pointer_sha256",
         trial.v82.ROOT / "robot/ape-current-design.json"),
    ):
        check(current[key] == sha(path), f"source {key}", checks)
    check(prior["runner_source_sha256"] == sha(
          trial.v82.v77.runner.__file__) and
          prior["source_sha256"] == sha(trial.v82.__file__) and
          probe["v82_report_sha256"] == sha(trial.v82.OUT / "report.json"),
          "prior source chain", checks)
    source = trial.v82.v77.runner.original()
    summary = source["summary"]
    policy = json.loads(trial.v82.v77.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    model = mujoco.MjModel.from_xml_path(str(
        trial.v82.v77.runner.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    prop_geom = model.geom("route_obstacle_geom").id
    prop_body = model.body("route_obstacle").id
    model.geom_size[prop_geom, 1] = \
        trial.v82.v77.v72.v64.v51.v46.PROP_HALF_HEIGHT_M
    model.body_pos[prop_body, 2] = \
        trial.v82.v77.v72.v64.v51.v46.PROP_CENTER_Z_M
    start, _, checkpoint_s, _, _ = \
        trial.v82.v77.v72.v64.v51.v46.low_prop_geometry(
            trial.v82.v77.runner.geometry_at)(
                summary, source["result"]["result"]["episode"],
                scenario, model, data)
    check(abs(checkpoint_s-current["checkpoint_s"]) < 1e-8,
          "source checkpoint", checks)
    findings = []
    for item in current["results"]:
        label = item["path"]
        old = next(row for row in prior["results"] if row["path"] == label)
        held_probe = next(row for row in probe["results"] if
                          row["path"] == label)
        run = item["run"]
        rows = json.loads(Path(run["trace"]["path"]).read_text())
        old_rows = json.loads(Path(old["run"]["trace"]["path"]).read_text())
        check(sha(run["trace"]["path"]) == run["trace"]["sha256"] and
              sha(run["video"]["path"]) == run["video"]["sha256"] and
              media.video_frames(run["video"]["path"], (528, 1040, 3)) ==
              run["video"]["frames"], f"{label} decoded full media", checks)
        flank = next(row for row in rows if row["stage"] ==
                     "flank_sensor_video")
        check(sha(flank["path"]) == flank["sha256"] and
              media.video_frames(flank["path"], (480, 640, 3)) ==
              flank["frames"], f"{label} decoded flank RGB-D", checks)
        hold = [row for row in rows if row["stage"] ==
                "post_alarm_range_watch"]
        steps = [row for row in rows if row["stage"] ==
                 "pre_show_restage_step"]
        waits = [row for row in rows if row["stage"] ==
                 "pre_show_restage_wait"]
        watch = [row for row in rows if row["stage"] ==
                 "second_inference_range_watch"]
        old_hold = [row for row in old_rows if row["stage"] ==
                    "post_alarm_range_watch"]
        old_steps = [row for row in old_rows if row["stage"] ==
                     "pre_show_restage_step"]
        check(len(hold) == len(old_hold) == 14 and
              len(steps) == item["restage_steps"] == 34 and
              len(old_steps) == 20 and len(waits) == 2 and
              len(watch) == trial.wait_runner.FIXED_MONITOR_TICKS and
              item["fresh_captures"] == 1 and
              item["item_valid_depth_pixels"] > 0,
              f"{label} complete owner clock", checks)
        for index, (a, b) in enumerate(zip(hold+steps[:20],
                                           old_hold+old_steps)):
            check(a["time_s"] == b["time_s"] and
                  a["actor_packet"] == b["actor_packet"] and
                  a.get("ape_xy_m") == b.get("ape_xy_m"),
                  f"{label} matched pre-block prefix {index}", checks)
        check(np.allclose(waits[0]["ape_xy_m"],
                          held_probe["held_xy_m"]) and
              abs(waits[0]["actor_decision"]["minimum_endpoint_gap_m"]-
                  next(row for row in old_rows if row["stage"] ==
                       "pre_show_restage_decision")["actor_decision"][
                           "minimum_endpoint_gap_m"]) < 1e-8,
              f"{label} exact blocked next step", checks)
        ordered = [row for row in rows if row["stage"] in
                   ("pre_show_restage_step", "pre_show_restage_wait")]
        prior_packet = hold[-1]["actor_packet"]
        prior_pose = np.asarray(start)
        for index, row in enumerate(ordered):
            proposed = prior_pose+.025*trial.v82.DIRECTION
            endpoints = trial.v82.v77.v59.v51.v41.JointHazardResponse.endpoints(
                prior_packet, prior_pose)
            gap = float(np.min(np.linalg.norm(
                endpoints-proposed, axis=1)))
            decision = row["actor_decision"]
            check(abs(gap-decision["minimum_endpoint_gap_m"]) < 1e-8 and
                  ((gap < trial.v82.v77.v59.v56.MIN_PATH_ENDPOINT_GAP_M)
                   == (row["stage"] == "pre_show_restage_wait")),
                  f"{label} prior-packet action {index}", checks)
            if row["stage"] == "pre_show_restage_step":
                check(np.allclose(row["ape_xy_m"], proposed, atol=1e-8),
                      f"{label} executed 2.5cm step {index}", checks)
                prior_pose = proposed
            else:
                check(np.allclose(row["ape_xy_m"], prior_pose, atol=1e-8),
                      f"{label} true held pose {index}", checks)
            prior_packet = row["actor_packet"]
        target = np.asarray(screen["all_eight_clear_candidates"][0][
            "target_xy_m"])
        check(np.allclose(steps[-1]["ape_xy_m"], target, atol=1e-8) and
              run["outcome"] == "new_show_metric_item_seen_unattributed",
              f"{label} reached task-visible destination", checks)
        # Reconstruct the piecewise-linear kinematic body path from the actual
        # 10 Hz commands rather than trusting evaluator sample pose fields.
        timeline = [(checkpoint_s, np.asarray(start)),
                    (run["resume_time_s"], np.asarray(start))]
        timeline.extend((row["time_s"], np.asarray(row["ape_xy_m"]))
                        for row in ordered+watch)
        timeline.sort(key=lambda item: item[0])
        times = np.asarray([t for t, _ in timeline])
        poses = np.asarray([xy for _, xy in timeline])
        endpoint = np.asarray(item["endpoint_xy_m"])
        prop_min = person_min = float("inf")
        for sample in run["evaluator_gap_samples"]:
            t = sample["time_s"]
            pose = np.asarray([np.interp(t, times, poses[:, axis])
                               for axis in range(2)])
            check(np.linalg.norm(pose-sample["ape_xy_m"]) < .00011,
                  f"{label} 20ms commanded pose {t:.2f}", checks)
            person = trial.v82.v77.runner.human_xy(
                trial.v82.v77.v72.v64.v54.CASE, summary, t,
                run["request_time_s"], run["second_inference_start_s"])
            trial.v82.v77.runner.v35.two.set_distractor(model, person)
            model.body_pos[model.body("audit_ape").id, :2] = pose
            model.body_pos[prop_body, :2] = trial.v82.v77.v65.prop_at(
                t, current["checkpoint_s"], endpoint,
                item["finish_after_checkpoint_s"])
            mujoco.mj_forward(model, data)
            pg = trial.v82.v77.runner.prop_gap(model, data, pose)
            hg = trial.v82.v77.runner.geometry.gap(model, data, pose, person)
            check(abs(pg-sample["prop_gap_m"]) < .00011 and
                  abs(hg-sample["gap_m"]) < .00011,
                  f"{label} 20ms scene geometry {t:.2f}", checks)
            prop_min = min(prop_min, pg)
            person_min = min(person_min, hg)
        check(abs(prop_min-run["minimum_prop_gap_m"]) < .00011 and
              abs(person_min-run["minimum_proxy_gap_m"]) < .00011 and
              prop_min >= .05 and person_min >= .05,
              f"{label} modeled minimum gaps", checks)
        findings.append({"path": label,
                         "outcome": run["outcome"],
                         "restage_wait_ticks": len(waits),
                         "restage_steps": len(steps),
                         "item_valid_depth_pixels":
                             item["item_valid_depth_pixels"],
                         "minimum_prop_gap_m": round(prop_min, 5),
                         "minimum_person_gap_m": round(person_min, 5),
                         "video": run["video"]["path"]})
    audit = {"schema": "robot-learning-lab.ape-radial-wait-resume-audit.v1",
             "passed": True, "checks": len(checks), "results": findings,
             "scope": "Historical-proxy V82 blocked prefix, V83 held-step observation, V84 actor prior-packet wait/move causality, 20ms MuJoCo geometry and decoded owner/flank videos. Destination selected by V81 evaluator futures, so this is not a learned or general actor planner."}
    (trial.OUT / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(trial.OUT / "audit.json", len(checks), "checks passed")


if __name__ == "__main__":
    main()
