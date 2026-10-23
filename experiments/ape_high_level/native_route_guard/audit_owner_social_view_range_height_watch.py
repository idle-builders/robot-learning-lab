"""Audit V89's height-only change, raw packets, task outcomes and videos."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import audit_owner_joint_hazard_response as media
import owner_social_view_range_height_watch as trial


def check(ok, label, checks):
    if not ok:
        raise AssertionError(label)
    checks.append(label)


def main():
    sha = trial.v86.v82.v77.v72.v52.sha
    suite = json.loads((trial.OUT / "report.json").read_text())
    checks = []
    check(suite["source_sha256"] == sha(trial.__file__) and
          suite["v87_report_sha256"] == sha(trial.v87.OUT / "report.json")
          and suite["v88_report_sha256"] == sha(trial.v88.OUT / "report.json")
          and tuple(suite["cases"]) == trial.CASES,
          "suite frozen provenance", checks)
    source = trial.v86.runner.original()
    summary = source["summary"]
    policy = json.loads(trial.v86.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    model = mujoco.MjModel.from_xml_path(str(
        trial.v86.runner.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    prop_body = model.body("route_obstacle").id
    model.geom_size[model.geom("route_obstacle_geom").id, 1] = \
        trial.v86.v82.v77.v72.v64.v51.v46.PROP_HALF_HEIGHT_M
    model.body_pos[prop_body, 2] = \
        trial.v86.v82.v77.v72.v64.v51.v46.PROP_CENTER_Z_M
    _, _, checkpoint_s, _, _ = \
        trial.v86.v82.v77.v72.v64.v51.v46.low_prop_geometry(
            trial.v86.runner.geometry_at)(
                summary, source["result"]["result"]["episode"],
                scenario, model, data)
    reference_reports = {
        "outward_normal": json.loads((trial.v87.OUT / "report.json").read_text()),
        "grid_x0_y2": json.loads((trial.v88.OUT / "report.json").read_text()),
        "grid_x1_y1": json.loads((trial.v88.OUT / "report.json").read_text())}
    all_watch = {}
    findings = []
    for mode, heights in trial.MODES.items():
        path = trial.OUT / mode / "report.json"
        report = json.loads(path.read_text())
        entry = next(x for x in suite["modes"] if x["height_mode"] == mode)
        check(entry["report_sha256"] == sha(path) and
              report["source_sha256"] == sha(trial.__file__) and
              report["v86_source_sha256"] == sha(trial.v86.__file__) and
              report["runner_source_sha256"] == sha(
                  trial.v86.runner.__file__) and
              tuple(report["held_watch_heights_m"]) == heights and
              tuple(report["pre_show_heights_m"]) == (.05, .30) and
              abs(report["checkpoint_s"]-checkpoint_s) < 1e-9,
              f"{mode} source and height contract", checks)
        sensor = trial.v86.v82.v77.v72.v64.v51.v46.plane_packet(
            trial.v86.runner.observed_packet, heights)
        for item in report["results"]:
            name = item["path"]
            label = f"{mode}/{name}"
            run = item["run"]
            rows = json.loads(Path(run["trace"]["path"]).read_text())
            reference = next(x for x in reference_reports[name]["results"]
                             if x["path"] == name)
            ref_rows = json.loads(Path(reference["run"]["trace"]["path"]).read_text())
            pre = [x for x in rows if x["stage"] in
                   ("post_alarm_range_watch", "pre_show_restage_step",
                    "pre_show_restage_wait", "fresh_new_show_capture")]
            ref_pre = [x for x in ref_rows if x["stage"] in
                       ("post_alarm_range_watch", "pre_show_restage_step",
                        "pre_show_restage_wait", "fresh_new_show_capture")]
            pre_normalized = [dict(x, raw_rgbd=x["raw_rgbd"]["sha256"])
                              if x["stage"] == "fresh_new_show_capture"
                              else x for x in pre]
            ref_normalized = [dict(x, raw_rgbd=x["raw_rgbd"]["sha256"])
                              if x["stage"] == "fresh_new_show_capture"
                              else x for x in ref_pre]
            check(pre_normalized == ref_normalized and len(pre) == 51 and
                  run["outcome"] == next(x for x in entry["outcomes"]
                                          if x["path"] == name)["outcome"],
                  f"{label} identical fused pre-show route", checks)
            check(sha(run["trace"]["path"]) == run["trace"]["sha256"] and
                  sha(run["video"]["path"]) == run["video"]["sha256"] and
                  media.video_frames(run["video"]["path"],
                                     (528, 1040, 3)) == run["video"]["frames"],
                  f"{label} decoded full video", checks)
            flank = next(x for x in rows if x["stage"] ==
                         "flank_sensor_video")
            check(sha(flank["path"]) == flank["sha256"] and
                  media.video_frames(flank["path"],
                                     (480, 640, 3)) == flank["frames"],
                  f"{label} decoded flank video", checks)
            watch = [x for x in rows if x["stage"] ==
                     "second_inference_range_watch"]
            check(len(watch) == trial.v86.runner.FIXED_MONITOR_TICKS,
                  f"{label} full image-watch clock", checks)
            _, endpoint, finish_s = next(x for x in trial.PATHS
                                         if x[0] == name)
            for index, row in enumerate(watch):
                t = row["time_s"]
                pose = np.asarray(row["ape_xy_m"])
                person = trial.v86.runner.human_xy(
                    trial.v86.v82.v77.v72.v64.v54.CASE, summary, t,
                    run["request_time_s"],
                    run["second_inference_start_s"])
                trial.v86.runner.v35.two.set_distractor(model, person)
                model.body_pos[model.body("audit_ape").id, :2] = pose
                model.body_pos[prop_body, :2] = trial.v86.v82.v77.v65.prop_at(
                    t, checkpoint_s, endpoint, finish_s)
                mujoco.mj_forward(model, data)
                packet = sensor(model, data, pose,
                                trial.v86.v82.v77.v72.v64.v54.CASE,
                                1000+index, person_xy=person)
                check(len(packet) == len(row["actor_packet"]) == 72 and
                      all((a is None and b is None) or
                          (a is not None and b is not None and
                           abs(a-b) < 1e-8)
                          for a, b in zip(packet, row["actor_packet"])),
                      f"{label} raw {heights} packet {index}", checks)
                pg = trial.v86.runner.prop_gap(model, data, pose)
                check(abs(pg-row["evaluator_prop_gap_m"]) < .00011,
                      f"{label} held prop geometry {index}", checks)
            all_watch[(mode, name)] = [row["actor_packet"] for row in watch]
            first_breach = next((x["time_s"] for x in run[
                "evaluator_gap_samples"] if x["prop_gap_m"] < .05), None)
            alarm = next((x["time_s"] for x in rows if x["stage"] ==
                          "near_body_escape_plan"), None)
            check(abs(run["minimum_prop_gap_m"]-reference[
                  "run"]["minimum_prop_gap_m"]) < 1e-9 and
                  (name == "outward_normal" or
                   (first_breach is not None and
                    (alarm is None or alarm > first_breach))),
                  f"{label} same geometry and alarm timing", checks)
            findings.append({"mode": mode, "path": name,
                             "outcome": run["outcome"],
                             "minimum_prop_gap_m":
                                 run["minimum_prop_gap_m"],
                             "first_prop_below_5cm_s": first_breach,
                             "near_body_alarm_s": alarm,
                             "watch_first_nearest_m": min(
                                 value for value in watch[0]["actor_packet"]
                                 if value is not None),
                             "video": run["video"]["path"]})
    mid = [all_watch[("mid_30cm", name)] for name in trial.CASES]
    low = [all_watch[("low_5cm", name)] for name in trial.CASES]
    check(mid[0] == mid[1] == mid[2],
          "all 36 mid-only packets identical across control and threats", checks)
    check(all(a != b for a, b in zip(low[0], low[1])) and
          all(a != b for a, b in zip(low[0], low[2])),
          "all 36 low-only packets distinguish changed turns", checks)
    audit = {"schema": "robot-learning-lab.ape-social-view-range-height-audit.v1",
             "passed": True, "checks": len(checks), "results": findings,
             "scope": "Matched fused pre-show route, regenerated height-specific raw 72-ray watch packets and MuJoCo prop geometry, alarm timing and decoded full/flank videos. Historical proxy and ideal rays only; no real sensor selection."}
    (trial.OUT / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(trial.OUT / "audit.json", len(checks), "checks passed")


if __name__ == "__main__":
    main()
