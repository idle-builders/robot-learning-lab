"""Independent sensor, fixture, timed-gap and media audit of V79."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import audit_owner_joint_hazard_response as media
import audit_owner_low_prop_scan_planes as sensor
import owner_northwest_new_turn_stress as trial


def check(ok, label, checks):
    if not ok:
        raise AssertionError(label)
    checks.append(label)


def first_below(rows, threshold):
    return next((row["time_s"] for row in rows if
                 row["prop_gap_m"] < threshold), None)


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    checkset = []
    sha = trial.v77.v72.v52.sha
    check(report["schema"] == "robot-learning-lab.ape-northwest-new-turn.v1",
          "schema", checkset)
    for key, path in (
        ("source_sha256", trial.__file__),
        ("v77_source_sha256", trial.v77.__file__),
        ("v78_source_sha256", trial.v78.__file__),
        ("runner_source_sha256", trial.v77.runner.__file__),
        ("scene_sha256", trial.v77.runner.v35.moving.SCENE_XML),
        ("current_design_pointer_sha256",
         trial.ROOT / "robot/ape-current-design.json"),
    ):
        check(report[key] == sha(path), f"source {key}", checkset)
    source = trial.v77.runner.original()
    summary = source["summary"]
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
            next(x for x in json.loads(trial.v77.runner.joined.CONFIG.read_text())
                 ["cases"] if x["id"] == "late_crossing"), model, data)
    check(abs(checkpoint_s-report["checkpoint_s"]) < 1e-8,
          "reproduced source checkpoint", checkset)
    scanner = trial.v77.v72.v64.v51.v46.plane_packet(
        trial.v77.runner.observed_packet, (.05, .30))
    findings = []
    holds = []
    for item in report["results"]:
        label = item["variant"]
        endpoint = np.asarray(item["endpoint_xy_m"])
        run = item["run"]
        rows = json.loads(Path(run["trace"]["path"]).read_text())
        check(sha(run["trace"]["path"]) == run["trace"]["sha256"] and
              sha(run["video"]["path"]) == run["video"]["sha256"] and
              media.video_frames(run["video"]["path"], (528, 1040, 3)) ==
              run["video"]["frames"], f"{label} full media", checkset)
        hold = [r for r in rows if r["stage"] == "post_alarm_range_watch"]
        steps = [r for r in rows if r["stage"] == "pre_show_restage_step"]
        watch = [r for r in rows if r["stage"] ==
                 "second_inference_range_watch"]
        holds.append(hold)
        check(len(hold) == 14 and len(steps) == 22 and len(watch) == 36 and
              item["actor_choice"] == "northwest_55cm",
              f"{label} complete actor clock and choice", checkset)
        # Packet reconstruction uses the saved pose and this script's
        # independent environment state. Future endpoint remains evaluator-only.
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
                report["turn_finish_after_checkpoint_s"])
            mujoco.mj_forward(model, data)
            observed = scanner(model, data, pose,
                               trial.v77.v72.v64.v54.CASE, 1000+index,
                               person_xy=person, hidden_return=False)
            sensor.same_packet(observed, row["actor_packet"])
            check(row["task_risk_decision"]["risk"] ==
                  row["near_body_observation"]["alarm"],
                  f"{label} task risk from raw track {index}", checkset)
        fixture_gap = float("inf")
        fixture_name = None
        for t in np.arange(report["checkpoint_s"]+
                           trial.v77.v65.TURN_START_S,
                           report["checkpoint_s"]+
                           report["turn_finish_after_checkpoint_s"]+.00001,
                           .02):
            model.body_pos[prop_body, :2] = trial.v77.v65.prop_at(
                float(t), report["checkpoint_s"], endpoint,
                report["turn_finish_after_checkpoint_s"])
            mujoco.mj_forward(model, data)
            for geom in range(model.ngeom):
                name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
                if (geom == prop_geom or name.startswith("audit_ape") or
                        "person" in name or "distractor" in name or
                        "floor" in name):
                    continue
                distance = float(mujoco.mj_geomDistance(
                    model, data, prop_geom, geom, 1., None))
                if distance < fixture_gap:
                    fixture_gap, fixture_name = distance, name
        check(abs(fixture_gap-item["fixture"]["minimum_gap_m"]) < 1e-8 and
              fixture_gap > .05 and fixture_name ==
              item["fixture"]["nearest_geom"],
              f"{label} fixture-clear path", checkset)
        prop_min = person_min = float("inf")
        samples = run["evaluator_gap_samples"]
        for sample in samples:
            t = sample["time_s"]
            pose = np.asarray(sample["ape_xy_m"])
            person = trial.v77.runner.human_xy(
                trial.v77.v72.v64.v54.CASE, summary, t,
                run["request_time_s"], run["second_inference_start_s"])
            trial.v77.runner.v35.two.set_distractor(model, person)
            model.body_pos[model.body("audit_ape").id, :2] = pose
            model.body_pos[prop_body, :2] = trial.v77.v65.prop_at(
                t, report["checkpoint_s"], endpoint,
                report["turn_finish_after_checkpoint_s"])
            mujoco.mj_forward(model, data)
            person_gap = trial.v77.runner.geometry.gap(
                model, data, pose, person)
            prop_gap = trial.v77.runner.prop_gap(model, data, pose)
            check(abs(person_gap-sample["gap_m"]) < .00011 and
                  abs(prop_gap-sample["prop_gap_m"]) < .00011,
                  f"{label} 20ms geometry {t:.2f}", checkset)
            person_min = min(person_min, person_gap)
            prop_min = min(prop_min, prop_gap)
        check(abs(person_min-run["minimum_proxy_gap_m"]) < .00011 and
              abs(prop_min-run["minimum_prop_gap_m"]) < .00011,
              f"{label} replayed minimum", checkset)
        final = next(r for r in rows if r["stage"] ==
                     "second_inference_result")
        alarm = final["near_body_alarm_s"]
        breach = first_below(samples, .05)
        if label == "near_northwest":
            check(prop_min < 0 and alarm is not None and breach is not None
                  and 0 <= breach-alarm <= .05,
                  "near path late alarm and overlap", checkset)
        else:
            check(prop_min >= .05 and breach is None and alarm is not None,
                  "clear path unnecessary escape", checkset)
        findings.append({"variant": label,
                         "fixture_gap_m": round(fixture_gap, 5),
                         "minimum_prop_gap_m": round(prop_min, 5),
                         "minimum_person_gap_m": round(person_min, 5),
                         "near_body_alarm_s": alarm,
                         "first_prop_below_5cm_s": breach,
                         "outcome": run["outcome"],
                         "video": run["video"]["path"]})
    for a, b in zip(*holds):
        sensor.same_packet(a["actor_packet"], b["actor_packet"])
        check(a["time_s"] == b["time_s"], "paired pre-turn clock", checkset)
    out = {"schema": "robot-learning-lab.ape-northwest-new-turn-audit.v1",
           "passed": True, "checks": len(checkset), "results": findings,
           "scope": "Adversarial same-kitchen endpoint change after V78; independently reconstructed low/mid range packets, static fixture gaps, 20ms ape/person/prop geometry and full videos. Historical kinematic proxy; no selected-body or held-out generalization."}
    (trial.OUT / "audit.json").write_text(json.dumps(out, indent=2)+"\n")
    print(trial.OUT / "audit.json", len(checkset), "checks passed")


if __name__ == "__main__":
    main()
