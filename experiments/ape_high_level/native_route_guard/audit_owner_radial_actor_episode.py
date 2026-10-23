"""Independent actor-choice, step causality, geometry and media audit for V85–V86."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import audit_owner_joint_hazard_response as media
import owner_radial_actor_episode as trial


def check(ok, label, checks):
    if not ok:
        raise AssertionError(label)
    checks.append(label)


def independent_choice(packet, history, start, angles, distances):
    """Rebuild the frozen tube score without calling V85's rank method."""
    v59 = trial.v85.v59
    start = np.asarray(start)
    endpoints = v59.v51.v41.JointHazardResponse.endpoints(packet, start)
    first, last = history[0]["packet"], history[-1]["packet"]
    surfaces = []
    for index in range(45, 62):
        old, now = first[index], last[index]
        if old is None or now is None or now <= 1.50 or old-now < .50:
            continue
        angle = math.radians(-180.+360.*index/len(last))
        surfaces.append(start+now*np.asarray([
            math.cos(angle), math.sin(angle)]))
    ranked = []
    for angle_deg in angles:
        direction = np.asarray([math.cos(math.radians(angle_deg)),
                                math.sin(math.radians(angle_deg))])
        for distance in distances:
            target = start+distance*direction
            gaps = [float(np.min(np.linalg.norm(
                endpoints-(start+f*distance*direction), axis=1)))
                for f in np.linspace(0., 1., 15)]
            station = float(np.linalg.norm(
                v59.v56.OWNER_SHOW_STATION_XY_M-target))
            if (sum(x is not None for x in packet) < 36 or
                    min(gaps) < .27 or gaps[-1] < .27 or station > 2.60):
                continue
            approach = float("inf")
            for surface in surfaces:
                for t in np.arange(0., distance/.25+3.50+.05, .10):
                    pose = start+min(t*.25/distance, 1.)*(target-start)
                    dx = max(abs(pose[0]-surface[0])-.40, 0.)
                    dy = max(surface[1]-pose[1]-1.20*t,
                             pose[1]-surface[1], 0.)
                    approach = min(approach, math.hypot(dx, dy)-.17)
            combined = min(gaps[-1]-.17, approach)
            ranked.append((combined, -distance, angle_deg, distance,
                           target))
    return max(ranked, key=lambda row: row[:2]), len(ranked), len(surfaces)


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    rank_report = json.loads((trial.v85.OUT / "report.json").read_text())
    screen = json.loads((trial.v85.v81.OUT / "report.json").read_text())
    sha = trial.v82.v77.v72.v52.sha
    checks = []
    check(report["schema"] == "robot-learning-lab.ape-radial-actor-episode.v1"
          and report["restage_wait_on_block"] and
          report["candidate_source"] == "V85 actor-only radial score",
          "actor radial contract", checks)
    for key, path in (
        ("source_sha256", trial.__file__),
        ("v85_report_sha256", trial.v85.OUT / "report.json"),
        ("runner_source_sha256", trial.runner.__file__),
        ("scene_sha256", trial.runner.v35.moving.SCENE_XML),
        ("current_design_pointer_sha256",
         trial.ROOT / "robot/ape-current-design.json")):
        check(report[key] == sha(path), f"source {key}", checks)
    check(rank_report["source_sha256"] == sha(trial.v85.__file__) and
          rank_report["v81_report_sha256"] == sha(trial.v85.v81.OUT /
                                                   "report.json"),
          "rank source chain", checks)
    check(rank_report["actor_candidate_count"] == 80 and
          rank_report["actor_eligible_count"] == 58 and
          rank_report["actor_chosen"]["angle_deg"] == 202.5 and
          rank_report["actor_chosen"]["distance_m"] == .85 and
          rank_report["post_choice_evaluator"]["worst_prop_gap_m"] == -.14,
          "actor choice and held-out evaluator diagnostic", checks)
    source = trial.runner.original()
    summary = source["summary"]
    policy = json.loads(trial.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    model = mujoco.MjModel.from_xml_path(str(
        trial.runner.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    prop_body = model.body("route_obstacle").id
    model.geom_size[model.geom("route_obstacle_geom").id, 1] = \
        trial.v82.v77.v72.v64.v51.v46.PROP_HALF_HEIGHT_M
    model.body_pos[prop_body, 2] = \
        trial.v82.v77.v72.v64.v51.v46.PROP_CENTER_Z_M
    start, _, checkpoint_s, _, _ = \
        trial.v82.v77.v72.v64.v51.v46.low_prop_geometry(
            trial.runner.geometry_at)(
                summary, source["result"]["result"]["episode"],
                scenario, model, data)
    check(abs(checkpoint_s-report["checkpoint_s"]) < 1e-8 and
          np.allclose(start, rank_report["start_xy_m"]),
          "source checkpoint and start pose", checks)
    findings = []
    for item in report["results"]:
        label = item["path"]
        run = item["run"]
        rows = json.loads(Path(run["trace"]["path"]).read_text())
        check(sha(run["trace"]["path"]) == run["trace"]["sha256"] and
              sha(run["video"]["path"]) == run["video"]["sha256"] and
              media.video_frames(run["video"]["path"], (528, 1040, 3)) ==
              run["video"]["frames"], f"{label} decoded full video", checks)
        flank = next(row for row in rows if row["stage"] ==
                     "flank_sensor_video")
        check(sha(flank["path"]) == flank["sha256"] and
              media.video_frames(flank["path"], (480, 640, 3)) ==
              flank["frames"], f"{label} decoded flank video", checks)
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
              len(waits) == item["restage_waits"] == 0 and
              len(watch) == trial.runner.FIXED_MONITOR_TICKS and
              item["fresh_captures"] == 1,
              f"{label} complete owner clock", checks)
        history = [{"time_s": row["time_s"],
                    "packet": row["actor_packet"]} for row in hold]
        independently_chosen, eligible_count, surface_count = \
            independent_choice(history[-1]["packet"], history, start,
                               screen["angles_deg"], screen["distances_m"])
        chosen, ranked, surfaces = trial.v85.rank(
            history[-1]["packet"], history, start)
        check(chosen == item["actor_chosen"] ==
              rank_report["actor_chosen"] and
              surfaces == rank_report["actor_dynamic_surfaces"] and
              ranked == rank_report["actor_candidates"] and
              eligible_count == rank_report["actor_eligible_count"] and
              surface_count == len(surfaces) and
              independently_chosen[2:4] == (
                  chosen["angle_deg"], chosen["distance_m"]) and
              np.allclose(independently_chosen[4],
                          chosen["target_xy_m"], atol=1e-9) and
              abs(independently_chosen[0]-chosen[
                  "actor_combined_margin_m"]) < 1e-9,
              f"{label} raw-packet actor choice", checks)
        first = steps[0]["actor_decision"]
        check(first["candidate_source"] ==
              "V85 actor range-history radial ranking" and
              first["candidate_scores"] == ranked and
              first["time_indexed_choice_margin_m"] ==
              chosen["actor_combined_margin_m"],
              f"{label} logged actor decision", checks)
        direction = np.asarray([
            np.cos(np.deg2rad(chosen["angle_deg"])),
            np.sin(np.deg2rad(chosen["angle_deg"]))])
        prior_packet, prior_pose = hold[-1]["actor_packet"], np.asarray(start)
        for index, row in enumerate(steps):
            proposed = prior_pose+.025*direction
            endpoints = trial.v85.v59.v51.v41.JointHazardResponse.endpoints(
                prior_packet, prior_pose)
            gap = float(np.min(np.linalg.norm(
                endpoints-proposed, axis=1)))
            check(abs(gap-row["actor_decision"][
                  "minimum_endpoint_gap_m"]) < 1e-8 and
                  gap >= trial.v85.v59.v56.MIN_PATH_ENDPOINT_GAP_M and
                  np.allclose(row["ape_xy_m"], proposed, atol=1e-8),
                  f"{label} prior-packet executed step {index}", checks)
            prior_packet, prior_pose = row["actor_packet"], proposed
        check(np.allclose(steps[-1]["ape_xy_m"],
                          chosen["target_xy_m"], atol=1e-8),
              f"{label} actor destination reached", checks)
        timeline = [(checkpoint_s, np.asarray(start)),
                    (run["resume_time_s"], np.asarray(start))]
        timeline.extend((row["time_s"], np.asarray(row["ape_xy_m"]))
                        for row in steps+watch)
        timeline.sort(key=lambda pair: pair[0])
        times = np.asarray([t for t, _ in timeline])
        poses = np.asarray([xy for _, xy in timeline])
        _, endpoint, finish_s = next(row for row in trial.v85.v81.PATHS
                                     if row[0] == label)
        prop_min = person_min = float("inf")
        for sample in run["evaluator_gap_samples"]:
            t = sample["time_s"]
            pose = np.asarray([np.interp(t, times, poses[:, axis])
                               for axis in range(2)])
            check(np.linalg.norm(pose-sample["ape_xy_m"]) < .00011,
                  f"{label} 20ms commanded pose {t:.2f}", checks)
            person = trial.runner.human_xy(
                trial.v82.v77.v72.v64.v54.CASE, summary, t,
                run["request_time_s"], run["second_inference_start_s"])
            trial.runner.v35.two.set_distractor(model, person)
            model.body_pos[model.body("audit_ape").id, :2] = pose
            model.body_pos[prop_body, :2] = trial.v82.v77.v65.prop_at(
                t, checkpoint_s, endpoint, finish_s)
            mujoco.mj_forward(model, data)
            pg = trial.runner.prop_gap(model, data, pose)
            hg = trial.runner.geometry.gap(model, data, pose, person)
            check(abs(pg-sample["prop_gap_m"]) < .00011 and
                  abs(hg-sample["gap_m"]) < .00011,
                  f"{label} 20ms scene geometry {t:.2f}", checks)
            prop_min, person_min = min(prop_min, pg), min(person_min, hg)
        check(abs(prop_min-run["minimum_prop_gap_m"]) < .00011 and
              abs(person_min-run["minimum_proxy_gap_m"]) < .00011,
              f"{label} independently scored minima", checks)
        screen_row = next(row for row in screen["candidates"] if
                          row["angle_deg"] == chosen["angle_deg"] and
                          row["distance_m"] == chosen["distance_m"])
        screen_path = next(row for row in screen_row["evaluator_paths"]
                           if row["path"] == label)
        # V81's fixed-clock screen starts the returning person from a
        # nominal distance/speed deadline. The full owner task resumes that
        # person from its actual capture/inference clock, so compare only the
        # identical scripted prop here; score person from the full replay.
        check(abs(screen_path["minimum_prop_gap_m"]-prop_min) < .001,
              f"{label} independent V81 prop comparison", checks)
        if label == "far_west_fast":
            check(prop_min < 0. and run["outcome"] ==
                  "near_body_escape_owner_turn_unresolved" and
                  item["item_valid_depth_pixels"] is None,
                  "changed fast turn overlap and no item claim", checks)
        else:
            check(prop_min >= .05 and run["outcome"] ==
                  "new_show_metric_item_seen_unattributed" and
                  item["item_valid_depth_pixels"] > 0,
                  f"{label} visual result with modeled clearance", checks)
        findings.append({"path": label, "outcome": run["outcome"],
                         "chosen_angle_deg": chosen["angle_deg"],
                         "chosen_distance_m": chosen["distance_m"],
                         "minimum_prop_gap_m": round(prop_min, 5),
                         "minimum_person_gap_m": round(person_min, 5),
                         "fresh_item_depth_pixels":
                             item["item_valid_depth_pixels"],
                         "video": run["video"]["path"]})
    audit = {"schema": "robot-learning-lab.ape-radial-actor-episode-audit.v1",
             "passed": True, "checks": len(checks), "results": findings,
             "scope": "Raw range-history radial choice, logged actor score, preceding-packet steps, 20ms MuJoCo person/prop gaps and decoded full/flank videos. Kinematic historical proxy and one scripted kitchen; no physical safety inference."}
    (trial.OUT / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(trial.OUT / "audit.json", len(checks), "checks passed")


if __name__ == "__main__":
    main()
