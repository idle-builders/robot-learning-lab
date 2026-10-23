"""Independently replay two moving props, full-body route and post-stop hold."""

from __future__ import annotations

import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from robocasa.models.scenes.kitchen_arena import KitchenArena
from robosuite.models.tasks import ManipulationTask

from audit_robocasa_layout2_routes_v409 import clearance
from run_robocasa_layout2_route_v407 import sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-layout2-two-props-v482"
CONFIG = Path(__file__).with_name("robocasa_layout2_two_props_v482.json")
SOURCE = Path(__file__).with_name("run_robocasa_layout2_two_props_v482.py")
PRIOR = ROOT / "artifacts/ape-robocasa-layout2-survey-v477/sensor-derived-map.npz"
POLICY = Path(__file__).with_name("run_robocasa_layout4_safe_side_v473.py")
STEP_S = .02


def secondary_at(case, time_s):
    start = np.asarray(case["secondary_from_xy_m"], dtype=float)
    end = np.asarray(case["secondary_to_xy_m"], dtype=float)
    ratio = min(1., max(0., (time_s - case["begin_s"]) /
                           (case["end_s"] - case["begin_s"])))
    return start + ratio * (end - start)


def evaluator_scene(case):
    random.seed(3201)
    np.random.seed(3201)
    arena = KitchenArena(layout_id=2, style_id=1,
                         rng=np.random.default_rng(3), clutter_mode=0)
    task = ManipulationTask(mujoco_arena=arena, mujoco_robots=[],
                            mujoco_objects=list(arena.fixtures.values()))
    for name, xy in (("route_obstacle", case["primary_xy_m"]),
                     ("second_obstacle", case["secondary_from_xy_m"])):
        body = ET.SubElement(task.worldbody, "body", name=name,
                             pos=f"{xy[0]} {xy[1]} .25")
        ET.SubElement(body, "geom", name=f"{name}_geom", type="cylinder",
                      size=".24 .25")
    ape = ET.SubElement(task.worldbody, "body", name="audit_ape", pos="1.05 -4.2 .25")
    ET.SubElement(ape, "geom", name="audit_ape_geom", type="cylinder",
                  size=".17 .25", rgba="0 0 0 0", contype="0", conaffinity="0")
    model = task.get_model()
    return model, mujoco.MjData(model)


def audit_case(case):
    scenario, episode = case["scenario"], case["episode"]
    model, data = evaluator_scene(scenario)
    ape_body = model.body("audit_ape").id
    ape_geom = model.geom("audit_ape_geom").id
    primary_body = model.body("route_obstacle").id
    secondary_body = model.body("second_obstacle").id
    primary_geom = model.geom("route_obstacle_geom").id
    secondary_geom = model.geom("second_obstacle_geom").id
    nearest = {"room_or_prop_gap_m_capped_at_10cm": .1,
               "nearest_geom": None, "time_s": None, "phase": None}
    prop_gaps = {"primary": .5, "secondary": .5}
    samples = 0

    def position(time_s):
        model.body_pos[primary_body, :2] = scenario["primary_xy_m"]
        model.body_pos[secondary_body, :2] = secondary_at(scenario, time_s)
        mujoco.mj_forward(model, data)

    def sample(time_s, pose, phase):
        nonlocal nearest, samples
        position(time_s)
        gap, geom = clearance(model, data, ape_body, ape_geom, pose)
        if gap < nearest["room_or_prop_gap_m_capped_at_10cm"]:
            nearest = {"room_or_prop_gap_m_capped_at_10cm": gap,
                       "nearest_geom": geom, "time_s": round(time_s, 4), "phase": phase}
        for label, prop_geom in (("primary", primary_geom), ("secondary", secondary_geom)):
            prop_gaps[label] = min(prop_gaps[label], float(mujoco.mj_geomDistance(
                model, data, ape_geom, prop_geom, .5, None)))
        samples += 1

    for row in episode["trace"]:
        pose = np.asarray(row["actual_xy_m"], dtype=float)
        sample(float(row["time_s"]), pose, "route_capture")
        if row["requested_xy_m"] is None:
            continue
        delta = ((np.asarray(row["requested_xy_m"]) -
                  np.asarray(row["estimated_xy_m"])) * (1. + .30 / 5.89))
        for index, sub in enumerate(row["motion_substeps"]):
            fraction = (index if sub["stopped_before_move"] else index + 1) / 20
            sample(float(sub["time_s"]), pose + fraction * delta, "executed_move")
    if episode["end_time_s"] < scenario["end_s"]:
        held = np.asarray(episode["end_true_xy_m"], dtype=float)
        for time_s in np.arange(episode["end_time_s"] + STEP_S,
                                scenario["end_s"] + STEP_S / 2, STEP_S):
            sample(float(time_s), held, "post_stop_hold")

    prop_fixture_gap = .3
    prop_pair_gap = .5
    for time_s in np.arange(0., max(float(episode["end_time_s"]), scenario["end_s"])
                               + STEP_S / 2, STEP_S):
        position(float(time_s))
        for prop_geom in (primary_geom, secondary_geom):
            prop_xy = data.geom_xpos[prop_geom, :2]
            for other in range(model.ngeom):
                if other in (ape_geom, primary_geom, secondary_geom) or model.geom(other).name.startswith("floor_"):
                    continue
                if np.linalg.norm(data.geom_xpos[other, :2] - prop_xy) > (
                        model.geom_rbound[other] + model.geom_rbound[prop_geom] + .3):
                    continue
                prop_fixture_gap = min(prop_fixture_gap, float(mujoco.mj_geomDistance(
                    model, data, prop_geom, other, .3, None)))
        prop_pair_gap = min(prop_pair_gap, float(mujoco.mj_geomDistance(
            model, data, primary_geom, secondary_geom, .5, None)))

    video = Path(episode["video"]["path"])
    reader = imageio.get_reader(video)
    frame_count = reader.count_frames()
    reader.close()
    track = [row for row in case["track_history"] if row["center_xy_m"] is not None]
    decision = case["safe_side_history"][-1] if case["safe_side_history"] else None
    return {"scenario_id": scenario["id"], "outcome": episode["outcome"],
            "nearest_ape": nearest, "ape_primary_gap_m_capped_at_50cm": prop_gaps["primary"],
            "ape_secondary_gap_m_capped_at_50cm": prop_gaps["secondary"],
            "prop_fixture_gap_m_capped_at_30cm": prop_fixture_gap,
            "prop_pair_gap_m_capped_at_50cm": prop_pair_gap,
            "track_count": len(track), "last_track": track[-1] if track else None,
            "safe_side_decision": decision,
            "ape_5cm_margin_pass": nearest["room_or_prop_gap_m_capped_at_10cm"] >= .05,
            "both_props_5cm_margin_pass": min(prop_gaps.values()) >= .05,
            "prop_paths_nonoverlap_pass": prop_fixture_gap >= 0. and prop_pair_gap >= 0.,
            "video_sha_pass": sha256(video) == episode["video"]["sha256"],
            "video_frames_pass": frame_count == episode["video"]["frames"] ==
            len(episode["trace"]) + 1,
            "geometry_samples": samples}


def main():
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text())
    cases = [audit_case(case) for case in report["cases"]]
    absent = cases[0]
    present = cases[1:]
    output = {"source_sha256": sha256(Path(__file__)),
              "trial_report_sha256": sha256(report_path),
              "trial_source_sha_pass": report["source_sha256"] == sha256(SOURCE),
              "config_source_sha_pass": report["config_sha256"] == sha256(CONFIG),
              "prior_map_sha_pass": report["prior_map_sha256"] == sha256(PRIOR),
              "frozen_policy_source_sha_pass": report["frozen_policy_source_sha256"] == sha256(POLICY),
              "same_present_initial_scene_pass": len({case["episode"]["scene_fingerprint"]
                                                      for case in report["cases"][1:]}) == 1,
              "all_geometry_pass": all(case["ape_5cm_margin_pass"] and
                                       case["both_props_5cm_margin_pass"] and
                                       case["prop_paths_nonoverlap_pass"] for case in cases),
              "all_videos_pass": all(case["video_sha_pass"] and case["video_frames_pass"]
                                     for case in cases),
              "absent_route_pass": absent["outcome"] == "goal_viewpoint_reached",
              "present_stage_before_motion_pass": all(
                  case["outcome"] == "request_owner_show_here_safe_side" and
                  case["safe_side_decision"] is not None and
                  case["safe_side_decision"]["should_stage"] and
                  report["cases"][i + 1]["episode"]["end_time_s"] <
                  report["cases"][i + 1]["scenario"]["begin_s"]
                  for i, case in enumerate(present)),
              "cases": cases}
    (OUT / "audit-v483.json").write_text(json.dumps(output, indent=2) + "\n")
    for case in cases:
        print(case["scenario_id"], case["outcome"],
              "room", round(case["nearest_ape"]["room_or_prop_gap_m_capped_at_10cm"], 3),
              "second", round(case["ape_secondary_gap_m_capped_at_50cm"], 3),
              "prop_room", round(case["prop_fixture_gap_m_capped_at_30cm"], 3),
              flush=True)
    print({key: value for key, value in output.items() if key.endswith("_pass")}, flush=True)


if __name__ == "__main__":
    main()
