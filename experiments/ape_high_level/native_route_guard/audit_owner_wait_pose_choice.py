"""Audit the full V33 wait-pose episodes against saved sensor and geometry data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

import owner_wait_pose_choice as trial


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def video_frames(path):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), path
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame is not None and frame.size
        n += 1
    cap.release()
    return n


def gap(model, data, ape_xy, bystander_xy):
    model.body_pos[model.body("audit_ape").id, :2] = ape_xy
    trial.two.set_distractor(model, bystander_xy)
    mujoco.mj_forward(model, data)
    ape = model.geom("audit_ape_geom").id
    names = ("distractor_torso_geom", "distractor_head_geom",
             "distractor_upper_arm_geom", "distractor_forearm_geom",
             "distractor_hand_geom")
    return min(float(mujoco.mj_geomDistance(
        model, data, ape, model.geom(name).id, 2., None)) for name in names)


def audit_case(s, result):
    ep = result["result"]["episode"]
    assert s["outcome"] == ep["outcome"]
    choice = s["wait_choice"]
    assert choice is not None and len(choice["range_packet"]) == 72
    actor_nearest = trial.side_range(choice["range_packet"])
    assert actor_nearest == choice["nearest_side_range_m"]
    known = all(x["free"] and not x["occupied"] for x in choice["known_free_cells"])
    assert known == choice["known_free_south"]
    moving = s["condition"].startswith("buffered") and known and actor_nearest is not None and actor_nearest < trial.RANGE_TRIGGER_M
    assert choice["action"] == ("move_south_then_show" if moving else "hold_current_pose")
    if moving:
        delta = np.asarray(s["ape_after_wait_xy_m"])-np.asarray(
            next(row for row in ep["trace"] if row["stage"] == "actor_wait_pose_motion")["start_xy_m"])
        assert np.linalg.norm(delta-[0., -trial.WAIT_SHIFT_M]) < 1e-8
        assert len(s["wait_motion_samples"]) == 10
        assert all(row["evaluator_person_gap_m"] > 0 for row in s["wait_motion_samples"])
    else:
        assert not s["wait_motion_samples"]
    assert not any(row.get("teacher_identity") == "confirmed_old_owner"
                   for row in ep["trace"])

    packets = s["range_packets"]
    changes = s["range_changes"]
    guard = next(row for row in ep["trace"] if row["stage"] == "new_show_inference_guard")
    assert len(packets) == len(changes) == len(guard["samples"]) > 0
    baseline = s["baseline_range_packet"]
    assert len(baseline) == 72
    alarm = None
    streak = 0
    for packet, change, sample in zip(packets, changes, guard["samples"]):
        assert len(packet["ranges_m"]) == 72
        assert abs(packet["time_s"]-change["time_s"]) < 1e-8
        count = trial.all_changed(baseline, packet["ranges_m"])
        assert count == change["all_changed"] == sample["changed_owner_sector_beams"]
        streak = streak+1 if count >= 3 else 0
        assert streak == sample["motion_streak"]
        if streak >= 2 and alarm is None:
            alarm = sample["time_s"]
    assert alarm == guard["alarm_time_s"] == s["guard_alarm_time_s"]
    if alarm is not None:
        assert s["outcome"] == "new_show_motion_unresolved"
        assert any(row["stage"] == "new_show_stale_image_discarded"
                   for row in ep["trace"])
    else:
        assert not any(row["stage"] == "new_show_stale_image_discarded"
                       for row in ep["trace"])

    g = s["inference_geometry_evaluator"]
    assert g is not None and len(g["samples"]) > 0
    model = mujoco.MjModel.from_xml_path(str(trial.moving.SCENE_XML))
    data = mujoco.MjData(model)
    ape = np.asarray(s["ape_after_episode_xy_m"])
    target = np.asarray(s["bystander_target_evaluator_xy_m"])
    for row in g["samples"]:
        t = row["time_s"]
        if s["condition"] == "buffered_no_side":
            bystander = np.array([-10., -10.])
        elif s["condition"] == "buffered_static":
            bystander = trial.SIDE_START
        else:
            fraction = min(1., trial.APPROACH_MPS*(t-s["worker_start_s"])/
                           np.linalg.norm(target-trial.SIDE_START))
            bystander = trial.SIDE_START+(target-trial.SIDE_START)*fraction
        assert abs(gap(model, data, ape, bystander)-row["gap_m"]) < .00011
    assert abs(min(x["gap_m"] for x in g["samples"])-g["minimum_gap_m"]) < 1e-8
    breach = next((x["time_s"] for x in g["samples"] if x["gap_m"] < .05), None)
    assert breach == g["first_below_5cm_s"]

    video = s["video"]
    assert video == ep["video"]
    assert sha(video["path"]) == video["sha256"]
    assert video_frames(video["path"]) == video["frames"]
    return {"condition": s["condition"], "outcome": s["outcome"],
            "choice": choice["action"], "nearest_side_range_m": actor_nearest,
            "min_gap_m": g["minimum_gap_m"], "first_below_5cm_s": breach,
            "alarm_after_motion_start_s": None if alarm is None else round(
                alarm-s["worker_start_s"], 3),
            "route_hash": hashlib.sha256(json.dumps(
                ep["route_episode"]["trace"], sort_keys=True).encode()).hexdigest(),
            "video_frames": video["frames"]}


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    assert report["schema"] == "robot-learning-lab.ape-owner-wait-pose.v1"
    assert tuple(report["conditions"]) == trial.CONDITIONS
    assert report["source_sha256"] == sha(trial.__file__)
    assert report["base_runner_sha256"] == sha(trial.reteach.__file__)
    assert report["scene_sha256"] == sha(trial.moving.SCENE_XML)
    assert report["current_design_pointer_sha256"] == sha(trial.ROOT / "robot/ape-current-design.json")
    assert len(report["summaries"]) == len(report["results"]) == len(trial.CONDITIONS)
    rows = [audit_case(s, r) for s, r in zip(report["summaries"], report["results"])]
    assert [x["condition"] for x in rows] == list(trial.CONDITIONS)
    assert len({x["route_hash"] for x in rows}) == 1
    audit = {"passed": True, "rows": rows,
             "limitation": "Pre-show movement uses a kinematic proxy. The current pointer names no selected mechanical design or qualified escape skill."}
    (trial.OUT / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
