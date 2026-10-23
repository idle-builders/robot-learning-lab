"""Recheck the saved paired side-entry episodes from raw range and video evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/closed_loop"))
import run_robocasa_social_reteach_v589 as reteach  # noqa: E402
import run_robocasa_two_moving_people_v586 as moving  # noqa: E402
import screen_robocasa_two_people_v582 as two  # noqa: E402
import run_robocasa_person_joined_v497 as joined  # noqa: E402

from run_inference_side_entry import OUT, CONDITIONS, CHANGE_M  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def direct_gap(model, data, ape_xy, bystander_xy):
    model.body_pos[model.body("audit_ape").id, :2] = ape_xy
    two.set_distractor(model, bystander_xy)
    mujoco.mj_forward(model, data)
    ape_geom = model.geom("audit_ape_geom").id
    person_geoms = [model.geom(name).id for name in (
        "distractor_torso_geom", "distractor_head_geom",
        "distractor_upper_arm_geom", "distractor_forearm_geom",
        "distractor_hand_geom")]
    return min(float(mujoco.mj_geomDistance(model, data, ape_geom, geom, 2., None))
               for geom in person_geoms)


def decode_count(path):
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened(), path
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        assert frame is not None and frame.size > 0
        count += 1
    capture.release()
    return count


def audit_case(summary, result):
    condition = summary["condition"]
    ep = result["result"]["episode"]
    guard = next(row for row in ep["trace"] if row["stage"] == "new_show_inference_guard")
    packets = summary["range_packets"]
    changes = summary["range_change_comparison"]
    baseline = summary["baseline_range_packet"]
    assert len(baseline) == 72
    assert len(packets) == len(changes) == len(guard["samples"]) == summary["guard_scan_count"]
    assert summary["outcome"] == ep["outcome"]
    assert summary["video"] == ep["video"]
    assert summary["old_owner_identity_restored"] is False

    streak = 0
    first_alarm = None
    for packet, change, row in zip(packets, changes, guard["samples"]):
        assert len(packet["ranges_m"]) == 72
        assert abs(packet["time_s"] - change["time_s"]) < 1e-8
        assert abs(packet["time_s"] - row["time_s"]) < 0.00011
        full_count = sum(((old is None) != (new is None)) or
                         (old is not None and new is not None and abs(old-new) >= CHANGE_M)
                         for old, new in zip(baseline, packet["ranges_m"]))
        assert full_count == change["all_angle_changed"]
        used_count = change["owner_sector_changed"] if condition == "owner_sector_moving" else full_count
        assert used_count == row["changed_owner_sector_beams"]
        streak = streak+1 if used_count >= 3 else 0
        assert streak == row["motion_streak"]
        if streak >= 2 and first_alarm is None:
            first_alarm = row["time_s"]
    alarm = summary["guard_alarm_time_s"]
    assert alarm == guard["alarm_time_s"]
    assert first_alarm == alarm

    model = mujoco.MjModel.from_xml_path(str(moving.SCENE_XML))
    data = mujoco.MjData(model)
    ape = np.asarray(summary["ape_xy_evaluator_m"])
    gaps = summary["geometry_evaluator"]["samples"]
    for row in gaps:
        t = row["time_s"]
        if condition == "all_angle_static":
            bystander = np.asarray(summary["side_start_xy_evaluator_m"])
        else:
            fraction = min(1., max(0., .62*(t-summary["moving_start_s"])/.65))
            start = np.asarray(summary["side_start_xy_evaluator_m"])
            target = np.asarray(summary["side_target_xy_evaluator_m"])
            bystander = start+(target-start)*fraction
        assert abs(direct_gap(model, data, ape, bystander)-row["gap_m"]) < 0.0001
    assert abs(min(row["gap_m"] for row in gaps)-
               summary["geometry_evaluator"]["min_bystander_ape_gap_m"]) < 1e-8
    first_breach = next((row["time_s"] for row in gaps if row["gap_m"] < .05), None)
    assert first_breach == summary["geometry_evaluator"]["first_below_5cm_s"]

    video = summary["video"]
    assert sha(video["path"]) == video["sha256"]
    assert decode_count(video["path"]) == video["frames"]
    shown = OUT / condition / "owner_reteach" / "new-show.png"
    assert shown.is_file()
    return {"condition": condition, "outcome": summary["outcome"],
            "route_trace_sha256": hashlib.sha256(json.dumps(
                ep["route_episode"]["trace"], sort_keys=True).encode()).hexdigest(),
            "shown_image_sha256": sha(shown),
            "range_packet_count": len(packets), "video_frames_decoded": video["frames"],
            "alarm_after_inference_start_s": None if alarm is None else round(
                alarm-summary["moving_start_s"], 3),
            "first_5cm_breach_after_start_s": None if first_breach is None else round(
                first_breach-summary["moving_start_s"], 3),
            "min_person_gap_m": summary["geometry_evaluator"]["min_bystander_ape_gap_m"]}


def main():
    report = json.loads((OUT / "report.json").read_text())
    assert report["schema"] == "robot-learning-lab.ape-owner-inference-side-entry.v1"
    assert tuple(report["conditions"]) == CONDITIONS
    assert report["source_sha256"] == sha(Path(__file__).with_name("run_inference_side_entry.py"))
    assert report["base_runner_sha256"] == sha(reteach.__file__)
    assert report["two_person_runner_sha256"] == sha(moving.__file__)
    assert report["scene_sha256"] == sha(moving.SCENE_XML)
    assert report["current_design_pointer_sha256"] == sha(ROOT / "robot/ape-current-design.json")
    assert len(report["summaries"]) == len(report["results"]) == 3
    rows = [audit_case(s, r) for s, r in zip(report["summaries"], report["results"])]
    assert {row["condition"] for row in rows} == set(CONDITIONS)
    assert len({row["route_trace_sha256"] for row in rows}) == 1
    assert len({row["shown_image_sha256"] for row in rows}) == 1
    baseline, moving_all, stationary = rows
    assert baseline["outcome"] == "new_show_metric_item_seen_unattributed"
    assert moving_all["outcome"] == "new_show_motion_unresolved"
    assert stationary["outcome"] == "new_show_metric_item_seen_unattributed"
    assert baseline["alarm_after_inference_start_s"] is None
    assert moving_all["alarm_after_inference_start_s"] < moving_all["first_5cm_breach_after_start_s"]
    assert stationary["alarm_after_inference_start_s"] is None
    assert baseline["min_person_gap_m"] <= 0 and moving_all["min_person_gap_m"] <= 0
    assert stationary["min_person_gap_m"] >= .05
    overview = OUT / "side-entry-overview.mp4"
    assert decode_count(overview) == 36
    result = {"passed": True, "rows": rows,
              "overhead_replay": {"path": str(overview), "sha256": sha(overview),
                                  "frames_decoded": 36},
              "limitation": "Image acceptance was guarded, but the kinematic ape never moved away; moving trials still overlap the person capsule."}
    (OUT / "audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
