"""Exercise Cove-50 view-backed memory with selected-body RGB model choices.

The owner confirmations/rejection and pose errors below are scripted status
inputs, not measured human or walking-ape behavior. The model candidate boxes
come from cove_rgb_transfer.py. Evaluator object positions are used only after
memory decisions to report whether the floor-ray estimate was plausible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from view_memory import Frame, Pose2, ViewMemory


ROOT = Path(__file__).resolve().parents[3]
DESIGN = ROOT / "robot/ape-current-design.json"
DEFAULT_RGB = ROOT / "artifacts/ape-cove50-rgb-transfer-20261008/report.json"
CALIBRATION = "virtual_640x480_fovy67_v1"
CAMERA = "chest_low50_x45"
NAME = "TomatoSauce"


def yaw_from_quaternion(q: list[float]) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def frame_from_view(frame_id: str, view: dict, camera: str, captured_at_ms: int,
                    *, pose_lag_ms: int = 10, epoch: str = "sim_map_1") -> Frame:
    p = view["view_pose"]
    if p["calibration_id"] != CALIBRATION:
        raise ValueError("Selected-body render calibration changed")
    base = p["body_world_xyz_m"]
    pose = Pose2(epoch, float(base[0]), float(base[1]),
                 yaw_from_quaternion(p["body_world_quat_wxyz"]),
                 captured_at_ms - pose_lag_ms, True,
                 0.01, 0.02)  # Replay-only supplied bounds, not measured ape errors.
    return Frame(frame_id, camera, view["rgb_sha256"], captured_at_ms,
                 CALIBRATION, True, pose)


def floor_point_in_body(box: list[float], view: dict) -> tuple[float, float] | None:
    """Bottom-center ray through a flat floor, using the saved virtual FK pose."""
    u, v = (box[0] + box[2]) / 2, box[3] - 2
    f = 480 / (2 * math.tan(math.radians(67) / 2))
    camera_ray = np.array([(u - 320) / f, -(v - 240) / f, -1.0])
    p = view["view_pose"]
    origin = np.asarray(p["camera_world_xyz_m"], dtype=float)
    rotation = np.asarray(p["camera_world_rotation"], dtype=float)
    world_ray = rotation @ camera_ray
    if world_ray[2] >= -1e-5:
        return None
    world = origin - origin[2] / world_ray[2] * world_ray
    base = np.asarray(p["body_world_xyz_m"][:2], dtype=float)
    yaw = yaw_from_quaternion(p["body_world_quat_wxyz"])
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = world[:2] - base
    return float(c * dx + s * dy), float(-s * dx + c * dy)


def row_for(rgb: dict, layout: str, name: str = NAME) -> dict:
    return next(r for r in rgb["rgb_model"]["rows"]
                if r["layout"] == layout and r["camera"] == CAMERA and r["name"] == name)


def owner_review_turn(*, layout: str, box: list[float],
                      captured_at_ms: int, confirmed: bool) -> tuple[dict, dict]:
    planner_dir = ROOT / "experiments/ape_high_level/planner"
    sys.path.insert(0, str(planner_dir))
    try:
        from turn_loop import Turn
    finally:
        sys.path.pop(0)
    turn = Turn(NAME, floor_camera=CAMERA)
    turn.teach(name_source="text_control", name_healthy=True,
               frame_id=f"teach:{NAME}", crop_confirmed=True)
    turn.find()
    review = turn.observe(frame_id=f"{layout}:{CAMERA}", camera=CAMERA,
                          captured_at_ms=captured_at_ms, age_ms=0,
                          image_healthy=True, view_pose_healthy=True,
                          candidate_box=box, owner_requested_review=True)
    if review["action"] != "ask_candidate_confirmation":
        raise AssertionError("Explicit owner review did not display the candidate")
    answer = turn.owner_answer(confirmed=confirmed)
    return review, answer


def candidate(memory: ViewMemory, rgb: dict, *, layout: str,
              captured_at_ms: int, pose_lag_ms: int = 10) -> tuple[dict, tuple[float, float] | None]:
    row = row_for(rgb, layout)
    view = rgb["views"]["search"][f"{layout}/{CAMERA}"]
    box = row["chosen_box"]
    if box is None:
        raise ValueError("Expected a saved model candidate in this replay case")
    frame = frame_from_view(f"{layout}:{CAMERA}", view, CAMERA,
                            captured_at_ms, pose_lag_ms=pose_lag_ms)
    point = floor_point_in_body(box, view)
    memory.observe_candidate(candidate_id=f"{layout}:{NAME}", name=NAME,
                             frame=frame, box_xyxy=tuple(box),
                             model_score=row["chosen_similarity"],
                             floor_xy_in_body_m=point,
                             floor_point_error_bound_m=0.02 if point else None)
    return row, point


def run(rgb_path: Path) -> dict:
    source = rgb_path.read_bytes()
    rgb = json.loads(source)
    design = json.loads(DESIGN.read_text())
    if rgb["design_id"] != design["id"] or rgb["selected_source_sha256"] != design["generator"]["sha256"]:
        raise ValueError("RGB choices do not match selected Cove-50 source")
    calibrations = {"upper_rgb": CALIBRATION, CAMERA: CALIBRATION}
    memory = ViewMemory(calibration_ids=calibrations)
    teach = rgb["rgb_model"]["teach"][NAME]
    if teach["selected_rgb_box"] is None:
        raise ValueError("No owner-click crop to bind")
    teach_frame = frame_from_view(f"teach:{NAME}", teach, "upper_rgb", 100)
    feature_hash = rgb["model_weights_sha256"]["dinov2-small"]
    memory.confirm_teaching(name=NAME, frame=teach_frame,
                            crop_xyxy=tuple(teach["selected_rgb_box"]),
                            feature_ref=f"dinov2-small:{feature_hash}:teach:{NAME}",
                            owner_confirmed_at_ms=120)
    events = [{"event": "owner_confirmed_teaching_crop",
               "status": memory.locate(NAME, now_ms=130, current_pose=teach_frame.pose)["status"]}]

    present, present_point = candidate(memory, rgb, layout="center_and_sides",
                                       captured_at_ms=1000)
    before = memory.locate(NAME, now_ms=1010, current_pose=Pose2(
        "sim_map_1", 0, 0, 0, 1000, True, 0.01, 0.02))
    events.append({"event": "rgb_candidate_before_owner_answer", "status": before["status"],
                   "action_target": before["action_target_body_xy_m"]})
    present_review, present_answer = owner_review_turn(
        layout="center_and_sides", box=present["chosen_box"],
        captured_at_ms=1000, confirmed=True)
    memory.display_candidate(f"center_and_sides:{NAME}",
                             displayed_frame_id=present_review["frame_id"],
                             displayed_at_ms=1015)
    memory.owner_answer(f"center_and_sides:{NAME}", confirmed=True,
                        displayed_frame_id=present_answer["candidate_frame"],
                        answered_at_ms=1020)
    confirmed = memory.locate(NAME, now_ms=1050, current_pose=Pose2(
        "sim_map_1", 0, 0, 0, 1040, True, 0.01, 0.02))
    events.append({"event": "scripted_owner_confirms_displayed_candidate",
                   "status": confirmed["status"],
                   "action_target": confirmed["action_target_body_xy_m"]})
    rotated = memory.locate(NAME, now_ms=1100, current_pose=Pose2(
        "sim_map_1", 0, 0, math.pi / 2, 1090, True, 0.01, 0.02))
    events.append({"event": "bounded_pose_update_after_body_turn",
                   "status": rotated["status"],
                   "action_target": rotated["action_target_body_xy_m"]})
    memory.calibration_ids[CAMERA] = "moved_chest_mount_uncalibrated"
    changed_mount = memory.locate(NAME, now_ms=1110, current_pose=Pose2(
        "sim_map_1", 0, 0, 0, 1100, True, 0.01, 0.02))
    events.append({"event": "camera_mount_changed_without_recalibration",
                   "status": changed_mount["status"],
                   "action_target": changed_mount["action_target_body_xy_m"]})
    memory.calibration_ids[CAMERA] = CALIBRATION

    absent, _ = candidate(memory, rgb, layout="tomato_absent", captured_at_ms=1150)
    pending = memory.locate(NAME, now_ms=1160, current_pose=Pose2(
        "sim_map_1", 0, 0, 0, 1150, True, 0.01, 0.02))
    events.append({"event": "new_scene_absent_target_model_suggests_distractor",
                   "status": pending["status"], "action_target": pending["action_target_body_xy_m"]})
    absent_review, absent_answer = owner_review_turn(
        layout="tomato_absent", box=absent["chosen_box"],
        captured_at_ms=1150, confirmed=False)
    memory.display_candidate(f"tomato_absent:{NAME}",
                             displayed_frame_id=absent_review["frame_id"],
                             displayed_at_ms=1155)
    memory.owner_answer(f"tomato_absent:{NAME}", confirmed=False,
                        displayed_frame_id=absent_answer["candidate_frame"],
                        answered_at_ms=1165)
    rejected = memory.locate(NAME, now_ms=1170, current_pose=Pose2(
        "sim_map_1", 0, 0, 0, 1160, True, 0.01, 0.02))
    events.append({"event": "owner_says_none_of_these",
                   "status": rejected["status"],
                   "action_target": rejected["action_target_body_xy_m"]})
    lost = memory.locate(NAME, now_ms=1180, current_pose=Pose2(
        "sim_map_1", 0, 0, 0, 1170, False, None, None))
    events.append({"event": "body_pose_lost", "status": lost["status"],
                   "action_target": lost["action_target_body_xy_m"]})
    new_epoch = memory.locate(NAME, now_ms=1200, current_pose=Pose2(
        "relocalized_map_2", 0, 0, 0, 1190, True, 0.01, 0.02))
    events.append({"event": "relocalized_without_new_object_image",
                   "status": new_epoch["status"],
                   "action_target": new_epoch["action_target_body_xy_m"]})

    # If capture-time pose is too old, even owner confirmation cannot produce
    # a metric action target. This branch has the same RGB candidate and new memory.
    stale_memory = ViewMemory(calibration_ids=calibrations)
    stale_memory.confirm_teaching(name=NAME, frame=teach_frame,
                                  crop_xyxy=tuple(teach["selected_rgb_box"]),
                                  feature_ref=f"dinov2-small:{feature_hash}:teach:{NAME}",
                                  owner_confirmed_at_ms=120)
    candidate(stale_memory, rgb, layout="center_and_sides", captured_at_ms=1000,
              pose_lag_ms=150)
    stale_memory.display_candidate(f"center_and_sides:{NAME}",
                                   displayed_frame_id=f"center_and_sides:{CAMERA}",
                                   displayed_at_ms=1015)
    stale_memory.owner_answer(f"center_and_sides:{NAME}", confirmed=True,
                              displayed_frame_id=f"center_and_sides:{CAMERA}",
                              answered_at_ms=1020)
    stale_capture_pose = stale_memory.locate(NAME, now_ms=1050, current_pose=Pose2(
        "sim_map_1", 0, 0, 0, 1040, True, 0.01, 0.02))
    events.append({"event": "owner_confirms_with_stale_capture_pose",
                   "status": stale_capture_pose["status"],
                   "action_target": stale_capture_pose["action_target_body_xy_m"]})

    # A later explicit correction replaces the contradicted appearance crop.
    # It does not reuse the old spatial sighting or call itself a trained model.
    correction_row = row_for(rgb, "swapped")
    correction_view = rgb["views"]["search"][f"swapped/{CAMERA}"]
    correction_frame = frame_from_view(f"correction:swapped:{CAMERA}", correction_view,
                                       CAMERA, 1300, epoch="relocalized_map_2")
    memory.replace_teaching(
        name=NAME, frame=correction_frame,
        crop_xyxy=tuple(correction_row["chosen_box"]),
        feature_ref=f"confirmed_crop:{correction_view['rgb_sha256']}:swapped:{NAME}",
        owner_confirmed_at_ms=1320)
    corrected = memory.locate(NAME, now_ms=1330, current_pose=Pose2(
        "relocalized_map_2", 0, 0, 0, 1320, True, 0.01, 0.02))
    events.append({"event": "scripted_owner_replaces_contradicted_teaching_crop",
                   "status": corrected["status"],
                   "action_target": corrected["action_target_body_xy_m"],
                   "old_view_marked_contradicted":
                       memory.teaching_history[-1].contradicted})

    expected = {
        "owner_confirmed_teaching_crop": "learned_name_no_confirmed_location",
        "rgb_candidate_before_owner_answer": "learned_name_no_confirmed_location",
        "scripted_owner_confirms_displayed_candidate": "fresh_owner_confirmed_location",
        "bounded_pose_update_after_body_turn": "fresh_owner_confirmed_location",
        "camera_mount_changed_without_recalibration": "location_unavailable",
        "new_scene_absent_target_model_suggests_distractor": "newer_candidate_unconfirmed",
        "owner_says_none_of_these": "newer_candidate_unconfirmed",
        "body_pose_lost": "location_unavailable",
        "relocalized_without_new_object_image": "location_unavailable",
        "owner_confirms_with_stale_capture_pose": "location_unavailable",
        "scripted_owner_replaces_contradicted_teaching_crop":
            "learned_name_no_confirmed_location",
    }
    if {row["event"]: row["status"] for row in events} != expected:
        raise AssertionError("Memory status replay changed")
    if present["chosen_overlap_evaluator_only"] < 0.25 or absent["chosen_overlap_evaluator_only"] != 0:
        raise AssertionError("Saved present/absent scenario evidence changed")
    if not (confirmed["action_target_body_xy_m"] and rotated["action_target_body_xy_m"] and
            math.isclose(rotated["action_target_body_xy_m"][0],
                         confirmed["action_target_body_xy_m"][1], abs_tol=1e-6) and
            math.isclose(rotated["action_target_body_xy_m"][1],
                         -confirmed["action_target_body_xy_m"][0], abs_tol=1e-6)):
        raise AssertionError("Bounded pose update did not rotate the stored target")

    return {"design_id": design["id"], "rgb_report_sha256": hashlib.sha256(source).hexdigest(),
            "scope": "Software evidence contract on selected-body synthetic RGB choices. "
                     "Optional candidate displays, owner responses, pose epochs and error bounds "
                     "are scripted (the current chest-view turn has no qualified body turn and "
                     "would offer show-again before this review); exact virtual "
                     "FK supplies camera pose and floor-plane ray. No walking odometry, "
                     "physical calibration, owner trial or skill execution. The 2 cm point, "
                     "1 cm pose and 0.02 rad yaw bounds are fixture values, not measured.",
            "evaluator_present_choice_overlap": present["chosen_overlap_evaluator_only"],
            "evaluator_absent_choice_overlap": absent["chosen_overlap_evaluator_only"],
            "optional_review_controller_actions": {
                "present": [present_review["action"], present_answer["action"]],
                "absent": [absent_review["action"], absent_answer["action"]]},
            "estimated_present_floor_xy_in_body_m": present_point,
            "evaluator_present_object_xy_m": [0.25, 0.0],
            "events": events,
            "memory": memory.record()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-report", type=Path, default=DEFAULT_RGB)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.rgb_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["events"], indent=2))


if __name__ == "__main__":
    main()
