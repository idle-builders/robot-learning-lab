"""V100: execute V99 information goal, then learned-object head sweep."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from audit_side_look_target_v81 import absent_capture, infer
from depth_voxel import DepthVoxelMap
from fast_range_guard import FastRangeGuard
from robot_geometry import RobotGeometry
from run import H, ROOT, W, Room, Vision, iou, sensed_range, sha
from run_live_local_bypass_v86 import run
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
RUNNER = SOURCE.with_name("run_live_local_bypass_v86.py")
V81 = ROOT / "artifacts/ape-closed-loop-cove50-v81"
V90 = ROOT / "artifacts/ape-closed-loop-cove50-v90"
V96 = ROOT / "artifacts/ape-closed-loop-cove50-v96"
V97 = ROOT / "artifacts/ape-closed-loop-cove50-v97"
V99 = ROOT / "artifacts/ape-closed-loop-cove50-v99/side-extension-view-audit.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v100"
HEAD_YAWS = (0., -.65, +.65)


def load_initial(seed: int, room: Room,
                 mask: SelfDepthMask) -> tuple[dict, dict]:
    prior_path = V96 / f"seed{seed}-repair/trace.json"
    view_path = V97 / f"seed{seed}/trace.json"
    prior = json.loads(prior_path.read_text())
    view = json.loads(view_path.read_text())
    v96_frames = sum("actor_input_sha256" in event
                     for event in prior["events"])
    paths = [V96 / f"seed{seed}-repair/frame-{v96_frames}.npz"] + [
        V97 / f"seed{seed}/view-{index}-actor-input.npz"
        for index in range(3)]
    voxel = DepthVoxelMap()
    input_hashes = []
    fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
    qpos = None
    pose = None
    stamp = None
    for path in paths:
        with np.load(path) as z:
            qpos = z["actor_qpos"].copy()
            pose = z["estimated_pose_xyyaw"].copy()
            depth = z["depth_m"].copy()
            camera = (z["camera_origin_m"].copy(),
                      z["camera_rotation"].copy())
            stamp = float(z["capture_time_s"])
        filtered, _ = mask.apply(depth, qpos)
        voxel.observe(filtered, camera, stamp, fovy_deg=fovy,
                      self_masked=True)
        input_hashes.append(sha(path))
    assert qpos is not None and pose is not None and stamp is not None
    side_trace = json.loads((V90 / f"seed{seed}-actor/trace.json").read_text())
    visited = {tuple(node) for node in side_trace["actor_graph_path_cells"]}
    visited |= {tuple(event["detail"]["plan_cells"][1]) for event in
                prior["events"] if event["action"] == "replanned_base_step"}
    initial = {"voxel": voxel,
               "actor_qpos": qpos,
               "estimated_pose_xyyaw": pose,
               "actual_pose_xyyaw_evaluator_only": np.asarray(
                   prior["evaluator_final_actual_xyyaw"], dtype=float),
               "capture_time_s": stamp,
               "visited_cells": visited,
               "provenance": {"v96_trace_sha256": sha(prior_path),
                              "v97_trace_sha256": sha(view_path),
                              "actor_input_sha256": input_hashes}}
    return initial, view


def endpoint_views(seed: int, room: Room, absent_room: Room,
                   vision: Vision, exemplar: np.ndarray,
                   target: np.ndarray) -> dict:
    branch = OUT / f"seed{seed}-endpoint-views"
    branch.mkdir(parents=True, exist_ok=True)
    monitor11 = FastRangeGuard(band_m=.05)
    monitor4 = FastRangeGuard(band_m=.05)
    serial = 0
    rows = []
    video_path = branch / "episode.mp4"

    def sense(height: float) -> list[float | None]:
        nonlocal serial
        serial += 1
        return sensed_range(room, seed, 1000000 + serial,
                            height_m=height)

    def guard_probe():
        monitor11.update(sense(.11), room.estimated, room.time_s)
        monitor4.update(sense(.04), room.estimated, room.time_s)

        def probe(substep: int, fraction: float) -> dict | None:
            upper = monitor11.update(sense(.11), room.estimated, room.time_s)
            lower = monitor4.update(sense(.04), room.estimated, room.time_s)
            if (upper["status"] == "near_body_change" or
                    lower["status"] == "near_body_change"):
                return {"upper": upper, "lower": lower,
                        "capture_time_s": room.time_s,
                        "action_fraction": fraction}
            return None
        return probe

    status = "completed_views"
    with imageio.get_writer(video_path, fps=4, codec="libx264", quality=7) as video:
        for index, yaw in enumerate(HEAD_YAWS):
            room.look(yaw, "lower_rgb", safety_probe=guard_probe())
            if room.safety_stop_observation is not None:
                status = "look_range_alarm"
            rgb, depth = room.capture("lower_rgb", depth=True)
            camera = room.camera_pose_estimate("lower_rgb")
            qpos = room.pose_data.qpos.copy()
            path = branch / f"view-{index}-actor-input.npz"
            np.savez_compressed(path, rgb=np.asarray(rgb), depth_m=depth,
                                camera_origin_m=camera[0],
                                camera_rotation=camera[1],
                                actor_qpos=qpos,
                                estimated_pose_xyyaw=room.estimated.copy(),
                                capture_time_s=room.time_s)
            present = infer(vision, exemplar, rgb, depth, camera, target,
                            None, branch / f"view-{index}-present-annotated.png")
            truth = room.truth_box("TomatoSauce", "lower_rgb")
            accepted_correct = (present["accepted_box"] is not None and
                                truth is not None and
                                iou(present["accepted_box"], truth) >= .25)
            frame = {"actor_qpos": qpos,
                     "estimated_pose_xyyaw": room.estimated.copy(),
                     "capture_time_s": room.time_s}
            absent_source = branch / f"view-{index}-absent"
            absent_rgb, absent_depth, absent_camera, _ = absent_capture(
                absent_room, seed, frame, room.actual.copy(), yaw,
                absent_source)
            absent = infer(vision, exemplar, absent_rgb, absent_depth,
                           absent_camera, target, None,
                           branch / f"view-{index}-absent-annotated.png")
            image = Image.fromarray(np.concatenate((np.asarray(rgb),
                                      np.asarray(room.capture_external())), axis=1))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, image.width, 34), fill=(0, 0, 0))
            draw.text((8, 8), f"V100 seed{seed} yaw {yaw:+.2f} | "
                      f"accepted {present['accepted_box'] is not None} | "
                      f"{room.time_s:.2f}s", fill=(255, 255, 255))
            for _ in range(4):
                video.append_data(np.asarray(image))
            rows.append({"head_yaw_rad": yaw,
                         "actor_input_sha256": sha(path),
                         "capture_time_s": round(room.time_s, 3),
                         "present_accepted_box": present["accepted_box"],
                         "present_accepted_similarity": present[
                             "accepted_similarity"],
                         "present_accepted_correct_evaluator_only":
                             accepted_correct,
                         "present_target_box_evaluator_only": truth,
                         "present_proposals": present["proposals"],
                         "absent_accepted": absent["accepted_box"] is not None,
                         "absent_actor_input_sha256": sha(
                             absent_source.with_suffix(".npz")),
                         "range_alarm": room.safety_stop_observation,
                         "evaluator_clearance_m": round(
                             room._current_obstacle_clearance()[0], 4)})
            print(seed, "view", yaw, "correct", accepted_correct,
                  "absent", absent["accepted_box"] is not None, flush=True)
            if status != "completed_views":
                break
    return {"status": status,
            "correct_target_reacquired": any(
                row["present_accepted_correct_evaluator_only"] for row in rows),
            "wrong_target_accepted": any(
                row["present_accepted_box"] is not None and
                not row["present_accepted_correct_evaluator_only"]
                for row in rows),
            "matched_absent_accepted": any(row["absent_accepted"]
                                           for row in rows),
            "evaluator_min_model_clearance_m": round(min(
                room.path_min_clearance[0],
                room._current_obstacle_clearance()[0]), 4),
            "views": rows,
            "video_sha256": sha(video_path)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selected = json.loads(V99.read_text())["row"]
    seed = int(selected["seed"])
    room = Room()
    absent_room = Room()
    mask = SelfDepthMask(room.model, camera="lower_rgb",
                         width=W, height=H)
    try:
        robot = RobotGeometry(room.model)
        initial, prior_view = load_initial(seed, room, mask)
        base = run(seed, {"actor_goal_cell": selected["actor_goal_cell"],
                          "blocked_cell": None}, room, robot,
                   output_root=OUT, video_label="V100",
                   goal_audit_path=V99, initial=initial,
                   self_mask=mask, allow_near_masked_depth=True)
        target = np.asarray(prior_view["actor_target_memory_xy_m"])
        exemplar_path = V81 / "owner-taught-exemplar.npy"
        vision = Vision("generic")
        views = (endpoint_views(seed, room, absent_room, vision,
                                np.load(exemplar_path), target)
                 if base["status"] == "local_goal_reached" else None)
    finally:
        mask.close()
        room.close()
        absent_room.close()
    gate = (base["status"] == "local_goal_reached" and
            base["evaluator_min_path_clearance_m"] >= .05 and
            not base["evaluator_model_overlap"] and
            views is not None and
            views["status"] == "completed_views" and
            views["correct_target_reacquired"] and
            not views["wrong_target_accepted"] and
            not views["matched_absent_accepted"] and
            views["evaluator_min_model_clearance_m"] >= .05)
    result = {"scope": "Live V99 actor-side information goal from V97 target-not-reacquired state, receding4cm route with fresh4/8/11cm selected-body checks, 10Hz independent4/11cm guard, scan localization and self-masked near RGB-D; then guarded endpoint head sweep and frozen owner-taught Grounding DINO+DINOv2 appearance/depth gate with matched absent. Kinematic, no target-area approach.",
              "source_sha256": sha(SOURCE),
              "runner_source_sha256": sha(RUNNER),
              "v99_goal_audit_sha256": sha(V99),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "base": {key: value for key, value in base.items()
                       if key != "events"},
              "views": views,
              "live_side_extension_reacquisition_gate_pass": gate}
    (OUT / "live-side-extension-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / "run_live_side_extension_v100_source.py").write_bytes(
        SOURCE.read_bytes())
    (OUT / "run_live_local_bypass_v86_imported_source.py").write_bytes(
        RUNNER.read_bytes())
    print("live side extension/reacquisition gate:", gate, flush=True)


if __name__ == "__main__":
    main()
