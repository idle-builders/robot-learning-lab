"""V105: target-memory-bearing guarded head sweep from V104 stops."""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from audit_side_look_target_v81 import absent_capture, infer
from fast_range_guard import FastRangeGuard
from run import ROOT, Room, Vision, iou, sensed_range, sha


SOURCE = Path(__file__)
V81 = ROOT / "artifacts/ape-closed-loop-cove50-v81"
V104 = ROOT / "artifacts/ape-closed-loop-cove50-v104"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v105"
HEAD_OFFSETS = (0., -.25, +.25, -.50, +.50)


def replay(seed: int, room: Room, absent_room: Room,
           vision: Vision, exemplar: np.ndarray) -> dict:
    prior_path = V104 / f"seed{seed}/trace.json"
    prior = json.loads(prior_path.read_text())
    final_frame = sum("actor_input_sha256" in event
                      for event in prior["events"])
    frame_path = V104 / f"seed{seed}/frame-{final_frame}.npz"
    with np.load(frame_path) as z:
        qpos = z["actor_qpos"].copy()
        stamp = float(z["capture_time_s"])
    pose = np.asarray(prior["final_estimated_pose_xyyaw"], dtype=float)
    actual = np.asarray(prior["evaluator_final_actual_xyyaw"], dtype=float)
    last_frame = next(event for event in reversed(prior["events"])
                      if "actor_target_xy_m" in event)
    target = np.asarray(last_frame["actor_target_xy_m"], dtype=float)
    bearing = math.atan2(target[1] - pose[1], target[0] - pose[0]) - pose[2]
    bearing = math.atan2(math.sin(bearing), math.cos(bearing))
    head_yaws = tuple(float(np.clip(bearing + offset, -1.2, 1.2))
                      for offset in HEAD_OFFSETS)
    room.reset()
    room.layout(seed, absent=False, furnished=True)
    room.model.body_pos[room.model.body("loop_table").id] = (
        .40, (1 if seed % 2 else -1) * .46, 0.)
    room.set_table_height(.37)
    room.data.qpos[:] = qpos
    room.actual[:] = actual
    room.estimated[:] = pose
    room.head = float(qpos[room.head_addr])
    room.time_s = stamp
    room._forward()
    monitor11 = FastRangeGuard(band_m=.05)
    monitor4 = FastRangeGuard(band_m=.05)
    serial = 0
    branch = OUT / f"seed{seed}"
    branch.mkdir(parents=True, exist_ok=True)
    rows = []
    video_path = branch / "episode.mp4"

    def sense(height: float) -> list[float | None]:
        nonlocal serial
        serial += 1
        return sensed_range(room, seed, 1050000 + serial,
                            height_m=height)

    def guard_probe():
        monitor11.update(sense(.11), room.estimated, room.time_s)
        monitor4.update(sense(.04), room.estimated, room.time_s)

        def probe(substep: int, fraction: float) -> dict | None:
            scan11 = sense(.11)
            scan4 = sense(.04)
            torso = monitor11.update(scan11, room.estimated, room.time_s)
            low = monitor4.update(scan4, room.estimated, room.time_s)
            if (torso["status"] == "near_body_change" or
                    low["status"] == "near_body_change"):
                return {"source": ("low4" if low["status"] ==
                                   "near_body_change" else "torso11"),
                        "torso": torso, "low": low,
                        "capture_time_s": room.time_s,
                        "action_fraction": fraction}
            return None
        return probe

    status = "completed_views"
    with imageio.get_writer(video_path, fps=4, codec="libx264", quality=7) as video:
        for view_index, yaw in enumerate(head_yaws):
            room.look(yaw, "lower_rgb", safety_probe=guard_probe())
            if room.safety_stop_observation is not None:
                status = "look_range_alarm"
            rgb, depth = room.capture("lower_rgb", depth=True)
            camera = room.camera_pose_estimate("lower_rgb")
            actor_qpos = room.pose_data.qpos.copy()
            source = branch / f"view-{view_index}-actor-input.npz"
            np.savez_compressed(source, rgb=np.asarray(rgb), depth_m=depth,
                                camera_origin_m=camera[0],
                                camera_rotation=camera[1],
                                actor_qpos=actor_qpos,
                                estimated_pose_xyyaw=room.estimated.copy(),
                                capture_time_s=room.time_s)
            present = infer(vision, exemplar, rgb, depth, camera, target,
                            None, branch / f"view-{view_index}-present-annotated.png")
            truth = room.truth_box("TomatoSauce", "lower_rgb")
            accepted_is_target = (present["accepted_box"] is not None and
                                  truth is not None and
                                  iou(present["accepted_box"], truth) >= .25)
            frame = {"actor_qpos": actor_qpos,
                     "estimated_pose_xyyaw": room.estimated.copy(),
                     "capture_time_s": room.time_s}
            absent_source = branch / f"view-{view_index}-absent"
            absent_rgb, absent_depth, absent_camera, _ = absent_capture(
                absent_room, seed, frame, room.actual.copy(), yaw,
                absent_source)
            absent = infer(vision, exemplar, absent_rgb, absent_depth,
                           absent_camera, target, None,
                           branch / f"view-{view_index}-absent-annotated.png")
            image = Image.fromarray(np.concatenate((np.asarray(rgb),
                                      np.asarray(room.capture_external())), axis=1))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, image.width, 34), fill=(0, 0, 0))
            draw.text((8, 8), f"V105 seed{seed} yaw {yaw:+.2f} | "
                      f"accepted {present['accepted_box'] is not None} | "
                      f"{room.time_s:.2f}s", fill=(255, 255, 255))
            for _ in range(4):
                video.append_data(np.asarray(image))
            rows.append({"head_yaw_rad": yaw,
                         "capture_time_s": round(room.time_s, 3),
                         "actor_input_sha256": sha(source),
                         "present": {"accepted_box": present["accepted_box"],
                                     "accepted_similarity": present[
                                         "accepted_similarity"],
                                     "accepted_world_xyz_m": next((
                                         item["depth_world_xyz_m"] for item in
                                         present["proposals"] if
                                         item["box"] == present["accepted_box"]),
                                         None),
                                     "evaluator_target_box": truth,
                                     "evaluator_accept_is_target":
                                         accepted_is_target,
                                     "proposal_count": len(present["proposals"]),
                                     "proposals": present["proposals"]},
                         "absent_accepted": absent["accepted_box"] is not None,
                         "absent_actor_input_sha256": sha(
                             absent_source.with_suffix(".npz")),
                         "range_alarm": room.safety_stop_observation,
                         "evaluator_clearance_m": round(
                             room._current_obstacle_clearance()[0], 4)})
            print(seed, yaw, "present", accepted_is_target,
                  "absent", absent["accepted_box"] is not None,
                  flush=True)
            if status != "completed_views":
                break
    min_clearance = min(room.path_min_clearance[0],
                        room._current_obstacle_clearance()[0])
    result = {"seed": seed, "v104_trace_sha256": sha(prior_path),
              "v104_final_frame_sha256": sha(frame_path),
              "actor_target_memory_xy_m": target.tolist(),
              "actor_memory_bearing_yaw_rad": round(bearing,4),
              "status": status,
              "correct_target_reacquired": any(
                  view["present"]["evaluator_accept_is_target"]
                  for view in rows),
              "wrong_target_accepted": any(
                  view["present"]["accepted_box"] is not None and
                  not view["present"]["evaluator_accept_is_target"]
                  for view in rows),
              "matched_absent_accepted": any(view["absent_accepted"]
                                             for view in rows),
              "evaluator_min_model_clearance_m": round(min_clearance, 4),
              "views": rows,
              "video_sha256": sha(video_path)}
    (branch / "trace.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    exemplar_path = V81 / "owner-taught-exemplar.npy"
    exemplar = np.load(exemplar_path)
    vision = Vision("generic")
    room = Room()
    absent_room = Room()
    try:
        rows = [replay(seed, room, absent_room, vision, exemplar)
                for seed in (85, 88)]
    finally:
        room.close()
        absent_room.close()
    gate = all(row["status"] == "completed_views" and
               row["correct_target_reacquired"] and
               not row["wrong_target_accepted"] and
               not row["matched_absent_accepted"] and
               row["evaluator_min_model_clearance_m"] >= .05
               for row in rows)
    result = {"scope": "Fresh guarded memory-bearing yaw and ±0.25/0.50rad lower-head RGB-D from exact V104 target-lost stops, cached V81 owner-taught exemplar and Grounding DINO+DINOv2 0.35 appearance/12cm depth-memory gate; matched target-absent render at same body/head state. Segmentation only post-decision score. Kinematic head; no base translation.",
              "source_sha256": sha(SOURCE),
              "v81_exemplar_sha256": sha(exemplar_path),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "rows": rows,
              "memory_bearing_reacquisition_gate_pass": gate}
    (OUT / "memory-bearing-look-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / "run_memory_bearing_look_v105_source.py").write_bytes(
        SOURCE.read_bytes())
    print("memory bearing reacquisition gate:", gate, flush=True)


if __name__ == "__main__":
    main()
