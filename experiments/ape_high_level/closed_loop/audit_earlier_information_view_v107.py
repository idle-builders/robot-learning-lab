"""V107: counterfactual recognition at fixed earlier V104 seed-85 waypoints."""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from audit_side_look_target_v81 import absent_capture, infer
from run import ROOT, Room, Vision, iou, sha

SOURCE = Path(__file__)
V81 = ROOT / "artifacts/ape-closed-loop-cove50-v81"
V104 = ROOT / "artifacts/ape-closed-loop-cove50-v104"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v107"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seed = 85
    trace_path = V104 / f"seed{seed}/trace.json"
    trace = json.loads(trace_path.read_text())
    first = next(event for event in trace["events"]
                 if event["action"] == "guarded_approach_step")
    route = first["detail"]["planner_goal"]["committed_path_cells"]
    latest_move = next(event for event in reversed(trace["events"])
                       if event["action"] == "guarded_approach_step")
    start_index = latest_move["detail"]["planner_goal"]["route_index"] + 1
    candidates = list(enumerate(route))[start_index:-1]
    target = np.asarray(latest_move["actor_target_xy_m"], dtype=float)
    final_frame_index = sum("actor_input_sha256" in event
                            for event in trace["events"])
    final_frame_path = V104 / f"seed{seed}/frame-{final_frame_index}.npz"
    with np.load(final_frame_path) as z:
        qpos = z["actor_qpos"].copy()
        stamp = float(z["capture_time_s"])
    pose = np.asarray(trace["final_estimated_pose_xyyaw"], dtype=float)
    actual = np.asarray(trace["evaluator_final_actual_xyyaw"], dtype=float)
    offset = actual[:2] - pose[:2]
    exemplar_path = V81 / "owner-taught-exemplar.npy"
    exemplar = np.load(exemplar_path)
    vision = Vision("generic")
    room = Room()
    absent_room = Room()
    rows = []
    video_path = OUT / "seed85-earlier-waypoints.mp4"
    try:
        with imageio.get_writer(video_path, fps=2, codec="libx264",
                                quality=7) as video:
            for index, node in candidates:
                goal = .04 * np.asarray(node, dtype=float)
                candidate_pose = np.asarray([goal[0], goal[1], pose[2]])
                yaw = math.atan2(target[1]-goal[1], target[0]-goal[0])-pose[2]
                yaw = float(np.clip(math.atan2(math.sin(yaw), math.cos(yaw)),
                                    -1.2, 1.2))
                room.reset()
                room.layout(seed, absent=False, furnished=True)
                room.model.body_pos[room.model.body("loop_table").id] = (
                    .40, (1 if seed % 2 else -1) * .46, 0.)
                room.set_table_height(.37)
                room.data.qpos[:] = qpos
                room.estimated[:] = candidate_pose
                room.actual[:] = [goal[0]+offset[0], goal[1]+offset[1],actual[2]]
                room.head = yaw
                room.time_s = stamp
                room._forward()
                rgb, depth = room.capture("lower_rgb", depth=True)
                camera = room.camera_pose_estimate("lower_rgb")
                actor_qpos = room.pose_data.qpos.copy()
                branch = OUT / f"waypoint-{index:02d}"
                branch.mkdir(parents=True, exist_ok=True)
                actor_path = branch / "actor-input.npz"
                np.savez_compressed(actor_path, rgb=np.asarray(rgb), depth_m=depth,
                                    camera_origin_m=camera[0],
                                    camera_rotation=camera[1],
                                    actor_qpos=actor_qpos,
                                    estimated_pose_xyyaw=candidate_pose,
                                    capture_time_s=stamp)
                present = infer(vision, exemplar, rgb, depth, camera, target,
                                None, branch / "present-annotated.png")
                truth = room.truth_box("TomatoSauce", "lower_rgb")
                correct = (present["accepted_box"] is not None and
                           truth is not None and
                           iou(present["accepted_box"], truth) >= .25)
                frame = {"actor_qpos": actor_qpos,
                         "estimated_pose_xyyaw": candidate_pose,
                         "capture_time_s": stamp}
                absent_source = branch / "absent"
                absent_rgb, absent_depth, absent_camera, _ = absent_capture(
                    absent_room, seed, frame, room.actual.copy(), yaw,
                    absent_source)
                absent = infer(vision, exemplar, absent_rgb, absent_depth,
                               absent_camera, target, None,
                               branch / "absent-annotated.png")
                clearance = float(room._current_obstacle_clearance()[0])
                combined = Image.fromarray(np.concatenate((np.asarray(rgb),
                        np.asarray(room.capture_external())), axis=1))
                draw = ImageDraw.Draw(combined)
                draw.rectangle((0,0,combined.width,34),fill=(0,0,0))
                draw.text((8,8), f"V107 waypoint {index} {node} | "
                          f"clear {clearance:.3f} | accepted {correct}",
                          fill=(255,255,255))
                for _ in range(3):
                    video.append_data(np.asarray(combined))
                row = {"route_index":index,"route_cell":node,
                       "candidate_xy_m":goal.tolist(),
                       "candidate_head_yaw_rad":round(yaw,4),
                       "actor_input_sha256":sha(actor_path),
                       "evaluator_model_clearance_m":round(clearance,4),
                       "present_accepted_box":present["accepted_box"],
                       "present_accepted_similarity":present[
                           "accepted_similarity"],
                       "evaluator_truth_box":truth,
                       "present_accept_correct_evaluator_only":correct,
                       "matched_absent_accepted":absent["accepted_box"] is not None,
                       "present_proposals":present["proposals"]}
                rows.append(row)
                print(index,node,"clear",row["evaluator_model_clearance_m"],
                      "correct",correct,"absent",row[
                          "matched_absent_accepted"],flush=True)
    finally:
        room.close()
        absent_room.close()
    gate = any(row["evaluator_model_clearance_m"] >= .05 and
               row["present_accept_correct_evaluator_only"] and
               not row["matched_absent_accepted"] for row in rows)
    result = {"scope":"Read-only evaluator screen of earlier fixed waypoints on the V104 seed85 actor-nominated locally-3D route. Selected joint state and estimated/actual offset, memory-bearing lower-head RGB-D and frozen taught-object gate, matched absent. No waypoint selected using evaluator results and no translation executed.",
              "source_sha256":sha(SOURCE),
              "v104_trace_sha256":sha(trace_path),
              "v104_final_frame_sha256":sha(final_frame_path),
              "v81_exemplar_sha256":sha(exemplar_path),
              "selected_design_sha256":sha(ROOT / "robot/ape-current-design.json"),
              "last_executed_route_index":start_index-1,
              "candidate_rows":rows,
              "video_sha256":sha(video_path),
              "earlier_safe_visible_waypoint_exists":gate}
    (OUT / "earlier-information-view-audit.json").write_text(
        json.dumps(result,indent=2)+"\n")
    (OUT / "audit_earlier_information_view_v107_source.py").write_bytes(
        SOURCE.read_bytes())
    print("earlier safe visible waypoint:",gate,flush=True)


if __name__ == "__main__":
    main()
