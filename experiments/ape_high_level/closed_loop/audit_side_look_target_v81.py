"""V81: image-grounded taught-object reacquisition after V80 side looks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from run import ROOT, Room, Vision, iou, locate_world, sha
from active_scan import DETECTOR, RETRIEVER


SOURCE = Path(__file__)
V80 = ROOT / "artifacts/ape-closed-loop-cove50-v80"
STOP_RUN = ROOT / "artifacts/ape-closed-loop-cove50-v72-stopstate"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v81"
CONFIG = "chest_rgbd_range_guard_goal_map"
SIMILARITY_MIN = .35
TARGET_DISTANCE_MAX_M = .12
HEAD_YAWS = (0., -.65, +.65)


def endpoint_frames(trace: dict) -> list[int]:
    numbered = [(index, event["action"])
                for index, event in enumerate(trace["events"], start=1)
                if "actor_input_sha256" in event]
    chosen = [index for index, action in numbered
              if action in ("reobserve_after_step", "endpoint_head_sweep")][-3:]
    if len(chosen) != 3:
        raise AssertionError("V80 endpoint does not have three RGB-D views")
    return chosen


def infer(vision: Vision, exemplar: np.ndarray,
          rgb: Image.Image, depth: np.ndarray,
          camera_pose: tuple[np.ndarray, np.ndarray],
          expected_xy: np.ndarray, evaluator_box: list[int] | None,
          annotated_path: Path) -> dict:
    proposals = vision.detect(rgb)
    features = (vision.feature([rgb.crop(row["box"]) for row in proposals])
                if proposals else np.zeros((0, len(exemplar))))
    rows = []
    for proposal, feature in zip(proposals, features):
        score = float(feature @ exemplar)
        world = locate_world(proposal["box"], camera_pose, depth)
        distance = (None if world is None else
                    float(np.linalg.norm(np.asarray(world[:2]) - expected_xy)))
        accepted = (score >= SIMILARITY_MIN and distance is not None and
                    distance <= TARGET_DISTANCE_MAX_M)
        rows.append({**proposal,
                     "similarity": round(score, 4),
                     "depth_world_xyz_m": world,
                     "distance_to_memory_m": (None if distance is None
                                              else round(distance, 4)),
                     "actor_gate_accept": accepted,
                     "evaluator_target_iou": round(
                         iou(proposal["box"], evaluator_box), 4)})
    rows.sort(key=lambda row: row["similarity"], reverse=True)
    selected = next((row for row in rows if row["actor_gate_accept"]), None)
    annotated = rgb.copy()
    draw = ImageDraw.Draw(annotated)
    for row in rows:
        color = (0, 255, 0) if row is selected else (255, 200, 0)
        draw.rectangle(row["box"], outline=color, width=3)
    annotated.save(annotated_path)
    return {"proposals": rows,
            "accepted_box": None if selected is None else selected["box"],
            "accepted_similarity": (None if selected is None else
                                    selected["similarity"]),
            "accepted_is_target_evaluator_only": (
                None if selected is None or evaluator_box is None else
                selected["evaluator_target_iou"] >= .25),
            "evaluator_target_box": evaluator_box,
            "annotated_sha256": sha(annotated_path)}


def absent_capture(room: Room, seed: int, frame: dict,
                   actual: np.ndarray, head_yaw: float,
                   out: Path) -> tuple[Image.Image, np.ndarray,
                                       tuple[np.ndarray, np.ndarray], list[int] | None]:
    room.reset()
    room.layout(seed, absent=True, furnished=True)
    room.model.body_pos[room.model.body("loop_table").id] = (
        .40, (1 if seed % 2 else -1) * .46, 0.)
    room.set_table_height(.37)
    room.data.qpos[:] = frame["actor_qpos"]
    room.actual[:] = actual
    room.estimated[:] = frame["estimated_pose_xyyaw"]
    room.head = head_yaw
    room.time_s = float(frame["capture_time_s"])
    room._forward()
    rgb, depth = room.capture("lower_rgb", depth=True)
    pose = room.camera_pose_estimate("lower_rgb")
    rgb.save(out.with_suffix(".png"))
    np.savez_compressed(out.with_suffix(".npz"),
                        rgb=np.asarray(rgb), depth_m=depth,
                        camera_origin_m=pose[0], camera_rotation=pose[1],
                        estimated_pose_xyyaw=room.estimated.copy(),
                        capture_time_s=room.time_s)
    return rgb, depth, pose, None


def present_truth_box(room: Room, seed: int, frame: dict,
                      actual: np.ndarray, head_yaw: float) -> list[int] | None:
    room.reset()
    room.layout(seed, absent=False, furnished=True)
    room.model.body_pos[room.model.body("loop_table").id] = (
        .40, (1 if seed % 2 else -1) * .46, 0.)
    room.set_table_height(.37)
    room.data.qpos[:] = frame["actor_qpos"]
    room.actual[:] = actual
    room.estimated[:] = frame["estimated_pose_xyyaw"]
    room.head = head_yaw
    room.time_s = float(frame["capture_time_s"])
    room._forward()
    return room.truth_box("TomatoSauce", "lower_rgb")


def replay(seed: int, side: str, vision: Vision,
           exemplar: np.ndarray, room: Room) -> dict:
    branch = V80 / f"seed{seed}-{side}"
    trace_path = branch / "trace.json"
    trace = json.loads(trace_path.read_text())
    if trace["status"] != "viewpoint_reached":
        return {"seed": seed, "side": side, "status": "viewpoint_not_reached"}
    stop_path = STOP_RUN / (f"furnished_table_corridor-seed{seed}-0-"
                            f"{CONFIG}-none/first-stop-actor-state.npz")
    with np.load(stop_path) as stop:
        expected_xy = stop["target_estimate_xy_m"].copy()
    actual = np.asarray(trace["final_actual_pose_xyyaw_evaluator_only"])
    rows = []
    for view_index, frame_index in enumerate(endpoint_frames(trace)):
        input_path = branch / f"frame-{frame_index}.npz"
        with np.load(input_path) as saved:
            frame = {key: saved[key].copy() for key in saved.files}
        rgb = Image.fromarray(frame["rgb"])
        depth = frame["depth_m"]
        pose = (frame["camera_origin_m"], frame["camera_rotation"])
        head_yaw = HEAD_YAWS[view_index]
        truth = present_truth_box(room, seed, frame, actual, head_yaw)
        present = infer(vision, exemplar, rgb, depth, pose, expected_xy,
                        truth, OUT / f"seed{seed}-{side}-view{view_index}-present-annotated.png")
        absent_source = OUT / f"seed{seed}-{side}-view{view_index}-absent"
        absent_rgb, absent_depth, absent_pose, _ = absent_capture(
            room, seed, frame, actual, head_yaw, absent_source)
        absent = infer(vision, exemplar, absent_rgb, absent_depth,
                       absent_pose, expected_xy, None,
                       OUT / f"seed{seed}-{side}-view{view_index}-absent-annotated.png")
        rows.append({"head_yaw_rad": head_yaw,
                     "capture_time_s": float(frame["capture_time_s"]),
                     "present_actor_input_sha256": sha(input_path),
                     "absent_actor_input_sha256": sha(
                         absent_source.with_suffix(".npz")),
                     "present": present, "absent": absent})
        print(seed, side, head_yaw,
              "present", present["accepted_is_target_evaluator_only"],
              "absent", absent["accepted_box"] is not None, flush=True)
    true_accept = any(row["present"]["accepted_is_target_evaluator_only"]
                      is True for row in rows)
    wrong_accept = any(row["present"]["accepted_box"] is not None and
                       row["present"]["accepted_is_target_evaluator_only"]
                       is not True for row in rows)
    absent_accept = any(row["absent"]["accepted_box"] is not None
                        for row in rows)
    return {"seed": seed, "side": side,
            "v80_trace_sha256": sha(trace_path),
            "expected_target_xy_m_from_stop": expected_xy.tolist(),
            "views": rows,
            "fresh_correct_target_accepted": true_accept,
            "wrong_object_accepted": wrong_accept,
            "absent_target_accepted": absent_accept}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision("generic")
    room = Room()
    try:
        room.teach()
        teach, _ = room.capture("upper_rgb", depth=False)
        crop = room.truth_box("TomatoSauce", "upper_rgb")
        if crop is None:
            raise AssertionError("Scripted owner cannot confirm teaching crop")
        teach.save(OUT / "owner-teach.png")
        exemplar = vision.feature([teach.crop(crop)])[0]
        np.save(OUT / "owner-taught-exemplar.npy", exemplar)
        rows = [replay(seed, side, vision, exemplar, room)
                for seed in (85, 88) for side in ("right", "left")]
    finally:
        room.close()
    gate = (all(not row["wrong_object_accepted"] and
                not row["absent_target_accepted"] for row in rows)
            and all(any(row["seed"] == seed and
                        row["fresh_correct_target_accepted"]
                        for row in rows) for seed in (85, 88)))
    result = {"scope": "Existing Grounding DINO + DINOv2 owner-crop and frozen 0.35 appearance/0.12m depth-memory gate on V80 reached lower-head endpoint frames, paired with rerendered target-absent control at identical body/head poses. Simulator segmentation only scores after inference; no approach executed.",
              "source_sha256": sha(SOURCE),
              "teach_rgb_sha256": sha(OUT / "owner-teach.png"),
              "owner_crop_xyxy_px": crop,
              "exemplar_sha256": sha(OUT / "owner-taught-exemplar.npy"),
              "vision_source_sha256": sha(Path(__file__).parents[1] /
                                          "teach_find/active_scan.py"),
              "detector_weights_sha256": sha(DETECTOR / "model.safetensors"),
              "retriever_weights_sha256": sha(RETRIEVER / "model.safetensors"),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "rows": rows,
              "fresh_target_reacquisition_gate_pass": gate}
    (OUT / "side-look-target-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / "audit_side_look_target_v81_source.py").write_bytes(
        SOURCE.read_bytes())
    print("fresh target reacquisition gate:", gate)


if __name__ == "__main__":
    main()
