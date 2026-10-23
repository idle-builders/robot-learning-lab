"""V159: evaluator swept-body and actor-camera screen of a needed turn."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from audit_side_look_target_v81 import infer
from build_scene_pairs_v133 import body_geoms, full_body_clearance, scene_geoms
from run import ROOT, Vision, iou, sha
from run_moved_props_v134 import MovedPropRoom


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v159"
V149 = ROOT / "artifacts/ape-closed-loop-cove50-v149"
V158 = ROOT / "artifacts/ape-closed-loop-cove50-v158"
SCENES = ROOT / "artifacts/ape-closed-loop-cove50-v133/paired-scene-validity.json"
SEED = 85
LIGHT = 1.35


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(SCENES.read_text())
    variant = next(v for row in scenes["rows"] if row["seed"] == SEED
                   for v in row["passable_variants"]
                   if v["variant"].endswith("-2"))
    trace_path = V158 / f"scene{SEED}/seed{SEED}/trace.json"
    trace = json.loads(trace_path.read_text())
    frame_count = sum(e.get("estimated_pose_xyyaw") is not None and
                      e.get("actor_input_sha256") is not None
                      for e in trace["events"])
    source_path = trace_path.parent / f"frame-{frame_count}.npz"
    with np.load(source_path) as z:
        qpos = z["actor_qpos"].copy()
        stamp = float(z["capture_time_s"])
    target = np.asarray(trace["final_actor_target_xy_m"])
    positions = {"CreamCheese": tuple(variant["cream_cheese_xy_m"]),
                 "Parmesan": tuple(variant["parmesan_xy_m"])}
    room = MovedPropRoom(positions)
    vision = Vision("generic")
    exemplar = np.load(V149 / "owner-taught-exemplar.npy")
    try:
        room.reset()
        room.layout(SEED, absent=False, furnished=True)
        room.model.body_pos[room.model.body("loop_table").id] = (
            .40, (1 if SEED % 2 else -1) * .46, 0.)
        room.set_table_height(.37)
        room.model.light_diffuse[0] *= LIGHT
        room.data.qpos[:] = qpos
        actual = np.asarray(trace["evaluator_final_actual_xyyaw"])
        estimated = np.asarray(trace["final_estimated_pose_xyyaw"])
        room.actual[:] = actual
        room.estimated[:] = estimated
        room.head = float(qpos[room.head_addr])
        room.time_s = stamp
        room._forward()
        scene = scene_geoms(room)
        robot = body_geoms(room, scene)
        rows = []
        for yaw in (.3, .5, .7, .9):
            swept = (float("inf"), "", "")
            for angle in np.linspace(actual[2], yaw, 25):
                room.actual[:] = actual
                room.actual[2] = angle
                room._forward()
                gap = full_body_clearance(room, scene, robot)
                if gap[0] < swept[0]:
                    swept = gap
            room.actual[:] = actual
            room.estimated[:] = estimated
            room.actual[2] = yaw
            room.estimated[2] = yaw
            room._forward()
            rgb, depth = room.capture("lower_rgb", depth=True)
            camera = room.camera_pose_estimate("lower_rgb")
            image_path = OUT / f"turn-yaw-{yaw:.1f}.png"
            annotated_path = OUT / f"turn-yaw-{yaw:.1f}-annotated.png"
            rgb.save(image_path)
            detected = infer(vision, exemplar, rgb, depth, camera, target,
                             None, annotated_path)
            truth = room.truth_box("TomatoSauce", "lower_rgb")
            correct = (detected["accepted_box"] is not None and
                       truth is not None and
                       iou(detected["accepted_box"], truth) >= .25)
            rows.append({"body_yaw_rad": yaw,
                         "head_yaw_rad": room.head,
                         "actor_target_bearing_after_turn_rad": round(
                             math.atan2(target[1]-estimated[1],
                                        target[0]-estimated[0])-yaw, 4),
                         "evaluator_min_full_scene_swept_clearance_m":
                             round(swept[0], 4),
                         "evaluator_limiting_geoms": list(swept[1:]),
                         "evaluator_truth_box": truth,
                         "actor_accepted_box": detected["accepted_box"],
                         "actor_similarity": detected["accepted_similarity"],
                         "evaluator_accepted_correct": correct,
                         "image_sha256": sha(image_path),
                         "annotated_sha256": sha(annotated_path)})
        result = {"scope": "V159 read-only body yaw sweep at V158 seed85 safe stop. Evaluator full-scene swept clearance and truth-box correctness; camera detector receives only rendered RGB-D, estimated pose and taught exemplar. No turn skill executed or range-stop qualified.",
                  "source_sha256": sha(SOURCE),
                  "v158_trace_sha256": sha(trace_path),
                  "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
                  "source_frame_sha256": sha(source_path),
                  "actual_start_xyyaw": actual.tolist(),
                  "estimated_start_xyyaw": estimated.tolist(),
                  "actor_target_xy_m": target.tolist(),
                  "rows": rows}
        (OUT / "body-turn-view-audit.json").write_text(
            json.dumps(result, indent=2) + "\n")
        (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
        for row in rows:
            print(row["body_yaw_rad"], row["evaluator_min_full_scene_swept_clearance_m"],
                  row["actor_similarity"], row["evaluator_accepted_correct"], flush=True)
    finally:
        room.close()


if __name__ == "__main__":
    main()
