"""V156: actor-only side/view routes from V155 safe-stop sensor states."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from audit_moved_goal_reselection_v135 import course_goal
from audit_origin_route_v152 import score_path
from depth_voxel import DepthVoxelMap
from robot_geometry import RobotGeometry
from run import H, ROOT, W, sensed_range, sha
from run_moved_props_v134 import MovedPropRoom
from self_depth_mask import SelfDepthMask
from two_height_clearance import TwoHeightClearance
from upper_obstacle_side import choose_side


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v156"
V151 = ROOT / "artifacts/ape-closed-loop-cove50-v151"
V155 = ROOT / "artifacts/ape-closed-loop-cove50-v155"
SCENES = ROOT / "artifacts/ape-closed-loop-cove50-v133/paired-scene-validity.json"
LIGHT = 1.35
MARGIN = .095


def audit(seed: int, variant: dict) -> dict:
    trace_path = V155 / f"scene{seed}/seed{seed}/trace.json"
    trace = json.loads(trace_path.read_text())
    accepted = [e for e in trace["events"] if e.get("accepted_box") is not None]
    target = np.asarray(accepted[-1]["actor_target_xy_m"], dtype=float)
    pose = np.asarray(trace["final_estimated_pose_xyyaw"], dtype=float)
    actual = np.asarray(trace["evaluator_final_actual_xyyaw"], dtype=float)
    positions = {"CreamCheese": tuple(variant["cream_cheese_xy_m"]),
                 "Parmesan": tuple(variant["parmesan_xy_m"])}
    room = MovedPropRoom(positions)
    mask = SelfDepthMask(room.model, camera="lower_rgb", width=W, height=H)
    try:
        room.reset()
        room.layout(seed, absent=False, furnished=True)
        room.model.body_pos[room.model.body("loop_table").id] = (
            .40, (1 if seed % 2 else -1) * .46, 0.)
        room.set_table_height(.37)
        room.model.light_diffuse[0] *= LIGHT
        fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
        voxel = DepthVoxelMap()
        input_paths = list(sorted((V151 / f"seed{seed}-light{LIGHT:.2f}-present")
                                  .glob("lower-angle-*.npz")))
        input_paths += list(sorted(trace_path.parent.glob("frame-*.npz"),
                                   key=lambda p: int(p.stem.split("-")[-1])))
        latest = None
        for path in input_paths:
            with np.load(path) as z:
                qpos = z["actor_qpos"].copy()
                filtered, _ = mask.apply(z["depth_m"], qpos)
                stamp = float(z["capture_time_s"])
                voxel.observe(filtered, (z["camera_origin_m"],
                                         z["camera_rotation"]), stamp,
                              fovy_deg=fovy, self_masked=True)
                latest = qpos, stamp
        assert latest is not None
        qpos, stamp = latest
        room.data.qpos[:] = qpos
        room.actual[:] = actual
        room.estimated[:] = pose
        room.head = float(qpos[room.head_addr])
        room.time_s = stamp
        room._forward()
        scans = {height: sensed_range(room, seed, 1560000 + int(height * 100),
                                      height_m=height)
                 for height in (.04, .08, .11)}
        robot = RobotGeometry(room.model)
        point = TwoHeightClearance(robot, qpos, pose, scans[.11], scans[.04],
                                   voxel, stamp, range_8=scans[.08])
        cue = choose_side(voxel, pose, target, stamp)
        rows = []
        for side in ("left", "right"):
            selected = course_goal(pose, target, side, scans, point, set(),
                                   margin_m=MARGIN)
            evaluated = score_path(room, actual.copy(), selected["path_cells"])
            room.actual[:] = actual
            room.estimated[:] = pose
            room._forward()
            view_screen = []
            cells = selected["path_cells"]
            for index in sorted({min(2, len(cells) - 1), len(cells) - 1}):
                if index <= 0:
                    continue
                xy = .04 * np.asarray(cells[index], dtype=float)
                room.actual[:2] = xy
                room.estimated[:2] = xy
                bearing = math.atan2(target[1] - xy[1], target[0] - xy[0])
                room.head = float(np.clip(bearing - pose[2], -1.2, 1.2))
                room._forward()
                rgb, _ = room.capture("lower_rgb", depth=False)
                truth = room.truth_box("TomatoSauce", "lower_rgb")
                image_path = OUT / f"seed{seed}-{side}-waypoint{index}.png"
                rgb.save(image_path)
                view_screen.append({"waypoint_index": index,
                                    "xy_m": xy.tolist(),
                                    "evaluator_truth_box": truth,
                                    "image_sha256": sha(image_path)})
            room.actual[:] = actual
            room.estimated[:] = pose
            room.head = float(qpos[room.head_addr])
            room._forward()
            rows.append({"side": side, "actor_route": selected,
                         "evaluator_route": evaluated,
                         "evaluator_view_screen": view_screen})
        return {"seed": seed, "stop_status": trace["status"],
                "actor_pose_xyyaw": pose.tolist(),
                "actor_target_xy_m": target.tolist(),
                "actor_side_cue": cue,
                "actor_input_sha256": [sha(p) for p in input_paths],
                "margin_m": MARGIN, "candidate_sides": rows}
    finally:
        mask.close()
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(SCENES.read_text())
    rows = []
    for seed in (85, 88):
        variant = next(v for row in scenes["rows"] if row["seed"] == seed
                       for v in row["passable_variants"]
                       if v["variant"].endswith("-2"))
        row = audit(seed, variant)
        rows.append(row)
        print(seed, row["stop_status"], row["actor_side_cue"],
              [(candidate["side"], candidate["actor_route"]["goal_cell"],
                None if candidate["evaluator_route"] is None else
                candidate["evaluator_route"]["min_full_scene_m"])
               for candidate in row["candidate_sides"]], flush=True)
    result = {"scope": "V156 read-only actor-side alternate information routes from V155 stopped states, using retained RGB-D/joint/pose frames and new 4/8/11 cm range readings. Evaluator route clearance afterward only. No route execution.",
              "source_sha256": sha(SOURCE), "v155_audit_sha256":
              sha(V155 / "origin-to-target-audit.json"), "rows": rows}
    (OUT / "recovery-view-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
