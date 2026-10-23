"""V207: sensor-built alternate information paths from revealed V206 view stops."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from audit_moved_goal_reselection_v135 import course_goal
from audit_origin_route_v152 import score_path
from depth_voxel import DepthVoxelMap
from robot_geometry import RobotGeometry
from run import H, ROOT, W, Vision, iou, sensed_range, sha
from run_color_bank_live_v201 import OwnerAnchoredColorViewBank
from run_moved_props_v134 import MovedPropRoom
from self_depth_mask import SelfDepthMask
from two_height_clearance import TwoHeightClearance
from upper_obstacle_side import choose_side


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v207"
V203 = ROOT / "artifacts/ape-closed-loop-cove50-v203/fourth-split-geometry-audit.json"
V204 = ROOT / "artifacts/ape-closed-loop-cove50-v204"
V206 = ROOT / "artifacts/ape-closed-loop-cove50-v206"
V149 = ROOT / "artifacts/ape-closed-loop-cove50-v149/owner-taught-exemplar.npy"
V150 = ROOT / "artifacts/ape-closed-loop-cove50-v150/entry-color-audit.json"
MARGIN = .095


def saved_frames(seed: int) -> list[Path]:
    origin = sorted((V204 / f"seed{seed}-light1.00-present").glob("lower-angle-*.npz"))
    live = sorted((V206 / f"scene{seed}/seed{seed}").glob("frame-*.npz"),
                  key=lambda p: int(p.stem.split("-")[-1]))
    return origin + live


def final_bank(seed: int, vision: Vision, exemplar: np.ndarray) -> list[np.ndarray]:
    events = json.loads((V206 / f"scene{seed}/seed{seed}/online-color-bank-events.json").read_text())
    accepted = [e for e in events if e["source"] == "original_owner_accepted_crop"][-2:]
    assert len(accepted) == 2
    crops = []
    for event in accepted:
        index = event["call_index"]
        with np.load(V206 / f"scene{seed}/seed{seed}/frame-{index}.npz") as z:
            image = Image.fromarray(z["rgb"])
        crops.append(image.crop(event["crop_box"]))
    features = vision.feature(crops)
    return [np.asarray(feature, dtype=float) for feature in features]


def audit(seed: int, vision: Vision, exemplar: np.ndarray,
          owner_hist: np.ndarray) -> dict:
    scene = next(row for row in json.loads(V203.read_text())["rows"]
                 if row["seed"] == seed)
    assert scene["label"] == "generous_kinematic"
    positions = scene["audits"]["0.065"]["positions_xy_m"]
    room = MovedPropRoom({"CreamCheese": tuple(positions["CreamCheese"]),
                          "Parmesan": tuple(positions["Parmesan"])})
    mask = SelfDepthMask(room.model, camera="lower_rgb", width=W, height=H)
    trace_path = V206 / f"scene{seed}/seed{seed}/trace.json"
    trace = json.loads(trace_path.read_text())
    assert trace["status"] == "information_goal_reached"
    pose = np.asarray(trace["final_estimated_pose_xyyaw"], dtype=float)
    actual = np.asarray(trace["evaluator_final_actual_xyyaw"], dtype=float)
    target = np.asarray(trace["final_actor_target_xy_m"], dtype=float)
    recent = final_bank(seed, vision, exemplar)
    input_paths = saved_frames(seed)
    try:
        room.reset()
        room.layout(seed, absent=False, furnished=True)
        room.model.body_pos[room.model.body("loop_table").id] = (
            .40, (1 if seed % 2 else -1) * .46, 0.)
        room.set_table_height(.37)
        fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
        voxel = DepthVoxelMap()
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
        scans = {height: sensed_range(room, seed, 2070000 + int(height * 100),
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
            cells = selected["path_cells"]
            evaluated = score_path(room, actual.copy(), cells)
            views = []
            for index in sorted({min(2, len(cells) - 1), len(cells) - 1}):
                if index <= 0:
                    continue
                xy = .04 * np.asarray(cells[index], dtype=float)
                room.actual[:2] = xy
                room.estimated[:2] = xy
                bearing = math.atan2(target[1] - xy[1], target[0] - xy[0])
                room.head = float(np.clip(bearing - pose[2], -1.2, 1.2))
                room._forward()
                rgb, depth = room.capture("lower_rgb", depth=True)
                truth = room.truth_box("TomatoSauce", "lower_rgb")
                branch = OUT / f"seed{seed}-{side}-waypoint{index}"
                branch.parent.mkdir(parents=True, exist_ok=True)
                rgb.save(branch.with_suffix(".png"))
                bank = OwnerAnchoredColorViewBank(exemplar, owner_hist)
                bank.recent = recent.copy()
                inferred = bank.infer(vision, exemplar, rgb, depth,
                                      room.camera_pose_estimate("lower_rgb"),
                                      target, None,
                                      branch.with_name(branch.name + "-annotated.png"))
                best_target = max(inferred["proposals"],
                                  key=lambda row: iou(row["box"], truth),
                                  default=None)
                views.append({"waypoint_index": index, "xy_m": xy.tolist(),
                              "target_truth_box_evaluator_only": truth,
                              "best_target_iou_evaluator_only":
                              None if best_target is None else round(iou(best_target["box"], truth), 4),
                              "best_target_similarity":
                              None if best_target is None else best_target["similarity"],
                              "best_target_hue_intersection":
                              None if best_target is None else best_target["owner_hue_intersection"],
                              "best_target_accepted":
                              bool(best_target is not None and best_target["actor_gate_accept"]),
                              "any_accepted": inferred["accepted_box"] is not None,
                              "image_sha256": sha(branch.with_suffix(".png"))})
                room.actual[:] = actual
                room.estimated[:] = pose
                room.head = float(qpos[room.head_addr])
                room._forward()
            rows.append({"side": side, "actor_route": selected,
                         "evaluator_route": evaluated, "endpoint_views": views})
        return {"seed": seed, "stop_status": trace["status"],
                "actor_pose_xyyaw": pose.tolist(), "actor_target_xy_m": target.tolist(),
                "actor_side_cue": cue, "margin_m": MARGIN,
                "actor_input_sha256": [sha(p) for p in input_paths],
                "candidate_sides": rows}
    finally:
        mask.close()
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())["owner_color"]["hue_histogram"])
    rows = [audit(seed, vision, exemplar, owner_hist) for seed in (111, 113)]
    result = {"scope": "V207 revealed-development read-only actor-side alternative routes from V206 seed111/113 final sensor frames. Actor routes use self-masked saved RGB-D, fresh virtual4/8/11cm range and selected-body9.5cm point margin. Hypothetical endpoint images/evaluator route clearance are future-view diagnostics only, not actor inputs or executed motion. Final two online original-owner prototype crops reconstructed from V206 frame images; unchanged V201 color bank scores images. No threshold adjustment or independent validation.",
              "source_sha256": sha(SOURCE),
              "v203_audit_sha256": sha(V203),
              "v206_audit_sha256": sha(V206 / "origin-to-target-audit.json"),
              "rows": rows}
    (OUT / "alternate-view-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["seed"], row["actor_side_cue"],
              [(r["side"], r["actor_route"].get("goal_cell"),
                None if r["evaluator_route"] is None else
                r["evaluator_route"].get("min_full_scene_m"),
                [(v["waypoint_index"], v["best_target_iou_evaluator_only"],
                  v["best_target_accepted"]) for v in r["endpoint_views"]])
               for r in row["candidate_sides"]], flush=True)


if __name__ == "__main__":
    main()
