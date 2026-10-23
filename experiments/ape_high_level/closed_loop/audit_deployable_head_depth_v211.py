"""V211: paired fixed-RGB, D405 range and head-adjacent 8x8 ToF optical screen."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from robot_geometry import RobotGeometry
from run import H, ROOT, W, sha
from run_moved_props_v134 import MovedPropRoom


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v211"
V203 = ROOT / "artifacts/ape-closed-loop-cove50-v203/fourth-split-geometry-audit.json"
V204 = ROOT / "artifacts/ape-closed-loop-cove50-v204"
DESIGN = ROOT / "robot/ape-current-design.json"
FRAME = ROOT / "experiments/compact_morphology/ape_frame50/frame_contract.yaml"
SHELL = ROOT / "artifacts/ape-frame50-cove-shell-v1/cove50-shell-visual.xml"
TOF_HEAD_LOCAL_M = np.array((.059, -.039, -.014))
TOF_FOV_RAD = math.radians(45)
ROOM_GROUP = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
SELF_GROUP = np.array([0, 1, 1, 0, 1, 0], dtype=np.uint8)


def segment(room: MovedPropRoom, name: str) -> np.ndarray:
    room.renderer.enable_segmentation_rendering()
    room.renderer.update_scene(room.data, camera="lower_rgb")
    segmentation = room.renderer.render().copy()
    room.renderer.disable_segmentation_rendering()
    gid = room.model.geom(f"loop_{name}_geom").id
    return ((segmentation[:, :, 0] == gid) &
            (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))


def stereo(room: MovedPropRoom, lower: np.ndarray,
           upper: np.ndarray, target_mask: np.ndarray) -> dict:
    lower_id = room.model.camera("lower_rgb").id
    upper_id = room.model.camera("upper_rgb").id
    lower_origin = room.data.cam_xpos[lower_id]
    upper_origin = room.data.cam_xpos[upper_id]
    lower_rotation = room.data.cam_xmat[lower_id].reshape(3, 3)
    upper_rotation = room.data.cam_xmat[upper_id].reshape(3, 3)
    focal_lower = H / (2 * math.tan(math.radians(
        float(room.model.cam_fovy[lower_id])) / 2))
    focal_upper = H / (2 * math.tan(math.radians(
        float(room.model.cam_fovy[upper_id])) / 2))
    rows, cols = np.mgrid[4:H:8, 4:W:8]
    distance = lower[rows, cols]
    valid = np.isfinite(distance) & (distance >= .08) & (distance <= 1.4)
    rays = np.stack(((cols - W / 2) / focal_lower,
                     -(rows - H / 2) / focal_lower,
                     -np.ones_like(distance)), axis=-1)
    world = lower_origin + (rays * distance[..., None]) @ lower_rotation.T
    camera = (world - upper_origin) @ upper_rotation
    forward = -camera[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        upper_u = focal_upper * camera[..., 0] / forward + W / 2
        upper_v = -focal_upper * camera[..., 1] / forward + H / 2
    geometric = (valid & (forward > .05) &
                 (upper_u >= 0) & (upper_u < W) &
                 (upper_v >= 0) & (upper_v < H))
    image_u = np.clip(np.rint(np.nan_to_num(upper_u, nan=0)).astype(int), 0, W - 1)
    image_v = np.clip(np.rint(np.nan_to_num(upper_v, nan=0)).astype(int), 0, H - 1)
    visible = geometric & np.isfinite(upper[image_v, image_u]) & (
        np.abs(upper[image_v, image_u] - forward) <= .03)
    target = target_mask[rows, cols] & valid
    return {"lower_valid_sampled_pixels": int(valid.sum()),
            "geometric_shared_pixels": int(geometric.sum()),
            "upper_depth_consistent_shared_pixels": int(visible.sum()),
            "target_lower_sampled_pixels": int(target.sum()),
            "target_geometric_shared_pixels": int((target & geometric).sum()),
            "target_upper_depth_consistent_shared_pixels": int((target & visible).sum())}


def first_hit(room: MovedPropRoom, origin: np.ndarray,
              direction: np.ndarray) -> tuple[float | None, int, bool]:
    room_geom = np.array([-1], dtype=np.int32)
    self_geom = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(room.model, room.data, origin, direction,
                             ROOM_GROUP, True, -1, room_geom)
    self_distance = mujoco.mj_ray(room.model, room.data, origin, direction,
                                  SELF_GROUP, True, -1, self_geom)
    if self_distance >= 0 and (distance < 0 or self_distance <= distance):
        return None, int(self_geom[0]), True
    if distance < .02 or distance > 4.:
        return None, -1, False
    return float(distance), int(room_geom[0]), False


def tof(room: MovedPropRoom) -> tuple[dict, dict[str, np.ndarray]]:
    head_id = room.model.body("head").id
    lower_id = room.model.camera("lower_rgb").id
    head_rotation = room.data.xmat[head_id].reshape(3, 3)
    origin = room.data.xpos[head_id] + head_rotation @ TOF_HEAD_LOCAL_M
    rotation = room.data.cam_xmat[lower_id].reshape(3, 3)
    robot = RobotGeometry(room.model)
    robot.set_state(room.data.qpos.copy())
    body_gap = RobotGeometry.solid_distance(origin, robot.primitives())
    depths = np.full((8, 8), np.nan, dtype=np.float32)
    self_first = np.zeros((8, 8), dtype=bool)
    center_ids = np.full((8, 8), -1, dtype=np.int32)
    optimistic_ids: list[set[int]] = [set() for _ in range(64)]
    for row in range(8):
        for col in range(8):
            for xoff, yoff in ((0., 0.), (-.25, -.25), (-.25, .25),
                               (.25, -.25), (.25, .25)):
                horizontal = (col + .5 + xoff - 4) * TOF_FOV_RAD / 8
                vertical = (3.5 - row - yoff) * TOF_FOV_RAD / 8
                local = np.array((math.tan(horizontal),
                                  math.tan(vertical), -1.))
                local /= np.linalg.norm(local)
                direction = rotation @ local
                distance, gid, blocked = first_hit(room, origin, direction)
                if xoff == 0. and yoff == 0.:
                    if distance is not None:
                        depths[row, col] = distance
                    center_ids[row, col] = gid if distance is not None else -1
                    self_first[row, col] = blocked
                if distance is not None:
                    optimistic_ids[row * 8 + col].add(gid)
    target_id = room.model.geom("loop_TomatoSauce_geom").id
    small_prop_ids = {room.model.geom(f"loop_{name}_geom").id
                      for name in ("CreamCheese", "Parmesan")}
    result = {"optical_origin_world_m": np.round(origin, 5).tolist(),
              "optical_origin_selected_body_gap_m": round(float(body_gap), 5),
              "valid_center_zones": int(np.isfinite(depths).sum()),
              "self_first_center_zones": int(self_first.sum()),
              "target_center_zones_evaluator_only": int((center_ids == target_id).sum()),
              "target_optimistic_subray_zones_evaluator_only": sum(
                  target_id in ids for ids in optimistic_ids),
              "small_prop_center_zones_evaluator_only": sum(
                  gid in small_prop_ids for gid in center_ids.ravel()),
              "small_prop_optimistic_subray_zones_evaluator_only": sum(
                  bool(ids & small_prop_ids) for ids in optimistic_ids)}
    return result, {"depth_m": depths, "self_first": self_first,
                    "optical_origin_world_m": origin,
                    "optical_rotation_world": rotation}


def one_view(room: MovedPropRoom, path: Path, *, seed: int,
             absent: bool) -> dict:
    with np.load(path) as z:
        qpos = z["actor_qpos"].copy()
        estimated = z["estimated_pose_xyyaw"].copy()
    room.data.qpos[:] = qpos
    room.actual[:] = 0.
    room.estimated[:] = estimated
    room.head = float(qpos[room.head_addr])
    room._forward()
    down_rgb, down_depth = room.capture("lower_rgb", depth=True)
    up_rgb, up_depth = room.capture("upper_rgb", depth=True)
    assert down_depth is not None and up_depth is not None
    target = segment(room, "TomatoSauce")
    pair = stereo(room, down_depth, up_depth, target)
    target_depth = down_depth[target]
    valid_target = np.isfinite(target_depth) & (target_depth > .02)
    metric = target_depth[valid_target]
    tof_result, tof_observation = tof(room)
    branch = OUT / f"seed{seed}-{'absent' if absent else 'present'}"
    branch.mkdir(parents=True, exist_ok=True)
    stem = path.stem.replace("lower-angle-", "yaw-")
    np.savez_compressed(branch / f"{stem}-tof-observation.npz",
                        **tof_observation)
    if abs(room.head) < .05:
        down_rgb.save(branch / "lower-rgb.png")
        up_rgb.save(branch / "upper-rgb.png")
    return {"seed": seed, "absent": absent, "head_yaw_rad": round(room.head, 3),
            "source_frame_sha256": sha(path),
            "stereo": pair,
            "true_can_visible_pixels_evaluator_only": int(valid_target.sum()),
            "true_can_median_depth_m_evaluator_only":
                None if not len(metric) else round(float(np.median(metric)), 4),
            "true_can_pixels_d405_ideal_7_to_50cm_evaluator_only": int(
                ((metric >= .07) & (metric <= .50)).sum()),
            "true_can_pixels_d430_ideal_30cm_to_3m_evaluator_only": int(
                ((metric >= .30) & (metric <= 3.)).sum()),
            "tof": tof_result,
            "tof_observation_sha256": sha(branch / f"{stem}-tof-observation.npz")}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(V203.read_text())["rows"]
    rows = []
    for scene in scenes:
        if scene["label"] != "generous_kinematic":
            continue
        seed = scene["seed"]
        positions = scene["audits"]["0.065"]["positions_xy_m"]
        for absent in (False, True):
            layout = {"CreamCheese": tuple(positions["CreamCheese"]),
                      "Parmesan": tuple(scene["absent_challenge"]["parmesan_xy_m"]
                                         if absent else positions["Parmesan"])}
            room = MovedPropRoom(layout)
            try:
                room.reset()
                room.layout(seed, absent=absent, furnished=True)
                room.model.body_pos[room.model.body("loop_table").id] = (
                    .40, (1 if seed % 2 else -1) * .46, 0.)
                room.set_table_height(.37)
                room._forward()
                branch = V204 / f"seed{seed}-light1.00-{'absent' if absent else 'present'}"
                paths = sorted(branch.glob("lower-angle-*.npz"))
                assert len(paths) == 5
                for path in paths:
                    row = one_view(room, path, seed=seed, absent=absent)
                    rows.append(row)
                    print(seed, "absent", absent, "yaw", row["head_yaw_rad"],
                          "stereo", row["stereo"]["upper_depth_consistent_shared_pixels"],
                          "can", row["true_can_visible_pixels_evaluator_only"],
                          "tof", row["tof"]["target_center_zones_evaluator_only"],
                          flush=True)
            finally:
                room.close()
    present = [row for row in rows if not row["absent"]]
    result = {"scope": "V211 same-scene static optical screen on V203 generous paired present/absent rooms and V204 five-angle origin head states. Direct Frame+shell fixed upper/lower RGB camera geometry. Stereo common pixels from withheld ideal depth, D405/D430 ideal target range masks from evaluator true target segmentation, and provisional 13x23x3mm head-adjacent VL53L8CX 8x8 45deg first-hit center/optimistic subrays. ToF actor observation stores only depth/self-hit, not geom labels. No motion, real depth device, physical ToF reflectance/crosstalk/mixed-zone behavior, module bracket or hardware stop qualification.",
              "source_sha256": sha(SOURCE), "v203_sha256": sha(V203),
              "v204_sha256": sha(V204 / "fourth-split-origin-audit.json"),
              "selected_design_sha256": sha(DESIGN),
              "frame_contract_sha256": sha(FRAME), "shell_visual_sha256": sha(SHELL),
              "candidate_tof_head_local_origin_m": TOF_HEAD_LOCAL_M.tolist(),
              "source_specs": {
                  "pi_camera3_wide": "https://datasheets.raspberrypi.com/camera/camera-module-3-product-brief.pdf",
                  "d405": "https://www.realsenseai.com/products/stereo-depth-camera-d405/",
                  "vl53l8cx_carrier": "https://www.pololu.com/product/3419"},
              "rows": rows,
              "summary": {
                  "paired_views": len(rows),
                  "lower_valid_sampled_pixels": sum(r["stereo"]["lower_valid_sampled_pixels"] for r in rows),
                  "stereo_geometric_shared_pixels": sum(r["stereo"]["geometric_shared_pixels"] for r in rows),
                  "stereo_depth_consistent_shared_pixels": sum(
                      r["stereo"]["upper_depth_consistent_shared_pixels"] for r in rows),
                  "present_target_visible_views": sum(
                      r["true_can_visible_pixels_evaluator_only"] > 0 for r in present),
                  "present_d405_any_target_depth_views": sum(
                      r["true_can_pixels_d405_ideal_7_to_50cm_evaluator_only"] > 0
                      for r in present),
                  "present_d430_any_target_depth_views": sum(
                      r["true_can_pixels_d430_ideal_30cm_to_3m_evaluator_only"] > 0
                      for r in present),
                  "present_tof_center_target_views": sum(
                      r["tof"]["target_center_zones_evaluator_only"] > 0
                      for r in present),
                  "present_tof_optimistic_target_views": sum(
                      r["tof"]["target_optimistic_subray_zones_evaluator_only"] > 0
                      for r in present)}}
    (OUT / "head-depth-optical-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
