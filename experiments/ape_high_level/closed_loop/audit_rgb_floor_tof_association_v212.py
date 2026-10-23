"""V212: associate RGB can boxes with floor rays, dense depth and coarse ToF."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from run import H, ROOT, W, locate_world, sha
from run_moved_props_v134 import MovedPropRoom


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v212"
V203 = ROOT / "artifacts/ape-closed-loop-cove50-v203/fourth-split-geometry-audit.json"
V204 = ROOT / "artifacts/ape-closed-loop-cove50-v204/fourth-split-origin-audit.json"
V211 = ROOT / "artifacts/ape-closed-loop-cove50-v211/head-depth-optical-audit.json"
TOF_ROOT = ROOT / "artifacts/ape-closed-loop-cove50-v211"
FOV_TOF_RAD = math.radians(45)
FOV_RGB_RAD = math.radians(67)


def tof_box_location(box: list[float], observation: Path) -> dict:
    focal = H / (2 * math.tan(FOV_RGB_RAD / 2))
    u = (box[0] + box[2]) / 2
    v = (box[1] + box[3]) / 2
    horizontal = math.atan((u - W / 2) / focal)
    vertical = math.atan(-(v - H / 2) / focal)
    cell = FOV_TOF_RAD / 8
    col = math.floor(4 + horizontal / cell)
    row = math.floor(4 - vertical / cell)
    result = {"zone_row_col": [row, col],
              "box_center_to_tof_fov": 0 <= row < 8 and 0 <= col < 8,
              "range_m": None, "estimated_xy_m": None}
    if not result["box_center_to_tof_fov"]:
        return result
    with np.load(observation) as z:
        distance = float(z["depth_m"][row, col])
        if not np.isfinite(distance) or bool(z["self_first"][row, col]):
            return result
        origin = z["optical_origin_world_m"].copy()
        rotation = z["optical_rotation_world"].copy()
    azimuth = (col + .5 - 4) * cell
    elevation = (3.5 - row) * cell
    local = np.array((math.tan(azimuth), math.tan(elevation), -1.))
    local /= np.linalg.norm(local)
    point = origin + distance * (rotation @ local)
    result["range_m"] = round(distance, 4)
    result["estimated_xy_m"] = np.round(point[:2], 5).tolist()
    return result


def origin_rows() -> list[dict]:
    scenes = json.loads(V203.read_text())["rows"]
    entries = json.loads(V204.read_text())["rows"]
    optical = json.loads(V211.read_text())["rows"]
    rows = []
    for scene in scenes:
        if scene["label"] != "generous_kinematic":
            continue
        seed = scene["seed"]
        truth = np.asarray(scene["audits"]["0.065"]["positions_xy_m"]["TomatoSauce"])
        entry = next(e for e in entries if e["seed"] == seed and not e["absent"])
        for view in entry["rows"]:
            box = view["accepted_box"]
            if box is None:
                continue
            yaw = float(view["yaw_command_rad"])
            path = (ROOT / f"artifacts/ape-closed-loop-cove50-v204/seed{seed}-light1.00-present/"
                    f"lower-angle-{yaw:+.1f}.npz")
            with np.load(path) as z:
                pose = (z["camera_origin_m"].copy(), z["camera_rotation"].copy())
                dense = z["depth_m"].copy()
            floor = locate_world(box, pose, None)
            dense_xyz = locate_world(box, pose, dense)
            d405_depth = dense.copy()
            d405_depth[(d405_depth < .07) | (d405_depth > .50)] = np.nan
            d405_xyz = locate_world(box, pose, d405_depth)
            observation = TOF_ROOT / f"seed{seed}-present/yaw-{yaw:+.1f}-tof-observation.npz"
            tof = tof_box_location(box, observation)
            candidate = next(r for r in optical if r["seed"] == seed and
                             not r["absent"] and r["head_yaw_rad"] == yaw)
            def error(xyz: list[float] | None) -> float | None:
                return None if xyz is None else round(float(np.linalg.norm(
                    np.asarray(xyz[:2]) - truth)), 4)
            rows.append({"seed": seed, "yaw_rad": yaw,
                         "accepted_box_actor": box,
                         "actor_frame_sha256": sha(path),
                         "true_xy_evaluator_only": truth.tolist(),
                         "dense_xyz_m": dense_xyz,
                         "dense_xy_error_m_evaluator_only": error(dense_xyz),
                         "rgb_floor_xyz_m": floor,
                         "rgb_floor_xy_error_m_evaluator_only": error(floor),
                         "d405_ideal_xyz_m": d405_xyz,
                         "d405_ideal_xy_error_m_evaluator_only": error(d405_xyz),
                         "tof": {**tof,
                                 "xy_error_m_evaluator_only": error(tof["estimated_xy_m"])},
                         "tof_center_target_hit_evaluator_only":
                             candidate["tof"]["target_center_zones_evaluator_only"] > 0})
    return rows


def raised_table_counterfactual() -> list[dict]:
    scenes = json.loads(V203.read_text())["rows"]
    rows = []
    for scene in scenes:
        if scene["label"] != "generous_kinematic":
            continue
        seed = scene["seed"]
        side = 1 if seed % 2 else -1
        positions = scene["audits"]["0.065"]["positions_xy_m"]
        room = MovedPropRoom({"CreamCheese": tuple(positions["CreamCheese"]),
                              "Parmesan": tuple(positions["Parmesan"])})
        try:
            room.reset()
            room.layout(seed, absent=False, furnished=True)
            room.model.body_pos[room.model.body("loop_table").id] = (
                .40, side * .46, 0.)
            room.set_table_height(.37)
            target_xy = np.array((.40, side * .46))
            room.model.body_pos[room.model.body("loop_TomatoSauce").id] = (
                target_xy[0], target_xy[1], .43)
            room.head = math.atan2(target_xy[1], target_xy[0])
            room._forward()
            image, depth = room.capture("upper_rgb", depth=True)
            box = room.truth_box("TomatoSauce", "upper_rgb")
            branch = OUT / f"seed{seed}-table-negative.png"
            image.save(branch)
            assert box is not None and depth is not None
            pose = room.camera_pose_estimate("upper_rgb")
            floor = locate_world(box, pose, None)
            dense = locate_world(box, pose, depth)
            def err(xyz: list[float] | None) -> float | None:
                return None if xyz is None else round(float(np.linalg.norm(
                    np.asarray(xyz[:2]) - target_xy)), 4)
            rows.append({"seed": seed, "scene": "can_above_table_upper_rgb_counterfactual",
                         "box_source": "evaluator segmentation upper bound, not actor detector",
                         "truth_box_evaluator_only": box,
                         "true_xy_evaluator_only": target_xy.tolist(),
                         "rgb_floor_xyz_m": floor,
                         "rgb_floor_xy_error_m_evaluator_only": err(floor),
                         "dense_xyz_m": dense,
                         "dense_xy_error_m_evaluator_only": err(dense),
                         "image_sha256": sha(branch)})
        finally:
            room.close()
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = origin_rows()
    assert len(rows) == 9
    counterfactual = raised_table_counterfactual()
    def count_within(name: str, margin: float) -> int:
        return sum(row[name] is not None and row[name] <= margin for row in rows)
    summary = {"accepted_origin_views": len(rows),
               "dense_6cm": count_within("dense_xy_error_m_evaluator_only", .06),
               "rgb_floor_6cm": count_within("rgb_floor_xy_error_m_evaluator_only", .06),
               "d405_ideal_6cm": count_within("d405_ideal_xy_error_m_evaluator_only", .06),
               "tof_available": sum(r["tof"]["estimated_xy_m"] is not None for r in rows),
               "tof_6cm": sum(r["tof"]["xy_error_m_evaluator_only"] is not None and
                              r["tof"]["xy_error_m_evaluator_only"] <= .06 for r in rows),
               "tof_over_12cm_error": sum(r["tof"]["xy_error_m_evaluator_only"] is not None and
                                          r["tof"]["xy_error_m_evaluator_only"] > .12
                                          for r in rows),
               "table_counterfactual_views": len(counterfactual)}
    result = {"scope": "V212 revealed static observation comparison on the nine V204 correctly accepted present origin RGB boxes. Current actor lower-RGB floor-plane ray (run.locate_world with no depth), exact simulator dense depth baseline, D405 7-50cm ideal mask, and nearest V211 provisional 8x8 ToF zone. Evaluator target XY scores only afterward. Separate table-supported can counterfactual uses evaluator exact box to expose floor-plane assumption, not a detector or live motion result. No route/stop authority, physical optics, noise or new threshold tuning.",
              "source_sha256": sha(SOURCE), "v203_sha256": sha(V203),
              "v204_sha256": sha(V204), "v211_sha256": sha(V211),
              "rows": rows, "table_counterfactual": counterfactual,
              "summary": summary}
    (OUT / "rgb-floor-tof-association-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
