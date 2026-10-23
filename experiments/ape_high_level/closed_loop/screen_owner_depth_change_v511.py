"""Screen an actor-observable RGB-D change cloud for the presented owner item.

Uses the saved V509 route endpoint, prior range track and the same owner
scene. Segmentation and true can pose are evaluator-only measurements.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import robocasa_person_interaction as base
import robocasa_standoff_turn_v507 as turn
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_person_joined_v497 as joined
from audit_robocasa_layout2_routes_v409 import sha256
from screen_robocasa_owner_person_v489 import evaluator_box
from screen_robocasa_owner_safe_stop_v494 import aim_camera


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-owner-depth-change-v511"
SOURCE = ROOT / "artifacts/ape-robocasa-person-standoff-bound-v509/report.json"
CAMERA = base.CAMERA


def changed_points(before, after, rotation, camera_xyz_est, track_xy):
    height, width = after.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    valid = np.isfinite(after) & (after >= .15) & (after <= 2.8)
    closer = (before - after >= .05) | (~np.isfinite(before))
    mask = valid & closer
    f = height / (2 * math.tan(math.radians(90.) / 2))
    uvz = np.column_stack(((u[mask] - width/2) * after[mask] / f,
                           (height/2 - v[mask]) * after[mask] / f,
                           -after[mask]))
    xyz = np.asarray(camera_xyz_est) + uvz @ np.asarray(rotation).T
    near_track = np.linalg.norm(xyz[:, :2] - track_xy, axis=1) <= .95
    above_floor = (xyz[:, 2] >= .24) & (xyz[:, 2] <= 1.55)
    keep = near_track & above_floor
    return mask, xyz[keep], (u[mask][keep], v[mask][keep])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ep = json.loads(SOURCE.read_text())["episodes"][0]
    actual = np.asarray(ep["route_episode"]["end_true_xy_m"])
    estimate = np.asarray(ep["route_episode"]["end_estimated_xy_m"])
    track = max(ep["multi_history"][-1]["tracks"], key=lambda row: row["hits"])
    model = joined.scene_with_owner_and_prop(ep["scenario"]["from_xy_m"])
    model.body_pos[model.body("audit_ape").id, :2] = actual
    aim_camera(model, CAMERA, estimate, track["center_xy_m"], 20)
    model.cam_pos[model.camera(CAMERA).id, :2] = actual
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    mujoco.mj_forward(model, data)
    before_rgb, before_depth, _ = owner.capture(renderer, data, CAMERA, True)
    turn._place_owner(model, 0.)
    mujoco.mj_forward(model, data)
    after_rgb, after_depth, _ = owner.capture(renderer, data, CAMERA, True)
    cam = model.camera(CAMERA).id
    camera_est = data.cam_xpos[cam].copy()
    camera_est[:2] = estimate
    mask, points, pixels = changed_points(before_depth, after_depth,
                                           data.cam_xmat[cam].reshape(3, 3),
                                           camera_est, np.asarray(track["center_xy_m"]))
    distance = np.linalg.norm(points[:, :2] - estimate, axis=1)
    order = np.argsort(distance)
    ordered = points[order]
    samples = {str(q): float(np.quantile(distance, q)) for q in (.01,.05,.1,.5)}
    candidates = []
    for retreat in np.arange(0., 1.01, .1):
        candidate = estimate + [0., -retreat]
        observed_surface = float(np.min(np.linalg.norm(points[:, :2] - candidate, axis=1)))
        candidates.append({"retreat_m": round(float(retreat), 3),
                           "nearest_changed_surface_m": observed_surface,
                           "bound_reserve_m": observed_surface - (.6+.17+.05+.03+.03)})
    can_box, can_pixels = evaluator_box(renderer, data, model, CAMERA, "shown_can_geom")
    chosen_mask = np.zeros_like(mask)
    chosen_mask[pixels[1], pixels[0]] = True
    overlay = after_rgb.copy()
    overlay[chosen_mask] = (np.asarray(overlay[chosen_mask], dtype=float)*.35 +
                            np.array([20,255,100])*.65).astype(np.uint8)
    paths = {}
    for name, image in (("before", before_rgb), ("after", after_rgb), ("changed", overlay)):
        path = OUT / f"{name}.png"
        Image.fromarray(image).save(path)
        paths[name] = {"path": str(path), "sha256": sha256(path)}
    report = {"source_sha256": sha256(Path(__file__)),
              "v509_report_sha256": sha256(SOURCE),
              "route_actual_xy_evaluator_m": actual.tolist(),
              "route_estimated_xy_actor_m": estimate.tolist(),
              "range_track_actor": track,
              "camera_pose_source": "range_localized_xy_plus_fixed_proxy_mount_and_aim",
              "capture_depth_change_pixels": int(mask.sum()),
              "track_near_height_filtered_points": int(len(points)),
              "nearest_changed_xyz_est_m": ordered[:10].tolist(),
              "xy_distance_quantiles_m": samples,
              "candidate_screen": candidates,
              "can_box_evaluator": can_box,
              "can_pixels_evaluator": can_pixels,
              "changed_pixels_in_can_box_evaluator": int(
                  chosen_mask[can_box[1]:can_box[3],can_box[0]:can_box[2]].sum())
                  if can_box else 0,
              "can_center_evaluator_m": model.body_pos[model.body("shown_can").id].tolist(),
              "images": paths,
              "scope": "one seeded before/after owner-presentation RGB-D change-cloud screen at actual route stop, using actor estimated XY, range track, camera calibration and depth; geometry labels/evaluator can pose only score the result"}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("points",len(points),"distance quantiles",samples,"can pixels",can_pixels,
          "changed in box",report["changed_pixels_in_can_box_evaluator"],flush=True)
    renderer.close()


if __name__ == "__main__":
    main()
