"""Test head/item visibility when the owner reaches toward the stopped ape.

The camera yaw aims from the actual stopped ape pose toward the owner's
torso. Evaluator segmentation labels accuracy after image-only recognition;
it is never passed to the detector or tracker.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import robocasa_owner_person as person
import run_robocasa_joined_owner_route_v410 as owner
import screen_robocasa_owner_person_v489 as screen
from run_robocasa_layout2_route_v407 import sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-owner-reach-v495"
ROUTE = ROOT / "artifacts/ape-robocasa-person-multitrack-v492/report.json"
CAMERAS = ("owner_wide20", "owner_wide25", "owner_wide30")
TORSO = [1., -2.4]
HAND = [1.30, -2.80, .55]


def aim_camera(model, camera, xy, target_xy, pitch_degrees):
    direction = np.asarray(target_xy, dtype=float) - np.asarray(xy, dtype=float)
    direction /= np.linalg.norm(direction)
    pitch = math.radians(pitch_degrees)
    right = np.array([direction[1], -direction[0], 0.])
    up = np.array([-direction[0] * math.sin(pitch),
                   -direction[1] * math.sin(pitch), math.cos(pitch)])
    back = np.cross(right, up)
    rotation = np.column_stack((right, up, back))
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rotation.ravel())
    cam = model.camera(camera).id
    model.cam_pos[cam, :2] = xy
    model.cam_quat[cam] = quat


def depth_point(box, depth, data, model, camera):
    x0, y0, x1, y1 = box
    u, v = (x0 + x1) / 2, (y0 + y1) / 2
    xlo, xhi = max(0, int(u - (x1-x0)*.18)), min(640, int(u + (x1-x0)*.18)+1)
    ylo, yhi = max(0, int(v - (y1-y0)*.18)), min(480, int(v + (y1-y0)*.18)+1)
    patch = depth[ylo:yhi, xlo:xhi]
    valid = patch[np.isfinite(patch) & (patch >= .15) & (patch <= 2.8)]
    if len(valid) < 9:
        return None
    z = float(np.quantile(valid, .25))
    focal = 480 / (2 * math.tan(math.radians(float(model.cam_fovy[model.camera(camera).id])/2)))
    camera_point = np.array([(u-320)*z/focal, (240-v)*z/focal, -z])
    cam = model.camera(camera).id
    world = data.cam_xpos[cam] + data.cam_xmat[cam].reshape(3, 3) @ camera_point
    return world.tolist()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    route = json.loads(ROUTE.read_text())
    ape_xy = np.asarray(route["cases"][0]["episode"]["end_true_xy_m"])
    model = person.scene_with_person(prop_xy=[-10., -10.])
    model.body_pos[model.body("audit_ape").id, :2] = ape_xy
    person.set_owner_pose(model, TORSO, HAND, show_can=True)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    person_room, person_ape = screen.gap_to_room(model, data)
    renderer = mujoco.Renderer(model, height=480, width=640)
    vision = owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(owner.TEACH_CROP).convert("RGB")])[0]
    views = []
    for camera, pitch in zip(CAMERAS, (20, 25, 30)):
        aim_camera(model, camera, ape_xy, TORSO, pitch)
        mujoco.mj_forward(model, data)
        rgb, depth, age = owner.capture(renderer, data, camera, True)
        image = Image.fromarray(rgb)
        proposals = vision.detect(image)
        scores = (vision.feature([image.crop(item["box"]) for item in proposals]) @ exemplar
                  if proposals else np.empty(0))
        ranked = sorted(zip(proposals, scores), key=lambda pair: float(pair[1]), reverse=True)
        chosen = ranked[0] if ranked and float(ranked[0][1]) >= owner.THRESHOLD else None
        can_box, can_pixels = screen.evaluator_box(renderer, data, model, camera,
                                                    "shown_can_geom")
        head_box, head_pixels = screen.evaluator_box(renderer, data, model, camera,
                                                      "owner_head_geom")
        frame = owner.owner_frame(rgb, f"safe stop; aim owner; {camera}",
                                  chosen[0]["box"] if chosen else None, can_box)
        path = OUT / f"{camera}.png"
        Image.fromarray(frame).save(path)
        point = (depth_point(chosen[0]["box"], depth, data, model, camera)
                 if chosen is not None else None)
        point_error = (float(np.linalg.norm(np.asarray(point) -
                                            model.body_pos[model.body("shown_can").id]))
                       if point is not None else None)
        view = {"camera": camera, "pitch_degrees": pitch,
                "raw_capture_age_s": age,
                "can_pixels_evaluator": can_pixels, "head_pixels_evaluator": head_pixels,
                "can_box_evaluator": can_box, "head_box_evaluator": head_box,
                "proposal_count": len(proposals),
                "top_similarity": float(ranked[0][1]) if ranked else None,
                "selected_box": chosen[0]["box"] if chosen else None,
                "selected_iou_evaluator": owner.iou(chosen[0]["box"] if chosen else None,
                                                     can_box),
                "depth_point_xyz_m": point, "depth_point_error_evaluator_m": point_error,
                "image": {"path": str(path), "sha256": sha256(path)}}
        views.append(view)
        print(camera, "can", can_pixels, "head", head_pixels,
              "similarity", view["top_similarity"], "iou", view["selected_iou_evaluator"],
              "depth_err", point_error, flush=True)
    renderer.close()
    report = {"source_sha256": sha256(Path(__file__)),
              "person_source_sha256": sha256(Path(person.__file__)),
              "route_report_sha256": sha256(ROUTE),
              "taught_crop_sha256": sha256(owner.TEACH_CROP),
              "scope": "static RGB-D owner/item screen at the V492 route stop; yaw aimed by known owner direction, not yet an image-estimated person bearing or joined live scene",
              "ape_xy_m": ape_xy.tolist(), "owner_torso_xy_m": TORSO,
              "owner_hand_xyz_m": HAND, "person_room": person_room,
              "person_ape": person_ape, "views": views}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
