"""Replay released person detections and candidate-local depth in V587 views."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image

import owner_depth_change_component_v516 as depth_change
import robocasa_owner_person as person
import robocasa_person_interaction as interaction
import run_owner_concurrent_reacquire_v571 as detector
import run_robocasa_two_moving_people_v586 as moving
import screen_robocasa_two_people_v582 as two
from screen_robocasa_owner_person_v489 import evaluator_box
from screen_robocasa_owner_safe_stop_v494 import aim_camera, depth_point


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-robocasa-post-retreat-view-sweep-v587/report.json"
OUT = ROOT / "artifacts/ape-robocasa-post-retreat-person-roi-v588"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def iou(a, b):
    if b is None:
        return 0.
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0., right-left) * max(0., bottom-top)
    aa = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
    bb = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
    return intersection / (aa+bb-intersection) if aa+bb>intersection else 0.


def roi_support(depth, box):
    x0, y0, x1, y1 = box
    cx, cy = (x0+x1)/2, (y0+y1)/2
    rw, rh = (x1-x0)*.30, (y1-y0)*.30
    lo_x, hi_x = max(0, int(cx-rw/2)), min(depth.shape[1], int(cx+rw/2)+1)
    lo_y, hi_y = max(0, int(cy-rh/2)), min(depth.shape[0], int(cy+rh/2)+1)
    patch = depth[lo_y:hi_y, lo_x:hi_x]
    valid = np.isfinite(patch) & (patch>=.15) & (patch<=2.8)
    return {"valid_fraction": float(valid.mean()),
            "valid_pixels": int(valid.sum()), "patch_pixels": int(valid.size),
            "xyxy": [lo_x, lo_y, hi_x, hi_y]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE.read_text())
    ep = source["result"]["episode"]
    estimate = np.asarray(ep["final_estimated_xy_m"])
    actual = np.asarray(ep["final_actual_xy_m"])
    model = mujoco.MjModel.from_xml_path(str(moving.SCENE_XML))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    views = []
    for record in source["views"]:
        with np.load(record["raw_rgbd"]["path"]) as raw:
            rgb, depth = raw["rgb"], raw["depth"]
        camera = record["camera"]
        pitch = record["pitch_degrees"]
        person.set_owner_pose(model, interaction.TORSO_XY + [0., -.4],
                              interaction.REST_HAND + [0., -.4, 0.], show_can=False)
        two.set_distractor(model, moving.distractor_xy_at(record["time_s"]))
        aim_camera(model, camera, estimate, record["aimed_at_anonymous_track_xy_est_m"], pitch)
        model.cam_pos[model.camera(camera).id, :2] = actual
        mujoco.mj_forward(model, data)
        evaluator = {who: evaluator_box(renderer, data, model, camera,
                     f"{who}_torso_geom")[0] for who in ("owner", "distractor")}
        result = detector.detect_person(rgb)
        detections = []
        annotated = rgb.copy()
        for index, det in enumerate(result["detections"]):
            box = det["box_xyxy"]
            point = depth_point(box, depth, data, model, camera)
            support = roi_support(depth, box)
            row = {**det, "rank": index, "person_roi_depth": support,
                   "point_xyz_est_m": (
                       (np.asarray(point) + np.r_[estimate-actual, 0.]).tolist()
                       if point is not None else None),
                   "torso_iou_evaluator": {who: iou(box, gt)
                                           for who, gt in evaluator.items()}}
            detections.append(row)
            cv2.rectangle(annotated, tuple(map(int, box[:2])),
                          tuple(map(int, box[2:])), (0, 255, 255), 2)
            cv2.putText(annotated, f"person {index} {det['score']:.2f}",
                        (int(box[0]), max(20, int(box[1])-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
        path = OUT / f"view-{record['index']:02d}.png"
        Image.fromarray(annotated).save(path)
        healthy, fractions = depth_change.packet_health(depth, depth)
        views.append({"index": record["index"], "camera": camera,
                      "pitch_degrees": pitch, "fovy_degrees": record["fovy_degrees"],
                      "global_central_depth_fraction": fractions[0],
                      "global_depth_healthy": healthy,
                      "detector_wall_s": result["worker_wall_s"],
                      "evaluator_torso_boxes": evaluator,
                      "detections": detections,
                      "annotated_image": {"path": str(path), "sha256": sha(path)},
                      "raw_rgbd": record["raw_rgbd"]})
        print(record["index"],camera,pitch,"global",round(fractions[0],3),
              "scores",[(round(x["score"],3),
                          round(x["person_roi_depth"]["valid_fraction"],3),
                          x["torso_iou_evaluator"]) for x in detections],flush=True)
    renderer.close()
    report = {"scope": "offline released person detector and per-box rendered depth quality on the six V587 post-retreat images; evaluator identity only for audit, not actor; no resumed show/finding",
              "source_sha256": sha(Path(__file__)),
              "source_report_sha256": sha(SOURCE),
              "model_weights_sha256": sha(detector.reacquire.DETECTOR / "model.safetensors"),
              "views": views}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
