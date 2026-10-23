"""Check whether a visible owner can stand and show a can at the safe side."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import robocasa_owner_person as person
import run_robocasa_joined_owner_route_v410 as owner
from run_robocasa_layout2_route_v407 import sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-owner-person-v489"
APE_XY = np.array([1.601, -2.136])
HAND_XYZ = np.array([1.10, -2.24, .43])
TORSO_CANDIDATES = ((1.0, -2.4), (1.0, -2.2), (1.0, -2.6), (1.2, -2.4))
CAMERAS = ("owner_view", "owner_up20", "owner_up25", "owner_up27",
           "owner_up28", "owner_up30", "owner_up35", "owner_wide25",
           "owner_wide30")


def gap_to_room(model, data):
    people = {model.geom(name).id for name in person.PERSON_GEOMS}
    ape = model.geom("audit_ape_geom").id
    can_body = model.body("shown_can").id
    can_geoms = {i for i in range(model.ngeom) if model.geom_bodyid[i] == can_body}
    exclusions = people | {ape} | can_geoms
    room = {"gap_m_capped_at_30cm": .3, "pair": None}
    ape_gap = {"gap_m_capped_at_50cm": .5, "person_geom": None}
    for pg in people:
        center = data.geom_xpos[pg]
        for other in range(model.ngeom):
            if other in exclusions or model.geom(other).name.startswith("floor_"):
                continue
            if np.linalg.norm(data.geom_xpos[other] - center) > (
                    model.geom_rbound[pg] + model.geom_rbound[other] + .3):
                continue
            distance = min(.3, float(mujoco.mj_geomDistance(
                model, data, pg, other, .3, None)))
            if distance < room["gap_m_capped_at_30cm"]:
                room = {"gap_m_capped_at_30cm": distance,
                        "pair": [model.geom(pg).name, model.geom(other).name]}
        distance = min(.5, float(mujoco.mj_geomDistance(
            model, data, pg, ape, .5, None)))
        if distance < ape_gap["gap_m_capped_at_50cm"]:
            ape_gap = {"gap_m_capped_at_50cm": distance,
                       "person_geom": model.geom(pg).name}
    return room, ape_gap


def evaluator_box(renderer, data, model, camera, geom_name):
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=camera)
    mask = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    ys, xs = np.nonzero(mask[:, :, 0] == model.geom(geom_name).id)
    return ([int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
            if len(xs) else None), int(len(xs))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = person.scene_with_person()
    data = mujoco.MjData(model)
    model.body_pos[model.body("audit_ape").id, :2] = APE_XY
    for name in CAMERAS:
        model.cam_pos[model.camera(name).id, :2] = APE_XY
    candidates = []
    for torso in TORSO_CANDIDATES:
        pose = person.set_owner_pose(model, torso, HAND_XYZ)
        mujoco.mj_forward(model, data)
        room, ape = gap_to_room(model, data)
        candidates.append({"torso_xy_m": list(torso), "pose": pose,
                           "person_room": room, "person_ape": ape,
                           "local_feasibility_pass": room["gap_m_capped_at_30cm"] >= .05
                           and ape["gap_m_capped_at_50cm"] >= .05})
        print(torso, "room", round(room["gap_m_capped_at_30cm"], 3),
              room["pair"], "ape", round(ape["gap_m_capped_at_50cm"], 3),
              flush=True)
    feasible = [row for row in candidates if row["local_feasibility_pass"]]
    if not feasible:
        raise RuntimeError("No locally clear owner posture; retain diagnostic table")
    chosen = feasible[0]
    person.set_owner_pose(model, chosen["torso_xy_m"], HAND_XYZ)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=480, width=640)
    vision = owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(owner.TEACH_CROP).convert("RGB")])[0]
    views = []
    for camera in CAMERAS:
        rgb, depth, age = owner.capture(renderer, data, camera, True)
        image = Image.fromarray(rgb)
        proposals = vision.detect(image)
        scores = (vision.feature([image.crop(item["box"]) for item in proposals]) @ exemplar
                  if proposals else np.empty(0))
        ranked = sorted(zip(proposals, scores), key=lambda item: float(item[1]), reverse=True)
        selected = ranked[0] if ranked and float(ranked[0][1]) >= owner.THRESHOLD else None
        gt_box, pixels = evaluator_box(renderer, data, model, camera,
                                       "shown_can_geom")
        head_box, head_pixels = evaluator_box(renderer, data, model, camera,
                                               "owner_head_geom")
        rendered = owner.owner_frame(rgb, f"articulated owner: {camera}",
                                     selected[0]["box"] if selected else None, gt_box)
        path = OUT / f"{camera}.png"
        Image.fromarray(rendered).save(path)
        views.append({"camera": camera, "raw_capture_age_s": age,
                      "shown_pixels_evaluator": pixels,
                      "head_pixels_evaluator": head_pixels,
                      "head_box_evaluator": head_box,
                      "evaluator_box": gt_box,
                      "proposal_count": len(proposals),
                      "top_similarity": float(ranked[0][1]) if ranked else None,
                      "selected_box": selected[0]["box"] if selected else None,
                      "selected_iou_evaluator": owner.iou(
                          selected[0]["box"] if selected else None, gt_box),
                      "depth_available": depth is not None,
                      "image": {"path": str(path), "sha256": sha256(path)}})
        print(camera, "can", pixels, "head", head_pixels,
              "top", views[-1]["top_similarity"],
              "iou", views[-1]["selected_iou_evaluator"], flush=True)
    renderer.close()
    report = {"source_sha256": sha256(Path(__file__)),
              "person_source_sha256": sha256(Path(person.__file__)),
              "frozen_owner_source_sha256": sha256(Path(owner.__file__)),
              "taught_crop_sha256": sha256(owner.TEACH_CROP),
              "scope": "local static feasibility and camera visibility of scripted articulated person presenting a taught can at the previously staged layout4 pose; no arrival path or owner motion",
              "ape_xy_m": APE_XY.tolist(), "hand_xyz_m": HAND_XYZ.tolist(),
              "candidates": candidates, "chosen_torso_xy_m": chosen["torso_xy_m"],
              "views": views}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
