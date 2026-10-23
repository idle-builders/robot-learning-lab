"""Challenge RGB-D standoff inference with varied articulated show poses.

The actor sees only calibrated before/after depth, the saved range track and
cells observed free in the joined route. Evaluator arm/item pose, segmentation
and 20 ms clearance stay outside the standoff choice.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import owner_depth_change_component_v516 as actor
import robocasa_owner_person as person
import robocasa_person_interaction as interaction
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_route_v407 as route
import run_robocasa_person_joined_v497 as joined
from audit_robocasa_layout2_routes_v409 import clearance, sha256
from screen_robocasa_owner_person_v489 import evaluator_box, gap_to_room
from screen_robocasa_owner_safe_stop_v494 import aim_camera


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-robocasa-person-depth-standoff-v517/report.json"
OUT = ROOT / "artifacts/ape-robocasa-owner-hand-pose-v519"
HAND_POSES = (
    ("reference", [1.30, -2.80, .55]),
    ("higher", [1.30, -2.80, .85]),
    ("lower", [1.30, -2.80, .45]),
    ("left", [1.10, -2.80, .55]),
    ("right", [1.50, -2.80, .55]),
    ("nearer", [1.30, -2.98, .55]),
    ("farther", [1.30, -2.62, .55]),
)


def presented(model, hand_xyz, south=0.):
    torso = interaction.TORSO_XY + [0., -south]
    hand = np.asarray(hand_xyz) + [0., -south, 0.]
    pose = person.set_owner_pose(model, torso, hand, show_can=True)
    direction = hand[:2] - torso
    direction /= np.linalg.norm(direction)
    model.body_pos[model.body("shown_can").id, :2] = hand[:2] + .14*direction
    pose["can_xyz_m"] = model.body_pos[model.body("shown_can").id].tolist()
    return pose


def screen_one(model, data, renderer, before_depth, baseline_pose,
               track, grid, hand_name, hand_xyz, vision, exemplar):
    actual, estimate = baseline_pose
    person.set_owner_pose(model, interaction.TORSO_XY,
                          interaction.REST_HAND, show_can=False)
    model.body_pos[model.body("audit_ape").id, :2] = actual
    aim_camera(model, interaction.CAMERA, estimate, track["center_xy_m"], 20)
    model.cam_pos[model.camera(interaction.CAMERA).id, :2] = actual
    try:
        pose = presented(model, hand_xyz)
    except ValueError as exc:
        return {"hand_pose": hand_name, "hand_xyz_m": hand_xyz,
                "valid_scripted_arm": False, "reason": str(exc)}
    mujoco.mj_forward(model, data)
    shown_rgb, after_depth, _ = owner.capture(renderer, data,
                                               interaction.CAMERA, True)
    cam = model.camera(interaction.CAMERA).id
    camera_est = data.cam_xpos[cam].copy()
    camera_est[:2] = estimate
    healthy, coverage = actor.packet_health(before_depth, after_depth)
    if healthy:
        mask, cloud, pixels = actor.changed_points(
            before_depth, after_depth, data.cam_xmat[cam].reshape(3, 3),
            camera_est, np.asarray(track["center_xy_m"]))
    else:
        mask, cloud, pixels = np.zeros_like(after_depth,dtype=bool), np.empty((0,3)), ([],[])
    chosen = None
    if len(cloud) >= 50:
        chosen, _ = actor.choose_standoff(grid, cloud, estimate)
    shown_can_box, shown_can_pixels = evaluator_box(
        renderer, data, model, interaction.CAMERA, "shown_can_geom")
    changed_in_can = (int(mask[shown_can_box[1]:shown_can_box[3],
                               shown_can_box[0]:shown_can_box[2]].sum())
                      if shown_can_box else 0)
    frame_path = OUT / f"{hand_name}-shown.png"
    Image.fromarray(shown_rgb).save(frame_path)
    minimum = None
    recognition = None
    if chosen is not None:
        retreat = chosen["distance_m"]
        moved_actual = actual + [0., -retreat]
        moved_est = estimate + [0., -retreat]
        model.body_pos[model.body("audit_ape").id, :2] = moved_actual
        aim_camera(model, interaction.CAMERA, moved_est,
                   track["center_xy_m"], 20)
        model.cam_pos[model.camera(interaction.CAMERA).id, :2] = moved_actual
        ape_body = model.body("audit_ape").id
        ape_geom = model.geom("audit_ape_geom").id
        minimum = {"gap_m_capped_at_10cm": .1, "south_m": None, "geom": None}
        for south in np.arange(0., .601, .02):
            presented(model, hand_xyz, float(south))
            gap, geom = clearance(model, data, ape_body, ape_geom, moved_actual)
            if gap < minimum["gap_m_capped_at_10cm"]:
                minimum = {"gap_m_capped_at_10cm": float(gap),
                           "south_m": round(float(south), 3), "geom": geom}
        presented(model, hand_xyz, .6)
        mujoco.mj_forward(model, data)
        rgb, _, _ = owner.capture(renderer, data, interaction.CAMERA, False)
        image = Image.fromarray(rgb)
        proposals = vision.detect(image)
        scores = (vision.feature([image.crop(item["box"]) for item in proposals]) @ exemplar
                  if proposals else np.empty(0))
        ranked = sorted(zip(proposals, scores), key=lambda pair: float(pair[1]),
                        reverse=True)
        selected = ranked[0] if ranked and float(ranked[0][1]) >= owner.THRESHOLD else None
        can_box, can_pixels = evaluator_box(renderer, data, model,
                                            interaction.CAMERA, "shown_can_geom")
        recognition = {"can_pixels_evaluator": can_pixels,
                       "top_similarity": float(ranked[0][1]) if ranked else None,
                       "selected_box": selected[0]["box"] if selected else None,
                       "selected_iou_evaluator": owner.iou(
                           selected[0]["box"] if selected else None, can_box)}
        post_path = OUT / f"{hand_name}-after-60cm.png"
        Image.fromarray(owner.owner_frame(rgb, f"{hand_name}: after 60 cm",
                                          recognition["selected_box"], can_box)).save(post_path)
        recognition["image"] = {"path": str(post_path), "sha256": sha256(post_path)}
    return {"hand_pose": hand_name, "hand_xyz_m": hand_xyz,
            "valid_scripted_arm": True,
            "shown_pose_evaluator": pose,
            "shown_can_pixels_evaluator": shown_can_pixels,
            "shown_changed_can_pixels_evaluator": changed_in_can,
            "depth_packet_healthy_actor": healthy,
            "central_valid_fractions": coverage,
            "changed_pixels_actor": int(mask.sum()),
            "changed_points_actor": int(len(cloud)),
            "changed_cloud_sha256": hashlib.sha256(cloud.tobytes()).hexdigest(),
            "chosen_actor": chosen,
            "minimum_gap_through_60cm_owner_approach_evaluator": minimum,
            "recognition_after_60cm": recognition,
            "shown_image": {"path": str(frame_path), "sha256": sha256(frame_path)}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = json.loads(SOURCE.read_text())
    ep = next(ep for ep in report["episodes"] if ep["response"] == "approach_60cm_ideal")
    track = max(ep["multi_history"][-1]["tracks"], key=lambda x: x["hits"])
    actual = np.asarray(ep["route_episode"]["end_true_xy_m"])
    estimate = np.asarray(ep["route_episode"]["end_estimated_xy_m"])
    grid = route.SensorGrid()
    selected = next(row["chosen"] for row in ep["trace"]
                    if row["stage"] == "choose_owner_standoff")
    for cell in selected["cells"]:
        if cell["free"] and not cell["occupied"]:
            grid.free[grid.cell(np.asarray(cell["xy_m"]))] = True
    model = joined.scene_with_owner_and_prop(ep["scenario"]["from_xy_m"])
    model.body_pos[model.body("audit_ape").id, :2] = actual
    aim_camera(model, interaction.CAMERA, estimate, track["center_xy_m"], 20)
    model.cam_pos[model.camera(interaction.CAMERA).id, :2] = actual
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    mujoco.mj_forward(model, data)
    _, before_depth, _ = owner.capture(renderer, data, interaction.CAMERA, True)
    vision = owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(owner.TEACH_CROP).convert("RGB")])[0]
    rows = []
    for name, hand in HAND_POSES:
        result = screen_one(model, data, renderer, before_depth,
                            (actual, estimate), track, grid, name, hand,
                            vision, exemplar)
        rows.append(result)
        gap = result.get("minimum_gap_through_60cm_owner_approach_evaluator")
        print(name, "standoff", (result.get("chosen_actor") or {}).get("distance_m"),
              "points", result.get("changed_points_actor"),
              "gap", gap, flush=True)
    renderer.close()
    output = {"source_sha256": sha256(Path(__file__)),
              "joined_report_sha256": sha256(SOURCE),
              "depth_change_source_sha256": sha256(Path(actor.__file__)),
              "person_source_sha256": sha256(Path(person.__file__)),
              "scope": "seven predeclared articulated hand show positions in one saved owner/route kitchen; fixed actor range track, camera pose and previously observed free corridor; actor RGB-D change-cloud standoff choice, evaluator 20ms owner approach clearance and can recognition after 60cm; static pose transfer only, no gait/late prop/video interaction loop",
              "rows": rows}
    (OUT / "report.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
