"""Two-person RGB-D screen for owner identity after a monitored retreat.

This uses the existing furnished kitchen and released person detector. The
owner and the second person are evaluator identities; neither is given to the
actor. Poses and ape motion are scripted collision proxies, not robot skills.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image

import robocasa_owner_person as person
import robocasa_person_interaction as interaction
import robocasa_first_return_person_reacquire_turn_v576 as actor
import run_robocasa_joined_owner_route_v410 as owner
import run_owner_concurrent_reacquire_v571 as detector
from screen_robocasa_owner_person_v489 import evaluator_box
from screen_robocasa_owner_safe_stop_v494 import aim_camera, depth_point


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-two-people-v582"
SOURCE = ROOT / "artifacts/ape-robocasa-first-return-person-reacquire-v566/report.json"
CAMERA = interaction.CAMERA
APE_AFTER_RETREAT_XY = np.array([1.3054671594930622, -4.239948757901364])
OLD_TRACK_XY = np.array([1.1117886222142441, -2.9390073752044352])
OWNER_TORSO_XY = np.array([1.75, -2.8])
DISTRACTOR_TORSO_XY = np.array([.85, -3.25])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scene_xml():
    OUT.mkdir(parents=True, exist_ok=True)
    base = person.scene_with_person((3., -1.05))
    path = OUT / "two-people.xml"
    mujoco.mj_saveLastXML(str(path), base)
    tree = ET.parse(path)
    world = tree.getroot().find("worldbody")
    original = world.find("body[@name='owner_person']")
    distractor = copy.deepcopy(original)
    distractor.set("name", "distractor_person")
    for geom in distractor.findall("geom"):
        geom.set("name", geom.get("name").replace("owner_", "distractor_"))
        if "torso" in geom.get("name") or "upper_arm" in geom.get("name"):
            geom.set("rgba", ".68 .20 .16 1")
    world.append(distractor)
    tree.write(path, encoding="unicode")
    return path


def set_distractor(model, torso_xy):
    torso = np.asarray(torso_xy)
    hand = np.array([torso[0] + .10, torso[1] + .16, .43])
    shoulder = np.array([torso[0], torso[1], 1.05])
    elbow = person.elbow_for(shoulder, hand)
    model.geom_pos[model.geom("distractor_torso_geom").id] = [*torso, .65]
    model.geom_pos[model.geom("distractor_head_geom").id] = [*torso, 1.34]
    person._capsule(model, "distractor_upper_arm_geom", shoulder, elbow)
    person._capsule(model, "distractor_forearm_geom", elbow, hand)
    model.geom_pos[model.geom("distractor_hand_geom").id] = hand


def box_iou(a, b):
    if a is None or b is None:
        return 0.
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    area = max(0., x1 - x0) * max(0., y1 - y0)
    aa = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    bb = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    return area / (aa + bb - area) if aa + bb > area else 0.


def gap(model, data, names1, names2):
    return min(float(mujoco.mj_geomDistance(model, data,
                   model.geom(a).id, model.geom(b).id, 2., None))
               for a in names1 for b in names2)


def main():
    xml = scene_xml()
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    model.body_pos[model.body("audit_ape").id, :2] = APE_AFTER_RETREAT_XY
    model.body_pos[model.body("shown_can").id] = [-10., -10., .43]
    owner_hand = np.array([OWNER_TORSO_XY[0] + .10,
                           OWNER_TORSO_XY[1] + .16, .43])
    person.set_owner_pose(model, OWNER_TORSO_XY, owner_hand, show_can=False)
    set_distractor(model, DISTRACTOR_TORSO_XY)
    # The owner is absent in the control. The same distractor remains in view.
    trials = (("distractor_only", False), ("owner_and_distractor", True))
    results = []
    video_frames = []
    for trial, owner_present in trials:
        for name in person.PERSON_GEOMS:
            model.geom_rgba[model.geom(name).id, 3] = 1. if owner_present else 0.
            model.geom_group[model.geom(name).id] = 0 if owner_present else 5
        views = []
        for pitch in (20, 15):
            aim_camera(model, CAMERA, APE_AFTER_RETREAT_XY, OLD_TRACK_XY, pitch)
            model.cam_pos[model.camera(CAMERA).id, :2] = APE_AFTER_RETREAT_XY
            mujoco.mj_forward(model, data)
            rgb, depth, capture_wall = owner.capture(renderer, data, CAMERA, True)
            found = detector.detect_person(rgb)
            gt = {}
            for who in ("owner", "distractor"):
                gt[who], _ = evaluator_box(renderer, data, model, CAMERA,
                                           f"{who}_torso_geom")
            mapped = []
            image = rgb.copy()
            for rank, det in enumerate(found["detections"]):
                box = det["box_xyxy"]
                point = depth_point(box, depth, data, model, CAMERA)
                d = {**det, "rank": rank,
                     "point_xyz_m": list(point) if point is not None else None,
                     "distance_from_old_track_m": (float(np.linalg.norm(
                         np.asarray(point)[:2] - OLD_TRACK_XY)) if point is not None else None),
                     "torso_iou_evaluator": {key: box_iou(box, value)
                                             for key, value in gt.items()}}
                mapped.append(d)
                cv2.rectangle(image, tuple(map(int, box[:2])),
                              tuple(map(int, box[2:])), (0, 255, 255), 2)
                cv2.putText(image, f"person {rank} {det['score']:.2f}",
                            (int(box[0]), max(20, int(box[1]) - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
            image_path = OUT / f"{trial}-up{pitch}.png"
            cv2.putText(image, f"{trial} / camera up {pitch} deg",
                        (12, 460), cv2.FONT_HERSHEY_SIMPLEX, .7,
                        (255, 255, 255), 2)
            Image.fromarray(image).save(image_path)
            video_frames.extend([image] * 8)
            raw_path = OUT / f"{trial}-up{pitch}.npz"
            np.savez_compressed(raw_path, rgb=rgb, depth=depth)
            views.append({"pitch_degrees": pitch, "detections": mapped,
                          "evaluator_torso_boxes": gt,
                          "capture_wall_s": capture_wall,
                          "detector_wall_s": found["worker_wall_s"],
                          "image": {"path": str(image_path), "sha256": sha(image_path)},
                          "raw_rgbd": {"path": str(raw_path), "sha256": sha(raw_path)}})
            print(trial, pitch, [(round(x["score"], 3),
                                 x["distance_from_old_track_m"],
                                 x["torso_iou_evaluator"]) for x in mapped], flush=True)
        first = [view["detections"][0] if view["detections"] else None
                 for view in views]
        points = [np.asarray(item["point_xyz_m"]) for item in first
                  if item is not None and item["point_xyz_m"] is not None]
        consistent = bool(len(points) == 2 and
                          np.linalg.norm(points[0] - points[1]) <= .15)
        candidate = np.mean(points, axis=0)[:2] if consistent else None
        displacement = (float(np.linalg.norm(candidate - OLD_TRACK_XY))
                        if candidate is not None else None)
        surface_uncertainty = .12200679964494986 + actor.POSE_UNCERTAINTY_M + actor.DEPTH_UNCERTAINTY_M
        recovered = bool(displacement is not None and
                         displacement > surface_uncertainty)
        minimum_score = min(item["score"] for item in first if item is not None) if all(first) else None
        bound = actor.OWNER_MOTION_BOUND_M + .15
        would_bind = bool(consistent and displacement <= bound and
                          minimum_score >= (.35 if recovered else .25))
        results.append({"trial": trial, "views": views,
                        "current_actor_association_replay": {
                            "two_pitch_3d_consistent": consistent,
                            "candidate_surface_xy_m": candidate.tolist() if candidate is not None else None,
                            "distance_from_old_track_m": displacement,
                            "association_bound_m": bound,
                            "restage_required": recovered,
                            "minimum_person_score": minimum_score,
                            "score_gate": .35 if recovered else .25,
                            "would_bind_person_to_owner_track": would_bind,
                            "evaluator_owner_present": owner_present,
                            "false_owner_association": would_bind and not owner_present}})
    for name in person.PERSON_GEOMS:
        model.geom_rgba[model.geom(name).id, 3] = 1.
        model.geom_group[model.geom(name).id] = 0
    mujoco.mj_forward(model, data)
    g = {"owner_ape_m": gap(model, data, person.PERSON_GEOMS, ("audit_ape_geom",)),
         "distractor_ape_m": gap(model, data,
                                tuple(x.replace("owner_", "distractor_") for x in person.PERSON_GEOMS),
                                ("audit_ape_geom",)),
         "owner_distractor_m": gap(model, data, person.PERSON_GEOMS,
                                   tuple(x.replace("owner_", "distractor_") for x in person.PERSON_GEOMS))}
    renderer.close()
    video_path = OUT / "two-person-camera-views.mp4"
    with imageio.get_writer(video_path, fps=8, codec="libx264",
                            macro_block_size=None) as writer:
        for frame in video_frames:
            writer.append_data(frame)
    report = {"scope": "two articulated people at one post-retreat kitchen pose; released detector and calibrated rendered RGB-D; scripted bodies and evaluator identity, no full route replay",
              "source_sha256": sha(Path(__file__)),
              "baseline_episode_sha256": sha(SOURCE),
              "person_detector_weights_sha256": sha(detector.reacquire.DETECTOR / "model.safetensors"),
              "scene_xml_sha256": sha(xml), "ape_xy_m": APE_AFTER_RETREAT_XY.tolist(),
              "old_track_xy_m": OLD_TRACK_XY.tolist(),
              "owner_torso_xy_m": OWNER_TORSO_XY.tolist(),
              "distractor_torso_xy_m": DISTRACTOR_TORSO_XY.tolist(),
              "evaluator_gaps": g, "trials": results,
              "camera_view_video": {"path": str(video_path), "sha256": sha(video_path),
                                    "is_motion_video": False}}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("gaps", g, flush=True)


if __name__ == "__main__":
    main()
