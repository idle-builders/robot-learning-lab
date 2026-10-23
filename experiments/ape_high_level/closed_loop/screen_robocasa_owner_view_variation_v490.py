"""Stress static head/item coverage near the safe-side owner meeting pose.

This geometric camera screen varies the person's stance/hand and the ape's
pose. It measures visible proxy head/can pixels and clipping, not face
recognition, exposure, optics or dynamic motion blur.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import robocasa_owner_person as person
from run_robocasa_layout2_route_v407 import sha256
from screen_robocasa_owner_person_v489 import evaluator_box, gap_to_room


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-owner-view-variation-v490"
BASE_APE = np.array([1.601, -2.136])
BASE_TORSO = np.array([1., -2.4])
BASE_HAND = np.array([1.10, -2.24, .43])
CAMERAS = ("owner_view", "owner_up20", "owner_up25", "owner_up27",
           "owner_up28", "owner_up30", "owner_up35",
           "owner_wide20", "owner_wide25", "owner_wide30")
VARIATIONS = (
    ("nominal", (0., 0.), (0., 0., 0.), (0., 0.)),
    ("torso_north20", (0., .20), (0., 0., 0.), (0., 0.)),
    ("torso_south20", (0., -.20), (0., 0., 0.), (0., 0.)),
    ("torso_east20", (.20, 0.), (0., 0., 0.), (0., 0.)),
    ("hand_up10", (0., 0.), (0., 0., .10), (0., 0.)),
    ("hand_down10", (0., 0.), (0., 0., -.10), (0., 0.)),
    ("hand_north10", (0., 0.), (0., .10, 0.), (0., 0.)),
    ("hand_south10", (0., 0.), (0., -.10, 0.), (0., 0.)),
    ("ape_north10", (0., 0.), (0., 0., 0.), (0., .10)),
    ("ape_south10", (0., 0.), (0., 0., 0.), (0., -.10)),
    ("ape_forward10", (0., 0.), (0., 0., 0.), (-.10, 0.)),
    ("ape_back10", (0., 0.), (0., 0., 0.), (.10, 0.)),
)


def fully_inside(box):
    return bool(box is not None and box[0] > 0 and box[1] > 0
                and box[2] < 640 and box[3] < 480)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = person.scene_with_person()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    rows = []
    for name, torso_delta, hand_delta, ape_delta in VARIATIONS:
        torso = BASE_TORSO + np.asarray(torso_delta)
        hand = BASE_HAND + np.asarray(hand_delta)
        ape = BASE_APE + np.asarray(ape_delta)
        pose = person.set_owner_pose(model, torso, hand)
        model.body_pos[model.body("audit_ape").id, :2] = ape
        for camera in CAMERAS:
            model.cam_pos[model.camera(camera).id, :2] = ape
        mujoco.mj_forward(model, data)
        room_gap, ape_gap = gap_to_room(model, data)
        views = []
        for camera in CAMERAS:
            can_box, can_pixels = evaluator_box(renderer, data, model, camera,
                                                 "shown_can_geom")
            head_box, head_pixels = evaluator_box(renderer, data, model, camera,
                                                   "owner_head_geom")
            view = {"camera": camera,
                    "can_pixels": can_pixels, "can_box": can_box,
                    "can_full_frame": fully_inside(can_box),
                    "head_pixels": head_pixels, "head_box": head_box,
                    "head_full_frame": fully_inside(head_box),
                    "both_full_frame": fully_inside(can_box) and fully_inside(head_box)}
            views.append(view)
            if camera in ("owner_view", "owner_up27", "owner_up30",
                          "owner_wide25") and name in (
                    "nominal", "hand_up10", "hand_down10", "ape_north10", "ape_south10"):
                renderer.update_scene(data, camera=camera)
                image = Image.fromarray(renderer.render().copy())
                path = OUT / f"{name}-{camera}.png"
                image.save(path)
                view["image"] = {"path": str(path), "sha256": sha256(path)}
        rows.append({"scenario": name, "torso_xy_m": torso.tolist(),
                     "hand_xyz_m": hand.tolist(), "ape_xy_m": ape.tolist(),
                     "person_pose": pose,
                     "person_room": room_gap, "person_ape": ape_gap,
                     "local_feasibility_pass": room_gap["gap_m_capped_at_30cm"] >= .05
                     and ape_gap["gap_m_capped_at_50cm"] >= .05,
                     "views": views})
        print(name, "room", round(room_gap["gap_m_capped_at_30cm"], 3),
              "ape", round(ape_gap["gap_m_capped_at_50cm"], 3),
              "both", [v["camera"] for v in views if v["both_full_frame"]], flush=True)
    renderer.close()
    summary = {camera: {"both_full_count": sum(
        next(v for v in row["views"] if v["camera"] == camera)["both_full_frame"]
        for row in rows),
                        "can_full_count": sum(
        next(v for v in row["views"] if v["camera"] == camera)["can_full_frame"]
        for row in rows),
                        "head_full_count": sum(
        next(v for v in row["views"] if v["camera"] == camera)["head_full_frame"]
        for row in rows)} for camera in CAMERAS}
    report = {"source_sha256": sha256(Path(__file__)),
              "person_source_sha256": sha256(Path(person.__file__)),
              "scope": "12 local pose variations around a scripted articulated owner's stationary show pose, seven ideal 70-degree pinhole camera pitches, geometric visible pixels/clipping only",
              "summary": summary, "cases": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
