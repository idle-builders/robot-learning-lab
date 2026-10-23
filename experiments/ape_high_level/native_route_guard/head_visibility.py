"""Evaluator-only head-camera coverage at V11 fast-crossing guard decisions.

Rendered RGB and depth are real simulated sensor images. MuJoCo segmentation
IDs are used only here to score visibility/crop, never by the route guard.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

import changed_speed_trial as holdout
import kitchen_native as kitchen
import kitchen_trial as trial


OUT = kitchen.OUT / "v12-head-visibility"
TIMES_S = (.16, .22, .30, .50, .90, 1.50)
CAMERAS = ("ape_upper_rgb", "ape_lower_rgb")


def score(segmentation: np.ndarray, depth: np.ndarray, geom_id: int) -> dict:
    person = segmentation[:, :, 0] == geom_id
    ys, xs = np.where(person)
    if len(xs) == 0:
        return {"visible_pixels": 0, "bbox_xyxy": None,
                "touches_left_border": False, "touches_any_border": False,
                "valid_person_depth_fraction": None, "person_depth_median_m": None}
    values = depth[person]
    valid = np.isfinite(values) & (values > 0)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return {"visible_pixels": len(xs), "bbox_xyxy": bbox,
            "touches_left_border": bbox[0] == 0,
            "touches_any_border": (bbox[0] == 0 or bbox[1] == 0 or
                                   bbox[2] == segmentation.shape[1]-1 or
                                   bbox[3] == segmentation.shape[0]-1),
            "valid_person_depth_fraction": round(float(valid.mean()), 5),
            "person_depth_median_m": (round(float(np.median(values[valid])), 5)
                                      if valid.any() else None)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(kitchen.OUT / "scene.xml"))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=240, width=320)
    person_id = model.geom("person_capsule").id
    image_rows = []
    rows = []
    with np.load(kitchen.TRACE, allow_pickle=False) as saved:
        for world in range(4):
            for t in TIMES_S:
                model.body_pos[model.body("approaching_person").id, :2] = (
                    holdout.hazard_xy("fast_clear_held", t))
                model.body_pos[model.body("low_pet").id, :2] = (10., -1.)
                pose = kitchen.put_native_pose(model, data, saved, world, t, 3.0)
                for camera in CAMERAS:
                    renderer.enable_segmentation_rendering()
                    renderer.update_scene(data, camera=camera)
                    segmented = renderer.render().copy()
                    renderer.disable_segmentation_rendering()
                    renderer.enable_depth_rendering()
                    renderer.update_scene(data, camera=camera)
                    depth = renderer.render().copy()
                    renderer.disable_depth_rendering()
                    row = {"world": world, "time_s": t, "camera": camera,
                           "root_xy_m": pose[:2].round(5).tolist(),
                           "person_xy_m": list(holdout.hazard_xy("fast_clear_held", t)),
                           **score(segmented, depth, person_id)}
                    if world in (1, 3) and t == .22:
                        renderer.update_scene(data, camera=camera)
                        rgb = renderer.render().copy()
                        path = OUT / f"{camera}-fast-world{world}-t022.png"
                        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                        row["rgb_path"] = str(path)
                        row["rgb_sha256"] = trial.sha(path)
                        image_rows.append(str(path))
                    rows.append(row)
    renderer.close()
    report = {"schema": "robot-learning-lab.ape-robocasa-head-coverage.v1",
              "task": "T-9v3p3n",
              "purpose": "Score only head-camera visibility and cropping at V11's fast-crossing packet times; segmentation IDs are evaluator-only and never fed to guard.",
              "image_size_px": [320, 240],
              "source_sha256": trial.sha(Path(__file__)),
              "scene_sha256": trial.sha(kitchen.OUT / "scene.xml"),
              "native_trace_sha256": trial.sha(kitchen.TRACE),
              "holdout_report_sha256": trial.sha(holdout.OUT / "report.json"),
              "images": image_rows,
              "rows": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(len(rows), "camera/time views")
    for row in rows:
        if row["world"] in (1, 3) and row["time_s"] in (.22, .30):
            print(row["world"], row["time_s"], row["camera"],
                  row["visible_pixels"], row["touches_left_border"],
                  row["valid_person_depth_fraction"])


if __name__ == "__main__":
    main()
