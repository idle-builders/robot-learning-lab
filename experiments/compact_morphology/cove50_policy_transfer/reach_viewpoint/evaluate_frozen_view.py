"""Evaluate one frozen virtual head view on later development and held-out states."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from screen_head_mount import (BODY, OUT, POINTER, ROOT, estimate_on_known_plane,
                               orange_bbox, project, sha, tilted_model)


def main() -> None:
    plan = json.loads((OUT / "plan.json").read_text())
    frozen = json.loads((OUT / "heldout-freeze.json").read_text())
    assert sha(BODY) == plan["selected_body_sha256"]
    assert sha(POINTER) == plan["selected_pointer_sha256"]
    selected = next(c for c in json.loads((OUT / "head-mount-screen.json").read_text())["cases"]
                    if c["case"] == frozen["chosen_case"])
    assert selected["entry_gate"]
    assert selected["head_local_camera_xyz_m"] == frozen["head_local_camera_xyz_m"]
    model = tilted_model(frozen["extra_downward_pitch_deg"])
    camera = model.camera("lower_rgb").id
    model.cam_pos[camera] = frozen["head_local_camera_xyz_m"]
    data = mujoco.MjData(model)
    head = model.joint("head_yaw").qposadr[0]
    option = mujoco.MjvOption()
    option.geomgroup[:] = 1
    cases = [("development", Path(p), [6, 7.5, 8.5, 10])
             for p in plan["development_traces"]]
    cases += [("heldout", Path(
        f"artifacts/ape-cove50-squat-learned-reach-20261010/v8/conditioned-route/reserved/seed-{seed}.npz"),
               [7.5]) for seed in frozen["heldout_cases_to_open"]]
    rows = []
    images = []
    with mujoco.Renderer(model, width=320, height=240) as renderer:
        for split, trace_name, times in cases:
            trace_path = ROOT / trace_name
            native = json.loads(trace_path.with_suffix(".json").read_text())
            assert native["body_sha256"] == plan["selected_body_sha256"]
            with np.load(trace_path) as trace:
                qpos = trace["qpos"].astype(float)
            assert qpos.shape[1] == 35
            for time_s in times:
                frame = round(time_s / .02)
                data.qpos[:] = qpos[frame]
                data.qpos[head] = frozen["fixed_head_yaw_rad"]
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)
                projection = project(model, data, "lower_rgb", data.qpos[28:31],
                                     320, 240)
                renderer.update_scene(data, camera="lower_rgb", scene_option=option)
                rgb = renderer.render().copy()
                bbox = orange_bbox(rgb)
                estimate = estimate_on_known_plane(bbox, model, data, "lower_rgb",
                                                   320, 240, .01)
                error = None if estimate is None else round(float(np.linalg.norm(
                    np.asarray(estimate)[:2] - data.qpos[28:30]) * 1000), 3)
                image_path = OUT / f"{split}-{trace_name.stem}-{time_s:g}s-frozen-head-view.png"
                if time_s in (6, 7.5, 10):
                    imageio.imwrite(image_path, rgb)
                    image = str(image_path.relative_to(ROOT))
                    images.append(image)
                else:
                    image = None
                passed = bbox is not None and projection["pixel_edge_margin"] is not None \
                    and projection["pixel_edge_margin"] >= 10 and error is not None \
                    and error <= 10 and bbox["pixels"] >= 10
                rows.append({
                    "case": f"{split}-{trace_name.stem}-{time_s:g}s", "split": split,
                    "trace": str(trace_name), "time_s": time_s,
                    "cube_truth_world_xyz_m": np.round(data.qpos[28:31], 5).tolist(),
                    "camera_world_xyz_m": np.round(data.cam_xpos[camera], 5).tolist(),
                    "camera_rotation_world_from_camera_rowmajor": np.round(
                        data.cam_xmat[camera], 6).tolist(),
                    "projection": projection, "orange_bbox": bbox,
                    "estimated_cube_xyz_m": estimate, "xy_error_mm": error,
                    "pose_invalid_returns_no_goal": estimate_on_known_plane(
                        bbox, model, data, "lower_rgb", 320, 240, .01,
                        pose_healthy=False) is None,
                    "pass": bool(passed), "image": image,
                })
    heldout = [r for r in rows if r["split"] == "heldout"]
    development = [r for r in rows if r["split"] == "development"]
    result = {
        "schema": "robot-learning-lab.cove50-frozen-virtual-head-view.v1",
        "classification": "Counterfactual retained-native-state image source screen; not selected sensor hardware, active gaze or image-driven native policy",
        "selected_body_sha256": plan["selected_body_sha256"],
        "frozen_choice": "artifacts/ape-cove50-reach-viewpoint-20261010/heldout-freeze.json",
        "method": "Render one frozen virtual lower RGB camera with visible robot self-occlusion at the exact saved 50 Hz robot/cube states. Change head yaw to -1.5 only for offline viewing; cube truth is evaluator-only. The floor-plane RGB centroid estimate uses exact simulator camera pose and known 10 mm center height.",
        "development_pass_count": sum(r["pass"] for r in development),
        "development_sample_count": len(development),
        "heldout_entry_pass_count": sum(r["pass"] for r in heldout),
        "heldout_entry_sample_count": len(heldout),
        "heldout_entry_gate": all(r["pass"] for r in heldout),
        "blank_image_returns_no_goal": estimate_on_known_plane(
            None, model, data, "lower_rgb", 320, 240, .01) is None,
        "pose_invalid_returns_no_goal_all_samples": all(
            r["pose_invalid_returns_no_goal"] for r in rows),
        "self_occluded_10s_development_abstentions": sum(
            r["time_s"] == 10 and r["orange_bbox"] is None and
            r["estimated_cube_xyz_m"] is None for r in development),
        "images": images, "cases": rows,
    }
    (OUT / "frozen-view-report.json").write_text(json.dumps(result, indent=2,
                                                           allow_nan=False) + "\n")
    (OUT / "frozen-view-matrix.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-frozen-view-matrix.v1",
        "cases": [{"case": r["case"], "split": r["split"], "time_s": r["time_s"],
                   "orange_pixels": r["orange_bbox"]["pixels"] if r["orange_bbox"] else 0,
                   "center_margin_px": r["projection"]["pixel_edge_margin"],
                   "xy_error_mm": r["xy_error_mm"], "pass": r["pass"],
                   "report": "artifacts/ape-cove50-reach-viewpoint-20261010/frozen-view-report.json",
                   "featured": r["split"] == "heldout"}
                  for r in rows]}, indent=2, allow_nan=False) + "\n")
    print("development",result["development_pass_count"],"/",len(development),
          "heldout entry",result["heldout_entry_pass_count"],"/",len(heldout))
    for row in rows:
        print(row["case"],row["orange_bbox"]["pixels"] if row["orange_bbox"] else 0,
              row["projection"]["pixel_edge_margin"],row["xy_error_mm"],row["pass"])


if __name__ == "__main__":
    main()
