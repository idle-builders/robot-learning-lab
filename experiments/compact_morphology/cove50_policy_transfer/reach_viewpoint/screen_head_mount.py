"""Frozen virtual head-camera position grid on retained native reach-entry poses.

All camera placement and target truth are evaluator-only. No actor or robot
state is changed except a fixed legal head yaw for the offline view screen.
"""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reach_perception"))
from screen_retained_rgb import BODY, POINTER, ROOT, estimate_on_known_plane, orange_bbox, project, sha
from screen_pitch_sensitivity import tilted_model

OUT = ROOT / "artifacts/ape-cove50-reach-viewpoint-20261010"


def main() -> None:
    plan = json.loads((OUT / "plan.json").read_text())
    assert sha(BODY) == plan["selected_body_sha256"]
    assert sha(POINTER) == plan["selected_pointer_sha256"]
    width, height = 320, 240
    traces = []
    for name in plan["development_traces"]:
        native = json.loads((ROOT / name).with_suffix(".json").read_text())
        assert native["body_sha256"] == plan["selected_body_sha256"]
        with np.load(ROOT / name) as trace:
            traces.append((Path(name).stem, trace["qpos"][round(7.5 / .02)].astype(float)))
    option = mujoco.MjvOption()
    option.geomgroup[:] = 1
    results = []
    for pitch in plan["extra_downward_pitch_deg"]:
        model = tilted_model(pitch)
        data = mujoco.MjData(model)
        head = model.joint("head_yaw").qposadr[0]
        camera = model.camera("lower_rgb").id
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            for xyz in product(*(plan["head_local_camera_position_grid_m"][axis]
                                 for axis in ("x", "y", "z"))):
                model.cam_pos[camera] = xyz
                samples = []
                for seed, state in traces:
                    data.qpos[:] = state
                    data.qpos[head] = -1.5
                    data.qvel[:] = 0
                    mujoco.mj_forward(model, data)
                    projection = project(model, data, "lower_rgb", data.qpos[28:31],
                                         width, height)
                    renderer.update_scene(data, camera="lower_rgb", scene_option=option)
                    rgb = renderer.render().copy()
                    bbox = orange_bbox(rgb)
                    estimate = estimate_on_known_plane(bbox, model, data, "lower_rgb",
                                                       width, height, 0.01)
                    error = None if estimate is None else round(float(np.linalg.norm(
                        np.asarray(estimate)[:2] - data.qpos[28:30]) * 1000), 3)
                    samples.append({"seed": seed,
                                    "cube_truth_world_xyz_m": np.round(data.qpos[28:31], 5).tolist(),
                                    "camera_world_xyz_m": np.round(data.cam_xpos[camera], 5).tolist(),
                                    "projection": projection, "orange_bbox": bbox,
                                    "estimated_xyz_m": estimate, "xy_error_mm": error,
                                    "pose_invalid_returns_no_goal": estimate_on_known_plane(
                                        bbox, model, data, "lower_rgb", width, height,
                                        0.01, pose_healthy=False) is None})
                gate = all(s["orange_bbox"] is not None and
                           s["projection"]["pixel_edge_margin"] is not None and
                           s["projection"]["pixel_edge_margin"] >= 10 and
                           s["xy_error_mm"] is not None and s["xy_error_mm"] <= 10
                           for s in samples)
                current = np.array([.048, -.015, -.014])
                results.append({
                    "case": f"head-p{pitch}-x{xyz[0]:.3f}-y{xyz[1]:+.3f}-z{xyz[2]:+.3f}",
                    "extra_pitch_deg": pitch, "head_local_camera_xyz_m": list(xyz),
                    "module_center_shift_mm": round(float(np.linalg.norm(np.array(xyz)-current)*1000), 2),
                    "head_sphere_center_distance_mm": round(float(np.linalg.norm(xyz)*1000), 2),
                    "entry_gate": gate,
                    "rgb_entry_detected": sum(s["orange_bbox"] is not None for s in samples),
                    "rgb_entry_under_10mm": sum(s["xy_error_mm"] is not None and
                                                s["xy_error_mm"] <= 10 for s in samples),
                    "min_rgb_pixels": min(s["orange_bbox"]["pixels"] if s["orange_bbox"]
                                          else 0 for s in samples),
                    "max_xy_error_mm": max((s["xy_error_mm"] if s["xy_error_mm"] is not None
                                            else 1e6) for s in samples),
                    "min_projected_center_margin_px": min(
                        s["projection"]["pixel_edge_margin"] if
                        s["projection"]["pixel_edge_margin"] is not None else -1e6
                        for s in samples),
                    "samples": samples,
                })
    passing = [r for r in results if r["entry_gate"]]
    ranked = sorted(passing or results, key=lambda r: (
        -r["rgb_entry_under_10mm"], -r["min_rgb_pixels"],
        r["max_xy_error_mm"], r["module_center_shift_mm"]))
    selected = ranked[0]
    model = tilted_model(selected["extra_pitch_deg"])
    data = mujoco.MjData(model)
    camera = model.camera("lower_rgb").id
    model.cam_pos[camera] = selected["head_local_camera_xyz_m"]
    head = model.joint("head_yaw").qposadr[0]
    images = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for seed, state in traces:
            data.qpos[:] = state
            data.qpos[head] = -1.5
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="lower_rgb", scene_option=option)
            path = OUT / f"{seed}-{selected['case']}-entry.png"
            imageio.imwrite(path, renderer.render())
            images.append(str(path.relative_to(ROOT)))
    report = {
        "schema": "robot-learning-lab.cove50-virtual-head-view-screen.v1",
        "classification": "Offline counterfactual camera placement on retained native states, not selected hardware, pose control or image-driven reach",
        "plan": str((OUT / "plan.json").relative_to(ROOT)),
        "selected_body_sha256": plan["selected_body_sha256"],
        "method": "Change only virtual lower_rgb head-local position and extra pitch; fix legal head yaw -1.5 on four retained 7.5 s development states. Render visual cube and visible robot geometry; target truth scores projection and XY error, not control. MuJoCo capture-time camera pose is exact simulator state.",
        "candidate_count": len(results), "entry_gate_pass_count": len(passing),
        "chosen_development_case": selected["case"],
        "chosen_case_is_entry_pass": selected["entry_gate"],
        "chosen_images": images,
        "cases": results,
    }
    (OUT / "head-mount-screen.json").write_text(json.dumps(report, indent=2,
                                                        allow_nan=False) + "\n")
    print("candidates",len(results),"entry passes",len(passing))
    for item in ranked[:10]:
        print(item["case"],item["rgb_entry_detected"],item["rgb_entry_under_10mm"],
              item["min_rgb_pixels"],item["max_xy_error_mm"],
              item["module_center_shift_mm"])


if __name__ == "__main__":
    main()
