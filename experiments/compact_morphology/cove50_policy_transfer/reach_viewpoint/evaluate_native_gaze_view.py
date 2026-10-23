"""Render the frozen virtual camera from a motor-attained native gaze trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from screen_head_mount import BODY, OUT, POINTER, ROOT, estimate_on_known_plane, orange_bbox, project, sha, tilted_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=OUT / "native-gaze/dev-9921.npz")
    args = parser.parse_args()
    plan = json.loads((OUT / "gaze-native-plan.json").read_text())
    freeze = json.loads((OUT / "heldout-freeze.json").read_text())
    assert sha(BODY) == plan["selected_body_sha256"]
    assert sha(POINTER) == plan["selected_pointer_sha256"]
    trace_path = args.trace.resolve()
    native = json.loads(trace_path.with_suffix(".json").read_text())
    assert native["body_sha256"] == plan["selected_body_sha256"]
    assert native["evaluator_sha256"] == plan["gaze_evaluator_sha256"]
    assert native["reach_checkpoint_sha256"] == plan["saved_reach_actor_sha256"]
    assert native["gaze_head_yaw_rad"] == freeze["fixed_head_yaw_rad"] + .2
    model = tilted_model(freeze["extra_downward_pitch_deg"])
    camera = model.camera("lower_rgb").id
    model.cam_pos[camera] = freeze["head_local_camera_xyz_m"]
    data = mujoco.MjData(model)
    head = model.joint("head_yaw").qposadr[0]
    option = mujoco.MjvOption()
    option.geomgroup[:] = 1
    with np.load(trace_path) as trace:
        qpos = trace["qpos"].astype(float)
    assert qpos.shape[1] == 35
    rows = []
    with mujoco.Renderer(model, width=320, height=240) as renderer:
        for time_s in [6, 7, 7.5, 8.5, 10]:
            frame = round(time_s / .02)
            data.qpos[:] = qpos[frame]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            projection = project(model, data, "lower_rgb", data.qpos[28:31], 320, 240)
            renderer.update_scene(data, camera="lower_rgb", scene_option=option)
            rgb = renderer.render().copy()
            bbox = orange_bbox(rgb)
            estimate = estimate_on_known_plane(bbox, model, data, "lower_rgb", 320, 240, .01)
            error_mm = None if estimate is None else round(float(np.linalg.norm(
                np.asarray(estimate)[:2] - data.qpos[28:30]) * 1000), 3)
            passed = bbox is not None and bbox["pixels"] >= 10 and \
                projection["pixel_edge_margin"] is not None and \
                projection["pixel_edge_margin"] >= 10 and error_mm is not None and error_mm <= 10
            image_path = OUT / f"native-gaze/{trace_path.stem}-{time_s:g}s-attained-view.png"
            imageio.imwrite(image_path, rgb)
            rows.append({
                "time_s": time_s, "head_yaw_rad": round(float(data.qpos[head]), 5),
                "camera_world_xyz_m": np.round(data.cam_xpos[camera], 5).tolist(),
                "orange_bbox": bbox, "projection": projection,
                "estimated_cube_xyz_m": estimate, "xy_error_mm": error_mm,
                "pass": bool(passed), "image": str(image_path.relative_to(ROOT)),
            })
    result = {
        "schema": "robot-learning-lab.cove50-native-attained-gaze-view.v1",
        "classification": "Native motor-attained head state, counterfactual massless camera; simulator-truth cube still controls reach",
        "selected_body_sha256": plan["selected_body_sha256"],
        "native_trace": str(trace_path.relative_to(ROOT)),
        "frozen_virtual_camera": str((OUT / "heldout-freeze.json").relative_to(ROOT)),
        "entry_7_5s_gate": next(r["pass"] for r in rows if r["time_s"] == 7.5),
        "cases": rows,
    }
    (OUT / f"native-gaze/{trace_path.stem}-attained-view-report.json").write_text(json.dumps(
        result, indent=2, allow_nan=False) + "\n")
    for row in rows:
        print(row["time_s"], row["head_yaw_rad"],
              row["orange_bbox"]["pixels"] if row["orange_bbox"] else 0,
              row["xy_error_mm"], row["pass"])


if __name__ == "__main__":
    main()
