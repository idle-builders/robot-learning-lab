"""Check whether a legal gaze could have seen the toy before saved reach entry.

This is an offline truth-scored visibility upper bound, not a gaze controller.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from screen_retained_rgb import BODY, OUT, ROOT, model_with_visual_cube, project, sha


TIMES = (0, 1, 2, 3, 4, 5, 6, 7, 7.5)


def main() -> None:
    plan = json.loads((OUT / "plan.json").read_text())
    assert sha(BODY) == plan["selected_body_sha256"]
    model = model_with_visual_cube()
    data = mujoco.MjData(model)
    head = model.joint("head_yaw").qposadr[0]
    yaw_grid = np.linspace(*model.joint("head_yaw").range, 61)
    rows = []
    for trace_name in plan["development_traces"]:
        with np.load(ROOT / trace_name) as trace:
            states = trace["qpos"].astype(float)
        for time_s in TIMES:
            data.qpos[:] = states[round(time_s / 0.02)]
            cube = data.qpos[28:31].copy()
            best = None
            for yaw in yaw_grid:
                data.qpos[head] = yaw
                mujoco.mj_forward(model, data)
                for camera in ("upper_rgb", "lower_rgb"):
                    result = project(model, data, camera, cube,
                                     plan["image_width_px"], plan["image_height_px"])
                    margin = result["pixel_edge_margin"]
                    if margin is not None and (best is None or margin > best["margin_px"]):
                        best = {"margin_px": margin, "yaw_rad": round(float(yaw), 2),
                                "camera": camera, "uv_px": result["uv_px"]}
            rows.append({"trace": trace_name, "time_s": time_s,
                         "best_legal_yaw_projection": best,
                         "any_camera_center_visible": best is not None and
                         best["margin_px"] >= 0})
    result = {
        "schema": "robot-learning-lab.cove50-early-legal-gaze-upper-bound.v1",
        "classification": "Offline truth-scored projection on retained native states; no active look or image-driven policy",
        "selected_body_sha256": plan["selected_body_sha256"],
        "method": "At each retained 0–7.5 s state, change only head yaw across its exact legal range in 0.05 rad increments, project the native toy center in both selected cameras using the capture-time camera pose, and report the best image-edge margin. Toy truth is evaluator-only.",
        "all_36_sampled_times_outside_both_cameras": all(not r["any_camera_center_visible"] for r in rows),
        "rows": rows,
    }
    (OUT / "early-visibility.json").write_text(json.dumps(result, indent=2,
                                                           allow_nan=False) + "\n")
    for trace_name in plan["development_traces"]:
        case = [r for r in rows if r["trace"] == trace_name]
        print(Path(trace_name).stem, "best margin", max(
            r["best_legal_yaw_projection"]["margin_px"] for r in case))
    print("all earlier samples outside both cameras",
          result["all_36_sampled_times_outside_both_cameras"])


if __name__ == "__main__":
    main()
