"""Frozen virtual view sensitivity to head-yaw and camera-pitch error."""

from __future__ import annotations

import json

import mujoco
import numpy as np

from screen_head_mount import (OUT, ROOT, estimate_on_known_plane, orange_bbox,
                               project, tilted_model)


def main() -> None:
    plan = json.loads((OUT / "perturbation-plan.json").read_text())
    frozen = json.loads((OUT / "heldout-freeze.json").read_text())
    traces = []
    for split, seeds in (("development", (9921, 9922, 9941, 9942)),
                         ("heldout", frozen["heldout_cases_to_open"])):
        native_split = "reserved" if split == "heldout" else split
        for seed in seeds:
            with np.load(ROOT / f"artifacts/ape-cove50-squat-learned-reach-20261010/v8/conditioned-route/{native_split}/seed-{seed}.npz") as trace:
                traces.append((split, seed, trace["qpos"][375].astype(float)))
    options = mujoco.MjvOption()
    options.geomgroup[:] = 1
    cases = []
    for pitch in plan["changed_factors"]["extra_camera_down_pitch_deg"]:
        model = tilted_model(pitch)
        camera = model.camera("lower_rgb").id
        model.cam_pos[camera] = frozen["head_local_camera_xyz_m"]
        data = mujoco.MjData(model)
        head = model.joint("head_yaw").qposadr[0]
        with mujoco.Renderer(model, width=320, height=240) as renderer:
            for yaw in plan["changed_factors"]["head_yaw_rad"]:
                rows = []
                for split, seed, state in traces:
                    data.qpos[:] = state
                    data.qpos[head] = yaw
                    data.qvel[:] = 0
                    mujoco.mj_forward(model, data)
                    projection = project(model, data, "lower_rgb", data.qpos[28:31],
                                         320, 240)
                    renderer.update_scene(data, camera="lower_rgb", scene_option=options)
                    bbox = orange_bbox(renderer.render().copy())
                    estimate = estimate_on_known_plane(bbox, model, data,
                                                       "lower_rgb", 320, 240, .01)
                    error = None if estimate is None else round(float(np.linalg.norm(
                        np.asarray(estimate)[:2] - data.qpos[28:30]) * 1000), 3)
                    passed = bbox is not None and bbox["pixels"] >= 10 and \
                        projection["pixel_edge_margin"] is not None and \
                        projection["pixel_edge_margin"] >= 10 and error is not None and error <= 10
                    rows.append({"split": split, "seed": seed,
                                 "orange_pixels": bbox["pixels"] if bbox else 0,
                                 "center_margin_px": projection["pixel_edge_margin"],
                                 "xy_error_mm": error, "pass": bool(passed)})
                cases.append({"case": f"pitch-{pitch}-yaw-{yaw:+.2f}",
                              "extra_pitch_deg": pitch, "head_yaw_rad": yaw,
                              "development_pass_count": sum(r["pass"] for r in rows
                                                            if r["split"] == "development"),
                              "heldout_pass_count": sum(r["pass"] for r in rows
                                                        if r["split"] == "heldout"),
                              "all_eight_entry_gate": all(r["pass"] for r in rows),
                              "minimum_orange_pixels": min(r["orange_pixels"] for r in rows),
                              "max_xy_error_mm": max((r["xy_error_mm"] if
                                                      r["xy_error_mm"] is not None else 1e6)
                                                     for r in rows),
                              "rows": rows})
    result = {"schema": "robot-learning-lab.cove50-frozen-view-yaw-pitch-sensitivity.v1",
              "classification": "Counterfactual static-image sensitivity on native states; no active yaw, physical mount, image-driven policy or measured uncertainty",
              "plan": "artifacts/ape-cove50-reach-viewpoint-20261010/perturbation-plan.json",
              "frozen_view": plan["frozen_view"], "cases": cases}
    (OUT / "view-perturbation.json").write_text(json.dumps(result, indent=2,
                                                         allow_nan=False) + "\n")
    print([(r["case"], r["development_pass_count"], r["heldout_pass_count"])
           for r in cases])


if __name__ == "__main__":
    main()
