"""Counterfactual lower-camera pitch and legal head-yaw visibility screen.

The saved actor/body/traces do not change. A virtual camera mount is compiled
only for offline image evidence; choosing yaw with toy truth is an upper bound.
"""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from screen_retained_rgb import (BODY, OUT, ROOT, orange_bbox,
                                 estimate_on_known_plane, project, sha)


PITCHES = (0, 10, 12, 15)
YAW_TARGETS = (-1.5, -1.4, -1.3)
TIMES = (7.5, 8.5, 10.0, 12.0, 15.0)


def tilted_model(pitch_deg: int) -> mujoco.MjModel:
    root = ET.parse(BODY).getroot()
    camera = root.find(".//camera[@name='lower_rgb']")
    assert camera is not None
    xyaxes = np.fromstring(camera.get("xyaxes"), sep=" ")
    x = xyaxes[:3] / np.linalg.norm(xyaxes[:3])
    y = xyaxes[3:] - x * np.dot(x, xyaxes[3:])
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    original = np.column_stack((x, y, z))
    rotated = original @ Rotation.from_euler("x", -pitch_deg, degrees=True).as_matrix()
    camera.set("xyaxes", " ".join(f"{v:.10f}" for v in
                                   np.r_[rotated[:, 0], rotated[:, 1]]))
    cube = ET.SubElement(root.find("worldbody"), "body", name="offline_visual_cube")
    ET.SubElement(cube, "freejoint", name="offline_visual_cube_free")
    ET.SubElement(cube, "geom", name="offline_visual_cube_geom", type="box",
                  size="0.01 0.01 0.01", mass="0.00625",
                  rgba="0.95 0.36 0.05 1", group="0")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    assert model.nq == 35
    for geom in range(model.ngeom):
        if (model.geom_bodyid[geom] != 0 and model.geom_group[geom] == 4
                and model.geom_rgba[geom, 3] == 0):
            model.geom_rgba[geom, 3] = 1
            model.geom_group[geom] = 0
    return model


def main() -> None:
    plan = json.loads((OUT / "plan.json").read_text())
    assert sha(BODY) == plan["selected_body_sha256"]
    traces = []
    for name in plan["development_traces"]:
        with np.load(ROOT / name) as trace:
            assert abs(float(trace["step_dt_s"]) - 0.02) < 1e-8
            traces.append((name, trace["qpos"].astype(float)))
    width, height = plan["image_width_px"], plan["image_height_px"]
    results = []
    for pitch in PITCHES:
        model = tilted_model(pitch)
        data = mujoco.MjData(model)
        head = model.joint("head_yaw").qposadr[0]
        for yaw in YAW_TARGETS:
            rows = []
            for trace_name, states in traces:
                for time_s in TIMES:
                    frame = round(time_s / 0.02)
                    data.qpos[:] = states[frame]
                    data.qpos[head] = yaw
                    data.qvel[:] = 0
                    mujoco.mj_forward(model, data)
                    projection = project(model, data, "lower_rgb", data.qpos[28:31],
                                         width, height)
                    rows.append({"trace": trace_name, "time_s": time_s,
                                 "cube_truth_world_xyz_m": np.round(data.qpos[28:31], 5).tolist(),
                                 **projection})
            min_margin = min((row["pixel_edge_margin"] if row["pixel_edge_margin"] is not None
                              else -1e6) for row in rows)
            results.append({"case": f"pitch-{pitch}-yaw-{yaw:+.1f}",
                            "additional_downward_pitch_deg": pitch,
                            "head_yaw_target_rad": yaw,
                            "all_20_centers_margin_10px": bool(min_margin >= 10),
                            "minimum_center_edge_margin_px": round(float(min_margin), 2),
                            "center_visible_samples": sum(r["center_in_frame"] for r in rows),
                            "samples": len(rows), "rows": rows})
        print(pitch, [(r["head_yaw_target_rad"], r["minimum_center_edge_margin_px"])
                      for r in results if r["additional_downward_pitch_deg"] == pitch])
    passing = [r for r in results if r["all_20_centers_margin_10px"]]
    option = mujoco.MjvOption()
    option.geomgroup[:] = 1
    rendered_cases = []
    for case in results:
        render_model = tilted_model(case["additional_downward_pitch_deg"])
        data = mujoco.MjData(render_model)
        head = render_model.joint("head_yaw").qposadr[0]
        rendered = []
        with mujoco.Renderer(render_model, height=height, width=width) as renderer:
            for trace_name, states in traces:
                seed = Path(trace_name).stem
                for time_s in TIMES:
                    frame = round(time_s / 0.02)
                    data.qpos[:] = states[frame]
                    data.qpos[head] = case["head_yaw_target_rad"]
                    data.qvel[:] = 0
                    mujoco.mj_forward(render_model, data)
                    renderer.update_scene(data, camera="lower_rgb", scene_option=option)
                    rgb = renderer.render().copy()
                    bbox = orange_bbox(rgb)
                    estimated = estimate_on_known_plane(
                        bbox, render_model, data, "lower_rgb", width, height,
                        plan["known_plane_cube_center_z_m"])
                    error_mm = None if estimated is None else round(float(np.linalg.norm(
                        np.asarray(estimated)[:2] - data.qpos[28:30]) * 1000), 3)
                    path = OUT / f"{seed}-lower-pitch{case['additional_downward_pitch_deg']}-yaw{case['head_yaw_target_rad']:+.1f}-{time_s:g}s.png"
                    if time_s in (7.5, 10.0) and seed in ("seed-9921", "seed-9941", "seed-9942"):
                        imageio.imwrite(path, rgb)
                        image = str(path.relative_to(ROOT))
                    else:
                        image = None
                    rendered.append({"trace": trace_name, "time_s": time_s,
                                     "orange_bbox": bbox, "estimated_cube_xyz_m": estimated,
                                     "xy_error_mm": error_mm,
                                     "pose_invalid_returns_no_goal": estimate_on_known_plane(
                                         bbox, render_model, data, "lower_rgb", width, height,
                                         plan["known_plane_cube_center_z_m"], pose_healthy=False) is None,
                                     "image": image})
        rendered_cases.append({
            "case": case["case"],
            "rgb_detected_samples": sum(r["orange_bbox"] is not None for r in rendered),
            "rgb_error_under_10mm_samples": sum(r["xy_error_mm"] is not None and
                                                r["xy_error_mm"] <= 10 for r in rendered),
            "all_20_rgb_gate": all(r["xy_error_mm"] is not None and
                                   r["xy_error_mm"] <= 10 for r in rendered),
            "rows": rendered,
        })
    selected = max(rendered_cases, key=lambda r: (
        r["rgb_error_under_10mm_samples"], r["rgb_detected_samples"],
        -next(c["additional_downward_pitch_deg"] for c in results
              if c["case"] == r["case"])))
    selected_rgb_pass = selected["all_20_rgb_gate"]
    result = {
        "schema": "robot-learning-lab.cove50-reach-camera-pitch-sensitivity.v1",
        "classification": "Counterfactual offline camera mount and truth-guided fixed yaw screen; not selected body, native rollout or deployable gaze",
        "plan": "artifacts/ape-cove50-reach-camera-20261010/plan.json",
        "selected_body_sha256": plan["selected_body_sha256"],
        "method": "On the same 20 retained development robot/cube poses, compile only a virtual lower_rgb mount rotated downward by the stated extra pitch, then set head yaw to one fixed legal target. Score center margin from exact MuJoCo capture pose and render the best development choice with visible physics proxies. The color mask detects only the synthetic orange cube; its centroid ray intersects a declared 10 mm cube-center plane. Toy truth selects/scores offline variants and never commands the robot.",
        "passing_center_margin_cases": [r["case"] for r in passing],
        "chosen_development_diagnostic": selected["case"],
        "chosen_diagnostic_rgb_error_gate_pass": selected_rgb_pass,
        "cases": results,
        "rendered_cases": rendered_cases,
    }
    (OUT / "pitch-sensitivity.json").write_text(json.dumps(result, indent=2,
                                                            allow_nan=False) + "\n")
    print("passing", result["passing_center_margin_cases"],
          "chosen", result["chosen_development_diagnostic"],
          "rgb error gate", selected_rgb_pass)
    print("rendered", [(r["case"], r["rgb_detected_samples"],
                        r["rgb_error_under_10mm_samples"]) for r in rendered_cases])


if __name__ == "__main__":
    main()
