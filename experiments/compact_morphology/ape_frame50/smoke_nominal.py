"""Run a tiny nominal-body stand and two-camera calibration scene.

This is a runnable interface check, not a learned policy or hardware validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image
import imageio


ROOT = Path(__file__).resolve().parents[3]


def main(leg_candidate=False, supported_leg=False, supported_body=False):
    if sum((leg_candidate, supported_leg, supported_body)) > 1:
        raise ValueError("choose one candidate variant")
    variant = "supported-body" if supported_body else "supported-leg" if supported_leg else "leg-kinematics" if leg_candidate else "bare"
    out = ROOT / f"artifacts/ape-frame50-{variant}"
    source = out / ("ape-frame50-nominal.xml" if variant == "bare" else f"ape-frame50-{variant}.xml")
    xml = ET.parse(source)
    world = xml.getroot().find("worldbody")
    for name, position, color, radius in (
        ("upper_view_marker", "0.60 0 0.78", "0.92 0.24 0.17 1", "0.060"),
        ("floor_view_marker", "0.50 0 0.035", "0.19 0.53 0.88 1", "0.035"),
    ):
        body = ET.SubElement(world, "body", name=name, pos=position)
        ET.SubElement(body, "geom", name=name + "_geom", type="sphere", size=radius,
                      rgba=color, contype="0", conaffinity="0")
    scene = out / "ape-frame50-camera-calibration.xml"
    ET.indent(xml.getroot(), space="  ")
    xml.write(scene, encoding="unicode", xml_declaration=True)

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    data.qpos[:7] = [0, 0, .231, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    camera_frames = {}
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        for name in ("upper_rgb", "lower_rgb"):
            renderer.update_scene(data, camera=name)
            frame = renderer.render()
            path = out / f"{name}-calibration.png"
            Image.fromarray(frame).save(path)
            camera_frames[name] = {"image": str(path.relative_to(ROOT)),
                                   "mean_brightness": round(float(frame.mean()), 2),
                                   "nonblack_fraction": round(float(np.any(frame > 0, axis=2).mean()), 4)}

    root_z = []
    peak_actuator_force = 0.0
    video = imageio.get_writer(out / "zero-command-stand.mp4", fps=25, codec="libx264",
                               quality=7, macro_block_size=16) if variant != "bare" else None
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0, 0, .22]
    camera.distance = .85
    camera.azimuth = 135
    camera.elevation = -12
    world_renderer = mujoco.Renderer(model, height=480, width=640) if video else None
    for i in range(1500):
        mujoco.mj_step(model, data)
        if video and i % 30 == 0:
            world_renderer.update_scene(data, camera=camera)
            video.append_data(world_renderer.render())
        if i % 50 == 0:
            root_z.append(float(data.qpos[2]))
            peak_actuator_force = max(peak_actuator_force,
                                      float(np.abs(data.actuator_force).max()))
    if video:
        video.close()
        world_renderer.close()
    roll_angles = {}
    for side in ("left", "right"):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_hip_roll")
        roll_angles[side] = round(float(data.qpos[model.jnt_qposadr[joint]]), 4)
    report = {
        "stage": f"{variant} interface smoke; not learned or mechanically qualified",
        "source_mjcf_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scene_mjcf_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
        "duration_s": 3.0,
        "camera_calibration": camera_frames,
        "zero_command_root_z_start_m": round(root_z[0], 5),
        "zero_command_root_z_end_m": round(root_z[-1], 5),
        "zero_command_root_x_end_m": round(float(data.qpos[0]), 5),
        "peak_actuator_force_nm": round(peak_actuator_force, 4),
        "end_hip_roll_rad": roll_angles,
        "world_video": str((out / "zero-command-stand.mp4").relative_to(ROOT)) if video else None,
    }
    (out / "nominal-smoke-report.json").write_text(json.dumps(report, indent=2) + "\n")
    if variant != "bare":
        shutil.copy2(Path(__file__).with_name(
            "supported_body_viewer.html" if supported_body else "supported_leg_viewer.html" if supported_leg else "leg_kinematics_viewer.html"),
            out / "index.html")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leg-candidate", action="store_true")
    parser.add_argument("--supported-leg", action="store_true")
    parser.add_argument("--supported-body", action="store_true")
    args = parser.parse_args()
    main(args.leg_candidate, args.supported_leg, args.supported_body)
