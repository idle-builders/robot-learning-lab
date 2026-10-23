"""Capture mounted RGB-D views while replaying saved native 50 cm body poses.

Room geoms are a visual counterfactual: they never enter the source rollout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
SOURCE = ROOT / "artifacts/ape-cove50-effort-envelope"
OUTPUT = ROOT / "artifacts/ape-onboard-motion-v1"
CASES = ("forward-stop-early", "forward-stop-late")
SIZE = (320, 240)
EVERY = 5  # 10 Hz from the saved 50 Hz actor trace.


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_model() -> mujoco.MjModel:
    root = ET.fromstring(BODY.read_text())
    root.remove(root.find("sensor"))  # Only for older local pose renderer.
    world = root.find("worldbody")
    # Fixed, varied visual structure in the robot's local replay room. Added
    # objects have no contact and are not claimed as rollout obstacles.
    ET.SubElement(world, "geom", name="rear_wall", type="box", pos="1.8 0 0.65",
                  size="0.02 1.2 0.65", rgba="0.56 0.59 0.61 1",
                  contype="0", conaffinity="0")
    for col in range(10):
        for row in range(6):
            y = -.96 + col * .21
            z = .15 + row * .19
            tone = .14 if (row + col) % 2 else .86
            ET.SubElement(world, "geom", name=f"wall_tile_{col}_{row}", type="box",
                          pos=f"1.776 {y:.4f} {z:.4f}", size="0.005 0.083 0.073",
                          rgba=f"{tone:.3f} {(.25+.58*((col*3+row)%5)/4):.3f} {1-tone:.3f} 1",
                          contype="0", conaffinity="0")
    for index, (x, y, z, r, g, b) in enumerate((
        (.65, -.62, .22, .82, .20, .15), (.90, .68, .32, .22, .38, .78),
        (1.18, -.69, .42, .20, .71, .38), (1.42, .53, .24, .77, .68, .21),
    )):
        ET.SubElement(world, "geom", name=f"furniture_{index}", type="box",
                      pos=f"{x} {y} {z}", size="0.11 0.11 0.17",
                      rgba=f"{r} {g} {b} 1", contype="0", conaffinity="0")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.geom_rgba[model.geom_rgba[:, 3] == 0, 3] = 1
    model.geom_group[:] = 0
    return model


def capture(case: str, world: int, model: mujoco.MjModel) -> dict:
    trace_path = SOURCE / f"{case}.npz"
    trace = np.load(trace_path)
    packet_times = trace["time_s"][::EVERY].astype(float)
    joint_addrs = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))]
                   for name in trace["leg_names"]]
    data = mujoco.MjData(model)
    origin = trace["root_pose"][0, world, :2].astype(float)
    frames, depths, camera_positions = [], [], []
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "upper_rgb")
    video_path = OUTPUT / f"{case}-world{world}-upper-rgb.mp4"
    writer = (cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, SIZE)
              if world == 0 else None)
    if writer is not None and not writer.isOpened():
        raise RuntimeError("Could not open RGB video writer")
    try:
        with mujoco.Renderer(model, width=SIZE[0], height=SIZE[1]) as renderer:
            for i in range(0, len(trace["time_s"]), EVERY):
                root = trace["root_pose"][i, world].astype(float).copy()
                root[:2] -= origin
                data.qpos[:] = model.qpos0
                data.qpos[:7] = root
                data.qpos[joint_addrs] = trace["joint_pos"][i, world]
                mujoco.mj_forward(model, data)
                camera_positions.append(data.cam_xpos[camera_id].copy())
                renderer.update_scene(data, camera="upper_rgb")
                rgb = renderer.render().copy()
                renderer.enable_depth_rendering()
                depth = renderer.render().copy()
                renderer.disable_depth_rendering()
                frames.append(rgb)
                depths.append(depth.astype(np.float32))
                if writer is not None:
                    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None:
            writer.release()
    npz_path = OUTPUT / f"{case}-world{world}-upper-rgbd.npz"
    np.savez_compressed(npz_path, time_s=packet_times, rgb=np.asarray(frames),
                        depth_m=np.asarray(depths),
                        evaluator_camera_world_pos_m=np.asarray(camera_positions), dt_s=.1)
    if world == 0:
        cv2.imwrite(str(OUTPUT / f"{case}-first-upper-rgb.png"),
                    cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
    return {"case": case, "world": world, "source_trace_sha256": sha(trace_path),
            "rgbd_path": str(npz_path.relative_to(ROOT)), "rgbd_sha256": sha(npz_path),
            "video_path": str(video_path.relative_to(ROOT)) if writer is not None else None,
            "video_sha256": sha(video_path) if writer is not None else None,
            "samples": len(frames)}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = make_model()
    cases = [capture(case, world, model) for case in CASES for world in range(4)]
    manifest = {"schema": "robot-learning-lab.ape-onboard-camera-capture.v1",
                "classification": "camera rendering of recorded native poses; added room never affects physics",
                "source_body_sha256": sha(BODY), "source_script_sha256": sha(Path(__file__)),
                "camera": "upper_rgb mounted on historical rejected Direct Frame",
                "frame_size": list(SIZE), "sample_period_s": .1, "room":
                "fixed checker-tile wall and four fixed colored furniture boxes, all contact-disabled",
                "cases": cases}
    (OUTPUT / "capture-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
