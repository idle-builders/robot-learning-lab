"""Run the pinned Berkeley Lite MuJoCo policy headlessly with upstream control code."""

import argparse
import importlib
import json
import math
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--vx", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.video:
        args.video = args.video.resolve()

    upstream = Path(__file__).resolve().parents[2] / ".cache/upstream/berkeley-humanoid-lite"
    assets = upstream / "source/berkeley_humanoid_lite_assets/data"
    mjcf = assets / "robots/berkeley_humanoid/berkeley_humanoid_lite/mjcf"

    # This pinned upstream revision ships meshes beside MJCF and uses a flat
    # data/mjcf path in its native simulator. Only resolve those file paths.
    flat_mjcf = assets / "mjcf"
    if not flat_mjcf.exists():
        flat_mjcf.symlink_to(mjcf.relative_to(assets), target_is_directory=True)
    mesh_alias = mjcf / "assets/merged"
    if not mesh_alias.exists():
        mesh_alias.parent.mkdir(exist_ok=True)
        mesh_alias.symlink_to("../../meshes", target_is_directory=True)

    sys.path[:0] = [
        str(upstream / "source/berkeley_humanoid_lite"),
        str(upstream / "source/berkeley_humanoid_lite_lowlevel"),
    ]
    native = importlib.import_module("berkeley_humanoid_lite.environments.mujoco")
    from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController

    class Viewer:
        def sync(self):
            pass

    class Command:
        def __init__(self):
            self.commands = {
                "mode_switch": 0,
                "velocity_x": args.vx,
                "velocity_y": 0.0,
                "velocity_yaw": 0.0,
            }

        def run(self):
            pass

    native.Se2Gamepad = Command
    mujoco.viewer.launch_passive = lambda _model, _data: Viewer()
    cfg = OmegaConf.load(upstream / "configs/policy_humanoid.yaml")
    cfg.policy_checkpoint_path = str(upstream / cfg.policy_checkpoint_path)

    old_cwd = Path.cwd()
    try:
        import os
        os.chdir(upstream)
        robot = native.MujocoSimulator(cfg)
        robot.reset()
        mujoco.mj_forward(robot.mj_model, robot.mj_data)
        obs = robot._get_observations()
        policy = RlController(cfg)
        policy.load_policy()

        xyz = []
        tilts = []
        contact_counts = []
        contact_pair_steps = {}
        renderer = mujoco.Renderer(robot.mj_model, height=240, width=320) if args.video else None
        writer = None
        if args.video:
            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(args.video, fps=5, codec="libx264")
        try:
            for step in range(args.steps):
                action = policy.update(obs.numpy())
                obs = robot.step(torch.as_tensor(action))
                xyz.append(robot.mj_data.qpos[:3].tolist())
                projected_gravity = robot._get_projected_gravity().numpy()
                tilts.append(math.acos(float(np.clip(-projected_gravity[2], -1, 1))))
                contact_counts.append(robot.mj_data.ncon)
                for contact in robot.mj_data.contact:
                    names = sorted((
                        mujoco.mj_id2name(robot.mj_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or str(contact.geom1),
                        mujoco.mj_id2name(robot.mj_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or str(contact.geom2),
                    ))
                    pair = " / ".join(names)
                    contact_pair_steps[pair] = contact_pair_steps.get(pair, 0) + 1
                if renderer and step % 5 == 0:
                    camera = mujoco.MjvCamera()
                    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                    camera.lookat[:] = robot.mj_data.qpos[:3] + np.array([0.0, 0.0, 0.4])
                    camera.distance = 2.0
                    camera.azimuth = 140
                    camera.elevation = -18
                    renderer.update_scene(robot.mj_data, camera=camera)
                    writer.append_data(renderer.render())
        finally:
            if writer:
                writer.close()
            if renderer:
                renderer.close()

        result = {
            "source": "HybridRobotics/Berkeley-Humanoid-Lite",
            "source_commit": "984741a3623c93b0583ccfdc479f1f8b1c4d900e",
            "asset_commit": "fc90fedd008b1e56a22e3c5221548d6b24f49707",
            "policy": "checkpoints/policy_humanoid.onnx",
            "policy_input": 75,
            "policy_output": 22,
            "steps": args.steps,
            "policy_dt": float(cfg.policy_dt),
            "velocity_command_x": args.vx,
            "video": str(args.video) if args.video else None,
            "root_start_xyz": xyz[0],
            "root_end_xyz": xyz[-1],
            "minimum_root_z": min(row[2] for row in xyz),
            "maximum_tilt_rad": max(tilts),
            "steps_with_contact": sum(count > 0 for count in contact_counts),
            "contact_pair_steps": contact_pair_steps,
            "root_xyz": xyz,
            "tilt_rad": tilts,
            "contact_counts": contact_counts,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k not in ("root_xyz", "tilt_rad", "contact_counts")}, indent=2))
    finally:
        import os
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
