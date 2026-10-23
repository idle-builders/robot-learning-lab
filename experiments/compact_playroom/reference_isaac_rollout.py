"""Play Berkeley Lite's released ONNX locomotion policy in its own Isaac Lab task.

Run from the pinned Berkeley workspace after installing its Python dependencies.
Unlike upstream play.py, this accepts the released ONNX checkpoint directly.
"""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output = args.output.resolve()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import onnxruntime as ort
import torch

from isaaclab_tasks.utils import parse_env_cfg
import berkeley_humanoid_lite.tasks  # noqa: F401 - registers the upstream task


def main():
    task = "Velocity-Berkeley-Humanoid-Lite-v0"
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env = gym.make(task, cfg=cfg)
    checkpoint = Path(__file__).resolve().parents[2] / ".cache/upstream/berkeley-humanoid-lite/checkpoints/policy_humanoid.onnx"
    if not checkpoint.exists():
        checkpoint = Path.cwd() / "checkpoints/policy_humanoid.onnx"
    policy = ort.InferenceSession(str(checkpoint), providers=["CPUExecutionProvider"])
    input_name = policy.get_inputs()[0].name
    obs, _ = env.reset()
    if obs["policy"].shape[-1] != 75 or env.action_space.shape[-1] != 22:
        raise RuntimeError(f"Unexpected Berkeley observation/action contract: {obs['policy'].shape}, {env.action_space}")

    robot = env.unwrapped.scene["robot"]
    trace = []
    try:
        for step in range(args.steps):
            state = obs["policy"].detach().cpu().numpy().astype(np.float32)
            action = policy.run(None, {input_name: state})[0]
            obs, reward, terminated, truncated, extras = env.step(torch.from_numpy(action).to(env.unwrapped.device))
            trace.append({
                "step": step,
                "base_xyz": robot.data.root_pos_w[0].detach().cpu().tolist(),
                "reward": float(reward[0].detach().cpu()),
                "terminated": bool(terminated[0].detach().cpu()),
                "truncated": bool(truncated[0].detach().cpu()),
            })
            if trace[-1]["terminated"] or trace[-1]["truncated"]:
                break
    finally:
        env.close()

    result = {
        "task": task,
        "robot_source": "HybridRobotics/Berkeley-Humanoid-Lite@984741a3623c93b0583ccfdc479f1f8b1c4d900e",
        "checkpoint": "checkpoints/policy_humanoid.onnx",
        "simulator": "Isaac Lab 2.3.2 / Isaac Sim 5.1",
        "trace": trace,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"steps": len(trace), "first": trace[0], "last": trace[-1]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
