"""One bounded exact 256-world A40 camera/contact gate before touch PPO."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from free_object_touch_env import ROOT
from train_free_object_touch_ppo import build_train_config


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    method = json.loads(args.method.read_text())
    body = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
    source = Path(__file__).with_name("free_object_touch_env.py")
    assert sha(body) == method["source"]["body_sha256"]
    assert sha(source) == method["source"]["environment_sha256"]
    _, cfg = build_train_config(worlds=256, updates=12,
                                seed=method["train"]["seed"])
    env = ManagerBasedRlEnv(cfg=cfg.env, device="cuda:0")
    try:
        obs, _ = env.reset(seed=35020)
        origins = env.scene.env_origins
        initial_xy = (env.scene["cube"].data.root_link_pos_w[:, :2] - origins[:, :2]).clone()
        initial_rgb = obs["actor"][:, -6:].clone()
        selected = torch.arange(0, 256, 2, device=env.device)
        kept = torch.arange(1, 256, 2, device=env.device)
        obs, _ = env.reset(env_ids=selected)
        after_xy = (env.scene["cube"].data.root_link_pos_w[:, :2] - origins[:, :2]).clone()
        unchanged = bool(torch.equal(initial_xy[kept], after_xy[kept]))
        changed = int((torch.linalg.norm(initial_xy[selected] - after_xy[selected], dim=1) > .001).sum())
        frames = [env.scene["touch_rgb"].data.rgb[0].detach().cpu().numpy().copy()]
        action = torch.zeros((256, env.action_space.shape[1]), device=env.device)
        torch.cuda.synchronize()
        start = time.monotonic()
        for _ in range(5):
            obs, reward, terminated, timed_out, _ = env.step(action)
            assert bool(torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all())
            assert bool(torch.isfinite(reward).all())
            assert not bool(terminated.any() or timed_out.any())
            frames.append(env.scene["touch_rgb"].data.rgb[0].detach().cpu().numpy().copy())
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        args.out.mkdir(parents=True, exist_ok=False)
        video = args.out / "camera-path.mp4"
        imageio.mimwrite(video, frames, fps=50, codec="libx264", quality=7,
                         macro_block_size=1)
        decoded = []
        with imageio.get_reader(video, format="ffmpeg") as reader:
            for frame in reader:
                decoded.append(float(np.std(frame)))
        visible = int(((initial_rgb[:, 3] == 1) & (initial_rgb[:, 5] == 0)).sum())
        unique_xy = int(torch.unique(initial_xy.round(decimals=5), dim=0).shape[0])
        report = {"schema": "robot-learning-lab.touch-cuda-capacity.v1",
                  "body_sha256": sha(body), "environment_sha256": sha(source),
                  "method_sha256": sha(args.method), "device": "cuda:0",
                  "gpu": torch.cuda.get_device_name(0), "worlds": 256,
                  "actor_shape": list(obs["actor"].shape),
                  "critic_shape": list(obs["critic"].shape),
                  "action_shape": list(action.shape),
                  "unique_initial_xy": unique_xy, "visible_initial_rgb": visible,
                  "partial_reset_selected_changed_gt_1mm": changed,
                  "partial_reset_unselected_unchanged": unchanged,
                  "five_step_elapsed_s": round(elapsed, 4),
                  "max_cuda_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 4),
                  "video_frames_decoded": len(decoded),
                  "min_decoded_frame_std": round(min(decoded), 3),
                  "video_sha256": sha(video),
                  "claim_limit": "Exact 256-world CUDA reset/camera/contact/finite-step and encode/decode path only; no learner outcome."}
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        if (report["gpu"] != "NVIDIA A40" or unique_xy < 220 or visible < 240
                or changed < 100 or not unchanged or len(decoded) != 6
                or min(decoded) < 1):
            raise RuntimeError("Touch CUDA entry, variation or graphics gate failed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
