"""Small local reset/observation check for the varied one-pad touch lesson."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv

from free_object_touch_env import ROOT, touch_env_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    body = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
    cfg = touch_env_cfg(body, entry_mode="stand-tabletop",
                        stand_cube_xy_range=((.046, .054), (-.144, -.136)))
    cfg.scene.num_envs = 8
    cfg.seed = 35001
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        obs, _ = env.reset(seed=35001)
        origins = env.scene.env_origins

        def snapshot() -> tuple[torch.Tensor, torch.Tensor]:
            xy = (env.scene["cube"].data.root_link_pos_w[:, :2] - origins[:, :2]).clone()
            rgb = obs["actor"][:, -6:].clone()
            return xy, rgb

        before_xy, before_rgb = snapshot()
        selected = torch.tensor([0, 2, 4, 6], device=env.device)
        obs, _ = env.reset(env_ids=selected)
        after_xy, after_rgb = snapshot()
        kept = torch.tensor([1, 3, 5, 7], device=env.device)
        action = torch.zeros((8, env.action_space.shape[1]), device=env.device)
        step_obs, reward, terminated, timed_out, _ = env.step(action)
        unique_xy = torch.unique(before_xy.round(decimals=5), dim=0).shape[0]
        unique_rgb = torch.unique(before_rgb[:, :2], dim=0).shape[0]
        assert unique_xy >= 6 and unique_rgb >= 3
        assert torch.all(before_xy[:, 0] >= .046 - 1e-4)
        assert torch.all(before_xy[:, 0] <= .054 + 1e-4)
        assert torch.all(before_xy[:, 1] >= -.144 - 1e-4)
        assert torch.all(before_xy[:, 1] <= -.136 + 1e-4)
        assert torch.equal(before_xy[kept], after_xy[kept])
        assert torch.any(torch.linalg.norm(before_xy[selected] - after_xy[selected], dim=1) > .001)
        assert bool((before_rgb[:, 3] == 1).all() and (before_rgb[:, 5] == 0).all())
        assert bool(torch.isfinite(step_obs["actor"]).all() and torch.isfinite(reward).all())
        assert not bool(terminated.any() or timed_out.any())
        report = {
            "schema": "robot-learning-lab.touch-varied-cpu-entry.v1",
            "body_sha256": hashlib.sha256(body.read_bytes()).hexdigest(),
            "environment_sha256": hashlib.sha256((Path(__file__).parent / "free_object_touch_env.py").read_bytes()).hexdigest(),
            "device": "cpu", "worlds": 8,
            "placement_range_xy_m": [[.046, .054], [-.144, -.136]],
            "source": "2026-10-13 nine-placement teacher screen; authored local sensitivity, not measured hardware tolerance",
            "unique_initial_xy": int(unique_xy), "unique_initial_rgb_centroids": int(unique_rgb),
            "before_xy_m": before_xy.cpu().tolist(), "before_rgb_target": before_rgb.cpu().tolist(),
            "after_partial_reset_xy_m": after_xy.cpu().tolist(),
            "untouched_world_xy_unchanged": bool(torch.equal(before_xy[kept], after_xy[kept])),
            "finite_one_step": True, "step_terminations": int(terminated.sum()),
            "claim_limit": "CPU reset diversity and one-step smoke only; no CUDA throughput, PPO, contact outcome or transfer.",
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("worlds", "unique_initial_xy", "unique_initial_rgb_centroids", "untouched_world_xy_unchanged")}))
    finally:
        env.close()


if __name__ == "__main__":
    main()
