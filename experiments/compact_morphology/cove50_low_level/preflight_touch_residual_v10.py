"""One full-contact-length 256-world CUDA gate for the V10 residual task."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from collect_free_object_touch import BODY
from free_object_touch_env import ROOT
from touch_reference_residual_v10 import residual_touch_env_cfg


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    method = json.loads(args.method.read_text())
    source = Path(__file__).with_name("touch_reference_residual_v10.py")
    assert sha(BODY) == method["body_sha256"]
    assert sha(source) == method["action_source_sha256"]
    assert sha(Path(__file__)) == method["sentinel_sha256"]
    cfg = residual_touch_env_cfg(cube_xy_range=((.046, .054), (-.144, -.136)))
    cfg.scene.num_envs, cfg.seed, cfg.auto_reset = 256, 36020, False
    assert cfg.scene.env_spacing == 0.0
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    try:
        obs, _ = env.reset(seed=36020)
        origins = env.scene.env_origins
        xy = (env.scene["cube"].data.root_link_pos_w[:, :2] - origins[:, :2]).clone()
        rgb = obs["actor"][:, -6:].clone()
        selected = torch.arange(0, 256, 2, device=env.device)
        kept = torch.arange(1, 256, 2, device=env.device)
        obs, _ = env.reset(env_ids=selected)
        after = (env.scene["cube"].data.root_link_pos_w[:, :2] - origins[:, :2]).clone()
        partial_kept = bool(torch.equal(xy[kept], after[kept]))
        partial_changed = int((torch.linalg.norm(xy[selected] - after[selected], dim=1) > .001).sum())
        obs, _ = env.reset(seed=36020)
        cube_start = env.scene["cube"].data.root_link_pos_w.clone()
        rng = np.random.default_rng(36020)
        action = np.zeros((256, 21), dtype=np.float32)
        for cohort, sigma in enumerate((0., .25, .5, 1.)):
            lo, hi = cohort * 64, (cohort + 1) * 64
            if sigma:
                action[lo:hi] = np.clip(rng.normal(0, sigma, (64, 21)), -1, 1)
        residual = torch.as_tensor(action, device=env.device)
        longest = torch.zeros(256, dtype=torch.int32, device=env.device)
        running = torch.zeros_like(longest)
        supported_frames = torch.zeros_like(longest)
        max_drift = torch.zeros(256, device=env.device)
        elapsed = []
        for step in range(200):
            torch.cuda.synchronize()
            begin = time.monotonic()
            obs, reward, terminated, timed_out, _ = env.step(residual)
            torch.cuda.synchronize()
            elapsed.append(time.monotonic() - begin)
            assert bool(torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all())
            assert bool(torch.isfinite(reward).all())
            if bool(terminated.any() or timed_out.any()):
                raise RuntimeError(f"unexpected sentinel termination at step {step + 1}")
            sensor = env.scene.sensors
            finger = sensor["right_finger_tip_cube_contact"].data.found.any(dim=-1)
            feet = sensor["feet_ground_contact"].data.found.all(dim=-1)
            table = sensor["cube_table_contact"].data.found.any(dim=-1)
            hand_floor = sensor["hand_ground_contact"].data.found.any(dim=-1)
            pelvis = env.scene["robot"].data.root_link_pos_w[:, 2]
            valid = finger & feet & table & ~hand_floor & (pelvis >= .10)
            running = torch.where(valid, running + 1, 0)
            longest = torch.maximum(longest, running)
            supported_frames += valid
            drift = torch.linalg.norm((env.scene["cube"].data.root_link_pos_w[:, :2] - cube_start[:, :2]) * 1000, dim=1)
            max_drift = torch.maximum(max_drift, drift)
        term = env.action_manager.get_term("joint_pos")
        cohorts = []
        for i, sigma in enumerate((0., .25, .5, 1.)):
            lo, hi = i * 64, (i + 1) * 64
            cohorts.append({"residual_sigma": sigma, "worlds": 64,
                            "supported_touch_ge_0p2s": int((longest[lo:hi] >= 10).sum()),
                            "supported_touch_ge_1s": int((longest[lo:hi] >= 50).sum()),
                            "gentle_ge_1s": int(((longest[lo:hi] >= 50) & (max_drift[lo:hi] <= 5)).sum()),
                            "median_max_touch_s": round(float(torch.median(longest[lo:hi].float()) * env.step_dt), 3),
                            "median_peak_drift_mm": round(float(torch.median(max_drift[lo:hi])), 3)})
        report = {"schema": "robot-learning-lab.touch-residual-v10-cuda-sentinel.v1",
                  "body_sha256": sha(BODY), "method_sha256": sha(args.method),
                  "action_source_sha256": sha(source), "device": "cuda:0",
                  "gpu": torch.cuda.get_device_name(0), "worlds": 256,
                  "env_spacing_m": cfg.scene.env_spacing,
                  "full_steps": 200, "actor_width": obs["actor"].shape[-1],
                  "unique_xy": int(torch.unique(xy.round(decimals=5), dim=0).shape[0]),
                  "visible_rgb_entries": int(((rgb[:, 3] == 1) & (rgb[:, 5] == 0)).sum()),
                  "partial_reset_kept": partial_kept, "partial_reset_changed_gt_1mm": partial_changed,
                  "first_step_s_including_ik": round(elapsed[0], 3),
                  "median_later_step_s": round(float(np.median(elapsed[1:])), 4),
                  "total_200_step_s": round(sum(elapsed), 3),
                  "rgb_entry_count": term.rgb_entry_count, "rgb_cue_count": term.rgb_cue_count,
                  "cohorts": cohorts,
                  "claim_limit": "Unscored CUDA capacity and policy-residual authority; fixed random biases are not a learned policy."}
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        if (report["actor_width"] != 79 or report["visible_rgb_entries"] < 240 or
                not partial_kept or partial_changed < 100 or
                cohorts[0]["supported_touch_ge_0p2s"] < 16 or
                sum(c["supported_touch_ge_0p2s"] for c in cohorts[1:]) < 8):
            raise RuntimeError("V10 capacity, observability or residual contact gate failed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
