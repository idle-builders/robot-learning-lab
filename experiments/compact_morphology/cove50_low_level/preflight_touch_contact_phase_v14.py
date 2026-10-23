"""Unscored V14 CUDA capacity and initial Gaussian actor-action screen."""

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
from touch_reference_residual_v14 import residual_touch_env_cfg


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    method = json.loads(args.method.read_text())
    assert method["schema"] == "robot-learning-lab.touch-contact-phase-v14.method.v1"
    assert sha(BODY) == method["body_sha256"]
    assert sha(Path(__file__)) == method["sentinel_sha256"]
    assert sha(Path(__file__).with_name("touch_reference_residual_v14.py")) == method["action_source_sha256"]
    for label, record in method["source_files"].items():
        if sha(Path(record["path"])) != record["sha256"]:
            raise RuntimeError(f"frozen source differs on worker: {label}")
    cfg = residual_touch_env_cfg(cube_xy_range=((.046, .054), (-.144, -.136)))
    cfg.scene.num_envs, cfg.seed, cfg.auto_reset = 256, 37120, False
    cfg.episode_length_s = 4.0
    assert cfg.scene.env_spacing == 0.0
    assert cfg.actions["joint_pos"].residual_gain == method["residual_gain"] == .005
    assert cfg.actions["joint_pos"].warmup_steps == 50
    assert cfg.actions["joint_pos"].ramp_steps == 15
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    try:
        obs, _ = env.reset(seed=37120)
        initial_actor = obs["actor"].clone()
        cube_xy = env.scene["cube"].data.root_link_pos_w[:, :2].clone()
        initial_rgb = initial_actor[:, -6:].clone()
        selected = torch.arange(0, 256, 2, device=env.device)
        kept = torch.arange(1, 256, 2, device=env.device)
        env.reset(env_ids=selected)
        after = env.scene["cube"].data.root_link_pos_w[:, :2]
        partial_kept = bool(torch.equal(cube_xy[kept], after[kept]))
        partial_changed = int((torch.linalg.norm(cube_xy[selected] - after[selected], dim=1) > .001).sum())
        obs, _ = env.reset(seed=37120)
        term = env.action_manager.get_term("joint_pos")
        cube_start = env.scene["cube"].data.root_link_pos_w[:, :2].clone()
        generator = torch.Generator(device=env.device).manual_seed(37120)
        longest = torch.zeros(256, dtype=torch.int32, device=env.device)
        running = torch.zeros_like(longest)
        supported_frames = torch.zeros_like(longest)
        table_frames = torch.zeros_like(longest)
        max_drift = torch.zeros(256, device=env.device)
        reward_pre = torch.zeros(256, device=env.device)
        reward_contact = torch.zeros(256, device=env.device)
        first_effective = None
        first_teacher = None
        step51_effective = None
        step65_effective = None
        step51_combined_gap = None
        step65_combined_gap = None
        step51_target_gap = None
        step65_target_gap = None
        elapsed = []
        for step in range(1, 201):
            # Upstream PPO starts at std 1.0. Draw new action noise each step;
            # the adapter itself clips to [-1, 1] and masks non-arm channels.
            action = torch.zeros((256, 21), device=env.device)
            action[128:] = torch.randn((128, 21), generator=generator, device=env.device)
            torch.cuda.synchronize()
            begin = time.monotonic()
            obs, reward, terminated, timed_out, _ = env.step(action)
            torch.cuda.synchronize()
            elapsed.append(time.monotonic() - begin)
            if not bool(torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all()
                        and torch.isfinite(reward).all()):
                raise RuntimeError(f"nonfinite actor, critic or reward at step {step}")
            if bool(terminated.any()) or (step < 200 and bool(timed_out.any())):
                raise RuntimeError(f"early termination or timeout at step {step}")
            effective = term.cfg.residual_gain * term.last_policy_residual
            if step == 1:
                first_effective = effective.clone()
                first_teacher = term.teacher_raw.clone()
            if step == 51:
                step51_effective = effective.clone()
            if step == 65:
                step65_effective = effective.clone()
            if step in (51, 65):
                expected_combined = (term.teacher_raw + effective).clamp(-1, 1)
                combined_gap = float((term._raw_actions - expected_combined).abs().max())
                expected_target = term._offset + term._scale * expected_combined
                target_gap = float((term._processed_actions - expected_target).abs().max())
                if step == 51:
                    step51_combined_gap, step51_target_gap = combined_gap, target_gap
                else:
                    step65_combined_gap, step65_target_gap = combined_gap, target_gap
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
            table_frames += table
            drift = torch.linalg.norm((env.scene["cube"].data.root_link_pos_w[:, :2]
                                       - cube_start) * 1000, dim=1)
            max_drift = torch.maximum(max_drift, drift)
            (reward_pre if step <= 65 else reward_contact).add_(reward)
        if first_effective is None or step51_effective is None or step65_effective is None:
            raise RuntimeError("missing staged actions")
        nonarm = [i for i in range(21) if i not in term._arm_indices]
        if (float(first_effective.abs().max()) != 0 or
                float(step51_effective[:, nonarm].abs().max()) != 0 or
                float(step65_effective[:, nonarm].abs().max()) != 0 or
                float(step51_effective.abs().max()) > .005 / 15 + 1e-7 or
                float(step65_effective.abs().max()) > .005 + 1e-7 or
                float(step51_effective[128:, term._arm_indices].abs().max()) == 0 or
                float(step65_effective[128:, term._arm_indices].abs().max()) == 0 or
                step51_combined_gap > 1e-6 or step65_combined_gap > 1e-6 or
                step51_target_gap > 1e-6 or step65_target_gap > 1e-6):
            raise RuntimeError("V14 staged action authority mismatch")
        cohorts = []
        for label, sl in (("zero", slice(0, 128)), ("gaussian_std1", slice(128, 256))):
            cohorts.append({"action_source": label, "worlds": 128,
                            "supported_touch_ge_0p2s": int((longest[sl] >= 10).sum()),
                            "supported_touch_ge_1s": int((longest[sl] >= 50).sum()),
                            "gentle_ge_1s": int(((longest[sl] >= 50) & (max_drift[sl] <= 5)).sum()),
                            "worlds_with_any_supported_frame": int((supported_frames[sl] > 0).sum()),
                            "table_supported_full_episode": int((table_frames[sl] == 200).sum()),
                            "mean_pre_step65_return": round(float(reward_pre[sl].mean()), 5),
                            "mean_post_step65_return": round(float(reward_contact[sl].mean()), 5),
                            "median_peak_drift_mm": round(float(torch.median(max_drift[sl])), 3)})
        report = {"schema": "robot-learning-lab.touch-contact-phase-v14-cuda-preflight.v1",
                  "claim_limit": "Unscored capacity and Gaussian-action diagnostic; no learner update or fresh skill score",
                  "body_sha256": sha(BODY), "method_sha256": sha(args.method),
                  "sentinel_sha256": sha(Path(__file__)), "device": "cuda:0",
                  "gpu": torch.cuda.get_device_name(0), "worlds": 256, "steps": 200,
                  "complete_episode_worlds": 256, "actor_width": obs["actor"].shape[-1],
                  "critic_width": obs["critic"].shape[-1],
                  "unique_cube_xy": int(torch.unique(cube_xy.round(decimals=5), dim=0).shape[0]),
                  "visible_rgb_entries": int(((initial_rgb[:, 3] == 1) & (initial_rgb[:, 5] == 0)).sum()),
                  "partial_reset_kept": partial_kept,
                  "partial_reset_changed_gt_1mm": partial_changed,
                  "first_teacher_abs_max": float(first_teacher.abs().max()),
                  "first_effective_abs_max": float(first_effective.abs().max()),
                  "step51_effective_abs_max": float(step51_effective.abs().max()),
                  "step65_effective_abs_max": float(step65_effective.abs().max()),
                  "step51_combined_raw_max_gap": step51_combined_gap,
                  "step65_combined_raw_max_gap": step65_combined_gap,
                  "step51_executed_target_max_gap": step51_target_gap,
                  "step65_executed_target_max_gap": step65_target_gap,
                  "first_step_s": round(elapsed[0], 3),
                  "median_later_step_s": round(float(np.median(elapsed[1:])), 4),
                  "total_200_step_s": round(sum(elapsed), 3),
                  "rgb_entry_count": term.rgb_entry_count, "rgb_cue_count": term.rgb_cue_count,
                  "cohorts": cohorts}
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        if (report["actor_width"] != 79 or report["visible_rgb_entries"] < 240 or
                not partial_kept or partial_changed < 100 or
                cohorts[0]["supported_touch_ge_0p2s"] < 16 or
                cohorts[1]["supported_touch_ge_0p2s"] < 16):
            raise RuntimeError("V14 capacity, observability or actual-noise contact gate failed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
