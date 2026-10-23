"""Opened-start reward and contact comparison for the V9 residual interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from touch_reference_residual_v9 import residual_touch_env_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = residual_touch_env_cfg(cube_xy=(.049, -.139))
    cfg.scene.num_envs, cfg.seed, cfg.auto_reset = 1, 33007, False
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    names = ("reference", "arm_negative", "arm_positive", "all_positive")
    try:
        term = env.action_manager.get_term("joint_pos")
        arm = [term._target_names.index(x) for x in ("right_shoulder_pitch", "right_elbow_pitch")]
        rows = []
        starts = []
        for name in names:
            obs, _ = env.reset(seed=33007)
            start = env.sim.data.qpos[0].detach().cpu().numpy().copy()
            starts.append(start)
            first_cube = env.scene["cube"].data.root_link_pos_w[0].detach().cpu().numpy().copy()
            action = torch.zeros((1, 21), device=env.device)
            if name in ("arm_negative", "arm_positive"):
                action[0, arm] = -1 if name == "arm_negative" else 1
            elif name == "all_positive":
                action[:] = 1
            total_reward = 0.0
            longest = running = contact_frames = table_frames = hand_floor_frames = 0
            max_drift = 0.0
            steps = 0
            for step in range(200):
                obs, reward, terminated, timed_out, _ = env.step(action)
                total_reward += float(reward[0])
                finger = bool(env.scene.sensors["right_finger_tip_cube_contact"].data.found[0].any())
                feet = bool(env.scene.sensors["feet_ground_contact"].data.found[0].all())
                table = bool(env.scene.sensors["cube_table_contact"].data.found[0].any())
                hand_floor = bool(env.scene.sensors["hand_ground_contact"].data.found[0].any())
                pelvis = float(env.scene["robot"].data.root_link_pos_w[0, 2])
                supported = finger and feet and table and not hand_floor and pelvis >= .10
                running = running + 1 if supported else 0
                longest = max(longest, running)
                contact_frames += supported
                table_frames += table
                hand_floor_frames += hand_floor
                cube = env.scene["cube"].data.root_link_pos_w[0].detach().cpu().numpy()
                max_drift = max(max_drift, float(np.linalg.norm((cube[:2] - first_cube[:2]) * 1000)))
                assert bool(torch.isfinite(obs["actor"]).all() and torch.isfinite(reward).all())
                steps = step + 1
                if bool(terminated[0] or timed_out[0]):
                    break
            rows.append({"mode": name, "steps": steps,
                         "full_native_return": round(total_reward, 5),
                         "supported_touch_s": round(longest * env.step_dt, 3),
                         "supported_frames": contact_frames, "table_frames": table_frames,
                         "hand_floor_frames": hand_floor_frames,
                         "peak_cube_drift_mm": round(max_drift, 3)})
        start_gap = float(max(np.max(np.abs(start - starts[0])) for start in starts))
        assert start_gap < 1e-7
        report = {"schema": "robot-learning-lab.touch-residual-v9-reward-audit.v1",
                  "device": "cpu", "seed": 33007, "cube_xy_m": [.049, -.139],
                  "planned_steps": 200, "entry_max_qpos_gap": start_gap,
                  "action_rate_rule": "MJLab action manager penalizes raw policy residual; native joint acceleration penalty remains active",
                  "rows": rows, "claim_limit": "Opened local reward and residual-authority check; no CUDA score or learner."}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
