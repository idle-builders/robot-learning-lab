"""Unscored eight-world fixed-placement V9 camera/reference/contact split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import mujoco
from mjlab.envs import ManagerBasedRlEnv

from collect_free_object_touch import BODY
from probe_observed_free_object_touch import entry_rgb_calibration
from touch_reference_residual_v9 import residual_touch_env_cfg
from ape_walk.physics.frame50 import require_direct_frame


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--zero-spacing", action="store_true")
    args = p.parse_args()
    cfg = residual_touch_env_cfg(cube_xy=(.049, -.139))
    cfg.scene.num_envs, cfg.seed, cfg.auto_reset = 8, 33007, False
    if args.zero_spacing:
        cfg.scene.env_spacing = 0.0
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    model = require_direct_frame(BODY)
    data = mujoco.MjData(model)
    tip = model.geom("right_finger_tip").id
    try:
        obs, _ = env.reset(seed=33007)
        rgb = obs["actor"][:, -6:].detach().cpu().numpy().copy()
        origins = env.scene.env_origins.detach().cpu().numpy().copy()
        cube_first = env.scene["cube"].data.root_link_pos_w.detach().cpu().numpy().copy()
        qpos_first = env.sim.data.qpos.detach().cpu().numpy().copy()
        coefficients, calibration = entry_rgb_calibration()
        estimate = np.column_stack((np.ones(8), rgb[:, :3])) @ coefficients
        actual_xy = cube_first[:, :2] - origins[:, :2]
        term = env.action_manager.get_term("joint_pos")
        running = torch.zeros(8, dtype=torch.int32, device=env.device)
        longest = torch.zeros_like(running)
        finger_frames = torch.zeros_like(running)
        max_drift = torch.zeros(8, device=env.device)
        samples = {}
        for step in range(200):
            obs, reward, terminated, timed_out, _ = env.step(torch.zeros((8, 21), device=env.device))
            sensor = env.scene.sensors
            finger = sensor["right_finger_tip_cube_contact"].data.found.any(dim=-1)
            table = sensor["cube_table_contact"].data.found.any(dim=-1)
            feet = sensor["feet_ground_contact"].data.found.all(dim=-1)
            hand_floor = sensor["hand_ground_contact"].data.found.any(dim=-1)
            pelvis = env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
            valid = finger & table & feet & ~hand_floor & (pelvis >= .10)
            running = torch.where(valid, running + 1, 0)
            longest = torch.maximum(longest, running)
            finger_frames += finger
            drift = torch.linalg.norm((env.scene["cube"].data.root_link_pos_w[:, :2] - torch.as_tensor(cube_first[:, :2], device=env.device)) * 1000, dim=1)
            max_drift = torch.maximum(max_drift, drift)
            if step in (0, 49, 99, 199):
                now_qpos = env.sim.data.qpos.detach().cpu().numpy().copy()
                cube_now = env.scene["cube"].data.root_link_pos_w.detach().cpu().numpy().copy()
                pad_gaps = []
                for i in range(8):
                    data.qpos[:] = now_qpos[i, :model.nq]
                    data.qpos[:3] -= origins[i]
                    mujoco.mj_forward(model, data)
                    pad_gaps.append(float(np.linalg.norm(data.geom_xpos[tip] - (cube_now[i] - origins[i]))))
                samples[str(step + 1)] = {"teacher_arm_raw": term.teacher_raw[:, term._arm_indices].detach().cpu().tolist(),
                                         "robot_root_xyz_local": (now_qpos[:, :3] - origins).tolist(),
                                         "robot_root_quat": now_qpos[:, 3:7].tolist(),
                                         "arm_qpos": [[float(now_qpos[i, model.joint(name).qposadr[0]]) for name in (
                                             "right_shoulder_yaw", "right_shoulder_roll", "right_shoulder_pitch",
                                             "right_elbow_pitch", "right_gripper")] for i in range(8)],
                                         "pad_cube_center_gap_m": pad_gaps,
                                         "rgb": obs["actor"][:, -6:].detach().cpu().tolist(),
                                         "finger": finger.detach().cpu().tolist(),
                                         "table": table.detach().cpu().tolist(),
                                         "feet": feet.detach().cpu().tolist(),
                                         "hand_floor": hand_floor.detach().cpu().tolist(),
                                         "pelvis_z_local_m": pelvis.detach().cpu().tolist()}
            assert bool(torch.isfinite(obs["actor"]).all() and torch.isfinite(reward).all())
            if bool(terminated.any() or timed_out.any()):
                raise RuntimeError(f"batch diagnostic termination at {step + 1}")
        rows = []
        for i in range(8):
            rows.append({"world": i, "entry_root_xyz_local": (qpos_first[i, :3] - origins[i]).tolist(),
                         "cube_xy_local_m": actual_xy[i].tolist(),
                         "entry_rgb": rgb[i].tolist(), "calibrated_xy_m": estimate[i].tolist(),
                         "calibration_error_mm": round(float(np.linalg.norm((estimate[i] - actual_xy[i]) * 1000)), 3),
                         "finger_frames": int(finger_frames[i]),
                         "longest_supported_touch_s": round(float(longest[i] * env.step_dt), 3),
                         "peak_cube_drift_mm": round(float(max_drift[i]), 3)})
        report = {"schema": "robot-learning-lab.touch-v9-batch-diagnostic.v1",
                  "device": args.device,
                  "env_spacing_m": cfg.scene.env_spacing,
                  "seed": 33007, "fixed_cube_xy_m": [.049, -.139], "worlds": 8, "steps": 200,
                  "calibration_source": calibration, "rgb_entry_count": term.rgb_entry_count,
                  "rgb_cue_count": term.rgb_cue_count, "rows": rows, "samples": samples,
                  "claim_limit": "Opened fixed-place eight-world CUDA diagnostic; not an independent contact score."}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"calibration_error_mm": [r["calibration_error_mm"] for r in rows],
                          "touch_s": [r["longest_supported_touch_s"] for r in rows],
                          "rgb_cue_count": report["rgb_cue_count"]}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
