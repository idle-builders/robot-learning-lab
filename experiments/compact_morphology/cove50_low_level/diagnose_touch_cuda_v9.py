"""One opened seed: explain V9 CUDA reference versus saved local CPU teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import mujoco
from mjlab.envs import ManagerBasedRlEnv

from collect_free_object_touch import BODY
from free_object_touch_env import physical_touch_gate_frame, STAND_TABLE_CUBE_XYZ
from probe_observed_free_object_touch import entry_rgb_calibration
from touch_reference_residual_v9 import residual_touch_env_cfg
from ape_walk.physics.frame50 import require_direct_frame


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cpu-reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    with np.load(args.cpu_reference) as saved:
        cpu = {key: saved[key].copy() for key in ("first_qpos", "first_actor", "action", "qpos", "qvel")}
    cfg = residual_touch_env_cfg(cube_xy=(.049, -.139))
    cfg.scene.num_envs, cfg.seed, cfg.auto_reset = 1, 33007, False
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    model = require_direct_frame(BODY)
    data = mujoco.MjData(model)
    tip = model.geom("right_finger_tip").id
    coefficients, calibration = entry_rgb_calibration()
    try:
        obs, _ = env.reset(seed=33007)
        first_qpos = env.sim.data.qpos[0].detach().cpu().numpy().copy()
        first_actor = obs["actor"][0].detach().cpu().numpy().copy()
        calibrated_xy = (np.array([1, *first_actor[-6:-3]]) @ coefficients).tolist()
        term = env.action_manager.get_term("joint_pos")
        action_gaps, qpos_gaps, qvel_gaps = [], [], []
        samples = []
        running = longest = frames = 0
        for step in range(200):
            obs, reward, terminated, timed_out, _ = env.step(torch.zeros((1, 21), device=env.device))
            action_gaps.append(float(np.max(np.abs(term.teacher_raw[0].detach().cpu().numpy() - cpu["action"][step]))))
            qpos_gaps.append(float(np.max(np.abs(env.sim.data.qpos[0].detach().cpu().numpy() - cpu["qpos"][step]))))
            qvel_gaps.append(float(np.max(np.abs(env.sim.data.qvel[0].detach().cpu().numpy() - cpu["qvel"][step]))))
            valid = physical_touch_gate_frame(env, support_z_m=STAND_TABLE_CUBE_XYZ[2])
            running = running + 1 if valid else 0
            longest = max(longest, running)
            frames += valid
            if step in (0, 24, 49, 68, 99, 149, 199):
                now_qpos = env.sim.data.qpos[0].detach().cpu().numpy().copy()
                data.qpos[:] = now_qpos[:model.nq]
                mujoco.mj_forward(model, data)
                cube_xyz = env.scene["cube"].data.root_link_pos_w[0].detach().cpu().numpy().copy()
                pad_gap = float(np.linalg.norm(data.geom_xpos[tip] - cube_xyz))
                samples.append({"step": step + 1, "teacher_action_gap": action_gaps[-1],
                                "qpos_gap": qpos_gaps[-1], "qvel_gap": qvel_gaps[-1],
                                "pad_cube_center_gap_m": pad_gap,
                                "cube_xyz_m": cube_xyz.tolist(),
                                "rgb": obs["actor"][0, -6:].detach().cpu().tolist(),
                                "teacher_arm_raw": term.teacher_raw[0, term._arm_indices].detach().cpu().tolist(),
                                "right_arm_qpos": [float(now_qpos[model.joint(name).qposadr[0]]) for name in (
                                    "right_shoulder_yaw", "right_shoulder_roll", "right_shoulder_pitch",
                                    "right_elbow_pitch", "right_gripper")],
                                "finger": bool(env.scene.sensors["right_finger_tip_cube_contact"].data.found[0].any()),
                                "table": bool(env.scene.sensors["cube_table_contact"].data.found[0].any())})
            if bool(terminated[0] or timed_out[0]):
                break
        report = {"schema": "robot-learning-lab.touch-v9-cuda-one-world-diagnostic.v1",
                  "seed": 33007, "cube_xy_m": [.049, -.139], "steps": len(action_gaps),
                  "cpu_first_qpos_max_gap": float(np.max(np.abs(first_qpos - cpu["first_qpos"]))),
                  "cpu_first_rgb_max_gap": float(np.max(np.abs(first_actor[-6:] - cpu["first_actor"][-6:]))),
                  "calibrated_entry_xy_m": calibrated_xy,
                  "calibration_source": calibration,
                  "cuda_first_rgb": first_actor[-6:].tolist(), "cpu_first_rgb": cpu["first_actor"][-6:].tolist(),
                  "max_teacher_action_gap": max(action_gaps), "max_qpos_gap": max(qpos_gaps),
                  "max_qvel_gap": max(qvel_gaps), "first_action_gap": action_gaps[0],
                  "longest_supported_touch_s": longest * env.step_dt, "supported_frames": frames,
                  "rgb_entry_count": term.rgb_entry_count, "rgb_cue_count": term.rgb_cue_count,
                  "samples": samples,
                  "claim_limit": "Opened seed cross-runtime diagnostic; CPU reference is not a CUDA score or frozen score replacement."}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("steps", "cpu_first_qpos_max_gap", "cpu_first_rgb_max_gap", "first_action_gap", "max_qpos_gap", "longest_supported_touch_s", "rgb_cue_count")}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
