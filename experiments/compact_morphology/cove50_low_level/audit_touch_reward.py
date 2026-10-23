"""Compare full native reward on an opened touch, hover and push-off."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from collect_free_object_touch import BODY
from free_object_touch_env import ROOT, STAND_TABLE_CUBE_XYZ, physical_touch_gate_frame, touch_env_cfg
from probe_observed_free_object_touch import entry_rgb_calibration
from screen_observed_free_object_touch_v4 import action_target, rgb_estimate
from ape_walk.physics.frame50 import require_direct_frame


CASES = (
    (34002, (.048, -.138), "gentle_touch"),
    (34002, (.048, -.138), "hover"),
    (34002, (.048, -.138), "no_hold"),
    (34003, (.052, -.142), "push_off"),
    (34002, (.048, -.138), "zero"),
)


def episode(seed: int, cube_xy: tuple[float, float], mode: str,
            coefficients: np.ndarray) -> dict:
    cfg = touch_env_cfg(BODY, entry_mode="stand-tabletop", stand_cube_xy=cube_xy)
    cfg.scene.num_envs = 1
    cfg.seed = seed
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        obs, _ = env.reset(seed=seed)
        first_qpos = env.sim.data.qpos[0].detach().cpu().numpy().copy()
        first_cube = env.scene["cube"].data.root_link_pos_w[0].detach().cpu().numpy().copy()
        initial_estimate = rgb_estimate(obs["actor"][0, -6:].detach().cpu().numpy(), coefficients)
        assert initial_estimate is not None
        target = action_target(env, require_direct_frame(BODY), initial_estimate)
        term = env.action_manager.get_term("joint_pos")
        offset, scale = term._offset[0], term._scale[0]
        last_rgb = obs["actor"][0, -6:].detach().cpu().numpy().copy()
        hold_alpha = None
        total_reward = 0.0
        longest = running = table_frames = hand_floor_frames = 0
        max_drift = 0.0
        steps = 0
        for step in range(1, 201):
            if mode == "zero":
                action = torch.zeros((1, env.action_space.shape[1]), device=env.device)
            else:
                alpha = min(1.0, step / 75) if hold_alpha is None else hold_alpha
                if mode == "hover":
                    alpha = min(alpha, 0.40)
                desired = offset + alpha * (target - offset)
                raw = torch.where(scale.abs() > 1e-8,
                                  (desired - offset) / scale.clamp_min(1e-8),
                                  torch.zeros_like(desired))
                action = raw.clamp(-1, 1)[None, :]
            obs, reward, terminated, timed_out, _ = env.step(action)
            steps = step
            total_reward += float(reward[0])
            valid = physical_touch_gate_frame(env, support_z_m=STAND_TABLE_CUBE_XYZ[2])
            running = running + 1 if valid else 0
            longest = max(longest, running)
            table_frames += int(bool(env.scene.sensors["cube_table_contact"].data.found[0].any()))
            hand_floor_frames += int(bool(env.scene.sensors["hand_ground_contact"].data.found[0].any()))
            cube = env.scene["cube"].data.root_link_pos_w[0].detach().cpu().numpy()
            max_drift = max(max_drift, float(np.linalg.norm((cube - first_cube)[:2]) * 1000))
            if mode in ("gentle_touch", "push_off"):
                rgb = obs["actor"][0, -6:].detach().cpu().numpy()
                du, dv = rgb[:2] - last_rgb[:2]
                if hold_alpha is None and rgb[3] == 1 and rgb[5] == 0 and du <= -.0015 and dv <= -.0015:
                    hold_alpha = alpha
                last_rgb = rgb.copy()
            if bool(terminated[0]) or bool(timed_out[0]):
                break
        return {"seed": seed, "mode": mode, "cube_xy_m": list(cube_xy),
                "first_qpos_sha256": hashlib.sha256(first_qpos.tobytes()).hexdigest(),
                "steps": steps, "full_native_return": round(total_reward, 5),
                "longest_supported_touch_s": round(longest * env.step_dt, 3),
                "table_contact_frames": table_frames, "hand_floor_frames": hand_floor_frames,
                "peak_cube_xy_drift_mm": round(max_drift, 3),
                "physical_gate": bool(longest >= 10 and table_frames >= 190 and
                                      hand_floor_frames == 0 and steps == 200),
                "strict_1s": bool(longest >= 50 and table_frames >= 190 and
                                  hand_floor_frames == 0 and steps == 200)}
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    coefficients, calibration = entry_rgb_calibration()
    rows = [episode(seed, xy, mode, coefficients) for seed, xy, mode in CASES]
    by_mode = {row["mode"]: row for row in rows}
    report = {"schema": "robot-learning-lab.touch-reward-counterfactual.v1",
              "device": "cpu", "body_sha256": hashlib.sha256(BODY.read_bytes()).hexdigest(),
              "environment_sha256": hashlib.sha256((Path(__file__).parent / "free_object_touch_env.py").read_bytes()).hexdigest(),
              "calibration": calibration, "rows": rows,
              "gentle_ranks_above_hover": by_mode["gentle_touch"]["full_native_return"] > by_mode["hover"]["full_native_return"],
              "gentle_ranks_above_push_off": by_mode["gentle_touch"]["full_native_return"] > by_mode["push_off"]["full_native_return"],
              "gentle_ranks_above_same_start_no_hold": by_mode["gentle_touch"]["full_native_return"] > by_mode["no_hold"]["full_native_return"],
              "claim_limit": "Opened local CPU counterfactuals; full native reward ranking only, no learner or CUDA contact score."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({r["mode"]: [r["full_native_return"], r["longest_supported_touch_s"],
                                  r["peak_cube_xy_drift_mm"]] for r in rows}))
    if (not report["gentle_ranks_above_hover"] or not report["gentle_ranks_above_push_off"]
            or not report["gentle_ranks_above_same_start_no_hold"]):
        raise RuntimeError("Touch reward misranks an opened shortcut")


if __name__ == "__main__":
    main()
