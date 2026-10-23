"""Check V11 zero residual is exactly the V10 reference, then probe capacity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

import ape_walk.task  # noqa: F401 - register native tasks
from collect_free_object_touch import BODY
from touch_reference_residual_v10 import residual_touch_env_cfg as v10_env_cfg
from touch_reference_residual_v11 import residual_touch_env_cfg as v11_env_cfg


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def zero_episode(factory, *, seed: int, steps: int, device: str) -> dict:
    cfg = factory(cube_xy=(.047, -.137))
    cfg.scene.num_envs, cfg.seed, cfg.auto_reset = 1, seed, False
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    try:
        obs, _ = env.reset(seed=seed)
        first = {
            "qpos": env.sim.data.qpos[0].detach().cpu().numpy().copy(),
            "qvel": env.sim.data.qvel[0].detach().cpu().numpy().copy(),
            "actor": obs["actor"][0].detach().cpu().numpy().copy(),
        }
        target, qpos, finger, reward = [], [], [], []
        zero = torch.zeros((1, 21), device=env.device)
        for step in range(steps):
            obs, r, terminated, timed_out, _ = env.step(zero)
            if bool(terminated[0] or timed_out[0]):
                raise RuntimeError(f"Zero-reference parity terminated at step {step + 1}")
            term = env.action_manager.get_term("joint_pos")
            target.append(term._processed_actions[0].detach().cpu().numpy().copy())
            qpos.append(env.sim.data.qpos[0].detach().cpu().numpy().copy())
            finger.append(bool(env.scene.sensors["right_finger_tip_cube_contact"].data.found[0].any()))
            reward.append(float(r[0]))
        return {**first, "target": np.asarray(target), "trajectory": np.asarray(qpos),
                "finger": np.asarray(finger), "reward": np.asarray(reward)}
    finally:
        env.close()


def capacity(*, worlds: int, seed: int, steps: int) -> dict:
    cfg = v11_env_cfg(cube_xy_range=((.046, .054), (-.144, -.136)))
    cfg.scene.num_envs, cfg.seed, cfg.auto_reset = worlds, seed, False
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    try:
        obs, _ = env.reset(seed=seed)
        rgb = obs["actor"][:, -6:]
        visible = int(((rgb[:, 3] == 1) & (rgb[:, 5] == 0)).sum())
        rng = np.random.default_rng(seed)
        # One zero-reference half and one exploratory half under the new mask/ramp.
        raw = np.zeros((worlds, 21), dtype=np.float32)
        raw[worlds // 2:, 16:21] = np.clip(
            rng.normal(0, 1, (worlds - worlds // 2, 5)), -1, 1)
        action = torch.as_tensor(raw, device=env.device)
        running = torch.zeros(worlds, dtype=torch.int32, device=env.device)
        longest = torch.zeros_like(running)
        first_nonarm_target_gap = None
        times = []
        for step in range(steps):
            torch.cuda.synchronize()
            begin = time.monotonic()
            obs, reward, terminated, timed_out, _ = env.step(action)
            torch.cuda.synchronize()
            times.append(time.monotonic() - begin)
            if not bool(torch.isfinite(obs["actor"]).all() and torch.isfinite(reward).all()):
                raise RuntimeError("Nonfinite V11 capacity state")
            if bool(terminated.any() or timed_out.any()):
                raise RuntimeError(f"V11 capacity reset before step {step + 1}")
            sensor = env.scene.sensors
            valid = (sensor["right_finger_tip_cube_contact"].data.found.any(dim=-1)
                     & sensor["feet_ground_contact"].data.found.all(dim=-1)
                     & sensor["cube_table_contact"].data.found.any(dim=-1)
                     & ~sensor["hand_ground_contact"].data.found.any(dim=-1))
            running = torch.where(valid, running + 1, 0)
            longest = torch.maximum(longest, running)
            if step == 0:
                # Full reference targets may differ between placements; compare
                # exploratory worlds to same-world zero replay in parity mode.
                first_nonarm_target_gap = float(action[:, :16].abs().max())
        half = worlds // 2
        return {"worlds": worlds, "steps": steps, "visible_rgb_entries": visible,
                "first_nonarm_action_max": first_nonarm_target_gap,
                "zero_basic_ge_0p2s": int((longest[:half] >= 10).sum()),
                "exploratory_basic_ge_0p2s": int((longest[half:] >= 10).sum()),
                "median_later_step_s": float(np.median(times[1:])),
                "total_step_wall_s": float(sum(times)),
                "gpu": torch.cuda.get_device_name(0)}
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("parity", "capacity"), required=True)
    parser.add_argument("--seed", type=int, default=36102)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--worlds", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "parity":
        old = zero_episode(v10_env_cfg, seed=args.seed, steps=args.steps,
                           device=args.device)
        new = zero_episode(v11_env_cfg, seed=args.seed, steps=args.steps,
                           device=args.device)
        gaps = {key: float(np.max(np.abs(old[key] - new[key])))
                for key in ("qpos", "qvel", "actor", "target", "trajectory", "reward")}
        result = {"mode": "zero_reference_parity", "seed": args.seed,
                  "device": args.device, "steps": args.steps, "max_abs_gaps": gaps,
                  "finger_exact": bool(np.array_equal(old["finger"], new["finger"])),
                  "pass": all(value == 0 for value in gaps.values())
                  and bool(np.array_equal(old["finger"], new["finger"]))}
    else:
        result = {"mode": "cuda_capacity", **capacity(worlds=args.worlds,
                                                   seed=args.seed, steps=args.steps)}
    result.update(schema="robot-learning-lab.touch-v11-preflight.v1",
                  body_sha256=sha(BODY),
                  v11_source_sha256=sha(Path(__file__).with_name("touch_reference_residual_v11.py")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.mode == "parity" and not result["pass"]:
        raise RuntimeError("V11 zero-reference parity failed")


if __name__ == "__main__":
    main()
