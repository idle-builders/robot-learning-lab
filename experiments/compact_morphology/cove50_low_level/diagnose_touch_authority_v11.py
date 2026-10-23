"""Local, opened-start diagnostic of V10 residual authority; no skill score."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

import ape_walk.task  # noqa: F401 - register task before importing reach
from ape_walk import reach
from collect_free_object_touch import BODY
from free_object_touch_env import STAND_TABLE_CUBE_XYZ, physical_touch_gate_frame
from touch_reference_residual_v10 import residual_touch_env_cfg
from train_touch_reference_residual_v10 import build_train_config


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/ape-low-level-training-20261015/object-touch-residual-v10"
METHOD = BASE / "method.json"
CHECKPOINT = BASE / "model_7.pt"
EXPECTED_CHECKPOINT_SHA = "5a4ad77627a6619bb122f0ede09df334a1cc397b8d2f929cecb674b11944b23c"
ARM = slice(16, 21)
MODES = ("reference", "v10", "arm_only", "delay25", "arm_only_delay25")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(case: dict, mode: str, agent, device: str) -> dict:
    cfg = residual_touch_env_cfg(cube_xy=tuple(case["cube_xy_m"]))
    cfg.scene.num_envs, cfg.seed, cfg.auto_reset = 1, int(case["seed"]), False
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device),
                            clip_actions=agent.clip_actions)
    native = env.unwrapped
    try:
        native.reset(seed=int(case["seed"]))
        qpos = native.sim.data.qpos[0].detach().cpu().numpy().copy()
        qvel = native.sim.data.qvel[0].detach().cpu().numpy().copy()
        rgb = env.get_observations()["actor"][0, -6:].detach().cpu().numpy().copy()
        policy = None
        if mode != "reference":
            runner = reach.ManipulationOnPolicyRunner(env, asdict(agent), device=device)
            runner.load(str(CHECKPOINT), strict=True, map_location=device)
            policy = runner.get_inference_policy(device=device)
        gate, finger, table, rewards = [], [], [], []
        first_raw = first_applied = first_teacher = None
        for step in range(200):
            obs = env.get_observations()
            if policy is None:
                raw = torch.zeros((1, 21), device=device)
            else:
                with torch.inference_mode():
                    raw = policy(obs)
            action = raw.clone()
            if "arm_only" in mode:
                action[:, :ARM.start] = 0
            if "delay25" in mode and step < 25:
                action.zero_()
            _, reward, done, _ = env.step(action)
            term = native.action_manager.get_term("joint_pos")
            if first_raw is None:
                first_raw = raw[0].detach().cpu().numpy().copy()
                first_applied = action[0].detach().cpu().numpy().copy()
                first_teacher = term.teacher_raw[0].detach().cpu().numpy().copy()
            gate.append(physical_touch_gate_frame(native, support_z_m=STAND_TABLE_CUBE_XYZ[2]))
            sensors = native.scene.sensors
            finger.append(bool(sensors["right_finger_tip_cube_contact"].data.found[0].any()))
            table.append(bool(sensors["cube_table_contact"].data.found[0].any()))
            rewards.append(float(reward[0]))
            if bool(done[0]):
                break
        streak = best = 0
        for value in gate:
            streak = streak + 1 if value else 0
            best = max(best, streak)
        discounted = sum(value * 0.99**index for index, value in enumerate(rewards))
        return {
            "seed": int(case["seed"]), "mode": mode, "device": device,
            "source_body_sha256": sha(BODY),
            "first_qpos_sha256": hashlib.sha256(qpos.tobytes()).hexdigest(),
            "first_qvel_sha256": hashlib.sha256(qvel.tobytes()).hexdigest(),
            "first_rgb": rgb.tolist(), "steps": len(rewards),
            "first_raw_arm_norm": float(np.linalg.norm(first_raw[ARM])),
            "first_raw_non_arm_norm": float(np.linalg.norm(first_raw[:ARM.start])),
            "first_applied_arm_norm": float(np.linalg.norm(first_applied[ARM])),
            "first_applied_non_arm_norm": float(np.linalg.norm(first_applied[:ARM.start])),
            "first_teacher_arm_norm": float(np.linalg.norm(first_teacher[ARM])),
            "first_finger_step": next((i + 1 for i, value in enumerate(finger) if value), None),
            "finger_frames": sum(finger), "table_frames": sum(table),
            "longest_supported_touch_s": best * native.step_dt,
            "basic_touch": best >= 10, "discounted_return": discounted,
            "undiscounted_return": sum(rewards),
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[36102, 36103])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    method = json.loads(METHOD.read_text())
    if sha(BODY) != method["body_sha256"] or sha(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise ValueError("V10 body or checkpoint changed")
    cases = {case["seed"]: case for case in method["development"]}
    if set(args.seeds) - cases.keys():
        raise ValueError("Diagnostic must use only opened V10 development seeds")
    _, config = build_train_config(seed=method["train_seed"])
    records = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for mode in args.modes:
            row = run(cases[seed], mode, config.agent, args.device)
            records.append(row)
            args.out.write_text(json.dumps({
                "schema": "robot-learning-lab.touch-v11-local-authority-diagnostic.v1",
                "claim_limit": "Opened-start local runtime mechanism check; not a CUDA score or new policy fit",
                "checkpoint_sha256": sha(CHECKPOINT), "v10_method_sha256": sha(METHOD),
                "records": records}, indent=2) + "\n")
            print(seed, mode, row["basic_touch"], row["longest_supported_touch_s"], flush=True)


if __name__ == "__main__":
    main()
