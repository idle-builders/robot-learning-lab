"""Fresh whole-episode native comparison for the bounded one-pad PPO pilot."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from ape_walk import reach
from ape_walk.physics.frame50 import require_direct_frame
from collect_free_object_touch import BODY
from evaluate_touch_pad_relative_v6 import loaded_actor
from free_object_touch_env import ROOT, STAND_TABLE_CUBE_XYZ, physical_touch_gate_frame, touch_env_cfg
from probe_observed_free_object_touch import entry_rgb_calibration
from screen_observed_free_object_touch_v4 import action_target, rgb_estimate
from touch_pad_relative_v6 import PadRelativeRgb
from train_free_object_touch_ppo import build_train_config


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episode(case: dict, mode: str, checkpoint: Path, method: dict,
            agent, coefficients: np.ndarray, out: Path, device: str) -> dict:
    seed, xy = int(case["seed"]), tuple(case["cube_xy_m"])
    cfg = touch_env_cfg(BODY, entry_mode="stand-tabletop", stand_cube_xy=xy)
    cfg.scene.num_envs = 1
    cfg.seed = seed
    cfg.auto_reset = False
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device),
                            clip_actions=agent.clip_actions)
    native = env.unwrapped
    try:
        native.reset(seed=seed)
        first_qpos = native.sim.data.qpos[0].detach().cpu().numpy().copy()
        first_qvel = native.sim.data.qvel[0].detach().cpu().numpy().copy()
        first_actor = env.get_observations()["actor"][0].detach().cpu().numpy().copy()
        first_cube = native.scene["cube"].data.root_link_pos_w[0].detach().cpu().numpy().copy()
        initial_est = rgb_estimate(first_actor[-6:], coefficients)
        assert initial_est is not None and first_actor[-3] == 1 and first_actor[-1] == 0
        if mode == "ppo":
            runner = reach.ManipulationOnPolicyRunner(env, asdict(agent), device=device)
            runner.load(str(checkpoint), strict=True, map_location=device)
            policy = runner.get_inference_policy(device=device)
        elif mode == "v6_bc":
            bc = ROOT / method["baselines"]["v6_bc_checkpoint"]
            model, mean, std, indices, outputs = loaded_actor(bc)
            adapter = PadRelativeRgb(native)
        elif mode == "teacher":
            target = action_target(native, require_direct_frame(BODY), initial_est)
            term = native.action_manager.get_term("joint_pos")
            offset, scale = term._offset[0], term._scale[0]
            hold_alpha = None
            last_rgb = first_actor[-6:].copy()
        frames = [native.scene["touch_rgb"].data.rgb[0].detach().cpu().numpy().copy()]
        actions, qpos, qvel, cube, valid_touch, rewards = [], [], [], [], [], []
        table, finger, fixed, feet, hand_floor, pelvis_z = [], [], [], [], [], []
        running = longest = 0
        for step in range(1, 201):
            obs = env.get_observations()
            if mode == "ppo":
                with torch.inference_mode():
                    action = policy(obs)
            elif mode == "v6_bc":
                augmented = adapter.augmented(native, obs["actor"][0])
                with torch.inference_mode():
                    prediction = model((torch.as_tensor(augmented[indices]) - mean) / std)
                action = torch.zeros((1, 21), device=device)
                action[0, outputs] = prediction.to(device)
            elif mode == "teacher":
                alpha = min(1.0, step / 75) if hold_alpha is None else hold_alpha
                desired = offset + alpha * (target - offset)
                raw = torch.where(scale.abs() > 1e-8,
                                  (desired - offset) / scale.clamp_min(1e-8),
                                  torch.zeros_like(desired))
                action = raw.clamp(-1, 1)[None, :]
            else:
                action = torch.zeros((1, 21), device=device)
            actions.append(action[0].detach().cpu().numpy().copy())
            _, reward, done, _ = env.step(action)
            if mode == "teacher":
                rgb = env.get_observations()["actor"][0, -6:].detach().cpu().numpy()
                du, dv = rgb[:2] - last_rgb[:2]
                if hold_alpha is None and rgb[3] == 1 and rgb[5] == 0 and du <= -.0015 and dv <= -.0015:
                    hold_alpha = alpha
                last_rgb = rgb.copy()
            gate = physical_touch_gate_frame(native, support_z_m=STAND_TABLE_CUBE_XYZ[2])
            running = running + 1 if gate else 0
            longest = max(longest, running)
            valid_touch.append(gate)
            rewards.append(float(reward[0]))
            sensors = native.scene.sensors
            finger.append(bool(sensors["right_finger_tip_cube_contact"].data.found[0].any()))
            fixed.append(bool(sensors["right_hand_cube_contact"].data.found[0].any()))
            feet.append(bool(sensors["feet_ground_contact"].data.found[0].all()))
            hand_floor.append(bool(sensors["hand_ground_contact"].data.found[0].any()))
            table.append(bool(sensors["cube_table_contact"].data.found[0].any()))
            pelvis_z.append(float(native.scene["robot"].data.root_link_pos_w[0, 2]))
            qpos.append(native.sim.data.qpos[0].detach().cpu().numpy().copy())
            qvel.append(native.sim.data.qvel[0].detach().cpu().numpy().copy())
            cube.append(native.scene["cube"].data.root_link_pos_w[0].detach().cpu().numpy().copy())
            frames.append(native.scene["touch_rgb"].data.rgb[0].detach().cpu().numpy().copy())
            if bool(done[0]):
                break
        stem = f"seed-{seed}-{mode}"
        out.mkdir(parents=True, exist_ok=True)
        trace, video = out / f"{stem}.npz", out / f"{stem}.mp4"
        np.savez_compressed(trace, first_qpos=first_qpos, first_qvel=first_qvel,
                            first_actor=first_actor, action=np.asarray(actions),
                            qpos=np.asarray(qpos), qvel=np.asarray(qvel),
                            cube_xyz_m=np.asarray(cube), valid_touch=np.asarray(valid_touch),
                            reward=np.asarray(rewards), table=np.asarray(table),
                            finger=np.asarray(finger), fixed=np.asarray(fixed),
                            feet=np.asarray(feet), hand_floor=np.asarray(hand_floor),
                            pelvis_z_m=np.asarray(pelvis_z))
        imageio.mimwrite(video, frames, fps=50, codec="libx264", quality=7,
                         macro_block_size=1)
        drift = np.linalg.norm((np.asarray(cube)[:, :2] - first_cube[:2]) * 1000, axis=1)
        pass_gate = bool(longest >= 10 and sum(table) >= 190 and table[-1]
                         and not any(hand_floor) and len(actions) == 200)
        return {"seed": seed, "mode": mode, "cube_xy_m": list(xy),
                "first_qpos_sha256": hashlib.sha256(first_qpos.tobytes()).hexdigest(),
                "first_qvel_sha256": hashlib.sha256(first_qvel.tobytes()).hexdigest(),
                "first_actor_sha256": hashlib.sha256(first_actor.tobytes()).hexdigest(),
                "steps": len(actions), "video_frames": len(frames),
                "native_reward_return": round(sum(rewards), 5),
                "longest_supported_touch_s": round(longest * native.step_dt, 3),
                "task_pass": pass_gate, "strict_1s_pass": bool(pass_gate and longest >= 50),
                "table_frames": sum(table), "table_at_end": table[-1],
                "finger_frames": sum(finger), "fixed_frames": sum(fixed),
                "both_feet_frames": sum(feet), "hand_floor_frames": sum(hand_floor),
                "min_pelvis_z_m": round(min(pelvis_z), 5),
                "peak_cube_xy_drift_mm": round(float(max(drift)), 3),
                "trace": str(trace.relative_to(ROOT)), "trace_sha256": sha(trace),
                "video": str(video.relative_to(ROOT)), "video_sha256": sha(video)}
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    method = json.loads(args.method.read_text())
    assert sha(BODY) == method["source"]["body_sha256"]
    assert sha(Path(__file__).with_name("free_object_touch_env.py")) == method["source"]["environment_sha256"]
    assert sha(Path(__file__)) == method["source"]["evaluator_sha256"]
    assert sha(args.checkpoint) != method["source"].get("body_sha256")
    _, train_cfg = build_train_config(seed=method["train"]["seed"])
    agent = train_cfg.agent
    coefficients, calibration = entry_rgb_calibration()
    records = []
    for case in method["development"]:
        for mode in ("zero", "teacher", "v6_bc", "ppo"):
            row = episode(case, mode, args.checkpoint, method, agent,
                          coefficients, args.out, args.device)
            records.append(row)
            (args.out / "partial.json").write_text(json.dumps({"records": records}, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("seed", "mode", "steps", "task_pass", "strict_1s_pass")}), flush=True)
    pair_identity = {str(case["seed"]): len({r["first_qpos_sha256"] for r in records
                                             if r["seed"] == case["seed"]}) == 1
                     for case in method["development"]}
    if not all(pair_identity.values()):
        raise RuntimeError("Matched initial physical states diverged")
    report = {"schema": "robot-learning-lab.touch-ppo-native-development.v1",
              "body_sha256": sha(BODY), "environment_sha256": method["source"]["environment_sha256"],
              "method_sha256": sha(args.method), "evaluator_sha256": sha(Path(__file__)),
              "checkpoint_sha256": sha(args.checkpoint), "device": args.device,
              "calibration": calibration, "pair_identity": pair_identity,
              "visits": len(records), "records": records,
              "passes_by_mode": {mode: sum(r["task_pass"] for r in records if r["mode"] == mode)
                                 for mode in ("zero", "teacher", "v6_bc", "ppo")},
              "claim_limit": "Fresh source-matched single-pad native simulation, not possession, lift or physical transfer."}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
