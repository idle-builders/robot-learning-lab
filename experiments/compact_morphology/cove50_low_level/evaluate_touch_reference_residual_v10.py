"""Fresh matched 4 s native teacher-reference versus residual-policy visits."""

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

import ape_walk.task  # register native tasks before reach imports grasp
from ape_walk import reach
from collect_free_object_touch import BODY
from free_object_touch_env import ROOT, STAND_TABLE_CUBE_XYZ, physical_touch_gate_frame
from touch_reference_residual_v10 import residual_touch_env_cfg
from train_touch_reference_residual_v10 import build_train_config


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def visit(case: dict, mode: str, checkpoint: Path, agent, out: Path, device: str) -> dict:
    seed, xy = int(case["seed"]), tuple(case["cube_xy_m"])
    cfg = residual_touch_env_cfg(cube_xy=xy)
    cfg.scene.num_envs, cfg.seed, cfg.auto_reset = 1, seed, False
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device),
                            clip_actions=agent.clip_actions)
    native = env.unwrapped
    try:
        native.reset(seed=seed)
        first_qpos = native.sim.data.qpos[0].detach().cpu().numpy().copy()
        first_qvel = native.sim.data.qvel[0].detach().cpu().numpy().copy()
        first_actor = env.get_observations()["actor"][0].detach().cpu().numpy().copy()
        first_cube = native.scene["cube"].data.root_link_pos_w[0].detach().cpu().numpy().copy()
        if mode == "residual":
            runner = reach.ManipulationOnPolicyRunner(env, asdict(agent), device=device)
            runner.load(str(checkpoint), strict=True, map_location=device)
            policy = runner.get_inference_policy(device=device)
        frames = [native.scene["touch_rgb"].data.rgb[0].detach().cpu().numpy().copy()]
        actions, teacher_actions, qpos, qvel, cube = [], [], [], [], []
        gate, table, finger, feet, hand_floor, pelvis_z, rewards = [], [], [], [], [], [], []
        running = longest = 0
        for step in range(1, 201):
            obs = env.get_observations()
            if mode == "residual":
                with torch.inference_mode():
                    action = policy(obs)
            else:
                action = torch.zeros((1, 21), device=device)
            actions.append(action[0].detach().cpu().numpy().copy())
            _, reward, done, _ = env.step(action)
            term = native.action_manager.get_term("joint_pos")
            teacher_actions.append(term.teacher_raw[0].detach().cpu().numpy().copy())
            valid = physical_touch_gate_frame(native, support_z_m=STAND_TABLE_CUBE_XYZ[2])
            running = running + 1 if valid else 0
            longest = max(longest, running)
            gate.append(valid)
            sensors = native.scene.sensors
            finger.append(bool(sensors["right_finger_tip_cube_contact"].data.found[0].any()))
            table.append(bool(sensors["cube_table_contact"].data.found[0].any()))
            feet.append(bool(sensors["feet_ground_contact"].data.found[0].all()))
            hand_floor.append(bool(sensors["hand_ground_contact"].data.found[0].any()))
            pelvis_z.append(float(native.scene["robot"].data.root_link_pos_w[0, 2]))
            qpos.append(native.sim.data.qpos[0].detach().cpu().numpy().copy())
            qvel.append(native.sim.data.qvel[0].detach().cpu().numpy().copy())
            cube.append(native.scene["cube"].data.root_link_pos_w[0].detach().cpu().numpy().copy())
            rewards.append(float(reward[0]))
            frames.append(native.scene["touch_rgb"].data.rgb[0].detach().cpu().numpy().copy())
            if bool(done[0]):
                break
        out.mkdir(parents=True, exist_ok=True)
        stem = f"seed-{seed}-{mode}"
        trace, video = out / f"{stem}.npz", out / f"{stem}.mp4"
        np.savez_compressed(trace, first_qpos=first_qpos, first_qvel=first_qvel,
                            first_actor=first_actor, residual_action=np.asarray(actions),
                            teacher_action=np.asarray(teacher_actions), qpos=np.asarray(qpos),
                            qvel=np.asarray(qvel), cube_xyz_m=np.asarray(cube),
                            physical_touch=np.asarray(gate), table=np.asarray(table),
                            finger=np.asarray(finger), feet=np.asarray(feet),
                            hand_floor=np.asarray(hand_floor), pelvis_z_m=np.asarray(pelvis_z),
                            reward=np.asarray(rewards))
        imageio.mimwrite(video, frames, fps=50, codec="libx264", quality=7,
                         macro_block_size=1)
        drift = np.linalg.norm((np.asarray(cube)[:, :2] - first_cube[:2]) * 1000, axis=1)
        pass_gate = bool(longest >= 10 and len(actions) == 200 and sum(table) >= 190
                         and table[-1] and not any(hand_floor))
        failure = []
        if longest < 10: failure.append("supported_finger_contact_under_0.2s")
        if len(actions) < 200: failure.append("early_termination")
        if sum(table) < 190 or not table[-1]: failure.append("lost_table_support")
        if any(hand_floor): failure.append("hand_floor_contact")
        if float(max(drift)) > 5: failure.append("cube_drift_over_5mm")
        return {"seed": seed, "mode": mode, "cube_xy_m": list(xy),
                "first_qpos_sha256": hashlib.sha256(first_qpos.tobytes()).hexdigest(),
                "first_qvel_sha256": hashlib.sha256(first_qvel.tobytes()).hexdigest(),
                "first_rgb_target": first_actor[-6:].tolist(), "actor_width": len(first_actor),
                "steps": len(actions), "video_frames": len(frames),
                "native_reward_return": round(sum(rewards), 5),
                "longest_supported_touch_s": round(longest * native.step_dt, 3),
                "task_pass": pass_gate, "strict_1s_pass": bool(pass_gate and longest >= 50),
                "gentle_pass": bool(pass_gate and longest >= 50 and float(max(drift)) <= 5),
                "table_frames": sum(table), "table_at_end": table[-1],
                "finger_frames": sum(finger), "both_feet_frames": sum(feet),
                "hand_floor_frames": sum(hand_floor),
                "min_pelvis_z_m": round(min(pelvis_z), 5),
                "peak_cube_xy_drift_mm": round(float(max(drift)), 3),
                "failure_reasons": failure, "trace": str(trace.relative_to(ROOT)),
                "trace_sha256": sha(trace), "video": str(video.relative_to(ROOT)),
                "video_sha256": sha(video)}
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
    assert sha(BODY) == method["body_sha256"]
    assert sha(Path(__file__)) == method["evaluator_sha256"]
    assert sha(Path(__file__).with_name("touch_reference_residual_v10.py")) == method["action_source_sha256"]
    assert sha(args.checkpoint) != method["body_sha256"]
    _, cfg = build_train_config(seed=method["train_seed"])
    agent = cfg.agent
    records = []
    for case in method["development"]:
        for mode in ("reference", "residual"):
            row = visit(case, mode, args.checkpoint, agent, args.out, args.device)
            records.append(row)
            (args.out / "partial.json").write_text(json.dumps({"records": records}, indent=2) + "\n")
            print(json.dumps({k: row[k] for k in ("seed", "mode", "task_pass", "strict_1s_pass", "gentle_pass")}), flush=True)
    matched = {str(c["seed"]): len({r["first_qpos_sha256"] for r in records if r["seed"] == c["seed"]}) == 1
               and len({r["first_qvel_sha256"] for r in records if r["seed"] == c["seed"]}) == 1
               and len({tuple(r["first_rgb_target"]) for r in records if r["seed"] == c["seed"]}) == 1
               for c in method["development"]}
    if not all(matched.values()):
        raise RuntimeError("matched native physical or RGB entries diverged")
    report = {"schema": "robot-learning-lab.touch-reference-residual-v10-development.v1",
              "claim_limit": "Fresh matched native one-pad development score on experimental Direct Frame; reference-guided residual is not standalone skill, acquisition, lift or physical transfer",
              "body_sha256": sha(BODY), "method_sha256": sha(args.method),
              "evaluator_sha256": sha(Path(__file__)), "checkpoint_sha256": sha(args.checkpoint),
              "device": args.device, "matched_entries": matched, "visits": len(records),
              "records": records,
              "passes_by_mode": {mode: sum(r["task_pass"] for r in records if r["mode"] == mode)
                                 for mode in ("reference", "residual")}}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
