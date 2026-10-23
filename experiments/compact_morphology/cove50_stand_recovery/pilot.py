"""Paired local native recovery-entry screen; no training or skill promotion."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from ape_walk import task as ape
from ape_walk.physics.frame50 import DIRECT_FRAME_JOINTS, require_direct_frame
from experiments.compact_morphology.cove50_policy_transfer.train_posture_arm import (
    reset_recovery_near_floor, retarget,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = ape.ELBOW_STAND_TASK_ID
BUCKETS = ("upright", "seated", "face_down", "face_up")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def make_env(body: Path, bucket: str, seed: int, worlds: int):
    cfg = load_env_cfg(TASK, play=True)
    retarget(cfg, body, require_direct_frame(body), "stand", variance=False)
    cfg.seed = seed
    cfg.scene.num_envs = worlds
    cfg.sim.mujoco.timestep = 0.002
    cfg.decimation = 10
    cfg.episode_length_s = 8.0
    cfg.auto_reset = False
    cfg.commands["twist"].resampling_time_range = (100.0, 100.0)
    reset = cfg.events["set_ground_state"]
    reset.func = reset_recovery_near_floor
    reset.params.update({
        "face_down_prob": float(bucket == "face_down"),
        "face_up_prob": float(bucket == "face_up"),
        "sitting_prob": float(bucket == "seated"),
        "standing_prob": float(bucket == "upright"),
        "prone_z_min": 0.020,
        "prone_z_max": 0.020,
        "sitting_z_min": 0.188,
        "sitting_z_max": 0.194,
        "face_up_roll_max": math.radians(60),
    })
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        ape.ContactSensorCfg(
            name="self_body_contact",
            primary=ape.ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            secondary=ape.ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            fields=("found", "force"), reduce="none", num_slots=1,
        ),
    )
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=cfg, device="cpu"),
        clip_actions=load_rl_cfg(TASK).clip_actions,
    )
    return cfg, env


def evaluate(body: Path, checkpoint: Path, out: Path, bucket: str, seed: int,
             worlds: int, method: str) -> dict:
    torch.manual_seed(seed)
    cfg, env = make_env(body, bucket, seed, worlds)
    robot = env.unwrapped.scene["robot"]
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    assert tuple(action_term.target_names) == DIRECT_FRAME_JOINTS
    sensors = env.unwrapped.scene.sensors
    obs = env.get_observations()
    assert obs["actor"].shape[1] == 72 and env.unwrapped.action_space.shape[1] == 21
    first_actor_hashes = [row_hash(v) for v in obs["actor"].detach().cpu().numpy()]
    initial_qpos = env.unwrapped.sim.data.qpos.detach().cpu().numpy().copy()
    initial_qvel = env.unwrapped.sim.data.qvel.detach().cpu().numpy().copy()
    if method == "saved_actor":
        runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cpu")
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True,
                    map_location="cpu")
        policy = runner.get_inference_policy(device="cpu")
    else:
        policy = lambda _obs: torch.zeros((worlds, 21), dtype=torch.float32)
    rated = np.asarray([act.effort_limit for act in
                        cfg.scene.entities["robot"].articulation.actuators[:10]])
    traces = [[] for _ in range(worlds)]
    first_actions = []
    first_targets = []
    start = time.monotonic()
    for step in range(400):
        with torch.inference_mode():
            raw_action = policy(obs)
            if step == 0:
                first_actions = raw_action.detach().cpu().numpy().tolist()
            obs, _, done, _ = env.step(raw_action)
        if step == 0:
            first_targets = action_term._processed_actions.detach().cpu().numpy().tolist()
        qpos = robot.data.root_link_pose_w.detach().cpu().numpy()
        xy = robot.data.root_link_pos_w[:, :2].detach().cpu().numpy()
        foot = sensors["feet_ground_contact"].data.found.detach().cpu().numpy()
        hand = sensors["hand_ground_contact"].data.force.norm(dim=-1).detach().cpu().numpy()
        body_contact = sensors["self_body_contact"].data.found.detach().cpu().numpy()
        effort = robot.data.actuator_force[:, :10].abs().detach().cpu().numpy()
        for w in range(worlds):
            quat = qpos[w, 3:7]
            tilt = math.degrees(math.acos(max(-1.0, min(1.0,
                1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)))))
            traces[w].append({
                "t_s": round((step + 1) * 0.02, 2),
                "pelvis_z_m": round(float(qpos[w, 2]), 5),
                "tilt_deg": round(tilt, 3),
                "xy_m": np.round(xy[w], 5).tolist(),
                "feet": [bool(v) for v in foot[w]],
                "hand_support": bool(np.max(hand[w]) >= 1.0),
                "body_contact": bool(np.any(body_contact[w])),
                "rated_leg": bool(np.any(effort[w] >= 0.995 * rated)),
                "mean_leg_effort_nm": round(float(np.mean(effort[w])), 4),
                "done": bool(done[w]),
            })
        if bool(done.all()):
            break
    episodes = []
    for w, rows in enumerate(traces):
        tail = [r for r in rows if r["t_s"] > 7.0]
        drift = float(np.linalg.norm(np.asarray(rows[-1]["xy_m"]) - initial_qpos[w, :2]))
        reached = bool(tail and len(tail) >= 45 and
                       max(r["tilt_deg"] for r in tail) < 20 and
                       np.mean([all(r["feet"]) for r in tail]) >= 0.9 and
                       all(not r["hand_support"] and not r["body_contact"] for r in rows) and
                       all(0.19 <= r["pelvis_z_m"] <= 0.26 for r in tail) and
                       drift <= 0.05 and
                       all(r["tilt_deg"] < 60 for r in rows if r["t_s"] > 1.0))
        episodes.append({
            "world": w, "initial_state_sha256": row_hash(np.r_[initial_qpos[w], initial_qvel[w]]),
            "initial_root_z_m": float(initial_qpos[w, 2]),
            "first_actor_sha256": first_actor_hashes[w],
            "first_raw_action": first_actions[w], "first_processed_target": first_targets[w],
            "duration_s": rows[-1]["t_s"], "recovered": reached,
            "final_drift_m": round(drift, 5),
            "tail_both_feet_fraction": round(float(np.mean([all(r["feet"]) for r in tail])), 3) if tail else None,
            "rated_leg_fraction": round(float(np.mean([r["rated_leg"] for r in rows])), 3),
            "hand_support_frames": sum(r["hand_support"] for r in rows),
            "body_contact_frames": sum(r["body_contact"] for r in rows),
            "trace": rows,
        })
    result = {
        "schema": "robot-learning-lab.cove50-posture-stage-a.v1",
        "method": method, "bucket": bucket, "seed": seed, "worlds": worlds,
        "body": str(body.relative_to(ROOT)), "body_sha256": sha(body),
        "checkpoint": str(checkpoint.relative_to(ROOT)) if method == "saved_actor" else None,
        "checkpoint_sha256": sha(checkpoint) if method == "saved_actor" else None,
        "source": {
            "mjlab": "1.3.0", "physics_dt_s": 0.002, "control_dt_s": 0.02,
            "actor_shape": list(obs["actor"].shape), "action_order": action_term.target_names,
            "motor": "mjlab DcMotorActuator; 12 V, zero command delay; nominal contact",
            "entry": "real native reset; recovery-near-floor correction for prone",
        },
        "elapsed_wall_s": round(time.monotonic() - start, 2),
        "episodes": episodes,
        "claim_limit": "Local CPU native simulator screen; no demonstration, new fit or physical robot qualification.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body", type=Path, default=ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-stand-loweffort198.pt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2201)
    parser.add_argument("--bucket", choices=BUCKETS)
    args = parser.parse_args()
    body = args.body.resolve()
    checkpoint = args.checkpoint.resolve()
    buckets = (args.bucket,) if args.bucket else BUCKETS
    for bucket in buckets:
        for method in ("zero", "saved_actor"):
            result = evaluate(body, checkpoint, args.out / f"{bucket}-{method}.json",
                              bucket, args.seed, args.worlds, method)
            print(bucket, method, sum(e["recovered"] for e in result["episodes"]),
                  "/", len(result["episodes"]), "wall", result["elapsed_wall_s"], flush=True)


if __name__ == "__main__":
    main()
