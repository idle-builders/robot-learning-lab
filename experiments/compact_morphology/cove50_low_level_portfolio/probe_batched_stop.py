"""Unscored native gait-to-stop-to-settle warmup and capacity probe.

One world may be compared with an original V2 CUDA trace. Larger world counts
measure the exact prior/router physics workload, not learner throughput or a
new physical score. No policy is fitted here.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
import torch

import ape_walk.task as task
from batched_controller import POSTURE, STOP, WALK, BatchedPortfolioRouter
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import dr
from mjlab.managers import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


ROOT = Path(__file__).resolve().parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=Path, default=ROOT /
                        "artifacts/ape-low-level-training-20261015/modular-portfolio-breadth-v2/method.json")
    parser.add_argument("--source-report", type=Path, default=ROOT /
                        "artifacts/ape-low-level-training-20261015/modular-portfolio-breadth-v2/native/seed-54801-original.json")
    parser.add_argument("--worlds", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare-original", action="store_true")
    args = parser.parse_args()
    if args.worlds < 1 or args.worlds > 4096 or not args.device.startswith("cuda"):
        parser.error("native warmup probe requires 1..4096 CUDA worlds")
    if args.compare_original and args.worlds != 1:
        parser.error("original CUDA prefix comparison requires one world")
    if args.out.exists() or args.out.with_suffix(".npz").exists():
        parser.error("probe outputs already exist")
    method = json.loads(args.method.read_text())
    source = json.loads(args.source_report.read_text())
    for key in ("body", "gait", "stop", "squat", "retarget"):
        path = ROOT / method["sources"][key]
        if sha(path) != method["source_sha256"][key]:
            raise ValueError(f"{key} source differs from V2 frozen method")
    if source["body_sha256"] != method["source_sha256"]["body"]:
        raise ValueError("original and method body hashes differ")
    if source["checkpoint_sha256"]["gait"] != method["source_sha256"]["gait"]:
        raise ValueError("original and method gait hashes differ")
    if source["checkpoint_sha256"]["stop"] != method["source_sha256"]["stop"]:
        raise ValueError("original and method stop hashes differ")
    if source["checkpoint_sha256"]["squat"] != method["source_sha256"]["squat"]:
        raise ValueError("original and method posture hashes differ")
    if source["device"] != "cuda:0" or source["ended_early"]:
        raise ValueError("compare only against a complete original CUDA route")

    sys.path.insert(0, str(ROOT))
    from experiments.compact_morphology.cove50_policy_transfer.train_posture_arm import (
        expose_squat_blend, mask_squat_action_history, retarget,
    )
    from experiments.compact_morphology.cove50_low_level.multicommand_bc import load_student
    from ape_walk.physics.frame50 import (
        clip_frame50_shoulder_targets, configure_frame50_motors,
    )

    body = ROOT / method["assets"]["body"]
    task_id = task.ELBOW_SQUAT_HEADING_OBS_TASK_ID
    cfg = load_env_cfg(task_id, play=True)
    retarget(cfg, body, mujoco.MjModel.from_xml_path(str(body)), "squat", variance=False)
    cfg.events["cove_sole_friction"] = EventTermCfg(
        mode="reset", func=dr.geom_friction,
        params={"asset_cfg": SceneEntityCfg("robot", geom_names=("left_sole", "right_sole")),
                "operation": "abs", "ranges": (source["sole_friction"], source["sole_friction"]),
                "axes": [0], "shared_random": True},
    )
    expose_squat_blend(cfg)
    mask_squat_action_history(cfg)
    configure_frame50_motors(cfg, body, rail_v=source["rail_v"],
                             delay_s=source["delay_ms"] / 1000)
    clip_frame50_shoulder_targets(cfg, body, margin_rad=0.05)
    # Match the original V2 contact sensor inventory even though this probe
    # does not use it to score a new physical episode.
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        task.ContactSensorCfg(
            name="handoff_self_contact",
            primary=task.ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            secondary=task.ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            fields=("found", "force"), reduce="none", num_slots=1,
        ),
        task.ContactSensorCfg(
            name="handoff_body_ground_contact",
            primary=task.ContactMatch(mode="geom", pattern=(
                r"^pelvis_frame_budget$", r"^torso_frame_compute_budget$",
                r"^(left|right)_(thigh|shin|upper_arm|forearm)_frame_budget$",
            ), entity="robot"),
            secondary=task.ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="none", num_slots=1,
        ),
    )
    cfg.scene.num_envs = args.worlds
    cfg.seed = source["seed"]
    cfg.auto_reset = not args.compare_original
    cfg.episode_length_s = 24.5
    cfg.commands["twist"].resampling_time_range = (100.0, 100.0)
    reset = cfg.events["set_ground_state"].params
    reset["standing_prob"], reset["sitting_prob"] = 1.0, 0.0
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=args.device),
                            clip_actions=load_rl_cfg(task_id).clip_actions)
    try:
        action_term = env.unwrapped.action_manager.get_term("joint_pos")
        names = list(action_term.target_names)
        router = BatchedPortfolioRouter(names, action_term.scale, args.worlds, args.device)
        squat_policy = load_student(
            ROOT / method["assets"]["squat"], args.device, names,
            action_term.scale, action_term.offset, "squat")
        gait_task = task.ELBOW_TASK_ID
        gait_cfg = load_env_cfg(gait_task, play=True)
        if not gait_cfg.actions["joint_pos"].use_default_offset or not cfg.actions["joint_pos"].use_default_offset:
            raise ValueError("physical target offset contract changed")
        gait_runner = load_runner_cls(gait_task)(
            env, asdict(load_rl_cfg(gait_task)), device=args.device)
        gait_runner.load(str(ROOT / method["assets"]["gait"]),
                         load_cfg={"actor": True}, strict=True, map_location=args.device)
        gait_policy = gait_runner.get_inference_policy(device=args.device)
        stop_runner = load_runner_cls(gait_task)(
            env, asdict(load_rl_cfg(gait_task)), device=args.device)
        stop_runner.load(str(ROOT / method["assets"]["stop"]),
                         load_cfg={"actor": True}, strict=True, map_location=args.device)
        stop_policy = stop_runner.get_inference_policy(device=args.device)
        command = env.unwrapped.command_manager.get_term("twist")
        dt = float(env.unwrapped.step_dt)
        if abs(dt - 0.02) > 1e-8:
            raise ValueError("saved route requires 50 Hz")
        walk_end = float(source["schedule_s"]["walk_end"])
        stop_end = float(source["schedule_s"]["stop_end"])
        settled_end = float(source["schedule_s"]["squat_start"])
        steps = round(settled_end / dt)
        zero_action = torch.zeros((args.worlds, 21), device=args.device)
        zero_flag = torch.zeros(args.worlds, device=args.device)
        forward = torch.full((args.worlds,), float(source["walk_velocity_m_s"]), device=args.device)
        yaw = torch.full((args.worlds,), float(source["walk_yaw_rate_rad_s"]), device=args.device)
        traces = {key: [] for key in ("pre_qpos", "post_qpos", "actor_obs", "gait_action", "applied_action")}
        initial_qpos = env.unwrapped.sim.data.qpos[0].cpu().numpy().copy()
        initial_qvel = env.unwrapped.sim.data.qvel[0].cpu().numpy().copy()
        done_count = torch.zeros((), dtype=torch.long, device=args.device)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for step in range(steps):
            # The one-world comparison follows gait, stop and settled-posture
            # handoffs from the original source, ending before squat begins.
            elapsed = env.unwrapped.episode_length_buf * dt
            walking = elapsed < walk_end
            stopping = (elapsed >= walk_end) & (elapsed < stop_end)
            mode = torch.where(walking, WALK, torch.where(stopping, STOP, POSTURE))
            command.vel_command_b[:, 0] = torch.where(walking, forward, 0.0)
            command.vel_command_b[:, 2] = torch.where(walking, yaw, 0.0)
            with torch.inference_mode():
                obs = env.get_observations()
                actor_obs = obs["actor"]
                squat_action = squat_policy(obs)
                gait_obs = obs.clone()
                gait_obs["actor"] = router.gait_observation(actor_obs, mode, forward, yaw)
                if args.compare_original:
                    if step * dt < walk_end:
                        prior = gait_policy(gait_obs)
                    elif step * dt < stop_end:
                        prior = stop_policy(gait_obs)
                    else:
                        prior = zero_action
                else:
                    gait_action = gait_policy(gait_obs)
                    stop_action = stop_policy(gait_obs)
                    prior = torch.where(walking[:, None], gait_action,
                                        torch.where(stopping[:, None], stop_action, zero_action))
                action = router.route(mode, squat_action, zero_action, prior, zero_flag)
                if args.compare_original:
                    pre_qpos = env.unwrapped.sim.data.qpos[0].cpu().numpy().copy()
                    actor_obs_0 = actor_obs[0].cpu().numpy().copy()
                    prior_0 = prior[0].cpu().numpy().copy()
                    action_0 = action[0].cpu().numpy().copy()
                _, _, dones, _ = env.step(action)
            if args.compare_original:
                for key, value in (("pre_qpos", pre_qpos),
                                   ("post_qpos", env.unwrapped.sim.data.qpos[0].cpu().numpy()),
                                   ("actor_obs", actor_obs_0),
                                   ("gait_action", prior_0),
                                   ("applied_action", action_0)):
                    traces[key].append(value.copy())
            done_count += dones.sum()
            if args.compare_original:
                if bool(dones.any()):
                    raise ValueError(f"original prefix terminated at step {step}")
            else:
                router.reset_where(dones)
        torch.cuda.synchronize()
        wall_s = time.perf_counter() - start
        parity = None
        parity_pass = None
        if args.compare_original:
            src = np.load(args.source_report.with_suffix(".npz"))
            trace = {key: np.stack(value) for key, value in traces.items()}
            parity = {
                "initial_qpos_max_gap": float(np.max(np.abs(initial_qpos - src["initial_qpos"]))),
                "initial_qvel_max_gap": float(np.max(np.abs(initial_qvel - src["initial_qvel"]))),
                "pre_qpos_max_gap": float(np.max(np.abs(trace["pre_qpos"] - src["pre_qpos"][:steps]))),
                "post_qpos_max_gap": float(np.max(np.abs(trace["post_qpos"] - src["post_qpos"][:steps]))),
                "actor_obs_max_gap": float(np.max(np.abs(trace["actor_obs"] - src["actor_obs"][:steps]))),
                "gait_action_max_gap": float(np.max(np.abs(trace["gait_action"] - src["gait_action"][:steps]))),
                "applied_action_max_gap": float(np.max(np.abs(trace["applied_action"] - src["applied_action"][:steps]))),
            }
            parity_pass = max(parity.values()) <= 1e-4
            args.out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.out.with_suffix(".npz"), **trace)
        report = {
            "schema": "robot-learning-lab.modular-batched-stop-probe.v1",
            "claim_limit": "unscored native warmup/throughput gate; no learner update or physical score",
            "source_report": str(args.source_report),
            "source_report_sha256": sha(args.source_report),
            "body_sha256": sha(body),
            "gait_sha256": sha(ROOT / method["assets"]["gait"]),
            "stop_sha256": sha(ROOT / method["assets"]["stop"]),
            "posture_sha256": sha(ROOT / method["assets"]["squat"]),
            "batched_router_sha256": sha(Path(__file__).with_name("batched_controller.py")),
            "worlds": args.worlds,
            "steps_per_world": steps,
            "native_steps": steps * args.worlds,
            "reset_events": int(done_count.item()),
            "wall_seconds_excluding_constructor": wall_s,
            "native_steps_per_second": steps * args.worlds / wall_s,
            "peak_cuda_allocated_bytes_during_loop": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes_during_loop": torch.cuda.max_memory_reserved(),
            "parity": parity,
            "parity_pass": parity_pass,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: report[key] for key in
                          ("worlds", "native_steps", "reset_events", "native_steps_per_second", "parity")}))
        if parity_pass is False:
            raise ValueError(f"batched warmup diverges from original CUDA source: {parity}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
