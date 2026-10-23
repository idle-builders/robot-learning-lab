"""One bounded parallel PPO pilot over saved gait→stop→settled-posture physics.

The first paid launch requires a source-matched native warmup parity report.
This script has not trained a checkpoint until its own report says completed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import mujoco
import torch

import ape_walk.task as task
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import dr
from mjlab.managers import EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from stop_feedback_env import StopFeedbackVecEnv
from stop_feedback_reward import (
    SettledStopTerminal, forbidden_contact_penalty, low_settle_speed,
    rated_leg_penalty, residual_magnitude, supported_settle,
)
from stop_feedback_rail import sample_cove_rail


ROOT = Path(__file__).resolve().parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", type=Path, default=ROOT /
                   "artifacts/ape-low-level-training-20261015/modular-portfolio-breadth-v2/method.json")
    p.add_argument("--native-parity-report", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--worlds", type=int, default=512)
    p.add_argument("--updates", type=int, default=16)
    p.add_argument("--seed", type=int, default=61001)
    p.add_argument("--residual-gain", type=float, default=0.15)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--smoke-only", action="store_true",
                   help="run one unscored full native episode with zero residual and no PPO update")
    args = p.parse_args()
    if not args.device.startswith("cuda") or not 64 <= args.worlds <= 4096 or not 1 <= args.updates <= 64:
        p.error("bounded pilot requires CUDA, 64..4096 worlds and 1..64 PPO updates")
    if args.out.exists():
        p.error("output directory already exists")
    parity = json.loads(args.native_parity_report.read_text())
    if (parity.get("schema") != "robot-learning-lab.modular-batched-stop-probe.v1" or
            parity.get("worlds") != 1 or parity.get("steps_per_world") != 300 or
            parity.get("parity_pass") is not True):
        raise ValueError("exact six-second native V2 CUDA warmup parity is required")
    method = json.loads(args.method.read_text())
    for key in ("body", "gait", "stop", "squat", "retarget"):
        path = ROOT / method["sources"][key]
        if sha(path) != method["source_sha256"][key]:
            raise ValueError(f"{key} differs from frozen V2 source")
    for parity_key, method_key in (("body_sha256", "body"),
                                    ("gait_sha256", "gait"),
                                    ("stop_sha256", "stop"),
                                    ("posture_sha256", "squat")):
        if parity.get(parity_key) != method["source_sha256"][method_key]:
            raise ValueError(f"native parity used a different {method_key} source")
    if parity.get("batched_router_sha256") != sha(Path(__file__).with_name("batched_controller.py")):
        raise ValueError("native parity used a different batched router")
    original = ROOT / "artifacts/ape-low-level-training-20261015/modular-portfolio-breadth-v2/native/seed-54801-original.json"
    if parity.get("source_report_sha256") != sha(original):
        raise ValueError("native parity did not compare with the pinned V2 original")
    reward_audit = ROOT / "artifacts/ape-low-level-training-20261015/modular-stop-feedback-prep/reward-alignment-audit.json"
    audited = json.loads(reward_audit.read_text())
    if (audited.get("source_evaluator_sha256") != method["source_sha256"]["evaluator"] or
            [audited.get(k) for k in ("original_reports", "stop_window_pass",
                                      "later_posture_window_pass", "whole_sequence_pass")]
            != [16, 8, 16, 6]):
        raise ValueError("saved stop reward-window audit changed")
    router_parity = json.loads((ROOT /
        "artifacts/ape-low-level-training-20261015/modular-stop-feedback-prep/router-parity.json").read_text())
    if router_parity["result"] != "pass" or router_parity["max_action_gap"] != 0:
        raise ValueError("batched router scalar parity is required")

    sys.path.insert(0, str(ROOT))
    from experiments.compact_morphology.cove50_low_level.multicommand_bc import load_student
    from experiments.compact_morphology.cove50_policy_transfer.train_posture_arm import (
        expose_squat_blend, mask_squat_action_history, retarget,
    )
    from ape_walk.physics.frame50 import (
        clip_frame50_shoulder_targets, configure_frame50_motors,
    )

    body = ROOT / method["assets"]["body"]
    model = mujoco.MjModel.from_xml_path(str(body))
    task_id = task.ELBOW_SQUAT_HEADING_OBS_TASK_ID
    cfg = load_env_cfg(task_id, play=True)
    retarget(cfg, body, model, "squat", variance=False)
    expose_squat_blend(cfg)
    mask_squat_action_history(cfg)
    configure_frame50_motors(cfg, body, rail_v=12.0, delay_s=0.0)
    clip_frame50_shoulder_targets(cfg, body, margin_rad=0.05)
    robot_cfg = cfg.scene.entities["robot"]
    robot_cfg.articulation.actuators = tuple(
        replace(act, delay_min_lag=0, delay_max_lag=4)
        for act in robot_cfg.articulation.actuators)
    bodies = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                   for i in range(1, model.nbody))
    cfg.events["cove_link_density"] = EventTermCfg(
        mode="reset", func=dr.pseudo_inertia,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=bodies),
                "alpha_range": (math.log(0.9) / 2, math.log(1.1) / 2)},
    )
    cfg.events["cove_sole_friction"] = EventTermCfg(
        mode="reset", func=dr.geom_friction,
        params={"asset_cfg": SceneEntityCfg("robot", geom_names=("left_sole", "right_sole")),
                "operation": "abs", "ranges": (0.65, 1.15),
                "axes": [0], "shared_random": True},
    )
    cfg.events["cove_rail"] = EventTermCfg(
        mode="reset", func=sample_cove_rail,
        params={"low_v": 10.0, "high_v": 12.0},
    )
    cfg.events["cove_pd"] = EventTermCfg(
        mode="reset", func=dr.pd_gains,
        params={"asset_cfg": SceneEntityCfg("robot"),
                "kp_range": (0.9, 1.1), "kd_range": (0.9, 1.1),
                "operation": "scale"},
    )
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
    cfg.rewards = {
        "supported_settle": RewardTermCfg(func=supported_settle, weight=3.0),
        "low_settle_speed": RewardTermCfg(func=low_settle_speed, weight=1.5,
                                           params={"target_m_s": 0.06}),
        "rated_leg_penalty": RewardTermCfg(func=rated_leg_penalty, weight=-2.0),
        "forbidden_contact_penalty": RewardTermCfg(func=forbidden_contact_penalty,
                                                    weight=-5.0),
        "residual_magnitude": RewardTermCfg(func=residual_magnitude, weight=-0.05),
        "settled_terminal": RewardTermCfg(func=SettledStopTerminal, weight=100.0),
    }
    cfg.curriculum = {}
    cfg.scene.num_envs = args.worlds
    cfg.seed = args.seed
    cfg.auto_reset = True
    cfg.episode_length_s = 6.5
    cfg.commands["twist"].resampling_time_range = (100.0, 100.0)
    cfg.events["set_ground_state"].params.update({"standing_prob": 1.0, "sitting_prob": 0.0})

    agent = load_rl_cfg(task_id)
    agent.actor.distribution_cfg["init_std"] = 0.05
    agent.experiment_name = "cove50_modular_stop_feedback_v1"
    agent.max_iterations = args.updates
    agent.save_interval = max(1, min(4, args.updates))
    agent.resume = False
    agent.logger = "tensorboard"
    args.out.mkdir(parents=True)
    manifest = {
        "schema": "robot-learning-lab.modular-stop-feedback-v1.train.v1",
        "status": "started",
        "claim_limit": "source-labelled experimental Direct Frame stop residual; no independent physical score",
        "body_sha256": sha(body),
        "source_actors_sha256": {key: sha(ROOT / method["assets"][key])
                                  for key in ("gait", "stop", "squat")},
        "training_source_sha256": {name: sha(Path(__file__).with_name(name)) for name in
                                   ("batched_controller.py", "stop_feedback_env.py",
                                    "stop_feedback_reward.py", "stop_feedback_rail.py",
                                    "train_stop_feedback_v1.py")},
        "native_parity_report": str(args.native_parity_report),
        "native_parity_report_sha256": sha(args.native_parity_report),
        "reward_alignment_audit_sha256": sha(reward_audit),
        "worlds": args.worlds,
        "updates_planned": args.updates,
        "smoke_only": args.smoke_only,
        "steps_per_update_per_world": agent.num_steps_per_env,
        "residual_gain": args.residual_gain,
        "reward_weights": {k: v.weight for k, v in cfg.rewards.items()},
        "configured_variation": {"speed_m_s": [0.20, 0.35],
                                 "walk_s": [1.5, 2.5], "stop_s": [1.5, 2.0],
                                 "link_inertia_scale": [0.9, 1.1],
                                 "sole_friction": [0.65, 1.15],
                                 "feetech_rail_v": [10.0, 12.0],
                                 "pd_scale": [0.9, 1.1],
                                 "motor_lag_steps": [0, 4]},
    }
    report_path = args.out / "train-report.json"
    report_path.write_text(json.dumps(manifest, indent=2) + "\n")
    env = None
    started = time.perf_counter()
    try:
        raw = ManagerBasedRlEnv(cfg=cfg, device=args.device)
        term = raw.action_manager.get_term("joint_pos")
        env = StopFeedbackVecEnv(
            raw, clip_actions=agent.clip_actions,
            target_names=list(term.target_names), action_scale=term.scale,
            seed=args.seed, residual_gain=args.residual_gain)
        gait_task = task.ELBOW_TASK_ID
        gait_runner = load_runner_cls(gait_task)(
            env, asdict(load_rl_cfg(gait_task)), device=args.device)
        gait_runner.load(str(ROOT / method["assets"]["gait"]),
                         load_cfg={"actor": True}, strict=True, map_location=args.device)
        stop_runner = load_runner_cls(gait_task)(
            env, asdict(load_rl_cfg(gait_task)), device=args.device)
        stop_runner.load(str(ROOT / method["assets"]["stop"]),
                         load_cfg={"actor": True}, strict=True, map_location=args.device)
        posture_policy = load_student(
            ROOT / method["assets"]["squat"], args.device, list(term.target_names),
            term.scale, term.offset, "squat")
        env.set_priors(gait_runner.get_inference_policy(device=args.device),
                       stop_runner.get_inference_policy(device=args.device),
                       posture_policy)
        if args.smoke_only:
            torch.cuda.reset_peak_memory_stats()
            finite = torch.ones((), dtype=torch.bool, device=args.device)
            for _ in range(350):
                obs, reward, _, _ = env.step(env.zero_action)
                finite &= torch.isfinite(reward).all() & torch.isfinite(obs["actor"]).all()
            terminal = raw.reward_manager.get_term_cfg("settled_terminal").func
            torch.cuda.synchronize()
            if not bool(finite) or int(env.reset_count.item()) == 0 or int(terminal.total_checks.item()) == 0:
                raise ValueError("native stop-feedback smoke lacked finite rollout, reset or stop check")
            manifest.update({
                "status": "smoke_completed",
                "updates_completed": 0,
                "nominal_transitions": 0,
                "unscored_native_steps": 350 * args.worlds,
                "episode_resets": int(env.reset_count.item()),
                "stop_window_checks": int(terminal.total_checks.item()),
                "stop_window_proxy_payments": int(terminal.total_payments.item()),
                "sampled_start_count": int(env.sample_count.item()),
                "sampled_speed_walk_stop_min": env.sample_min.cpu().tolist(),
                "sampled_speed_walk_stop_max": env.sample_max.cpu().tolist(),
                "sampled_rail_v_min": float(raw.stop_feedback_rail_min.item()),
                "sampled_rail_v_max": float(raw.stop_feedback_rail_max.item()),
                "sampled_rail_count": int(raw.stop_feedback_rail_samples.item()),
                "sampled_leg_continuous_nm_min": float(raw.stop_feedback_leg_rating_min.item()),
                "sampled_leg_continuous_nm_max": float(raw.stop_feedback_leg_rating_max.item()),
                "wall_seconds_including_constructor": time.perf_counter() - started,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            })
            return
        runner = load_runner_cls(task_id)(env, asdict(agent), str(args.out), device=args.device)
        torch.cuda.reset_peak_memory_stats()
        # Random episode offsets would skip the required physical gait warmup.
        runner.learn(num_learning_iterations=args.updates, init_at_random_ep_len=False)
        torch.cuda.synchronize()
        manifest.update({
            "status": "completed",
            "updates_completed": args.updates,
            "nominal_transitions": args.worlds * agent.num_steps_per_env * args.updates,
            "wall_seconds_including_constructor": time.perf_counter() - started,
            "episode_resets": int(env.reset_count.item()),
            "sampled_start_count": int(env.sample_count.item()),
            "sampled_speed_walk_stop_min": env.sample_min.cpu().tolist(),
            "sampled_speed_walk_stop_max": env.sample_max.cpu().tolist(),
            "sampled_rail_v_min": float(raw.stop_feedback_rail_min.item()),
            "sampled_rail_v_max": float(raw.stop_feedback_rail_max.item()),
            "sampled_rail_count": int(raw.stop_feedback_rail_samples.item()),
            "sampled_leg_continuous_nm_min": float(raw.stop_feedback_leg_rating_min.item()),
            "sampled_leg_continuous_nm_max": float(raw.stop_feedback_leg_rating_max.item()),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "stop_window_checks": int(raw.reward_manager.get_term_cfg("settled_terminal").func.total_checks.item()),
            "stop_window_proxy_payments": int(raw.reward_manager.get_term_cfg("settled_terminal").func.total_payments.item()),
            "checkpoint_paths": [str(x) for x in sorted(args.out.glob("model_*.pt"))],
        })
    except Exception as exc:
        manifest.update({"status": "failed", "error_type": type(exc).__name__,
                         "error": str(exc),
                         "wall_seconds_including_constructor": time.perf_counter() - started})
        raise
    finally:
        report_path.write_text(json.dumps(manifest, indent=2) + "\n")
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
