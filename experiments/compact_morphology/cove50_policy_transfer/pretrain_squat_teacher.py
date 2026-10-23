"""Distill a motor-feasible squat target controller into the native posture actor.

The teacher is an explicitly scripted position-target probe. This produces a
behavior-cloned policy candidate, not PPO evidence or physical qualification.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity import mdp

from ape_walk import task as ape
from ape_walk.physics.frame50 import clip_frame50_leg_targets
from experiments.compact_morphology.cove50_policy_transfer.train_posture_arm import (
    SQUAT_ACTION_SCALE, SQUAT_COUNTERBALANCE_ARMS, SQUAT_LEGS, retarget,
    expose_squat_blend, mask_squat_action_history,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = ape.ELBOW_SQUAT_HEADING_OBS_TASK_ID


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-xml", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=256)
    parser.add_argument("--collect-steps", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3187)
    parser.add_argument("--tag", default="blendmask")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    body = args.body_xml.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_env_cfg(TASK, play=False)
    retarget(cfg, body, mujoco.MjModel.from_xml_path(str(body)), "squat")
    expose_squat_blend(cfg)
    mask_squat_action_history(cfg)
    clip_frame50_leg_targets(cfg, body, margin_rad=0.02)
    cfg.seed = args.seed
    cfg.scene.num_envs = args.worlds
    cfg.sim.mujoco.timestep = 0.002
    cfg.decimation = 10
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.7,
                "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",))},
    )
    cfg.terminations["hand_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": "hand_ground_contact", "force_threshold": 1.0},
    )
    rl_cfg = load_rl_cfg(TASK)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"),
                              clip_actions=rl_cfg.clip_actions)
    runner = load_runner_cls(TASK)(env, asdict(rl_cfg), device="cuda:0")
    actor = runner.alg.get_policy()
    robot = env.unwrapped.scene["robot"]
    command = env.unwrapped.command_manager.get_term("twist")
    default_pos = robot.data.default_joint_pos
    targets = SQUAT_LEGS | SQUAT_COUNTERBALANCE_ARMS
    samples = []
    labels = []
    terminated = 0
    with torch.inference_mode():
        for step in range(args.collect_steps):
            obs = env.get_observations()
            actor.update_normalization(obs)
            samples.append(torch.cat([obs[group] for group in actor.obs_groups], dim=-1).clone())
            action = torch.zeros_like(robot.data.joint_pos)
            alpha = command.alpha
            for index, target in targets.items():
                action[:, index] = alpha * (target - default_pos[:, index]) / SQUAT_ACTION_SCALE[index]
            labels.append(action.clone())
            _, _, done, _ = env.step(action)
            terminated += int(done.sum())
            if (step + 1) % 100 == 0:
                print(f"collected {step + 1}/{args.collect_steps} steps, terminations {terminated}", flush=True)
    inputs = torch.cat(samples)
    target_actions = torch.cat(labels)
    del samples, labels
    actor.train()
    optimizer = torch.optim.Adam(actor.mlp.parameters(), lr=1e-3)
    weights = torch.ones(21, device=inputs.device)
    weights[list(targets)] = 3.0
    batch_size = 4096
    losses = []
    for epoch in range(args.epochs):
        order = torch.randperm(inputs.shape[0], device=inputs.device)
        total_loss = 0.0
        count = 0
        for indices in order.split(batch_size):
            latent = actor.obs_normalizer(inputs[indices])
            output = actor.distribution.deterministic_output(actor.mlp(latent))
            loss = (((output - target_actions[indices]) ** 2) * weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.mlp.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
            count += len(indices)
        losses.append(round(total_loss / count, 7))
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"epoch {epoch + 1}/{args.epochs} weighted_mse {losses[-1]}", flush=True)
    saved = runner.alg.save()
    saved["iter"] = 0
    saved["infos"] = {"teacher": "scripted_native_cove_squat_target",
                      "method": "behavior_cloning", "seed": args.seed}
    checkpoint = out / f"squat-teacher-bc-{args.tag}.pt"
    torch.save(saved, checkpoint)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "schema": "robot-learning-lab.cove50-squat-teacher-bc.v1",
        "method": "behavior_cloning_of_scripted_joint_target_teacher",
        "task": TASK,
        "selected_design_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "body_xml": str(body.relative_to(ROOT)),
        "body_sha256": sha(body),
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha(checkpoint),
        "seed": args.seed,
        "worlds": args.worlds,
        "collect_steps": args.collect_steps,
        "observations": len(inputs),
        "terminated_teacher_episodes": terminated,
        "epochs": args.epochs,
        "weighted_mse": losses,
        "action_target": "alpha-blended stand to feasible squat pose in 21-axis native action coordinates",
        "actor_observation": "21-axis native posture observations with target blend in the x command slot and zeroed previous-action channels",
        "training_variance": "retarget() non-accumulating sampled mass, grip, effort, PD and lag sensitivity",
        "qualification": "Training loss and scripted teacher are not closed-loop policy success; evaluate saved BC actor on reserved native simulations",
    }
    (out / f"squat-teacher-bc-{args.tag}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "weighted_mse"}, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    main()
