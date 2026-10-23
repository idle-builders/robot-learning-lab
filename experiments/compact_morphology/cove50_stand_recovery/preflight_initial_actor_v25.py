"""Record the actual first untrained seated PPO action on its training device.

This is an interface diagnostic, not a rollout score or an optimizer update.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.scripts.train import TrainConfig
from mjlab.tasks.registry import load_rl_cfg

from experiments.compact_morphology.cove50_policy_transfer import train_posture_arm as seated


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-xml", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worlds", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--untrained-checkpoint-out", type=Path)
    args = parser.parse_args()

    body = args.body_xml.resolve()
    model = seated.require_direct_frame(body)
    cfg = seated.build_seated_rise_env_cfg(body, model=model)
    agent = deepcopy(load_rl_cfg(seated.TASKS["squat"]))
    agent.experiment_name = "ape_cove50_seated_rise"
    agent.logger = "tensorboard"
    agent.upload_model = False
    agent.seed = args.seed
    task_id = "Mjlab-Cove50-seated-rise-v25-preflight"
    seated.ape.register_mjlab_task(
        task_id=task_id, env_cfg=cfg, play_env_cfg=cfg,
        rl_cfg=agent, runner_cls=seated.ape.VelocityOnPolicyRunner,
    )
    train = TrainConfig.from_task(task_id)
    train.env.scene.num_envs = args.worlds
    train.env.sim.mujoco.timestep = 0.002
    train.env.decimation = 10
    train.env.seed = args.seed
    torch.manual_seed(args.seed)

    env = ManagerBasedRlEnv(cfg=train.env, device=args.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=train.agent.clip_actions)
    runner = seated.ape.VelocityOnPolicyRunner(
        wrapped, asdict(train.agent), None, args.device,
    )
    obs = wrapped.get_observations().to(args.device)
    actor = obs["actor"]
    critic = obs["critic"]
    robot = env.scene["robot"]
    torso = int(robot.indexing.body_ids[robot.body_names.index("torso")])
    soles = [int(robot.indexing.geom_ids[robot.geom_names.index(f"{side}_sole")])
             for side in ("left", "right")]
    initial_qpos = env.sim.data.qpos[0].detach().cpu().tolist()
    initial_qvel = env.sim.data.qvel[0].detach().cpu().tolist()
    initial_physics = {
        "torso_mass_kg": float(env.sim.model.body_mass[0, torso]),
        "sole_slide_friction": [
            float(env.sim.model.geom_friction[0, geom, 0]) for geom in soles
        ],
    }
    initial_motors = {
        "target_names": [act.target_names[0] for act in robot.actuators],
        "effort_factor": [float(act.force_limit[0, 0] / act.default_force_limit[0, 0])
                          for act in robot.actuators],
        "stiffness_factor": [float(act.stiffness[0, 0] / act.default_stiffness[0, 0])
                             for act in robot.actuators],
        "damping_factor": [float(act.damping[0, 0] / act.default_damping[0, 0])
                           for act in robot.actuators],
    }
    with torch.inference_mode():
        deterministic = runner.alg.actor(obs, stochastic_output=False)
        sampled = runner.alg.act(obs)
        _, reward, done, _ = wrapped.step(sampled)
    reload_max_abs_gap = None
    if args.untrained_checkpoint_out is not None:
        args.untrained_checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        # Save the native runner payload without its optional ONNX side export.
        # This runner has no initialized logger because preflight does no learning.
        MjlabOnPolicyRunner.save(runner, str(args.untrained_checkpoint_out))
        runner.load(str(args.untrained_checkpoint_out), load_cfg={"actor": True})
        with torch.inference_mode():
            replayed = runner.alg.actor(obs, stochastic_output=False)
        reload_max_abs_gap = float(torch.max(torch.abs(replayed - deterministic)))
        if reload_max_abs_gap > 1e-6:
            raise RuntimeError(f"Untrained actor checkpoint reload mismatch: {reload_max_abs_gap}")
    processed = env.action_manager.get_term("joint_pos")._processed_actions
    observed_lags = sorted({int(act._delay_buffer.current_lags[0])
                            for act in robot.actuators if act._delay_buffer is not None})
    named = {name: int(env.termination_manager.get_term(name).sum())
             for name in env.termination_manager.active_terms}
    report = {
        "schema": "robot-learning-lab.cove50-seated-initial-actor.v25",
        "claim_limit": "Untrained first actor sample and one step on the training device; no optimization or policy score.",
        "device": args.device,
        "worlds": args.worlds,
        "seed": args.seed,
        "body_sha256": sha(body),
        "trainer_sha256": sha(Path(seated.__file__)),
        "preflight_sha256": sha(Path(__file__)),
        "task": task_id,
        "actor_shape": list(actor.shape),
        "critic_shape": list(critic.shape),
        "sampled_action_shape": list(sampled.shape),
        "first_actor_observation": actor[0].detach().cpu().tolist(),
        "first_critic_observation": critic[0].detach().cpu().tolist(),
        "first_sampled_action": sampled[0].detach().cpu().tolist(),
        "first_deterministic_action": deterministic[0].detach().cpu().tolist(),
        "first_processed_target": processed[0].detach().cpu().tolist(),
        "first_qpos": initial_qpos,
        "first_qvel": initial_qvel,
        "first_physics": initial_physics,
        "first_motors": initial_motors,
        "first_observed_motor_lags": observed_lags,
        "first_reward": float(reward[0]),
        "first_done_count": int(done.sum()),
        "first_named_termination_counts": named,
        "untrained_checkpoint": (str(args.untrained_checkpoint_out)
                                 if args.untrained_checkpoint_out is not None else None),
        "untrained_checkpoint_sha256": (sha(args.untrained_checkpoint_out)
                                        if args.untrained_checkpoint_out is not None else None),
        "same_observation_reload_deterministic_max_abs_gap": reload_max_abs_gap,
        "nonfinite_counts": {
            "actor": int((~torch.isfinite(actor)).sum()),
            "critic": int((~torch.isfinite(critic)).sum()),
            "sampled_action": int((~torch.isfinite(sampled)).sum()),
            "deterministic_action": int((~torch.isfinite(deterministic)).sum()),
            "processed_target": int((~torch.isfinite(processed)).sum()),
            "reward": int((~torch.isfinite(reward)).sum()),
        },
    }
    env.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "actor_shape": report["actor_shape"],
        "critic_shape": report["critic_shape"],
        "first_done_count": report["first_done_count"],
        "nonfinite_counts": report["nonfinite_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
