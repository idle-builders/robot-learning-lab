"""Motion-imitation PPO for one ape step, with no reference joints in the actor.

MJLab's published tracker provides the training reward and reset distribution.
The small on-policy BC pass follows Pollen's expert-BC pattern: label states
visited by the current actor with the already physics-validated joint target.
At replay the actor receives only phase, IMU, joints, and previous action.
"""

from copy import deepcopy
from dataclasses import dataclass

import torch
from mjlab.envs.mdp import projected_gravity
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.rl import RslRlPpoAlgorithmCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from rsl_rl.algorithms import PPO

from .step import env_cfg as step_env_cfg, foot_rl_cfg


def skill_phase(env):
    motion = env.command_manager.get_term("motion")
    return (motion.time_steps.float() / max(motion.motion.time_step_total - 1, 1)).unsqueeze(-1)


def teacher_joint_target(env):
    return env.command_manager.get_term("motion").joint_pos


def env_cfg(side: str, play: bool = False):
    cfg = step_env_cfg(side, play=play)
    cfg.actions["joint_pos"] = JointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",), scale=1.0,
        use_default_offset=False,
    )
    old = cfg.observations["actor"].terms
    cfg.observations["actor"].terms = {
        "phase": ObservationTermCfg(func=skill_phase),
        "base_lin_vel": old["base_lin_vel"],
        "base_ang_vel": old["base_ang_vel"],
        "projected_gravity": ObservationTermCfg(func=projected_gravity),
        "joint_pos": old["joint_pos"],
        "joint_vel": old["joint_vel"],
        "actions": old["actions"],
    }
    cfg.observations["teacher_target"] = ObservationGroupCfg(
        terms={"joint_pos": ObservationTermCfg(func=teacher_joint_target)},
        concatenate_terms=True, enable_corruption=False,
    )
    return cfg


@dataclass
class ReferenceBcCfg(RslRlPpoAlgorithmCfg):
    bc_learning_rate: float = 3e-4
    bc_epochs: int = 1
    bc_mini_batches: int = 4
    class_name: str = "ape_walk.step_distill.ReferenceBcPPO"


class ReferenceBcPPO(PPO):
    """Pollen-style BC on the student's own rollout, after each native PPO update."""

    def __init__(self, actor, critic, storage, *args, bc_learning_rate=3e-4,
                 bc_epochs=1, bc_mini_batches=4, **kwargs):
        super().__init__(actor, critic, storage, *args, **kwargs)
        self.bc_epochs = bc_epochs
        self.bc_mini_batches = bc_mini_batches
        self.bc_optimizer = torch.optim.Adam(self.actor.parameters(), lr=bc_learning_rate)

    def update(self):
        stats = super().update()
        obs = self.storage.observations.flatten(0, 1)
        target = obs["teacher_target"].detach()
        count = target.shape[0]
        batch_size = max(512, count // self.bc_mini_batches)
        losses = []
        for _ in range(self.bc_epochs):
            for indices in torch.randperm(count, device=target.device).split(batch_size):
                prediction = self.actor(obs[indices])
                loss = (prediction - target[indices]).square().mean()
                self.bc_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.bc_optimizer.step()
                losses.append(loss.detach())
        stats["reference_bc_mse"] = torch.stack(losses).mean().item()
        return stats


class DistilledStepRunner(VelocityOnPolicyRunner):
    """Accept MJLab's tracking metadata without exporting motion in the actor."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu", registry_name=None):
        super().__init__(env, train_cfg, log_dir, device)


for _side in ("left", "right"):
    _rl_cfg = deepcopy(foot_rl_cfg)
    _rl_cfg.experiment_name = f"ape_step_distill_{_side}"
    _rl_cfg.run_name = "on_policy_reference_bc_v1"
    _rl_cfg.algorithm = ReferenceBcCfg(**{
        key: value for key, value in vars(_rl_cfg.algorithm).items()
        if key != "class_name"
    })
    _rl_cfg.max_iterations = 300
    _rl_cfg.save_interval = 25
    _rl_cfg.actor.distribution_cfg["init_std"] = 0.1
    _rl_cfg.upload_model = False
    register_mjlab_task(
        task_id=f"Mjlab-Distilled-Ape-Step-{_side.title()}",
        env_cfg=env_cfg(_side), play_env_cfg=env_cfg(_side, play=True),
        rl_cfg=_rl_cfg, runner_cls=DistilledStepRunner,
    )
