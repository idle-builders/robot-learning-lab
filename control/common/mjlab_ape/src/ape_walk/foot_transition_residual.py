"""Reference-centered MJLab motion tracking for the ape foot transition.

The published MJLab tracking task uses absolute joint position actions. This
small action adapter keeps its scene, rewards, observations, PPO runner and
motion command, but applies the policy's correction around the reference pose.
"""

from dataclasses import dataclass
from copy import deepcopy

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .foot_transition_tracking import tracking_env_cfg, runner_cfg


TASK_ID = "Mjlab-Tracking-Ape-Foot-Transition-Residual"


@dataclass(kw_only=True)
class MotionResidualJointActionCfg(JointPositionActionCfg):
    use_default_offset: bool = False

    def build(self, env):
        return MotionResidualJointAction(self, env)


class MotionResidualJointAction(JointPositionAction):
    def apply_actions(self):
        reference = self._env.command_manager.get_term("motion").joint_pos
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        target = reference[:, self._target_ids] + self._processed_actions - encoder_bias
        self._entity.set_joint_position_target(target, joint_ids=self._target_ids)


def env_cfg(play=False):
    cfg = tracking_env_cfg(play=play)
    prior = cfg.actions["joint_pos"]
    cfg.actions["joint_pos"] = MotionResidualJointActionCfg(
        entity_name="robot", actuator_names=prior.actuator_names,
        scale=0.05, use_default_offset=False,
    )
    if not play:
        motion = cfg.commands["motion"]
        motion.pose_range = {
            "x": (-0.005, 0.005), "y": (-0.005, 0.005),
            "roll": (-0.015, 0.015), "pitch": (-0.015, 0.015),
        }
        motion.joint_position_range = (-0.015, 0.015)
    return cfg


rl_cfg = deepcopy(runner_cfg)
rl_cfg.experiment_name = "ape_foot_transition_residual"
rl_cfg.run_name = "reference_centered_v1"
rl_cfg.actor.distribution_cfg["init_std"] = 0.05
rl_cfg.max_iterations = 300
rl_cfg.save_interval = 50

register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=env_cfg(),
    play_env_cfg=env_cfg(play=True),
    rl_cfg=rl_cfg,
    runner_cls=MotionTrackingOnPolicyRunner,
)
