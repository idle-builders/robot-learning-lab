"""Bind upstream MJLab manager terms to the exact batched R3 boundary."""
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import torch

from config import Interface
from mjlab.managers import ObservationTermCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg


def runtime_package():
    name = 'cove50_r3_runtime'
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / 'runtime'
        spec = importlib.util.spec_from_file_location(name, path / '__init__.py', submodule_search_locations=[str(path)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class MappedAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.interface = Interface()
        self._raw = torch.zeros(env.num_envs, 24, device=self.device)
        self.requested_target = self.interface.target(self._raw)
        self.ids = torch.tensor([self._entity.joint_names.index(n) for n in self.interface.current_names], device=self.device)

    @property
    def action_dim(self): return 24

    @property
    def raw_action(self): return self._raw

    def process_actions(self, actions):
        self._raw[:] = actions
        rt = runtime_package().get_runtime(self._env)
        # A newly reset controller waits one control period for an actual
        # delayed packet. Keep this hold inside the episode and transition
        # accounting; retain the sampled raw output separately as history.
        home = self.interface.target(torch.zeros_like(actions))
        self.requested_target[:] = torch.where(rt.valid[:, None], self.interface.target(actions), home)

    def apply_actions(self):
        runtime_package().get_runtime(self._env).arm_physics_step()
        self._entity.set_joint_position_target(self.requested_target, joint_ids=self.ids)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0
        self.requested_target[ids] = self.interface.target(self._raw[ids])


@dataclass(kw_only=True)
class MappedActionCfg(ActionTermCfg):
    def build(self, env): return MappedAction(self, env)


def mapped_values(env):
    rt = runtime_package().get_runtime(env)
    term = env.action_manager.get_term('joint_pos')
    values = rt.last_values
    return term.interface.observe(values[:, :22], values[:, 22:44], values[:, 44:47],
        values[:, 47:50], env.action_manager.action,
        env.command_manager.get_command('twist'))


def native_mapped_values(env):
    """Privileged source features use native truth, independently of actor delay."""
    term = env.action_manager.get_term('joint_pos')
    data = env.scene['robot'].data
    return term.interface.observe(data.joint_pos[:, term.ids], data.joint_vel[:, term.ids],
        data.root_link_ang_vel_b, data.projected_gravity_b, env.action_manager.action,
        env.command_manager.get_command('twist'))


def part(env, start, stop, native=False):
    return (native_mapped_values(env) if native else mapped_values(env))[:, start:stop]


def critic_linvel(env):
    # Saved privileged velocimeter was at pelvis-local (0,0,.065), identity
    # orientation. Preserve that feature meaning as a virtual point on R3's
    # root; this changes no physical site, mass or sensor available to actor.
    data = env.scene['robot'].data
    offset = data.root_link_lin_vel_b.new_tensor([0., 0., .065]).expand_as(data.root_link_ang_vel_b)
    return data.root_link_lin_vel_b + torch.cross(data.root_link_ang_vel_b, offset, dim=-1)


def legacy_action_rate(env):
    """Same source21 squared raw-action difference; new heads stay separate."""
    return torch.sum(torch.square(env.action_manager.action[:, :21] - env.action_manager.prev_action[:, :21]), dim=1)


def bind(cfg):
    cfg.actions = {'joint_pos': MappedActionCfg(entity_name='robot')}
    slices = dict(base_ang_vel=(0, 3), projected_gravity=(3, 6), joint_pos=(6, 27),
        joint_vel=(27, 48), actions=(48, 69), command=(69, 72))
    for group_name, group in cfg.observations.items():
        for name, (start, stop) in slices.items():
            group.terms[name].func = part
            group.terms[name].params = dict(start=start, stop=stop, native=group_name == 'critic')
        group.terms['new_axes'] = ObservationTermCfg(func=part, params=dict(start=72, stop=81, native=group_name == 'critic'))
    cfg.observations['critic'].terms['base_lin_vel'].func = critic_linvel
    cfg.observations['critic'].terms['base_lin_vel'].params = {}
    cfg.rewards['action_rate_l2'].func = legacy_action_rate
    return cfg
