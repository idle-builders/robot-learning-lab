"""Pip-specific boundaries around the pinned AGILE sim2mujoco runtime.

LEAPP still owns normalization, history, inference and action processing. AGILE
owns state extraction, command schedules and the CLI loop. This adapter adds
mixed position/velocity outputs and reuses the existing shared electrical law.
"""
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from agile.sim2mujoco.leapp import LeappPolicyController, _flatten_output
from agile.sim2mujoco.simulation import JointCommand, MuJocoSimulation
from lab.electromechanical import ElectromechanicalConfig, ElectromechanicalPack


@dataclass
class MixedJointCommand(JointCommand):
    velocity: torch.Tensor


class PipPolicyController(LeappPolicyController):
    @property
    def action_dim(self):
        return sum(int(b.desc['shape'][-1]) for b in self.output_bindings
                   if b.desc.get('kind', '').startswith('target/'))

    def process(self, state):
        inputs = {b.key: self._build_input(b.desc, state, self.input_devices[b.key])
                  for b in self.input_bindings}
        outputs = self.manager.run_policy(inputs)
        fields = {'target/joint/position': self.default_position.clone(),
                  'target/joint/velocity': torch.zeros_like(self.default_position),
                  'kp': self.default_kp.clone(), 'kd': self.default_kd.clone()}
        targets = []
        for binding in self.output_bindings:
            kind = binding.desc.get('kind')
            if kind not in fields:
                raise ValueError(f'Unsupported actuator output: {binding.desc}')
            values = _flatten_output(outputs[binding.key], self.device)
            self._scatter_joint_output(fields[kind], values, binding.desc)
            if kind.startswith('target/'):
                targets.append(values)
        # These are processed targets, not raw network actions.
        self.last_action = torch.cat(targets).detach().clone()
        return MixedJointCommand(position=fields['target/joint/position'],
            velocity=fields['target/joint/velocity'], kp=fields['kp'], kd=fields['kd'])


class PipSimulation(MuJocoSimulation):
    def __init__(self, *args, contract, electrical, **kwargs):
        self.contract = contract
        super().__init__(*args, **kwargs)
        if self.actuator_type != 'torque' or set(self.joint_names) != set(contract['joint_order']):
            raise ValueError('Expected the sealed 16-joint torque-actuated Pip MJCF')
        joints = {j['name']: j for j in contract['joints']}
        cfg = ElectromechanicalConfig(dt_s=self.physics_dt, delay_steps=electrical['delay_steps'],
            **electrical['pack_rc'], **electrical['servo_proxy'])
        self.pack = ElectromechanicalPack(
            [joints[n]['kp'] for n in self.joint_names],
            [joints[n]['kd'] for n in self.joint_names],
            [joints[n]['effort_limit_nm'] for n in self.joint_names],
            [electrical['servo_proxy']['no_load_speed_rad_s']] * len(self.joint_names), cfg)
        self.electrical_telemetry = None

    def _set_initial_configuration(self, config):
        friction = self.mj_model.dof_frictionloss.copy()
        damping = self.mj_model.dof_damping.copy()
        super()._set_initial_configuration(config)
        self._default_qpos[:3] = self.contract['root_position_m']
        self.mj_model.dof_frictionloss[:] = friction
        self.mj_model.dof_damping[:] = damping

    def reset(self, initial_state=None):
        super().reset(initial_state)
        self.pack.reset()
        self._prime_delay = True
        self.electrical_telemetry = None

    def step(self, command):
        if self._push_sign:
            raise NotImplementedError('Physical push protocol not connected in this bounded comparison')
        state = self.get_state()
        qd = command.position.detach().cpu().numpy()
        vd = command.velocity.detach().cpu().numpy()
        # The executed Isaac actuator fills its delay buffer from the first
        # command after reset. Keep that startup behavior explicit here.
        if self._prime_delay:
            self.pack._queue[:] = np.stack((qd, vd, np.zeros_like(qd)), axis=1)
            self._prime_delay = False
        self.pack.kp[:] = command.kp.detach().cpu().numpy()
        self.pack.kd[:] = command.kd.detach().cpu().numpy()
        self.electrical_telemetry = self.pack.step(state.joint_pos.cpu().numpy(),
            state.joint_vel.cpu().numpy(), qd, vd)
        # The upstream list is joint -> actuator index; scatter rather than
        # gather so a shuffled actuator declaration preserves joint ownership.
        self.mj_data.ctrl[self._joint_to_ctrl_indices] = self.electrical_telemetry['applied_effort_nm']
        mujoco.mj_step(self.mj_model, self.mj_data)
