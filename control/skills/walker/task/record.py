"""Retain each complete/censored episode, including terminal transitions.

Control-rate native snapshots (50 Hz), not every 4 kHz integration substep.
The input observation is paired with its sampled raw action and reached state.
"""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from mjlab.managers.recorder_manager import RecorderTerm
from boundary import runtime_package


class Episodes(RecorderTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.directory = Path(cfg.params['directory'])
        self.directory.mkdir(parents=True, exist_ok=True)
        self.buffers = {}; self.initial = {}; self.initial_velocity = {}; self.previous_obs = {}; self.previous_packet = {}; self.serial = 0
        self.transitions = 0; self.completed = 0; self.censored = 0
        self.manifest = self.directory / 'episodes.jsonl'
        self.inputs = None

    def cache_inputs(self, value):
        """One batched host copy before the step, including terminal pairs."""
        env = self._env; rt = runtime_package().get_runtime(env)
        tensors = dict(input_observation81=env.obs_buf['actor'], input_privileged_observation96=env.obs_buf['critic'],
            input_value_prediction=value, input_sensor_valid=rt.valid,
            input_sensor_capture_s=rt.capture_time_s, input_imu_capture_s=rt.imu_capture_time_s)
        self.inputs = {k:v.detach().cpu().numpy().copy() for k,v in tensors.items()}

    def record_post_reset(self, env_ids):
        qpos = self._env.sim.data.qpos[env_ids].cpu().numpy().copy()
        qvel = self._env.sim.data.qvel[env_ids].cpu().numpy().copy()
        for j,i in enumerate(env_ids.cpu().tolist()):
            if i in self.buffers and self.buffers[i]: self.finish(i, 'external_reset_censoring')
            self.buffers[i] = defaultdict(list)
            self.initial[i] = qpos[j]
            self.initial_velocity[i] = qvel[j]

    def capture(self, ids):
        env = self._env; rt = runtime_package().get_runtime(env)
        term = env.action_manager.get_term('joint_pos')
        feet = env.scene['feet_ground_contact'].data
        hand = env.scene['hand_ground_contact'].data
        values = dict(qpos=env.sim.data.qpos, qvel=env.sim.data.qvel,
            root_velocity_body_m_s=env.scene['robot'].data.root_link_lin_vel_b,
            root_ang_velocity_body_rad_s=env.scene['robot'].data.root_link_ang_vel_b,
            raw_action24=env.action_manager.action, requested_target22_rad=term.requested_target,
            accepted_target22_rad=rt.previous_command, applied_target22_rad=rt.target,
            torque22_Nm=rt.torque, current22_A=rt.current, saturated22=rt.saturated,
            command3=env.command_manager.get_command('twist'),
            feet_force_N=feet.force, feet_found=feet.found, distal_forearm_force_N=hand.force,
            sensor_valid=rt.valid, sensor_capture_s=rt.capture_time_s,
            imu_capture_s=rt.imu_capture_time_s, runtime_time_s=rt.tick * rt.physics_dt_s,
            native_time_s=env.sim.data.time,
            delayed_values50=rt.last_values, reward=env.reward_buf,
            reward_terms=env.reward_manager._step_reward,
            source_terminated=env.reset_terminated, source_timeout=env.reset_time_outs)
        # One batched copy per field; retain every world, never select by reward.
        arrays = {k: v[ids].detach().cpu().numpy().copy() for k, v in values.items()}
        for j, i in enumerate(ids.cpu().tolist()):
            if self.inputs is None: raise RuntimeError('Missing pre-action actor/value observation capture')
            for key,value in self.inputs.items(): self.buffers[i][key].append(value[i])
            for key, value in arrays.items(): self.buffers[i][key].append(value[j])
            self.transitions += 1

    def record_pre_reset(self, env_ids):
        self.capture(env_ids)
        for i in env_ids.cpu().tolist():
            reason = 'source_timeout' if self._env.reset_time_outs[i] else 'source_physical_termination'
            self.finish(i, reason)

    def record_post_step(self):
        env = self._env
        ids = (~env.reset_buf).nonzero(as_tuple=False).squeeze(-1) if env.cfg.auto_reset else torch.arange(env.num_envs, device=env.device)
        if len(ids): self.capture(ids)

    def finish(self, i, reason):
        b = self.buffers.get(i)
        if not b or not b['qpos']: return
        self.serial += 1
        name = f'episode-{self.serial:06d}-slot-{i:04d}'
        arrays = {k: np.asarray(v) for k, v in b.items()}
        arrays['initial_qpos'] = self.initial[i]
        arrays['initial_qvel'] = self.initial_velocity[i]
        path = self.directory / (name + '.npz')
        np.savez_compressed(path, **arrays)
        error = arrays['qpos'][-1, :2] - self.initial[i][:2]
        row = dict(id=name, world_slot=i, trace=path.name, terminal_reason=reason,
            control_transitions=len(arrays['qpos']), duration_s=len(arrays['qpos'])*self._env.step_dt,
            final_native_clock_s=float(arrays['native_time_s'][-1]),
            requested_horizon_s=self._env.cfg.episode_length_s,
            delta_xy_m=error.tolist(), minimum_root_height_m=float(arrays['qpos'][:, 2].min()),
            command_velocity_unit='m/s,m/s,rad/s', raw_action_width=24, physical_joint_width=22,
            capture_period_s=self._env.step_dt, native_substeps_per_capture=self._env.cfg.decimation,
            source_termination_observed=bool(arrays['source_terminated'].any()),
            reward_terms=self._env.reward_manager.active_terms)
        with self.manifest.open('a') as f: f.write(json.dumps(row) + '\n')
        if reason.endswith('censoring'): self.censored += 1
        else: self.completed += 1
        self.buffers[i] = defaultdict(list)

    def close(self):
        for i in list(self.buffers): self.finish(i, 'learner_budget_censoring')
