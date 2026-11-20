"""Four intact whole22 donor visits in the selected MJLab R3 runtime.

No learner, value model, optimizer, reference splicing or physical-state repair.
Execute only after the lead releases this exact source on the sole native lane.
"""
import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

from reference import Donor, Goals, command

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATA = ROOT / 'data/coordinated-arm-reference-20261030'


def ref(path):
    path = Path(path).absolute()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha, bytes=path.stat().st_size)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def verify(config_path, seal_path, *, payload):
    cfg = json.loads(config_path.read_text())
    seal = json.loads(seal_path.read_text())
    if ref(config_path) != seal['configuration']:
        raise ValueError('Configuration differs from frozen seal')
    for row in seal['source'] + cfg['inputs'] + ([cfg['checkpoint']] if payload else []):
        if ref(ROOT / row['path']) != row:
            raise ValueError('Exact source/input differs: ' + row['path'])
    if payload:
        for row in cfg['physical_meshes']:
            if ref(ROOT / row['path']) != row:
                raise ValueError('Selected physical source differs: ' + row['path'])
    if cfg['planned'] != dict(visits=4, world_slots=1, controls_per_visit=500,
            controls=2000, physics_substeps=160000, optimizer_updates=0, new_weight_sets=0):
        raise ValueError('Unexpected finite feasibility budget')
    return cfg


def load_native():
    task = ROOT / 'control/skills/walker/task'
    sys.path.insert(0, str(task))
    spec = importlib.util.spec_from_file_location('coordinated_arm_native_source', task / 'run.py')
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    return native


def components(native, cfg):
    import numpy as np
    import torch
    from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
    from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

    class DirectMotor(ActionTerm):
        def __init__(self, config, env):
            super().__init__(config, env)
            self.ids = torch.tensor([self._entity.joint_names.index(n) for n in cfg['joint_order']], device=self.device)
            self._raw = torch.zeros((env.num_envs, 22), device=self.device)
            self.requested_target = self._raw.clone()
            self.bootstrap_target = self._raw.clone()
            self.saturation_sum = self._raw.clone()
            self.peak_abs_torque = self._raw.clone()
            self.peak_current = self._raw.clone()
            self.interval_substeps = 0

        @property
        def action_dim(self): return 22

        @property
        def raw_action(self): return self._raw

        def process_actions(self, actions):
            self._raw.copy_(actions)
            rt = native.runtime_package().get_runtime(self._env)
            self.requested_target.copy_(torch.where(rt.valid[:, None], actions, self.bootstrap_target))
            self.saturation_sum.zero_(); self.peak_abs_torque.zero_(); self.peak_current.zero_()
            self.interval_substeps = 0

        def apply_actions(self):
            native.runtime_package().get_runtime(self._env).arm_physics_step()
            self._entity.set_joint_position_target(self.requested_target, joint_ids=self.ids)

        def reset(self, env_ids=None):
            ids = slice(None) if env_ids is None else env_ids
            self._raw[ids] = 0.
            # Event writes precede ActionManager.reset in the existing env.
            # This startup hold has no online geometric/native correction.
            self.bootstrap_target[ids] = self._entity.data.joint_pos[ids][:, self.ids]
            self.requested_target[ids] = self.bootstrap_target[ids]

        def sampled_substep(self):
            rt = native.runtime_package().get_runtime(self._env)
            self.saturation_sum.add_(rt.saturated)
            torch.maximum(self.peak_abs_torque, rt.torque.abs(), out=self.peak_abs_torque)
            torch.maximum(self.peak_current, rt.current, out=self.peak_current)
            self.interval_substeps += 1

    @dataclass(kw_only=True)
    class DirectMotorCfg(ActionTermCfg):
        def build(self, env): return DirectMotor(self, env)

    def beta_observation(env):
        k = env.episode_length_buf
        t = k.to(torch.float32) * .02
        beta = torch.where(t < 6., ((t - 2.) / 2.).clamp(0., 1.), (1. - (t - 6.) / 2.).clamp(min=0.))
        return beta[:, None]

    def reset_joints(env, env_ids=None):
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
        robot = env.scene['robot']; case = env._coordinated_case
        native_ids = [robot.joint_names.index(n) for n in cfg['joint_order']]
        q = robot.data.default_joint_pos[ids].clone()
        qd = robot.data.default_joint_vel[ids].clone()
        q[:, native_ids] += q.new_tensor(case['joint_offsets22_rad'])
        qd[:, native_ids] = qd.new_tensor(case['joint_velocities22_rad_s'])
        limits = robot.data.soft_joint_pos_limits[ids]
        unclipped = q.clone()
        q.clamp_(min=limits[:, :, 0], max=limits[:, :, 1])
        env._coordinated_reset = dict(joint_order=cfg['joint_order'],
            default_q22_rad=robot.data.default_joint_pos[ids][:, native_ids].cpu().tolist(),
            requested_q22_rad=unclipped[:, native_ids].cpu().tolist(),
            actual_q22_rad=q[:, native_ids].cpu().tolist(),
            actual_qd22_rad_s=qd[:, native_ids].cpu().tolist(),
            soft_limits22_rad=limits[:, native_ids].cpu().tolist(),
            clamped22=(q[:, native_ids] != unclipped[:, native_ids]).cpu().tolist())
        robot.write_joint_state_to_sim(q, qd, env_ids=ids)

    class Episodes(native.Episodes):
        def cache_inputs(self, beta, beta_dot, motor, goals):
            env = self._env; rt = native.runtime_package().get_runtime(env)
            action = env.action_manager.get_term('joint_pos')
            tensor = dict(delayed_values50=rt.last_values, input_sensor_valid=rt.valid,
                input_sensor_capture_s=rt.capture_time_s, input_imu_capture_s=rt.imu_capture_time_s,
                own_previous_motor_output22_rad=env.action_manager.action)
            self.inputs = {k: v.detach().cpu().numpy().copy() for k, v in tensor.items()}
            self.inputs.update(input_beta=np.asarray([[beta]], dtype=np.float32), input_beta_dot_s_inv=np.asarray([beta_dot]),
                motor_reference22_rad=motor.detach().cpu().numpy()[None].copy(),
                q_pose_goal22_rad=goals['q_pose_goal22_rad'][None].copy(),
                geometric_root_xyz_m=goals['geometric_root_xyz_m'][None].copy(),
                geometric_com_xyz_m=goals['geometric_com_xyz_m'][None].copy(),
                input_local_control=env.episode_length_buf.cpu().numpy().copy(),
                sensor_age_s=(rt.tick * rt.physics_dt_s - rt.capture_time_s).cpu().numpy().copy())

        def capture(self, ids):
            env = self._env; rt = native.runtime_package().get_runtime(env)
            action = env.action_manager.get_term('joint_pos')
            robot = env.scene['robot'].data
            fields = dict(qpos=env.sim.data.qpos, qvel=env.sim.data.qvel,
                root_velocity_body_m_s=robot.root_link_lin_vel_b,
                root_ang_velocity_body_rad_s=robot.root_link_ang_vel_b,
                measured_q22_rad=robot.joint_pos[:, action.ids], measured_qd22_rad_s=robot.joint_vel[:, action.ids],
                teacher_motor_output22_rad=env.action_manager.action,
                requested_target22_rad=action.requested_target, accepted_target22_rad=rt.previous_command,
                applied_target22_rad=rt.target, torque22_Nm=rt.torque, current22_A=rt.current,
                saturated22=rt.saturated, substep_saturation_fraction22=action.saturation_sum / max(1, action.interval_substeps),
                substep_peak_abs_torque22_Nm=action.peak_abs_torque, substep_peak_current22_A=action.peak_current,
                feet_force_N=env.scene['feet_ground_contact'].data.force,
                feet_found=env.scene['feet_ground_contact'].data.found,
                distal_forearm_force_N=env.scene['hand_ground_contact'].data.force,
                nonfoot_force_N=env.scene['nonfoot_ground_contact'].data.force,
                nonfoot_found=env.scene['nonfoot_ground_contact'].data.found,
                runtime_time_s=rt.tick * rt.physics_dt_s, native_time_s=env.sim.data.time,
                source_terminated=env.reset_terminated, source_timeout=env.reset_time_outs)
            arrays = {k: v[ids].detach().cpu().numpy().copy() for k, v in fields.items()}
            for j, i in enumerate(ids.cpu().tolist()):
                for key, value in self.inputs.items(): self.buffers[i][key].append(value[i])
                for key, value in arrays.items(): self.buffers[i][key].append(value[j])
                self.buffers[i]['captured_physics_substeps'].append(action.interval_substeps)
                self.transitions += 1

        def finish(self, i, reason):
            b = self.buffers.get(i)
            if not b or not b['qpos']: return
            self.serial += 1
            name = f'episode-{self.serial:06d}-slot-{i:04d}'
            arrays = {k: np.asarray(v) for k, v in b.items()}
            arrays.update(initial_qpos=self.initial[i], initial_qvel=self.initial_velocity[i])
            path = self.directory / (name + '.npz')
            np.savez_compressed(path, **arrays)
            row = dict(id=name, world_slot=i, case=self._env._coordinated_case,
                trace=path.name, terminal_reason=reason, control_transitions=len(arrays['qpos']),
                requested_horizon_s=10., duration_s=len(arrays['qpos']) * .02,
                source_termination_observed=bool(arrays['source_terminated'].any()),
                source_timeout_observed=bool(arrays['source_timeout'].any()),
                physical_joint_width=22, actor_input_width=1, optimizer_updates=0,
                capture_period_s=.02, native_substeps_per_capture=80, joint_order=cfg['joint_order'])
            with self.manifest.open('a') as stream: stream.write(json.dumps(row) + '\n')
            if reason.endswith('censoring'): self.censored += 1
            else: self.completed += 1
            self.buffers[i] = defaultdict(list)

    return DirectMotorCfg, beta_observation, reset_joints, Episodes, ObservationGroupCfg, ObservationTermCfg


def run(cfg, config_path, seal_path, out, device):
    import numpy as np
    import torch
    from mjlab.managers.recorder_manager import RecorderTermCfg
    from mjlab.sensor.contact_sensor import ContactSensorCfg, ContactMatch
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'config.json', cfg)
    native = load_native()
    action_cfg, observation, reset_joints, episodes, Group, Term = components(native, cfg)
    donor = Donor(ROOT / cfg['checkpoint']['path'], cfg['joint_order'])
    goals = Goals(ROOT / cfg['goal_table']['path'], cfg['joint_order'])
    robot = native.runtime_package().robot_cfg(seed=cfg['cases'][0]['seed'])
    env_cfg = native.morphology_cfg(robot, play=True)
    env_cfg.actions = {'joint_pos': action_cfg(entity_name='robot')}
    env_cfg.observations = {'actor': Group(terms={'beta': Term(func=observation)})}
    env_cfg.rewards = {}; env_cfg.curriculum = {}; env_cfg.commands = {}; env_cfg.metrics = {}
    env_cfg.auto_reset = False; env_cfg.episode_length_s = 10.; env_cfg.scene.num_envs = 1
    env_cfg.seed = cfg['cases'][0]['seed']
    env_cfg.scene.entities['robot'].init_state.joint_pos = {n: 0. for n in cfg['joint_order']}
    env_cfg.scene.entities['robot'].init_state.pos = tuple(cfg['root_position_m'])
    env_cfg.events = {name: env_cfg.events[name] for name in ('reset_base', 'reset_robot_joints')}
    env_cfg.events['reset_robot_joints'].func = reset_joints
    env_cfg.events['reset_robot_joints'].params = {}
    env_cfg.events['reset_base'].params['velocity_range'] = {}
    env_cfg.scene.sensors += (ContactSensorCfg(name='nonfoot_ground_contact',
        primary=ContactMatch(mode='body', pattern=r'^(?!(R|L)_foot$).*$', entity='robot'),
        secondary=ContactMatch(mode='body', pattern='terrain'),
        fields=('found', 'force'), reduce='netforce', num_slots=1),)
    env_cfg.recorders = {'full_episodes': RecorderTermCfg(func=episodes, params={'directory': str(out / 'episodes')})}
    report = dict(schema='cove50.coordinated-arm-reference.v1', configuration=ref(config_path),
        source_seal=ref(seal_path), status='starting', cases=[], planned=cfg['planned'],
        counts=dict(simulator_controls=0, physics_substeps=0, optimizer_updates=0, new_weight_sets=0),
        semantics=cfg['semantics'])
    env = None
    try:
        env = native.ManagerBasedRlEnv(cfg=env_cfg, device=device)
        action = env.action_manager.get_term('joint_pos')
        recorder = env.recorder_manager._terms[0]
        original_step = env.sim.step
        def counted_step(*a, **kw):
            result = original_step(*a, **kw)
            report['counts']['physics_substeps'] += 1
            action.sampled_substep()
            return result
        env.sim.step = counted_step
        for case in cfg['cases']:
            env._coordinated_case = case
            env.event_manager.get_term_cfg('reset_base').params['pose_range'] = {
                name: (value, value) for name, value in zip(('roll', 'pitch', 'yaw'), case['root_rpy_rad'])}
            rt = native.runtime_package().get_runtime(env)
            if rt.names != cfg['joint_order']:
                raise ValueError('Current runtime named22 motor order differs')
            rt.generator.manual_seed(case['seed'])
            env.reset(seed=case['seed'])
            initial = dict(qpos=env.sim.data.qpos, qvel=env.sim.data.qvel,
                beta_input1=env.obs_buf['actor'], bootstrap_target22_rad=action.bootstrap_target,
                runtime_generator_state=rt.generator.get_state())
            robot_data = env.scene['robot'].data
            sole_ids = [env.scene['robot'].site_names.index(n) for n in ('R_sole_contact_frame', 'L_sole_contact_frame')]
            initial['sole_positions_m'] = robot_data.site_pos_w[:, sole_ids]
            for key in ('tick', 'pending_reset', 'target_history', 'target', 'initial_target',
                    'previous_command', 'gravity_estimate', 'sample_gyro', 'accel_bias', 'gyro_bias',
                    'next_imu', 'next_joint', 'last_imu_time', 'seq', 'packet_values', 'packet_capture',
                    'packet_imu_capture', 'last_values', 'capture_time_s', 'imu_capture_time_s', 'valid'):
                initial['runtime_' + key] = getattr(rt, key)
            initial_path = out / ('initial-' + case['name'] + '.npz')
            np.savez_compressed(initial_path, **{k: v.detach().cpu().numpy().copy() for k, v in initial.items()})
            row = dict(case=case, reset=env._coordinated_reset, initialization=ref(initial_path),
                common_counter_start=env.common_step_counter, controls=0, physics_substeps=0,
                physics_substeps_start=report['counts']['physics_substeps'])
            report['cases'].append(row)
            before = report['counts']['physics_substeps']
            for control in range(500):
                beta, beta_dot = command(control)
                # The external clock computes beta once. Only this scalar is
                # supplied to the retained network; goals/sensors are telemetry.
                motor = donor.motor_reference(beta)
                annotation = goals.at(beta)
                recorder.cache_inputs(beta, beta_dot, motor, annotation)
                env.step(motor.to(device=env.device)[None])
                env._manual_reset_pending[:] = False
                row['controls'] += 1; report['counts']['simulator_controls'] += 1
                if not bool(torch.isfinite(env.sim.data.qpos).all()):
                    raise FloatingPointError('Nonfinite native state; retain exact partial episode')
            recorder.finish(0, 'requested_horizon')
            row.update(physics_substeps=report['counts']['physics_substeps'] - before,
                common_counter_end=env.common_step_counter, status='whole_horizon_retained')
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()[-8000:])
        if env is not None:
            path = out / 'interrupted-native-state.npz'
            rt = native.runtime_package().get_runtime(env)
            state = dict(qpos=env.sim.data.qpos, qvel=env.sim.data.qvel, runtime_tick=rt.tick,
                requested22_rad=action.requested_target, accepted22_rad=rt.previous_command,
                applied22_rad=rt.target, delayed_values50=rt.last_values)
            np.savez_compressed(path, **{k: v.detach().cpu().numpy().copy() for k, v in state.items()})
            report['interrupted_native_state'] = ref(path)
        raise
    finally:
        if report['cases']:
            last = report['cases'][-1]
            last['physics_substeps'] = report['counts']['physics_substeps'] - last['physics_substeps_start']
        if env is not None:
            try: env.close()
            except Exception as error: report['close_error'] = str(error)
        report.update(wall_s=time.monotonic() - started, donor_model_instances=1,
            donor_checkpoint_loads=1, donor_actor_queries=donor.queries,
            distinct_fixed_policies=1, actual_world_slots=1 if env is not None else 0,
            independent_start_families=len(report['cases']),
            original_episodes=[ref(p) for p in sorted((out / 'episodes').glob('*.npz'))])
        write(out / 'report.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--config', type=Path, default=DATA / 'config.json')
    parser.add_argument('--seal', type=Path, default=DATA / 'source-seal.json')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    cfg = verify(args.config, args.seal, payload=args.execute)
    if not args.execute:
        print(json.dumps(dict(source_configuration=ref(args.config), source_seal=ref(args.seal),
            planned=cfg['planned'], activity='Source/hash verification only; no PT, NN, models or physics')))
        return
    if not args.out or not args.work:
        parser.error('--execute needs managed --out and --work')
    if not os.environ.get('LAB_EVIDENCE_DIR') or not os.environ.get('LAB_WORK_DIR'):
        parser.error('Native producer must run in infra/lab storage run')
    if not args.out.absolute().is_relative_to(Path(os.environ['LAB_EVIDENCE_DIR']).absolute()):
        parser.error('Native originals belong inside the managed evidence directory')
    try:
        run(cfg, args.config, args.seal, args.out, args.device)
    except BaseException as error:
        # Import/checkpoint/configuration failure can precede native entry.
        # Keep its exact original failure; do not invent native zero counts.
        if args.out.exists() and not (args.out / 'report.json').exists():
            write(args.out / 'setup-failure.json', dict(configuration=ref(args.config),
                source_seal=ref(args.seal), error_type=type(error).__name__, error=str(error),
                traceback=traceback.format_exc()[-8000:], native_counts=None,
                note='Setup failed before the native report finalizer; source closure/log determine activity. No retry.'))
        raise


if __name__ == '__main__': main()
