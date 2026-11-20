"""Thin composition of the retained R3 runner; no replacement PPO loop.

Replay accepts an explicit version/checkpoint and profile for actual use after
qualitative acceptance. Both modes preserve the original sensing/action path.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

from schedule import PROFILES, Schedule, configure, recorder_class

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / 'control/skills/walker/task'
DEFAULT = ROOT / 'control/skills/forward-stop-learning/setup/config.json'


def ref(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha, bytes=path.stat().st_size)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_state(left, right, trail=''):
    """Compare loaded live state to the exact source, including all Adam tensors."""
    import torch
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left.cpu(), right.cpu()):
            raise ValueError('Restored state differs: ' + trail)
    elif isinstance(left, dict):
        if left.keys() != right.keys(): raise ValueError('Restored keys differ: ' + trail)
        for key in left: exact_state(left[key], right[key], trail + '/' + str(key))
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right): raise ValueError('Restored length differs: ' + trail)
        for i, value in enumerate(left): exact_state(value, right[i], trail + '/' + str(i))
    elif left != right:
        raise ValueError('Restored scalar differs: ' + trail)


def normalizers(runner):
    result = {}
    for name in ('actor', 'critic'):
        model = getattr(runner.alg, name)
        states = model.state_dict()
        result[name] = {k: v.detach().cpu().tolist() for k, v in states.items()
                       if 'count' in k or ('normal' in k and v.numel() == 1)}
        result[name]['update_normalization'] = type(model).__module__ + '.' + type(model).__qualname__
        result[name]['until'] = model.obs_normalizer.until
        result[name]['eps'] = model.obs_normalizer.eps
    return result


def live_rewards(env, native):
    return {name: dict(weight=env.reward_manager.get_term_cfg(name).weight,
        params=native.serializable(env.reward_manager.get_term_cfg(name).params))
        for name in env.reward_manager.active_terms}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True, choices=('train', 'replay'))
    p.add_argument('--config', type=Path, default=DEFAULT)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--profile', choices=tuple(PROFILES))
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    seal_path = ROOT / 'data/forward-stop-learning-20261030/source-seal.json'
    seal = json.loads(seal_path.read_text())
    for row in seal['files']:
        if ref(ROOT / row['path']) != row:
            raise ValueError('Changed sealed source: ' + row['path'])
    for row in config['exact_inputs']:
        if ref(ROOT / row['path']) != row: raise ValueError('Changed input: ' + row['path'])
    if args.mode == 'train':
        if ref(args.checkpoint) != config['initial_checkpoint'] or args.seed != config['train_seed']:
            raise ValueError('Training must use the frozen source and seed')
        if args.profile: p.error('Training uses all three frozen cohorts')
    elif args.profile is None:
        p.error('Replay requires an explicit profile')
    # Replay is also the public executable interface for later saved versions;
    # caller records checkpoint identity, never silently substitutes a shelf.
    args.out.mkdir(parents=True, exist_ok=False)
    args.work.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(TASK))
    native = load(TASK / 'run.py', 'cove50_forward_stop_original_runner')
    sys.path.insert(0, str(ROOT / 'control/skills/basic-gait'))
    replay = load(ROOT / 'control/skills/basic-gait/replay.py', 'cove50_forward_stop_layout')
    import torch
    profile = 'train' if args.mode == 'train' else args.profile
    report = dict(schema='cove50.forward-stop-native.v1', mode=args.mode,
        profile=profile, seed=args.seed, checkpoint=ref(args.checkpoint), config=ref(args.config),
        status='starting', original_source=ref(TASK / 'run.py'),
        source_seal=ref(seal_path),
        semantics='input_command3 is authoritative for every issued action. Inherited command3 is the next request on ordinary post-step captures; terminal training pre-reset capture still contains the issued request.',
        motor_sampling='Inherited current/torque/saturated fields are the last native substep sampled at50Hz; they are not4kHz peak or clipping-duty aggregates.',
        native_report_note='Inherited contract/report factory prose is historical; actual restored counter and effective weights below are authoritative.')
    original_bind = native.bind
    def bind(cfg):
        cfg = configure(original_bind(cfg), profile)
        replay.validate_command_columns(cfg)
        return cfg
    native.bind = bind
    native.Episodes = recorder_class(native.Episodes)
    original_init = native.initialize
    def initialize(runner, env, **kw):
        transfer = original_init(runner, env, **kw)
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        current = runner.alg.save()
        for key in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict'):
            exact_state(saved[key], current[key], key)
        if env.common_step_counter != saved['infos']['env_state']['common_step_counter']:
            raise ValueError('Saved common counter not retained')
        if args.mode == 'train' and env.common_step_counter != 25200:
            raise ValueError('Unexpected training counter')
        env._forward_stop_schedule = Schedule(env, profile, args.out)
        import numpy as np
        rt = native.runtime_package().get_runtime(env)
        state = dict(qpos=env.sim.data.qpos, qvel=env.sim.data.qvel,
            actor81=env.obs_buf['actor'], critic96=env.obs_buf['critic'],
            previous_raw24=env.action_manager.action,
            previous_previous_raw24=env.action_manager.prev_action)
        for key in ('tick', 'pending_reset', 'target_history', 'target', 'initial_target',
                'previous_command', 'gravity_estimate', 'sample_gyro', 'accel_bias', 'gyro_bias',
                'next_imu', 'next_joint', 'last_imu_time', 'seq', 'packet_values',
                'packet_capture', 'packet_imu_capture', 'last_values', 'capture_time_s',
                'imu_capture_time_s', 'valid'):
            state['runtime_' + key] = getattr(rt, key)
        np.savez_compressed(args.out / 'initialization.npz',
            **{k: v.detach().cpu().numpy().copy() for k, v in state.items()})
        report.update(restored_state=dict(exact_actor_critic_optimizer=True,
            adam_parameters=len(runner.alg.optimizer.state),
            adam_step_counts=sorted({int(s['step']) for s in runner.alg.optimizer.state.values()}),
            completed_rounds=runner.current_learning_iteration, common_counter=env.common_step_counter,
            learning_rate=runner.alg.learning_rate, normalizers=normalizers(runner)),
            initial_effective_rewards=live_rewards(env, native))
        process = runner.alg.process_env_step
        def process_env_step(obs, *a, **kw):
            # Native command_manager.compute already updated reset-local
            # requests before observation computation. Check the exact packet
            # consumed by stock normalization and PPO storage, including resets.
            for name, start in (('actor', 69), ('critic', 72)):
                if not torch.equal(obs[name][:, start:start + 3], env._forward_stop_schedule.term.vel_command_b):
                    raise ValueError('Returned next-observation command differs')
            return process(obs, *a, **kw)
        runner.alg.process_env_step = process_env_step
        update = runner.alg.update
        def journaled_update(*a, **kw):
            result = update(*a, **kw)
            row = dict(common_counter=env.common_step_counter,
                adam_step_counts=sorted({int(s['step']) for s in runner.alg.optimizer.state.values()}),
                learning_rate=runner.alg.learning_rate, losses=native.serializable(result),
                normalizers=normalizers(runner))
            with (args.out / 'optimizer-rounds.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            return result
        runner.alg.update = journaled_update
        report['_live'] = (env, runner)
        return transfer
    native.initialize = initialize
    original_runner = native.MjlabOnPolicyRunner
    def runner_factory(*a, **kw):
        kw['log_dir'] = str(args.work / 'tensorboard')
        return original_runner(*a, **kw)
    native.MjlabOnPolicyRunner = runner_factory
    argv = ['retained-native', '--mode', args.mode, '--device', args.device,
        '--worlds', '32' if args.mode == 'train' else '1', '--updates', '50',
        '--seed', str(args.seed), '--checkpoint', str(args.checkpoint), '--out', str(args.out / 'native')]
    previous_argv = sys.argv
    try:
        sys.argv = argv
        native.main()
        report['status'] = 'completed'
    finally:
        sys.argv = previous_argv
        live = report.pop('_live', None)
        if live:
            env, runner = live
            report.update(final_common_counter=env.common_step_counter,
                final_effective_rewards=live_rewards(env, native),
                final_normalizers=normalizers(runner))
        inherited = args.out / 'native/report.json'
        if inherited.exists():
            report['native'] = json.loads(inherited.read_text())
        (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
