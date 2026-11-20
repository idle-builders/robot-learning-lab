"""Freeze turn inputs and check saved actor/decoder parity without physics."""
import ast
from contextlib import ExitStack
from dataclasses import asdict
import importlib.util
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import mujoco

from profiles import PROFILES
from replay import ROOT, TASK, DATA, CHECKPOINT, CHECKPOINT_SHA, ref, library_ref, command_observations

RUN = ROOT / 'artifacts/cove50-walk-resume/runpod-fit-r2-20261026'
HERE = Path(__file__).parent


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


class SavedActor(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.register_buffer('mean', state['obs_normalizer._mean'].clone())
        self.register_buffer('std', state['obs_normalizer._std'].clone())
        self.layers = nn.Sequential(nn.Linear(state['mlp.0.weight'].shape[1], 512), nn.ELU(), nn.Linear(512, 256),
            nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, state['mlp.6.weight'].shape[0]))
        self.layers.load_state_dict({k.removeprefix('mlp.'): v for k, v in state.items() if k.startswith('mlp.')})

    def forward(self, obs):
        return self.layers((obs - self.mean) / (self.std + .01))


def frozen_interface(contract):
    """Use unchanged Interface methods, hydrating its already retained map.

    Calling Interface() would compile/forward models. This source-only path
    deliberately uses the exact saved native contract instead.
    """
    path = TASK.parent / 'policy/adapter.py'
    spec = importlib.util.spec_from_file_location('turn_prep_source_interface', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    interface = module.Interface.__new__(module.Interface)
    interface.source_names = contract['source_order']
    interface.current_names = contract['physical_order']
    interface.rows = contract['mapping']
    interface.new_names = contract['new_axes']
    interface.home = torch.tensor(contract['source_home_rad'], dtype=torch.float32)
    interface.new_rows = {name: 21 + i for i, name in enumerate(interface.new_names)}
    interface.source_clips = {k: tuple(v) for k, v in contract['retained_source_target_clips_rad'].items()}
    interface._device_tensors = {}
    return interface


def decoder_reference(raw, interface):
    old = interface.home[None, :].repeat(len(raw), 1)
    old[:, :10] += .3 * raw[:, :10]
    for name, limits in interface.source_clips.items():
        index = interface.source_names.index(name)
        old[:, index].clamp_(*limits)
    columns = []
    for row in interface.rows:
        source = row['source_index']
        columns.append(row['sign'] * old[:, source] if source is not None
            else .3 * raw[:, interface.new_rows[row['current_name']]])
    return torch.stack(columns, dim=1)


def cached_switch_check(actor, critic):
    """Exercise installed cache/wrapper methods without an environment."""
    from mjlab.managers.observation_manager import ObservationManager
    from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
    from tensordict import TensorDict
    class Cache:
        def __init__(self):
            self.group_calls = 0
            self._obs_buffer = dict(actor=torch.arange(81, dtype=torch.float32)[None, :],
                critic=torch.arange(96, dtype=torch.float32)[None, :])
            self._obs_buffer['actor'][:, 69:72] = torch.tensor([.25, 0., 0.])
            self._obs_buffer['critic'][:, 72:75] = torch.tensor([.25, 0., 0.])
        def compute(self):
            return ObservationManager.compute(self, update_history=False)
        def compute_group(self, *args):
            self.group_calls += 1
            raise AssertionError('Command refresh must not recompute noisy/delayed groups')
    cache = Cache()
    original = TensorDict(cache._obs_buffer, batch_size=[1]).clone()
    fake_env = SimpleNamespace(observation_manager=cache, sensor_noise_state=['retained'],
        delay_queue=['retained'], previous_raw=torch.arange(24.), native_time_s=.9)
    fake_vec = SimpleNamespace(unwrapped=fake_env, num_envs=1)
    term = SimpleNamespace(vel_command_b=torch.zeros(1, 3), vel_command_w=torch.zeros(1, 3),
        is_standing_env=torch.zeros(1, dtype=torch.bool))
    requested = PROFILES['forward-to-left'].apply(term, 45)
    stale = RslRlVecEnvWrapper.get_observations(fake_vec)
    fresh = command_observations(stale, requested)
    fake_env.obs_buf = fresh  # Same object handed to recorder.cache_inputs.
    actor_seen, value_seen = [], []
    hook = actor.register_forward_pre_hook(lambda model, args: actor_seen.append(args[0].clone()))
    value_hook = critic.register_forward_pre_hook(lambda model, args: value_seen.append(args[0].clone()))
    with torch.inference_mode():
        actor(fresh['actor'])
        critic(fresh['critic'])
    hook.remove()
    value_hook.remove()
    masks = {'actor': torch.ones(81, dtype=torch.bool), 'critic': torch.ones(96, dtype=torch.bool)}
    masks['actor'][69:72] = False
    masks['critic'][72:75] = False
    noncommand_delta = {name: float((fresh[name][:, mask] - original[name][:, mask]).abs().max())
        for name, mask in masks.items()}
    assert all(value == 0 for value in noncommand_delta.values())
    assert all(torch.equal(cache._obs_buffer[name], original[name]) for name in masks)
    assert torch.equal(actor_seen[0][:, 69:72], torch.tensor(requested)[None, :])
    assert torch.equal(value_seen[0][:, 72:75], torch.tensor(requested)[None, :])
    assert torch.equal(fake_env.obs_buf['critic'][:, 72:75], torch.tensor(requested)[None, :])
    assert cache.group_calls == 0
    return dict(installed_compute_returns_stale_command=stale['actor'][0, 69:72].tolist(),
        requested_switch_command=list(requested), actor_input_command=actor_seen[0][0, 69:72].tolist(),
        value_input_command=value_seen[0][0, 72:75].tolist(),
        recorder_obs_buf_command=fake_env.obs_buf['critic'][0, 72:75].tolist(),
        applied_command_buffer=term.vel_command_b[0].tolist(),
        noncommand_max_abs_delta=noncommand_delta, original_cache_unchanged=True,
        group_recomputations=cache.group_calls, sensor_noise_state=fake_env.sensor_noise_state,
        delay_queue=fake_env.delay_queue, previous_raw_unchanged=bool(torch.equal(fake_env.previous_raw, torch.arange(24.))),
        native_clock_s=fake_env.native_time_s, no_extra_history_noise_or_delay_push=True)


def offline_checks():
    blocked_calls = []
    def forbidden(*args, **kwargs):
        blocked_calls.append('forbidden native operation')
        raise AssertionError('Preparation may not construct a model or run physics')
    class NoModel:
        from_xml_path = staticmethod(forbidden)
        from_xml_string = staticmethod(forbidden)
    with ExitStack() as stack:
        stack.enter_context(patch.object(mujoco, 'MjModel', NoModel))
        for name in ('mj_forward', 'mj_step', 'mj_integratePos', 'mj_inverse', 'mj_kinematics'):
            stack.enter_context(patch.object(mujoco, name, forbidden))
        saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
        if ref(CHECKPOINT)['sha256'] != CHECKPOINT_SHA:
            raise ValueError('Candidate identity changed')
        state = saved['actor_state_dict']
        native_report = json.loads((RUN / 'evidence/fit-through-200/report.json').read_text())
        interface = frozen_interface(native_report['selected_interface'])
        actor = SavedActor(state).eval()
        cached_switch = cached_switch_check(actor, SavedActor(saved['critic_state_dict']).eval())
        # Use an existing complete native replay as source data, never rerun it.
        trace = sorted((RUN / 'evidence/after-136101/episodes').glob('*.npz'))[0]
        with np.load(trace) as source:
            observations = torch.tensor(source['input_observation81'].copy())
            raw_native = torch.tensor(source['raw_action24'].copy())
            valid = torch.tensor(source['input_sensor_valid'].copy()).bool().reshape(-1)
            requested_native = torch.tensor(source['requested_target22_rad'].copy())
        with torch.inference_mode():
            raw = actor(observations)
            expected = observations
            expected = (expected - state['obs_normalizer._mean']) / (state['obs_normalizer._std'] + .01)
            for layer in (0, 2, 4, 6):
                expected = torch.nn.functional.linear(expected, state[f'mlp.{layer}.weight'], state[f'mlp.{layer}.bias'])
                if layer != 6:
                    expected = torch.nn.functional.elu(expected)
            decoder = interface.target(raw_native)
            reference = decoder_reference(raw_native, interface)
            home = interface.target(torch.zeros_like(raw_native))
            expected_native = torch.where(valid[:, None], decoder, home)
        fake = SimpleNamespace(vel_command_b=torch.ones(1, 3), vel_command_w=torch.ones(1, 3),
            is_standing_env=torch.ones(1, dtype=torch.bool), is_world_env=torch.ones(1, dtype=torch.bool),
            is_forward_env=torch.ones(1, dtype=torch.bool), is_heading_env=torch.ones(1, dtype=torch.bool),
            previous_raw=torch.arange(24.), queues=['retained'], native_clock=.9)
        history = fake.previous_raw.clone()
        schedules = []
        for profile in PROFILES.values():
            commands = [profile.command(i) for i in range(round(profile.duration_s / .02))]
            profile.apply(fake, 45)
            schedules.append(dict(profile=profile.name, controls=len(commands), first=commands[0],
                last=commands[-1], switch_control=45 if profile.switch_s is not None else None,
                command_at44=profile.command(44), command_at45=profile.command(45),
                command_buffer_matches=bool(torch.equal(fake.vel_command_b[0], torch.tensor(profile.command(45)))),
                pure_yaw_not_marked_standing=not bool(fake.is_standing_env[0])))
        assert torch.equal(history, fake.previous_raw) and fake.queues == ['retained'] and fake.native_clock == .9
        # Hydrated named decoder and the original target method must agree.
        assert torch.equal(decoder, reference)
        assert torch.equal(raw, expected)
        assert all(x['command_buffer_matches'] and x['pure_yaw_not_marked_standing'] for x in schedules)
        return dict(schema='cove50.basic-gait.offline-parity.v1', checkpoint=ref(CHECKPOINT),
            source_trace=ref(trace), retained_input_frames=len(observations),
            actor_vs_direct_saved_MLP_max_abs_raw=float((raw - expected).abs().max()),
            actor_cpu_vs_retained_cuda_raw_max_abs=float((raw - raw_native).abs().max()),
            decoder_vs_independent_named_formula_max_abs_rad=float((decoder - reference).abs().max()),
            requested_target_vs_retained_native_max_abs_rad=float((expected_native - requested_native).abs().max()),
            normalizer_buffers_unchanged=bool(torch.equal(actor.mean, state['obs_normalizer._mean']) and torch.equal(actor.std, state['obs_normalizer._std'])),
            schedule_checks=schedules, history_queue_clock_preserved_in_command_update=True,
            cached_command_switch=cached_switch,
            prohibited_native_calls=len(blocked_calls), new_native_episodes=0, new_controls=0,
            new_physics_steps=0, new_optimizer_updates=0,
            limit='Offline inference and decoder/source checks only. Native constructor and command-switch consumption remain unexecuted for this harness.')


def main():
    torch.set_num_threads(1)
    DATA.mkdir(parents=True, exist_ok=True)
    files = ['profiles.py', 'replay.py', 'prepare.py', 'turn-plan.json', 'README.md']
    inputs = [CHECKPOINT, ROOT / 'robot/model/current-model.json',
        ROOT / 'data/basic-skills-retry-20261027/gait-source-audit/report.json',
        ROOT / 'data/basic-skills-retry-20261027/gait-source-audit/current-command-exposure.json',
        RUN / 'evidence/fit-through-200/report.json', RUN / 'evidence/after-136101/report.json',
        ROOT / 'artifacts/cove50-walk-resume/local-clock-corrected-smoke-r1-20261026/report.json']
    inputs += [DATA / 'repair-preimages.json']
    selected = json.loads((ROOT / 'robot/model/current-model.json').read_text())
    inputs += [ROOT / selected[key]['path'] for key in ('rigid_MJCF', 'actuators', 'sensors', 'timing')]
    inputs += [(ROOT / selected['rigid_MJCF']['path']).with_name('evidence-seal.json'),
        ROOT / 'artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml',
        ROOT / 'artifacts/ape-cove50-recovery-20261008/ankleonly100-params/env.yaml',
        ROOT / 'artifacts/ape-cove50-recovery-20261008/ankleonly100-params/agent.yaml']
    shared = [TASK / name for name in ('run.py', 'config.py', 'boundary.py', 'checkpoint.py', 'record.py')]
    shared += [TASK.parent / 'policy/adapter.py'] + sorted((TASK.parent / 'runtime').glob('*.py'))
    preimages = dict(schema='cove50.basic-gait.source-preimages.v1',
        original_shared_sources=[ref(p) for p in shared],
        new_scoped_paths=[dict(path=str((HERE / name).relative_to(ROOT)), preimage='absent before this task') for name in files],
        policy='No original recorder/task/runtime or active basic-skills/arms file edited.')
    write(DATA / 'preimages.json', preimages)
    offline = offline_checks()
    write(DATA / 'offline.json', offline)
    contract = dict(schema='cove50.basic-gait.turn-preparation.v1', status='offline prepared; no native execution',
        inputs=[ref(p) for p in inputs], implementation=[ref(HERE / name) for name in files] + preimages['original_shared_sources'],
        library_sources=[library_ref(p) for p in ('managers/observation_manager.py', 'rl/vecenv_wrapper.py', 'tasks/velocity/mdp/observations.py')],
        profiles=[asdict(x) for x in PROFILES.values()], plan=ref(HERE / 'turn-plan.json'),
        offline=ref(DATA / 'offline.json'), preimages=ref(DATA / 'preimages.json'),
        recorder='Unmodified walk-resume/task/record.py Episodes, original R3 task/boundary/checkpoint.initialize.',
        bootstrap='Original native raw24 sampling/history at control0; mapped-home hold while sensor invalid. Basic-skills standing recorder bootstrap is not used.',
        command_alignment='Write requested command buffers. vec.get_observations returns the existing cache; clone it and replace actor69:72/value72:75 only after exact layout/dependency validation. Assign that clone to env.obs_buf for recorder/value/actor. Never compute_group, push history/delay or resample sensor noise. No robot-state/queue copy at .9s.',
        command_dependencies='Interface.observe concatenates command only in mapped69:72; bound noncommand parts use disjoint slices. Native virtual-IMU velocity and privileged feet read physics/sensors only. Command term has no noise/clip/scale/history/delay. Installed ObservationManager cache and wrapper source are hash-frozen and exercised unbound on a fake populated cache.',
        native_scope='Full requested horizon, source terminal flags retained, source no-reset resting/fall tails preserved. One world and no learning.',
        compute='CPU possible: retained source smoke completed2controls/160physics in10.6927s including startup. Full3–4s harness performance and first constructor remain unmeasured. CUDA optional; no provider query or allocation here.',
        runtime_requirements=dict(local_python='3.12 through .venv-control/bin/python',
            local_versions={name: importlib.metadata.version(name) for name in ('mjlab', 'mujoco', 'mujoco-warp', 'torch', 'rsl-rl-lib')},
            original_cuda_versions=json.loads((RUN / 'evidence/fit-through-200/report.json').read_text())['runtime_versions'],
            note='Same source contracts are frozen. CPU/CUDA device and binary differences remain explicit; offline inference parity does not replace the first native constructor/runtime check.'),
        regeneration='.venv-control/bin/python -B control/skills/basic-gait/prepare.py')
    write(DATA / 'contract.json', contract)
    write(DATA / 'retention.json', dict(schema='cove50.basic-gait.preparation-retention.v1',
        required=[ref(DATA / name) for name in ('preimages.json', 'repair-preimages.json', 'offline.json', 'contract.json')],
        regenerate=contract['regeneration'], cleanup='Only compact required preparation records generated; no new episodes, rendering, downloads or caches. Existing native source inputs retained.'))
    print(json.dumps(dict(contract=ref(DATA / 'contract.json'), offline=offline)))


if __name__ == '__main__':
    main()
