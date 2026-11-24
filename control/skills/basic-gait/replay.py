"""Replay one frozen gait condition through the original R3 native recorder.

Native dependencies are lazy: importing this module does not build a model.
The collection entry point is reserved for the subsequent physical skill slot.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

from profiles import PROFILES

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / 'control/skills/walker/task'
DATA = ROOT / 'data/basic-skills-retry-20261027/turn-preparation'
CHECKPOINT = ROOT / 'artifacts/cove50-walk-resume/runpod-fit-r2-20261026/evidence/fit-through-200/candidate.pt'
CHECKPOINT_SHA = '467b12a17a71f06d951294b3d9185503ea78c6edda431ae7e7870aa648fc02bd'
SOURCE_CHECKPOINT = ROOT / 'artifacts/ape-low-level-training-20261015/shared-forward-turn/native/bc/bc-best.pt'


def ref(path):
    path = Path(path).absolute()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path.relative_to(ROOT)), sha256=digest, bytes=path.stat().st_size)


def library_ref(module_path):
    """Portable source identity within the installed MJLab distribution."""
    library = Path(importlib.util.find_spec('mjlab').origin).parent
    path = library / module_path
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(module='mjlab/' + module_path, sha256=digest, bytes=path.stat().st_size)


def validate_command_columns(cfg):
    """Confirm the exact untransformed command layout before patching cache.

    The frozen Interface.observe only concatenates command in columns69:72.
    Other bound actor/native terms select disjoint slices; privileged foot
    terms and virtual IMU velocity read physics/sensors, not this command.
    """
    core = ['base_ang_vel', 'projected_gravity', 'joint_pos', 'joint_vel', 'actions', 'command']
    for name in ('actor', 'critic'):
        group = cfg.observations[name]
        expected = core + ['new_axes'] if name == 'actor' else ['base_lin_vel'] + core + [
            'foot_height', 'foot_air_time', 'foot_contact', 'foot_contact_forces', 'new_axes']
        if list(group.terms) != expected or not group.concatenate_terms or group.history_length not in (None, 0):
            raise ValueError('Observation group differs from the inspected command layout')
        for term_name in core + ['new_axes']:
            term = group.terms[term_name]
            func = term.func
            if func.__name__ != 'part' or Path(func.__code__.co_filename).resolve() != (TASK / 'boundary.py').resolve():
                raise ValueError('Mapped observation dependency changed: ' + term_name)
            slices = dict(base_ang_vel=(0, 3), projected_gravity=(3, 6), joint_pos=(6, 27),
                joint_vel=(27, 48), actions=(48, 69), command=(69, 72), new_axes=(72, 81))
            start, stop = slices[term_name]
            # Original source config aliases only actions/command across groups.
            # Their selected values are identical in Interface.observe for native
            # and delayed paths; all sensed state/new-axis slices stay distinct.
            native = True if term_name in ('actions', 'command') else name == 'critic'
            if term.params != dict(start=start, stop=stop, native=native):
                raise ValueError('Unexpected mapped state/history slice: ' + name + '/' + term_name)
        if name == 'critic':
            if group.terms['base_lin_vel'].func.__name__ != 'critic_linvel':
                raise ValueError('Privileged velocity dependency changed')
            for term_name in ('foot_height', 'foot_air_time', 'foot_contact', 'foot_contact_forces'):
                func = group.terms[term_name].func
                if func.__name__ != term_name or func.__module__ != 'mjlab.tasks.velocity.mdp.observations':
                    raise ValueError('Privileged foot dependency changed: ' + term_name)
        command = group.terms['command']
        if command.func.__name__ != 'part' or Path(command.func.__code__.co_filename).resolve() != (TASK / 'boundary.py').resolve():
            raise ValueError('Command observation function changed')
        if any(getattr(command, key) is not None for key in ('noise', 'clip', 'scale')):
            raise ValueError('Command observation has an unhandled transform')
        if any(getattr(command, key) != 0 for key in ('delay_min_lag', 'delay_max_lag',
                'delay_hold_prob', 'delay_update_period', 'history_length')):
            raise ValueError('Command observation has delay or history')


def command_observations(cached, requested):
    """Clone the cached packet and replace only direct command columns.

    Do not recompute groups: that can resample noise or push delay buffers.
    Do not mutate ObservationManager._obs_buffer or previous raw actions.
    """
    observations = cached.clone()
    for name, width, start in (('actor', 81, 69), ('critic', 96, 72)):
        values = observations[name]
        if values.ndim != 2 or values.shape[1] != width:
            raise ValueError('Unexpected observation shape for command refresh')
        values[:, start:start + 3] = values.new_tensor(requested)
    return observations


def load_native():
    """Load existing task/recorder without copying or editing their code."""
    sys.path.insert(0, str(TASK))
    spec = importlib.util.spec_from_file_location('ape0_original_gait_launcher', TASK / 'run.py')
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    return native


def verify_inputs(method_path=None):
    method = json.loads(Path(method_path or DATA / 'contract.json').read_text())
    for entry in method['inputs'] + method['implementation']:
        if ref(ROOT / entry['path']) != entry:
            raise ValueError('Frozen source changed: ' + entry['path'])
    for entry in method['library_sources']:
        if library_ref(entry['module'].removeprefix('mjlab/')) != entry:
            raise ValueError('Installed MJLab source changed: ' + entry['module'])
    if ref(CHECKPOINT)['sha256'] != CHECKPOINT_SHA:
        raise ValueError('Candidate identity changed')
    pointer = ROOT / 'robot/model/current-model.json'
    if Path(os.environ.get('APE0_MODEL_POINTER', pointer)).resolve() != pointer.resolve():
        raise ValueError('This preparation pins the selected model pointer')
    selected = json.loads(pointer.read_text())
    seal = json.loads((ROOT / selected['rigid_MJCF']['path']).with_name('evidence-seal.json').read_text())
    for section in ('models', 'meshes', 'contracts'):
        for entry in seal[section]:
            actual = ref(ROOT / entry['path'])
            if actual['sha256'] != entry['sha256'] or actual['bytes'] != entry['bytes']:
                raise ValueError('Selected model dependency changed: ' + entry['path'])
    return method


def collect(profile_name, seed, out, device='cpu', method_path=None, policy_name='candidate200'):
    """One world, one full attempted episode; never called by preparation."""
    method_path = Path(method_path or DATA / 'contract.json')
    frozen = verify_inputs(method_path)
    if policy_name not in frozen.get('permitted_policies', ['candidate200']):
        raise ValueError('Policy comparison has not been frozen in this method')
    if profile_name not in frozen.get('permitted_profiles', PROFILES):
        raise ValueError('Profile is outside the frozen comparison')
    plan = json.loads((Path(__file__).parent / 'turn-plan.json').read_text())
    if seed not in plan['proposed_start_seeds']:
        raise ValueError('Seed is outside the frozen proposed families')
    profile = PROFILES[profile_name]
    out = Path(out)
    if out.exists():
        raise FileExistsError('Keep each original visit in a fresh directory')
    out.mkdir(parents=True)
    report = dict(schema='cove50.basic-gait.turn-replay.v1', profile=asdict(profile),
        seed=seed, device=device, method=ref(method_path), policy=policy_name, status='starting',
        optimizer_updates=0, ppo_rounds=0, distinct_new_policies=0,
        native_physics_steps=0, simulator_transitions=0, reset_at_command_change=False)
    native = env = recorder = None
    start = time.monotonic()
    try:
        native = load_native()
        torch = native.torch
        torch.manual_seed(seed)
        cfg = native.bind(native.morphology_cfg(native.runtime_package().robot_cfg(seed=seed), play=True))
        cfg.scene.num_envs = 1
        cfg.seed = seed
        profile.configure(cfg)
        validate_command_columns(cfg)
        cfg.recorders = {'full_episodes': native.RecorderTermCfg(
            func=native.Episodes, params={'directory': str(out / 'episodes')})}
        env = native.ManagerBasedRlEnv(cfg=cfg, device=device)
        original_step = env.sim.step
        def counted_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            report['native_physics_steps'] += 1
            return result
        env.sim.step = counted_step
        recorder = env.recorder_manager._terms[0]
        vec = native.RslRlVecEnvWrapper(env)
        agent = native.agent_config()
        runner = native.MjlabOnPolicyRunner(vec, asdict(agent), log_dir=None, device=device)
        report['transfer'] = native.initialize(runner, env, checkpoint=CHECKPOINT) if policy_name == 'candidate200' else native.initialize(runner, env, source_checkpoint=SOURCE_CHECKPOINT)
        report['runtime_versions'] = native.runtime_versions()
        report['rewards'] = native.reward_recipe(cfg)
        report['numerical_options'] = native.serializable(cfg.sim)
        policy = runner.get_inference_policy(device=device)
        term = env.command_manager.get_term('twist')
        with torch.inference_mode():
            for control in range(round(profile.duration_s / env.step_dt)):
                requested = profile.apply(term, control)
                # MJLab returns its existing observation cache here. Preserve
                # that exact sensor/history packet; refresh only direct command.
                observations = command_observations(vec.get_observations(), requested)
                command = torch.tensor(requested, device=device, dtype=observations['actor'].dtype)[None, :]
                if not torch.equal(observations['actor'][:, 69:72], command):
                    raise ValueError('Actor command differs from the pre-action schedule')
                if not torch.equal(observations['critic'][:, 72:75], command):
                    raise ValueError('Value command differs from the pre-action schedule')
                env.obs_buf = observations
                recorder.cache_inputs(runner.alg.critic(observations))
                env.step(policy(observations, stochastic_output=False))
                # Same full-motion continuation used by original native replay.
                # Terminal flags remain in the recorder; no reset is performed.
                env._manual_reset_pending[:] = False
        recorder.finish(0, 'requested_horizon')
        report['status'] = 'completed'
    except BaseException as exc:
        report.update(status='interrupted', error=type(exc).__name__ + ': ' + str(exc),
                      traceback=traceback.format_exc()[-5000:])
        if recorder is not None:
            recorder.finish(0, 'collector_interruption_censoring')
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                report['close_error'] = str(exc)
        report.update(wall_s=time.monotonic() - start,
            world_slots_constructed=int(env is not None),
            independent_start_families=int(recorder is not None and bool(recorder.initial)),
            simulator_transitions=recorder.transitions if recorder is not None else 0,
            complete_or_terminated_episodes=recorder.completed if recorder is not None else 0,
            censored_episodes=recorder.censored if recorder is not None else 0)
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--method', type=Path)
    parser.add_argument('--policy', choices=('candidate200', 'source-bc'), default='candidate200')
    args = parser.parse_args()
    collect(args.profile, args.seed, args.out, args.device, args.method, args.policy)


if __name__ == '__main__':
    main()
