"""Freeze original72/87 shared BC expansion and four matched pure turns.

Offline model inference only; native initialize is exercised with actual RSL
models and an inert curriculum stub, never an environment or physics step.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel

from replay import ROOT, TASK, SOURCE_CHECKPOINT, ref, load_native
from prepare import frozen_interface, decoder_reference

HERE = Path(__file__).parent
SOURCE_SHA = '0e8c6862108bd880e330fd8c07d19bf012d4a034f8be6a2ef808de190e5f301f'


def model(observation, group, outputs, stochastic=False):
    return MLPModel(observation, dict(actor=['actor'], critic=['critic']), group, outputs,
        hidden_dims=[512, 256, 128], activation='elu', obs_normalization=True,
        distribution_cfg=dict(class_name='rsl_rl.modules.distribution.GaussianDistribution',
            init_std=.2, std_type='scalar') if stochastic else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidate-method', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('Keep each original source comparison freeze')
    torch.set_num_threads(1)
    if ref(SOURCE_CHECKPOINT)['sha256'] != SOURCE_SHA:
        raise ValueError('Original shared BC source changed')
    blocked = []
    def forbidden(*args, **kwargs):
        blocked.append('native operation')
        raise AssertionError('Source preparation does not construct or step a model')
    class NoModel:
        from_xml_path = staticmethod(forbidden)
        from_xml_string = staticmethod(forbidden)
    with ExitStack() as stack:
        stack.enter_context(patch.object(mujoco, 'MjModel', NoModel))
        for name in ('mj_forward', 'mj_step', 'mj_integratePos', 'mj_inverse', 'mj_kinematics'):
            stack.enter_context(patch.object(mujoco, name, forbidden))
        native = load_native()
        trace = ROOT / 'artifacts/cove50-walk-resume/runpod-fit-r2-20261026/evidence/after-136101/episodes/episode-000001-slot-0000.npz'
        with np.load(trace) as n:
            observations = TensorDict(dict(actor=torch.tensor(n['input_observation81'].copy()),
                critic=torch.tensor(n['input_privileged_observation96'].copy())), batch_size=[400])
        actor, critic = model(observations, 'actor', 24, True), model(observations, 'critic', 1)
        optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
        runner = SimpleNamespace(device='cpu', alg=SimpleNamespace(actor=actor, critic=critic, optimizer=optimizer))
        curriculum_calls = []
        env = SimpleNamespace(num_envs=1, device='cpu',
            curriculum_manager=SimpleNamespace(compute=lambda **kw: curriculum_calls.append(kw['env_ids'].tolist())))
        transfer = native.initialize(runner, env, source_checkpoint=SOURCE_CHECKPOINT)
        source = torch.load(SOURCE_CHECKPOINT, map_location='cpu', weights_only=False)
        old_obs = TensorDict(dict(actor=observations['actor'][:, :72], critic=observations['critic'][:, :87]), batch_size=[400])
        old_actor, old_critic = model(old_obs, 'actor', 21, True), model(old_obs, 'critic', 1)
        old_actor.load_state_dict(source['actor_state_dict']); old_critic.load_state_dict(source['critic_state_dict'])
        for instance in (actor, critic, old_actor, old_critic): instance.eval()
        with torch.inference_mode():
            expanded_raw, original_raw = actor(observations), old_actor(old_obs)
            expanded_value, original_value = critic(observations), old_critic(old_obs)
            altered = observations.clone()
            altered['actor'][:, 72:] += 100.
            altered['critic'][:, 87:] -= 100.
            altered_raw, altered_value = actor(altered), critic(altered)
        first = actor.state_dict()['mlp.0.weight']; last = actor.state_dict()['mlp.6.weight']
        assert torch.equal(first[:, :72], source['actor_state_dict']['mlp.0.weight'])
        assert torch.count_nonzero(first[:, 72:]) == 0
        assert torch.count_nonzero(last[21:]) == 0 and torch.count_nonzero(expanded_raw[:, 21:]) == 0
        assert torch.equal(expanded_raw, altered_raw) and torch.equal(expanded_value, altered_value)
        torch.testing.assert_close(expanded_raw[:, :21], original_raw, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(expanded_value, original_value, atol=1e-5, rtol=1e-5)
        report_path = ROOT / 'artifacts/cove50-walk-resume/runpod-fit-r2-20261026/evidence/fit-through-200/report.json'
        interface = frozen_interface(json.loads(report_path.read_text())['selected_interface'])
        targets = interface.target(expanded_raw)
        reference = decoder_reference(torch.cat([original_raw, torch.zeros(400, 3)], dim=1), interface)
        previous = expanded_raw
        values = interface.observe(torch.zeros(400, 22), torch.zeros(400, 22), torch.zeros(400, 3),
            torch.zeros(400, 3), previous, observations['actor'][:, 69:72])
        assert torch.equal(values[:, 48:69], previous[:, :21])
        assert torch.equal(values[:, 78:81], previous[:, 21:])
        parity = dict(schema='cove50.basic-gait.original-bc-expansion-parity.v1', source=ref(SOURCE_CHECKPOINT),
            actual_initializer=ref(TASK/'checkpoint.py'), transfer=transfer,
            retained_input_trace=ref(trace), input_frames=400,
            actor21_prefix_max_abs_difference=float((expanded_raw[:,:21]-original_raw).abs().max()),
            critic_value_max_abs_difference=float((expanded_value-original_value).abs().max()),
            decoded_target_max_abs_difference_rad=float((targets-reference).abs().max()),
            added_mean_output_max_abs=float(expanded_raw[:,21:].abs().max()),
            added_features_do_not_change_output=True, old_action_history21_preserved=True,
            actor_normalizer_prefix_unchanged=bool(torch.equal(actor.obs_normalizer._mean[:,:72],source['actor_state_dict']['obs_normalizer._mean']) and torch.equal(actor.obs_normalizer._std[:,:72],source['actor_state_dict']['obs_normalizer._std'])),
            critic_normalizer_prefix_unchanged=bool(torch.equal(critic.obs_normalizer._mean[:,:87],source['critic_state_dict']['obs_normalizer._mean']) and torch.equal(critic.obs_normalizer._std[:,:87],source['critic_state_dict']['obs_normalizer._std'])),
            source_iteration_label=source['iter'], new_ppo_rounds=runner.current_learning_iteration,
            fresh_optimizer_moment_count=len(optimizer.state), source_curriculum_counter=env.common_step_counter,
            curriculum_stub_calls=curriculum_calls, prohibited_native_calls=len(blocked),
            new_native_episodes=0, new_physics_steps=0, optimizer_updates=0,
            interpretation='Original shared BC actor/value expanded through unchanged native initialize(source_checkpoint). All original21 raw outputs/history retained; extra9 input columns/new3 means zero. No fit or expanded checkpoint written.')
    args.out.mkdir(parents=True)
    parity_path = args.out/'parity.json'; parity_path.write_text(json.dumps(parity, indent=2)+'\n')
    method = deepcopy(json.loads(args.candidate_method.read_text()))
    method.update(schema='cove50.basic-gait.original-bc-pure-turn-method.v1',
        permitted_policies=['source-bc'], permitted_profiles=['pure-left','pure-right'],
        source_checkpoint=ref(SOURCE_CHECKPOINT), source_expansion_parity=ref(parity_path),
        matched_candidate_method=ref(args.candidate_method),
        counts='Four matched original shared BC pure-turn visits:2seeds×2directions,1world sequentially,800controls/64000physics planned,0fit.')
    method['implementation'] += [ref(Path(__file__))]
    method['inputs'] += [ref(parity_path), ref(args.candidate_method)]
    method_path = args.out/'method.json'; method_path.write_text(json.dumps(method, indent=2)+'\n')
    print(json.dumps(dict(method=ref(method_path), parity=parity)))


if __name__ == '__main__': main()
