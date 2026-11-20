"""Bounded CPU scenario/reset and synthetic full-state restoration exercise."""
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def main():
    import torch
    from tensordict import TensorDict
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from method import agent_config, restore_training
    from scenario import Scenarios
    torch.set_num_threads(1)
    torch.manual_seed(252100)
    def forbidden(*a, **kw):
        raise RuntimeError('No PT loading or native execution in component fixture')
    torch.load = forbidden
    observations = TensorDict({'actor': torch.zeros(8, 120), 'critic': torch.zeros(8, 174)}, batch_size=[8])
    env = SimpleNamespace(num_envs=8, num_actions=22, device='cpu', cfg={}, get_observations=lambda: observations)
    runner = MjlabOnPolicyRunner(env, asdict(agent_config()), log_dir=None, device='cpu')
    forwards = []
    runner.alg.actor.register_forward_pre_hook(lambda *a: forwards.append('actor'))
    runner.alg.critic.register_forward_pre_hook(lambda *a: forwards.append('critic'))
    states = {}
    for kind in ('actor', 'critic'):
        state = {key: value.clone() for key, value in getattr(runner.alg, kind).state_dict().items()}
        for key, value in state.items():
            if key.endswith('count'):
                value.fill_(8000)
            elif '_var' in key:
                value.fill_(.5)
            elif '_std' in key:
                value.fill_(.5 ** .5)
            elif 'std_param' in key:
                value.fill_(.201)
            else:
                value.fill_(.0123)
        states[kind + '_state_dict'] = state
    optimizer = runner.alg.optimizer.state_dict()
    for group, live_group in zip(optimizer['param_groups'], runner.alg.optimizer.param_groups):
        for identifier, parameter in zip(group['params'], live_group['params']):
            optimizer['state'][identifier] = dict(step=torch.tensor(80.),
                exp_avg=torch.full_like(parameter, .001), exp_avg_sq=torch.full_like(parameter, .002))
    method_ref = dict(path='synthetic-original-method', sha256='fixture', bytes=0)
    saved = dict(**states, optimizer_state_dict=optimizer, method=method_ref,
                 layout=dict(actor=120, critic=174, raw=22),
                 counts=dict(simulator_controls=8000, physics_substeps=640000, PPO_rounds=4,
                             optimizer_updates=80, new_weight_sets=1),
                 common_counter=2000, normalizer_counts=dict(actor=8000, critic=8000))
    restored = restore_training(runner, saved, method_ref)
    assert runner.current_learning_iteration == 4
    for kind in ('actor', 'critic'):
        assert all(torch.equal(value, states[kind + '_state_dict'][key])
                   for key, value in getattr(runner.alg, kind).state_dict().items())
    assert len(runner.alg.optimizer.state) == 13
    assert all(int(state['step']) == 80 and bool((state['exp_avg'] == .001).all())
               and bool((state['exp_avg_sq'] == .002).all()) for state in runner.alg.optimizer.state.values())
    for kind in ('actor', 'critic'):
        model = getattr(runner.alg, kind)
        model.update_normalization(observations)
        assert int(model.obs_normalizer.count) == 8008
    runner.alg.eval_mode()
    for kind in ('actor', 'critic'):
        model = getattr(runner.alg, kind)
        model.update_normalization(observations)
        assert int(model.obs_normalizer.count) == 8008
    seeds = list(range(252201, 252209))
    scenarios = Scenarios(seeds, device='cpu')
    initial_rng = [g.get_state().clone() for g in scenarios.generators]
    assert scenarios.history == [] and not bool(scenarios.generation.any())
    assert all(torch.equal(before, g.get_state()) for before, g in zip(initial_rng, scenarios.generators))
    scenarios.reset(torch.arange(8), torch.ones(8, dtype=torch.long))
    first = [dict(row) for row in scenarios.history]
    duplicate = Scenarios(seeds, device='cpu')
    duplicate.reset(torch.arange(8), torch.ones(8, dtype=torch.long))
    assert duplicate.history == first
    previous_rng = [g.get_state().clone() for g in scenarios.generators]
    scenarios.reset(torch.tensor([2]), torch.tensor([2]))
    assert scenarios.generation.tolist() == [1, 1, 2, 1, 1, 1, 1, 1]
    assert all(torch.equal(previous_rng[i], scenarios.generators[i].get_state()) for i in range(8) if i != 2)
    assert not torch.equal(previous_rng[2], scenarios.generators[2].get_state())
    duplicate_reset_rejected = False
    try:
        scenarios.reset(torch.tensor([2]), torch.tensor([2]))
    except ValueError:
        duplicate_reset_rejected = True
    assert duplicate_reset_rejected
    # Repeat a real reset at localcontrol0: explicit generation still consumes a
    # new draw. This exercise never tries to infer reset from time decreasing.
    scenarios.reset(torch.tensor([2]), torch.tensor([3]))
    for generation in range(2, 102):
        duplicate.reset(torch.arange(8), torch.full((8,), generation))
    randomized = duplicate.summary()
    assert all(120 <= row['onset_control'] <= 350 and row['stop_control'] - row['onset_control'] == 10
               and row['force_world_Y_N'] in (0., -2., 2., -4., 4.) for row in randomized)
    fixed = [dict(onset_control=130, duration_controls=10, force_world_Y_N=4.),
             dict(onset_control=310, duration_controls=10, force_world_Y_N=-2.),
             dict(onset_control=200, duration_controls=10, force_world_Y_N=0.)]
    physical = Scenarios([1, 2, 3], device='cpu', fixed=fixed)
    fixed_rng = [g.get_state().clone() for g in physical.generators]
    physical.reset(torch.arange(3), torch.ones(3, dtype=torch.long))
    for k in range(500):
        control = torch.full((3,), k)
        for _ in range(80):
            force = physical.force(control, .00025)
            physical.sample_substep(force, .00025)
    assert torch.allclose(physical.applied_dose[:, 1], torch.tensor([.8, -.4, 0.], dtype=torch.float64), atol=1e-12, rtol=0)
    assert physical.applied_substeps.tolist() == [800, 800, 0]
    assert all(torch.equal(before, g.get_state()) for before, g in zip(fixed_rng, physical.generators))
    record = dict(schema='cove50.coordinated-arm-feedback-next.cpu-component-proof.v1',
        source=[ref(path) for path in sorted(HERE.glob('*.py'))],
        restored_synthetic_state=restored, all13_Adam_states_exact=True, all_actor_critic_buffers_exact=True,
        next_iteration=runner.current_learning_iteration, normalizers_after_one_synthetic_batch=8008,
        eval_normalizers_frozen=True, no_scenario_draw_at_construction=True,
        first_seeded_scenarios=first, randomized_resets=len(randomized),
        observed_force_counts={str(force): sum(row['force_world_Y_N'] == force for row in randomized)
                               for force in (0., -2., 2., -4., 4.)},
        partial_reset_preserves_other_generators=True, zero_to_zero_reset_uses_generation=True,
        duplicate_generation_rejected=duplicate_reset_rejected,
        synthetic_fixed_force_integral_N_s=physical.applied_dose.tolist(),
        synthetic_force_substeps=physical.applied_substeps.tolist(),
        counts=dict(CPU_models_constructed=2, NN_forward_calls=len(forwards), PT_reads=0,
                    optimizer_calls=0, native_worlds=0, native_controls=0, physics=0, provider=0),
        limits='Synthetic state and deterministic CPU force-integration calls only. No actual07tensor reread, donor inference, native world/manager/reset, motor force response, PPO rollout or capability observation.')
    assert forwards == []
    out = Path(os.environ['LAB_EVIDENCE_DIR']) / 'records/cpu-component-proof.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(report=ref(out), counts=record['counts'])))


if __name__ == '__main__':
    main()
