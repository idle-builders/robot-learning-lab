"""Regenerate a mapped seed and compact interface/reward audit, with no fit."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch
from adapter import Interface, CHECKPOINT, ROOT, expand_checkpoint, ref

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'task'))
from config import source_forward_cfg, reward_recipe, serializable, SAVED_ENV


def mlp(state, obs):
    x = (obs - state['obs_normalizer._mean']) / (state['obs_normalizer._std'] + .01)
    for layer in (0, 2, 4, 6):
        x = torch.nn.functional.linear(x, state[f'mlp.{layer}.weight'], state[f'mlp.{layer}.bias'])
        if layer != 6: x = torch.nn.functional.elu(x)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/cove50-walk-resume/r1-20261026')
    parser.add_argument('--report', type=Path, default=ROOT / 'data/walk-resume-r1-20261026/preparation.json')
    args = parser.parse_args()
    interface = Interface()
    expected = '0e8c6862108bd880e330fd8c07d19bf012d4a034f8be6a2ef808de190e5f301f'
    if ref(CHECKPOINT)['sha256'] != expected: raise ValueError('Source checkpoint differs')
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    expanded = expand_checkpoint(saved)
    torch.manual_seed(135726)
    obs = torch.randn(128, 81)
    with torch.inference_mode():
        old = mlp(saved['actor_state_dict'], obs[:, :72])
        new = mlp(expanded['actor_state_dict'], obs)
        old_critic = mlp(saved['critic_state_dict'], torch.zeros(128, 87))
        new_critic = mlp(expanded['critic_state_dict'], torch.zeros(128, 96))
    # Explicitly exercise the pre-existing 21 raw outputs, including channels
    # without physical motors, rather than reconstructing action history.
    q = torch.zeros(2, 22); qd = torch.zeros_like(q)
    history = torch.arange(48, dtype=torch.float32).reshape(2, 24) / 100
    observed = interface.observe(q, qd, torch.zeros(2, 3),
        torch.tensor([[0., 0., -1.]]).expand(2, -1), history, torch.zeros(2, 3))
    if not torch.equal(observed[:, 48:69], history[:, :21]):
        raise ValueError('Legacy raw action history was altered')
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / 'mapped-seed.pt'
    torch.save(expanded, checkpoint)
    cfg = source_forward_cfg()
    report = dict(schema='cove50.walk-resume-preparation.v1', interface=interface.contract(),
        generator=ref(Path(__file__)), adapter=ref(Path(__file__).with_name('adapter.py')),
        source_task=ref(ROOT / 'control/common/mjlab_ape/src/ape_walk/task.py'),
        source_environment=ref(SAVED_ENV), source_agent=ref(SAVED_ENV.parent / 'agent.yaml'),
        task_adapter=ref(ROOT / 'control/skills/walker/task/config.py'),
        checkpoint=ref(checkpoint), actor_observation_shape=[128, 81], raw_action_shape=list(new.shape),
        physical_target_shape=list(interface.target(new).shape),
        legacy_actor_prefix_max_abs_difference=float((old-new[:, :21]).abs().max()),
        new_head_max_abs_initial_mean=float(new[:, 21:].abs().max()),
        legacy_critic_prefix_max_abs_difference=float((old_critic-new_critic).abs().max()),
        source_rewards=reward_recipe(cfg), source_commands=serializable(cfg.commands),
        source_events=serializable(cfg.events), source_terminations=serializable(cfg.terminations),
        source_curriculum=serializable(cfg.curriculum),
        numerical_physics_s=.00025, policy_period_s=.02, source_policy_period_s=.02,
        accounting=dict(simulator_transitions=0, optimizer_updates=0, distinct_new_trained_policies=0),
        evidence_grade=dict(interface_preservation=dict(grade_0_to_5=4., confidence='high for static tensors',
            limitation='Exact actor prefix on 128 synthetic observations; no native closed-loop trace.'),
            changed_body_motion=dict(grade_0_to_5=0., confidence='unmeasured', limitation='No new native episode in this preparation.')),
        regeneration_command='.venv-control/bin/python control/skills/walker/policy/prepare.py',
        limitations=['GPU/runtime parity remains a separate dependency.',
            'The mapped seed is cheap regeneration; retain source checkpoint and source bodies.',
            'No conclusion about walking or URDF mechanics comes from these static tensors.'])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('checkpoint','legacy_actor_prefix_max_abs_difference',
        'new_head_max_abs_initial_mean','legacy_critic_prefix_max_abs_difference','accounting')}))


if __name__ == '__main__': main()
