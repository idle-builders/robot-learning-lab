"""Exact supported272 continuation; stock48-step PPO and causal reward capture."""
from compose import project

_source = project('control/skills/forward-stop-learning/run.py',
    '9642ea8ce6a16642859f1a123abbe8961eb90454590f6d5e720e9ac2240389a2', [
        ('env.common_step_counter != 25200', 'env.common_step_counter != 26928', 1),
        ('from schedule import PROFILES, Schedule, configure, recorder_class',
         'from schedule import PROFILES, Schedule, configure, recorder_class, verify_live, cycle_term', 1),
        ('data/forward-stop-learning-20261030', 'data/forward-heading-correction-20261030', 2),
        ("schema='cove50.forward-stop-native.v1'", "schema='cove50.forward-heading-correction-native.v1'", 1),
        ("'--updates', '50'", "'--updates', '32'", 1),
        ('original_runner = native.MjlabOnPolicyRunner',
         'original_agent_config = native.agent_config\n    def agent_config():\n        agent = original_agent_config()\n        agent.num_steps_per_env = 48\n        return agent\n    native.agent_config = agent_config\n    original_runner = native.MjlabOnPolicyRunner', 1),
        ('env._forward_stop_schedule = Schedule(env, profile, args.out)',
         "env._forward_stop_schedule = Schedule(env, profile, args.out)\n        verify_live(env)\n        report['actual_rollout'] = dict(configured=runner.cfg['num_steps_per_env'], storage=runner.alg.storage.num_transitions_per_env)\n        if args.mode == 'train' and report['actual_rollout'] != dict(configured=48, storage=48):\n            raise ValueError('Actual stock48 rollout required')", 1),
        ("np.savez_compressed(args.out / 'initialization.npz',",
         "state.update({'cycle_'+k: v for k, v in cycle_term(env).snapshot().items()})\n        np.savez_compressed(args.out / 'initialization.npz',", 1),
        ('def journaled_update(*a, **kw):\n            result = update(*a, **kw)',
         'def journaled_update(*a, **kw):\n            verify_live(env)\n            result = update(*a, **kw)\n            verify_live(env)', 1),
        ('learning_rate=runner.alg.learning_rate, losses=native.serializable(result),',
         'learning_rate=runner.alg.learning_rate, losses=native.serializable(result),\n                lesson_live_recipe=verify_live(env),', 1),
        ('env, runner = live\n            report.update',
         "env, runner = live\n            verify_live(env)\n            import numpy as np\n            snapshot = cycle_term(env).snapshot()\n            np.savez_compressed(args.out / 'cycle-final-state.npz', **{k: v.cpu().numpy().copy() for k, v in snapshot.items()})\n            report['cycle_final_state_finite'] = bool(snapshot['finite'].all())\n            report['actual_stock_rollout_controls'] = runner.cfg['num_steps_per_env']\n            report.update", 1),
    ])
exec(compile(_source, __file__ + ':retained-runner-projection', 'exec'), globals())
