"""Run the new lesson using a counted projection of the proven R2 runner."""
from compose import project

_source = project('control/skills/forward-stop-learning/run.py',
    '9642ea8ce6a16642859f1a123abbe8961eb90454590f6d5e720e9ac2240389a2', [
        ('from schedule import PROFILES, Schedule, configure, recorder_class',
         'from schedule import PROFILES, Schedule, configure, recorder_class, verify_live', 1),
        ('data/forward-stop-learning-20261030', 'data/forward-entry-braking-next-20261030', 2),
        ("schema='cove50.forward-stop-native.v1'", "schema='cove50.forward-entry-braking-native.v1'", 1),
        ("'--updates', '50'", "'--updates', '24'", 1),
        ("env._forward_stop_schedule = Schedule(env, profile, args.out)",
         "env._forward_stop_schedule = Schedule(env, profile, args.out)\n        verify_live(env)", 1),
        ("def journaled_update(*a, **kw):\n            result = update(*a, **kw)",
         "def journaled_update(*a, **kw):\n            verify_live(env)\n            result = update(*a, **kw)\n            verify_live(env)", 1),
        ("learning_rate=runner.alg.learning_rate, losses=native.serializable(result),",
         "learning_rate=runner.alg.learning_rate, losses=native.serializable(result),\n                lesson_live_recipe=verify_live(env),", 1),
        ("env, runner = live\n            report.update", "env, runner = live\n            verify_live(env)\n            report.update", 1),
    ])
exec(compile(_source, __file__ + ':retained-runner-projection', 'exec'), globals())
