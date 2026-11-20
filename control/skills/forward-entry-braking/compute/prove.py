"""Actual extracted projection startup, CPU lesson fixture and interrupted owner."""
from common import project
_source = project('control/skills/forward-stop-learning/compute/prove.py',
    '1098419a9721d35e2153e4fdf620b8b4aa660a3ef646c6e8ac666cadf94ce6ad', [
        ('control/skills/forward-stop-learning', 'control/skills/forward-entry-braking', 3),
        ('data/forward-stop-learning-20261030', 'data/forward-entry-braking-next-20261030', 1),
        ("startup['stages']!=21", "startup['stages']!=31", 1),
    ])
exec(compile(_source, __file__ + ':retained-proof-projection', 'exec'), globals())
