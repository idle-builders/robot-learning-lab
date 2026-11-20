"""Same one-submit/sparse-status/authenticated-restore protocol, new named job."""
from common import project

_source = project('control/skills/forward-stop-learning/compute/launch.py',
    'c6c450cfdb902ed2fa138500f9611c034cc465c14546e84197d5df140279e04e', [
        ("worker_code=(HERE/'worker.py').read_text().split(\"\\n\\nif __name__ == '__main__':\")[0]",
         "from worker import WORKER_SOURCE\n    worker_code=WORKER_SOURCE.split(\"\\n\\nif __name__ == '__main__':\")[0]", 1),
        ("execution/native-index-r2.json", "execution/native-index-r1.json", 1),
        ('robot-learning-lab/v1/forward-stop-learning/', 'robot-learning-lab/v1/forward-entry-braking/', 1),
    ])
exec(compile(_source, __file__ + ':retained-launch-projection', 'exec'), globals())
