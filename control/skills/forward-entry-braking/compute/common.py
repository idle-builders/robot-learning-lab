"""New finite lesson identities, using the unchanged retained custody functions."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compose import project

_source = project('control/skills/forward-stop-learning/compute/common.py',
    'db0de534e23373397684625d5333f4467faa0e47172e65ebfd14f30d2759a581', [
        ('data/forward-stop-learning-20261030', 'data/forward-entry-braking-next-20261030', 1),
        ('cove50-forward-stop-20261030-r2', 'cove50-forward-entry-braking-20261030-r1', 1),
        ('(50,38400,1000,20,45000,3600000)', '(24,18432,480,30,28332,2266560)', 1),
        ("['train50'] + ['after-' + p + '-' + str(s) for s in cfg['family_seeds']", "['train24'] + ['after-' + p + '-' + str(s) for s in cfg['after_family_seeds']", 1),
        ("!= [5000] or outer['final_common_counter'] != 26400", "!= [4480] or outer['final_common_counter'] != 25776", 1),
    ])
exec(compile(_source, __file__ + ':retained-common-projection', 'exec'), globals())
