"""Exact finite named lesson, retained command/validation/custody helpers."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from compose import project

_source = project('control/skills/forward-stop-learning/compute/common.py', 'db0de534e23373397684625d5333f4467faa0e47172e65ebfd14f30d2759a581', [('data/forward-stop-learning-20261030', 'data/forward-heading-next-20261030', 1), ('cove50-forward-stop-20261030-r2', 'cove50-forward-heading-20261030-r1', 1), ('(50,38400,1000,20,45000,3600000)', '(48,36864,960,20,43464,3477120)', 1), ("['train50'] + ['after-' + p + '-' + str(s) for s in cfg['family_seeds']", "['train48'] + ['after-' + p + '-' + str(s) for s in cfg['after_family_seeds']", 1), ('(17,[4000],200,25200)', '(17,[4480],224,25776)', 1), ("!= [5000] or outer['final_common_counter'] != 26400", "!= [5440] or outer['final_common_counter'] != 26928", 1)])
exec(compile(_source, __file__+':retained-projection', 'exec'), globals())
