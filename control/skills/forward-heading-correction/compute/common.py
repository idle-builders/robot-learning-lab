"""Exact finite named lesson, retained command/validation/custody helpers."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from compose import project

_source = project('control/skills/forward-stop-learning/compute/common.py', 'db0de534e23373397684625d5333f4467faa0e47172e65ebfd14f30d2759a581', [('data/forward-stop-learning-20261030', 'data/forward-heading-correction-20261030', 1), ('cove50-forward-stop-20261030-r2', 'cove50-forward-heading-correction-20261030-r1', 1), ('(50,38400,1000,20,45000,3600000)', '(32,49152,640,20,55752,4460160)', 1), ("['train50'] + ['after-' + p + '-' + str(s) for s in cfg['family_seeds']", "['train32'] + ['after-' + p + '-' + str(s) for s in cfg['after_family_seeds']", 1), ('(17,[4000],200,25200)', '(17,[5440],272,26928)', 1), ("!= [5000] or outer['final_common_counter'] != 26400", "!= [6080] or outer['final_common_counter'] != 28464", 1)])
exec(compile(_source, __file__+':retained-projection', 'exec'), globals())
