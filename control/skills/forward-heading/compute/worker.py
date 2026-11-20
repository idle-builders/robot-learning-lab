"""Actual serialized remote function is this exact retained projection."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from compose import project

WORKER_SOURCE = project('control/skills/forward-stop-learning/compute/worker.py', '5255d296ab94cf93aa46259fb37397c8bbc9a28029fda8473da44eb6a32436a7', [('control/skills/forward-stop-learning', 'control/skills/forward-heading', 4), ('train50', 'train48', 2), ("=='baseline200'", "=='baseline224'", 2)])
exec(compile(WORKER_SOURCE, __file__+':retained-projection', 'exec'), globals())
