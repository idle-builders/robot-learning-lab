"""Same durable finite protocol, rebound to the selected next arm lesson."""
from pathlib import Path

BASE = Path(__file__).resolve().parents[4] / 'control/skills/coordinated-arm-feedback/compute/common.py'
source = BASE.read_text()
changes = [
    ('data/coordinated-arm-feedback-20261030', 'data/coordinated-arm-feedback-next-20261030', 1),
    ('cove50-coordinated-arm-feedback-20261030-r1', 'cove50-coordinated-arm-feedback-next-20261030-r1', 1),
    ('simulator_controls=8000,physics_substeps=640000,PPO_rounds=4,optimizer_updates=80',
     'simulator_controls=48000,physics_substeps=3840000,PPO_rounds=12,optimizer_updates=240', 1)]
for old, new, count in changes:
    if source.count(old) != count:
        raise ValueError('Exact retained common anchor differs: ' + old)
    source = source.replace(old, new)
exec(compile(source, str(__file__), 'exec'), globals())
