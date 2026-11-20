"""Same actual isolated extracted worker entry proof, exact17next commands."""
from pathlib import Path

BASE = Path(__file__).resolve().parents[4] / 'control/skills/coordinated-arm-feedback/compute/prove.py'
source = BASE.read_text()
changes = [('control/skills/coordinated-arm-feedback/compute/worker.py',
            'control/skills/coordinated-arm-feedback-next/compute/worker.py', 1),
           ("startup['stages']!=9", "startup['stages']!=17", 1),
           ('cove50.coordinated-arm-feedback-extracted-startup-proof.v1',
            'cove50.coordinated-arm-feedback-next-extracted-startup-proof.v1', 1),
           ('exact9childcommands', 'exact17childcommands', 1)]
for old, new, count in changes:
    if source.count(old) != count:
        raise ValueError('Exact retained extracted-proof anchor differs: ' + old)
    source = source.replace(old, new)
exec(compile(source, str(__file__), 'exec'), globals())
