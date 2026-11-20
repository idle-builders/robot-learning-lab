"""Exact shared one-submit/sparse-status/restore with explicit counted changes."""
from common import FORWARD,ROOT
source=(ROOT/FORWARD/'launch.py').read_text()
changes=[
    ("def submit(evidence, work, release):\n    plan=", "def submit(evidence, work, release):\n    from runpod_flash.runtime.serialization import serialize_kwargs\n    plan=",1),
    ("function_name='run_forward_stop'","function_name='run_coordinated_arm_feedback'",1),
    ('robot-learning-lab/v1/forward-stop-learning/','robot-learning-lab/v1/coordinated-arm-feedback/',1),
    ('cove50.forward-stop.closed-native-index.v1','cove50.coordinated-arm-feedback.closed-native-index.v1',1),
    ("worker_code=(HERE/'worker.py').read_text().split(","worker_code=load(HERE/'worker.py','feedback_composed_dispatch').source().split(",1),
    ('native-index-r2.json','native-index-r1.json',1)]
for old,new,count in changes:
    if source.count(old)!=count:raise ValueError('Exact shared launcher anchor differs: '+old)
    source=source.replace(old,new)
exec(compile(source,str(__file__),'exec'),globals())
