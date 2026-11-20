"""Same one-submit/terminal restore client; finite nine sparse checks."""
from pathlib import Path

BASE = Path(__file__).resolve().parents[4] / 'control/skills/coordinated-arm-feedback/compute/launch.py'
# Its composition deliberately imports our next common and uses our HERE.
source = BASE.read_text()
changes = [('run_coordinated_arm_feedback', 'run_coordinated_arm_feedback_next', 1),
           ('robot-learning-lab/v1/coordinated-arm-feedback/', 'robot-learning-lab/v1/coordinated-arm-feedback-next/', 1),
           ('cove50.coordinated-arm-feedback.closed-native-index.v1', 'cove50.coordinated-arm-feedback-next.closed-native-index.v1', 1)]
for old, new, count in changes:
    if source.count(old) != count:
        raise ValueError('Exact retained launcher anchor differs: ' + old)
    source = source.replace(old, new)
# Count both the outer insertion and the inner shared-source changes.
anchor = "exec(compile(source,str(__file__),'exec'),globals())"
if source.count(anchor) != 1:
    raise ValueError('Exact retained launcher composition boundary differs')
source = source.replace(anchor, """for old,new in [
    ("record.get('inspection_attempts',0)>=8","record.get('inspection_attempts',0)>=9"),
    ('Finite eight foreground inspection attempts exhausted','Finite nine foreground inspection attempts exhausted')]:
    if source.count(old)!=1:raise ValueError('Exact sparse inspection anchor differs: '+old)
    source=source.replace(old,new)
exec(compile(source,str(__file__),'exec'),globals())""")
exec(compile(source, str(__file__), 'exec'), globals())
