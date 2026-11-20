"""Reuse the exercised explicit launcher/session transport, with counted labels.

Only the dispatched function name, private namespace and returned index schema
change. All commands, native semantics and budgets belong to this common.py.
"""
from common import FORWARD, ROOT

source = (ROOT/FORWARD/'launch.py').read_text()
for old, new, count in (
        ("function_name='run_forward_stop'", "function_name='run_coordinated_arm_reference'", 1),
        ('robot-learning-lab/v1/forward-stop-learning/', 'robot-learning-lab/v1/coordinated-arm-reference/', 1),
        ('cove50.forward-stop.closed-native-index.v1', 'cove50.coordinated-arm-reference.closed-native-index.v1', 1),
        ("record.get('inspection_attempts',0)>=8", "record.get('inspection_attempts',0)>=4", 1),
        ('Finite eight foreground inspection attempts exhausted', 'Finite four foreground inspection attempts exhausted', 1)):
    if source.count(old) != count: raise ValueError('Inherited launcher anchor differs: ' + old)
    source = source.replace(old, new)
exec(compile(source, str(__file__), 'exec'), globals())
