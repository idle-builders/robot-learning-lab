"""Narrow immutable retained source projection."""
from common import project

_source = project('control/skills/forward-heading/compute/session.py', '63d4998bf6fa9f40de766ed76bf70d59ad236b5d99db7b31f2b5bf851e8269d1', [('heading_retained_session', 'cycle_heading_retained_session', 1)])
exec(compile(_source, __file__+':retained-projection', 'exec'), globals())
