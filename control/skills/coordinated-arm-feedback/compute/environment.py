"""Unchanged isolated pinned environment operations, rebound compact config."""
from common import ARM, ROOT
exec(compile((ROOT/ARM/'environment.py').read_text(),str(__file__),'exec'),globals())
