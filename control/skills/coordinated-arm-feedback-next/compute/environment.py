"""Reuse exercised isolated pinned environment operations and exact recipe."""
from common import ARM, ROOT
exec(compile((ROOT / ARM / 'environment.py').read_text(), str(__file__), 'exec'), globals())
