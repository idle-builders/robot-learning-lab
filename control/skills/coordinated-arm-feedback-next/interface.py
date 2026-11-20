"""Byte-exact previously reviewed120/174 residual interface implementation."""
from pathlib import Path

source = (Path(__file__).resolve().parents[3] / 'control/skills/coordinated-arm-feedback/interface.py').read_text()
exec(compile(source, str(__file__), 'exec'), globals())
