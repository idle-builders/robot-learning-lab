"""Exact, counted projections of the retained runner; no copied runtime forest."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def project(relative, sha256, replacements):
    path = ROOT / relative
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError('Retained projection source differs: ' + relative)
    source = raw.decode()
    for old, new, count in replacements:
        if source.count(old) != count:
            raise ValueError('Projection occurrence count differs: ' + repr(old))
        source = source.replace(old, new)
    return source
