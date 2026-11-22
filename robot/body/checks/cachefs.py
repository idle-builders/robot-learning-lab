"""Where the content-addressed check caches live, and writing them safely while several processes use them.

Every cache under it is keyed by the sha256 of the file it was made from (STEP tessellations, FEM meshes, closed screening
solids, standard-part registrations), so one store can serve the main checkout and all its Git worktrees: a worktree
session no longer starts cold (the standard-part registrations alone cost 177 s on one part, measured 2026-11-22) or
copies 300-500 MB of cache by hand. LAB_SHARED_CACHE=0 keeps a checkout's caches to itself; LAB_SHARED_CACHE=<dir> uses
<dir> as the .cache/work of the caches (an empty one measures a cold start).
Writers go through a temporary file and an atomic rename, so a reader in another process never opens half a file; readers
treat an unreadable entry as missing and rebuild it."""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _main_checkout() -> Path:
    """The main checkout of this repository (ROOT itself unless ROOT is a linked worktree)."""
    try:
        common = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ROOT
    if not common:
        return ROOT
    main = (ROOT / common).resolve().parent
    return main if (main / ".git").is_dir() else ROOT


_WORK = None


def work() -> Path:
    """.cache/work of the main checkout (shared with the worktrees), or of this checkout with LAB_SHARED_CACHE=0."""
    global _WORK
    if _WORK is None:
        v = os.environ.get("LAB_SHARED_CACHE", "1")
        _WORK = (ROOT / ".cache/work") if v == "0" else Path(v) if v not in ("", "1") else _main_checkout() / ".cache/work"
    return _WORK


def _tmp(f: Path) -> Path:
    return f.with_name(f".{f.name}.{os.getpid()}.tmp")


def savez(path: Path, /, **arrays):
    """np.savez_compressed to path atomically (positional-only: the caches name an array `f`)."""
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = _tmp(path)
    with open(t, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(t, path)


def load_npz(f: Path):
    """{name: array} of a cache entry, or None when it is missing or unreadable (written by an old non-atomic writer)."""
    import numpy as np
    try:
        with np.load(f) as d:
            return {k: d[k] for k in d.files}
    except (OSError, ValueError, EOFError, KeyError):
        return None


def write_text(f: Path, text: str):
    f = Path(f)
    f.parent.mkdir(parents=True, exist_ok=True)
    t = _tmp(f)
    t.write_text(text)
    os.replace(t, f)


def read_json(f: Path, default=None):
    try:
        return json.loads(Path(f).read_text())
    except (OSError, ValueError):
        return default


def update_json(f: Path, updates: dict) -> dict:
    """Merge updates into the JSON object in f under an exclusive lock (other processes' entries are kept); returns it."""
    f = Path(f)
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f.with_name(f".{f.name}.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        d = read_json(f, {}) or {}
        d.update(updates)
        write_text(f, json.dumps(d))
    return d
