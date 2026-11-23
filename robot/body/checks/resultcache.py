"""Result cache of the design checks: a check whose inputs have not changed returns its earlier result instead of running.

The inputs are not declared by the checks; they are recorded while a check runs (owner 2026-11-22: "if none of the inputs
to the check changed we should just return the cached result"):
- every file the check child reads below the repository (Python's `open` audit event, which covers json, numpy, trimesh and
  the import of every module), every path resolved through bodylib.src() or hashed through core.sha(), and every STEP
  handed to OpenCascade (build123d.import_step) or gmsh, whose reads are in C++ and invisible to the audit hook;
- the same for Python subprocesses (mount_check under `screws`, the render script): they inherit a sitecustomize that
  records their reads into a log the child merges (robot/body/checks/depsite/);
- named derived sets whose growth changes a result: the body definition files found under robot/ (a new `.bom.yaml`);
- the run options that change results (pose set, search distance, process overrides, FEM variant / mesh size, joints, ...).
A child records the union of what it has read so far for each check it finishes (later checks reuse solids built by
earlier ones in memory), minus the other checks' modules (every module is imported at start; a check depends on its own
module and the modules it imports by name). The manifest, the part settings and the body definitions are files the child
reads, so a change to any of them is a change of input.
Entry key: manifest path, check, the exact scope, the options. An entry is used only if every recorded file still has its
recorded sha256 (size / mtime index of core.sha: cheap), every named set is unchanged and every evidence file it names
still exists. Results with a failure, failed parts, or a time-limited measurement (fit pairs over their budget) are not
stored. Entries live in the shared cache (checks/cachefs.py), with repository-relative paths, so a worktree whose files
are identical gets the main checkout's results. --no-cache (run.py) or CHECKS_RESULT_CACHE=0 runs everything.
Not seen: files read by native code of a subprocess that is not Python (the slicer reads its own profiles in .tools/),
and Python packages outside the repository (.venv): an upgrade there does not invalidate."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

from checks import cachefs

ROOT = cachefs.ROOT
STORE = cachefs.work() / "cad-speed/results"
DEPSITE = Path(__file__).resolve().parent / "depsite"
VERSION = 1
# the checks' own caches and reports (content-addressed: never an input); scratch files elsewhere under .cache/work (an
# iteration's manifest or STEP) are inputs like any other (T-vftxyn: a scratch STEP rebuilt in place returned the old result)
SKIP = (".cache/work/cad-speed/", ".cache/work/mount-check/", ".cache/work/bom/", ".venv/", ".git/", ".bookkeeping/", ".tools/")
# blob-store indexes: they only say where a missing file is restored from, and the file then read is recorded itself;
# as inputs they would drop every stored result whenever any session uploads anything
INDEXES = {"data/private-archive.json", "data/regenerable-artifacts.json", "infra/storage/path-moves.json"}
ROOT_S = str(ROOT) + os.sep


def enabled():
    """Off with CHECKS_RESULT_CACHE=0, and whenever the definitions or part settings come from an override folder / file
    (LAB_DEFINITIONS_ROOT, LAB_PART_SETTINGS): those usually sit in .cache, whose reads are not recorded."""
    return os.environ.get("CHECKS_RESULT_CACHE", "1") != "0" and not os.environ.get("LAB_DEFINITIONS_ROOT") \
        and not os.environ.get("LAB_PART_SETTINGS") and not os.environ.get("LAB_CHECKS_DATA")


def rel_or_none(p):
    """Repository-relative path of a read worth recording, else None."""
    try:
        s = os.path.abspath(os.fsdecode(p))
    except (TypeError, ValueError):
        return None
    if not s.startswith(ROOT_S):
        return None
    r = s[len(ROOT_S):]
    if r.startswith(SKIP) or r.endswith((".pyc", ".lock")) or r in INDEXES or "__path_hook__" in r:
        return None
    return r


# ------------------------------------------------------------------------------------------------ recording (check child)
READ: set = set()


def note(path):
    """Record that path was read (called by bodylib.src, core.sha, the import_step / gmsh wrappers and the audit hook)."""
    r = rel_or_none(path)
    if r is not None:
        READ.add(r)


def _audit(event, args):
    if event == "open":
        mode = args[1] if len(args) > 1 else None
        if isinstance(mode, str) and any(c in mode for c in "wax+"):
            return
        if mode is None and len(args) > 2 and isinstance(args[2], int) and args[2] & (os.O_WRONLY | os.O_RDWR):
            return
        try:
            note(args[0])
        except Exception:  # noqa: BLE001  an audit hook must never raise
            pass


def install():
    """Start recording in this process (child mode of run.py), before the checks are imported."""
    sys.addaudithook(_audit)
    note(Path(__file__).resolve().parent / "run.py")
    log = cachefs.work() / "cad-speed/deps-logs" / f"{os.getpid()}"
    log.mkdir(parents=True, exist_ok=True)
    os.environ["CHECKS_DEPS_LOG"] = str(log)
    os.environ["PYTHONPATH"] = str(DEPSITE) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    try:
        import build123d
        orig = build123d.import_step

        def import_step(path, *a, **k):
            note(path)
            return orig(path, *a, **k)
        import_step.__doc__ = orig.__doc__
        build123d.import_step = import_step
    except ImportError:
        pass


def merge_subprocess_reads():
    """Fold in the reads the Python subprocesses of this child logged (depsite/sitecustomize.py)."""
    d = os.environ.get("CHECKS_DEPS_LOG")
    if not d:
        return
    for f in Path(d).glob("*.txt"):
        try:
            for ln in f.read_text().splitlines():
                if ln:
                    READ.add(ln)
            f.unlink()
        except OSError:
            pass


def module_file(check_fn):
    m = sys.modules.get(check_fn.__module__)
    return getattr(m, "__file__", None)


def _module_deps(mod_file):
    """The check's own module and the check modules it names (`checks.modules.X`, `from checks.modules import X`)."""
    out, todo = set(), [mod_file]
    while todo:
        f = todo.pop()
        r = rel_or_none(f)
        if not r or r in out:
            continue
        out.add(r)
        try:
            src = Path(f).read_text()
        except OSError:
            continue
        names = set(re.findall(r"checks\.modules\.(\w+)", src))
        for m in re.findall(r"from checks\.modules import ([\w ,]+(?:as \w+)?)", src):
            names |= {x.strip().split(" ")[0] for x in m.split(",")}
        todo += [str(Path(__file__).resolve().parent / "modules" / f"{n}.py") for n in names]
    return out


MODULES_DIR = "robot/body/checks/modules/"


def snapshot(check_fn):
    """{rel path: sha256} of what the check depends on, as read so far in this process."""
    from checks.core import sha
    merge_subprocess_reads()
    own = _module_deps(module_file(check_fn))
    files = {}
    for r in sorted(READ):
        if r.startswith(MODULES_DIR) and r not in own:
            continue
        p = ROOT / r
        try:
            files[r] = sha(p) if p.is_file() else "missing"
        except OSError:
            files[r] = "missing"
    for r in own:
        if r not in files:
            files[r] = sha(ROOT / r)
    return files


# ------------------------------------------------------------------------------------------------ named sets
def _definition_files():
    sys.path.insert(0, str(ROOT / "robot/body/bom"))
    import definitions as D
    return sorted(str(Path(p).relative_to(ROOT)) for p in D.discover())


SETS = {"body definition files": _definition_files}


def sets_now():
    out = {}
    for k, fn in SETS.items():
        try:
            out[k] = hashlib.sha256("\n".join(fn()).encode()).hexdigest()
        except Exception:  # noqa: BLE001
            out[k] = "unavailable"
    return out


# ------------------------------------------------------------------------------------------------ keys, store, lookup
OPTION_KEYS = ("poses", "search_mm", "exact_press", "include_shells", "fem_variant", "fem_mesh_h", "fem_rigid_load", "process",
               "joints", "step_deg", "hold", "fit_pair_budget_s", "render_timeout")


def options_of(a):
    return {k: getattr(a, k, None) for k in OPTION_KEYS}


def key(manifest_rel, check, scope, options):
    blob = json.dumps(dict(v=VERSION, manifest=manifest_rel, check=check, scope=sorted(scope), options=options), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def storable(res):
    if res.get("failed") or res.get("failed_parts") or res.get("stopped_in"):
        return False
    if (res.get("bad_events") or {}).get("pairs_timed_out"):       # depends on how loaded the machine was
        return False
    return res.get("cacheable", True)


KEEP = 8      # input versions kept per key: switching back to an earlier candidate finds its results again


def store(k, check, scope, deps, sets, res, label):
    entry = dict(version=VERSION, check=check, scope=sorted(scope), deps=deps, sets=sets, result=res,
                 at=time.strftime("%Y-%m-%d %H:%M:%S"), label=label, checkout=str(ROOT))
    digest = hashlib.sha256(json.dumps([deps, sets], sort_keys=True).encode()).hexdigest()[:16]
    d = STORE / k
    cachefs.write_text(d / f"{digest}.json", json.dumps(entry, default=float))
    old = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[KEEP:]
    for p in old:
        p.unlink(missing_ok=True)


def _valid(e):
    """None when every input recorded by entry e is unchanged, else why not."""
    from checks.core import sha
    for r, h in e["deps"].items():
        p = ROOT / r
        try:
            now = sha(p) if p.is_file() else "missing"
        except OSError:
            now = "missing"
        if now != h:
            return f"input changed: {r}"
    now_sets = sets_now()
    for s_, h in e.get("sets", {}).items():
        if now_sets.get(s_) != h:
            return f"input changed: {s_}"
    for ev in e["result"].get("evidence") or []:
        if isinstance(ev, str) and "/" in ev and not (ROOT / ev).exists():
            return f"evidence gone: {ev}"
    return None


def lookup(k):
    """(result, entry) of the newest stored version of key k whose recorded inputs are all unchanged; else (None, why)."""
    d = STORE / k
    files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if d.is_dir() else []
    why = "no earlier result"
    for f in files:
        e = cachefs.read_json(f)
        if not e or e.get("version") != VERSION:
            continue
        w = _valid(e)
        if w is None:
            return e["result"], e
        why = w if why == "no earlier result" else why
    return None, why
