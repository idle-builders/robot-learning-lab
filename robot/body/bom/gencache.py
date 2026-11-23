"""The generator cache: a generator's outputs are reused, from whatever checkout, branch or commit made them, as long as
nothing it read has changed (owner 2026-11-22: "No matter what commit, branch or whatever a part comes from if it hasn't
changed we shouldn't have to recreate it"), and outputs nobody needs any more are deleted ("Once a part isn't needed
anymore those artifacts should disappear").

One store serves the main checkout and every worktree: `.cache/work/gen-cache/` of the main checkout (checks/cachefs.py).
- blobs/<sha256[:2]>/<sha256>: each output file once, by content. On APFS a blob is a copy-on-write clone of the file it
  was taken from, and restoring is a clone back, so a blob costs disk only once the checkout's file has moved on.
- entries/<generator>/<digest>.json: one run of a generator: its command, every file it read with the sha256 it had when
  read (recorded by robot/body/bom/gensite/sitecustomize.py, not declared), the folders it listed, and the files it wrote
  with their sha256. An entry is valid in a checkout when every read file there still hashes the same (and every listed
  folder holds the same names); then its outputs are cloned into place instead of running the generator.
- checkouts/<id>.json: per checkout (main and each worktree) the entry last made or restored there per generator.
- sha-index.sqlite: sha256 per (path, size, mtime_ns, inode), so unchanged files are not hashed twice.

GC (make.py gc, and after every land): an entry is kept while a live checkout (git worktree list) holds it, or for
GRACE_DAYS after it was last made or used; blobs no kept entry names are deleted. Derived outputs are never uploaded
(owner 2026-11-22): the store is local only."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "robot/body"))
from checks import cachefs  # noqa: E402

VERSION = 2          # 2: an entry lists every file its run wrote (1 listed only the files whose content changed)
GRACE_DAYS = float(os.environ.get("GENCACHE_GRACE_DAYS", "3"))


def store() -> Path:
    return Path(os.environ.get("GENCACHE_STORE") or cachefs.work() / "gen-cache")


# ------------------------------------------------------------------------------------------------ hashing
def content_sha(p: Path) -> str:
    """sha256 of what a file says, not of the bytes a writer stamps on it: a STEP without its HEADER (OpenCascade writes the
    time into it), an npz by its arrays (zip members carry their write time), a binary STL without its 80-byte header.
    Without this every rerun of a builder reads as a change and nothing downstream is ever reused. Prefixed so a content
    hash never equals a raw one."""
    suf = p.suffix.lower()
    if suf in (".step", ".stp"):
        b = re.sub(rb"HEADER;.*?ENDSEC;", b"", p.read_bytes(), count=1, flags=re.S)
        return "c" + hashlib.sha256(b).hexdigest()[1:]
    if suf == ".npz":
        try:
            import numpy as np
            h = hashlib.sha256()
            with np.load(p, allow_pickle=False) as z:
                for k in sorted(z.files):
                    a = z[k]; h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
            return "c" + h.hexdigest()[1:]
        except Exception:  # noqa: BLE001  unreadable: fall back to the bytes
            pass
    if suf == ".stl":
        b = p.read_bytes()
        if not b[:5].lower() == b"solid":
            return "c" + hashlib.sha256(b[80:]).hexdigest()[1:]
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class ShaIndex:
    def __init__(self):
        p = store() / "sha-index.sqlite"
        p.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(p), timeout=60, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS sha (path TEXT PRIMARY KEY, size INT, mtime INT, ino INT, sha TEXT)")
        self.new = []

    def sha(self, p: Path) -> str:
        st = p.stat()
        k = str(p)
        row = self.db.execute("SELECT size, mtime, ino, sha FROM sha WHERE path=?", (k,)).fetchone()
        if row and row[0] == st.st_size and row[1] == st.st_mtime_ns and row[2] == st.st_ino:
            return row[3]
        d = content_sha(p)
        self.new.append((k, st.st_size, st.st_mtime_ns, st.st_ino, d))
        if len(self.new) >= 200:
            self.flush()
        return d

    def known(self, p: Path, sha: str):
        """p was just written with content sha (a restore): index it without reading it."""
        st = p.stat()
        self.new.append((str(p), st.st_size, st.st_mtime_ns, st.st_ino, sha))

    def flush(self):
        if self.new:
            self.db.executemany("INSERT OR REPLACE INTO sha VALUES (?,?,?,?,?)", self.new)
            self.new = []


_IDX = None


def idx() -> ShaIndex:
    global _IDX
    if _IDX is None:
        _IDX = ShaIndex()
    return _IDX


def file_state(root: Path, rel: str) -> str:
    p = root / rel
    try:
        return idx().sha(p) if p.is_file() else ("dir" if p.is_dir() else "missing")
    except OSError:
        return "missing"


GLOB_SKIP = {"__pycache__", ".git", "node_modules"}     # as definitions.SKIP_DIRS


_VALUES_CACHE: dict = {}


def values_state(root: Path, rel: str) -> str:
    """State of a "values:joints" / "values:config:<definition>" key (gensite, robot/body/bom/bom_config.py): sha256 of the
    declared joint ranges / the definition's config block, read from the definitions under root."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"_defs_{len(_VALUES_CACHE)}", root / "robot/body/bom/definitions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    files = mod.discover(root)
    key = tuple((str(f), f.stat().st_mtime_ns, f.stat().st_size) for f in files)
    if key not in _VALUES_CACHE:
        _VALUES_CACHE[key] = mod.Definitions(files)
    D = _VALUES_CACHE[key]
    what = rel[len("values:"):]
    if what == "joints":
        v = {n: d["range_deg"] for n, d in sorted(D.joints.items())}
    elif what.startswith("config:"):
        v = {k: x.get("value") for k, x in sorted(D.config.get(what[len("config:"):], {}).items())}
    elif what.startswith("part:"):                      # the claiming entry's declared values (bom_config.part_entry)
        import fnmatch
        n = what[len("part:"):]
        e = next((e for e in D.entries if any(fnmatch.fnmatchcase(n, p) for p in mod.names_of(e["match"]))), None)
        v = {k: x for k, x in (e or {}).items() if k not in ("file", "index")} if e else None
    else:
        return "unknown-values-key"
    return hashlib.sha256(json.dumps(v, sort_keys=True, default=str).encode()).hexdigest()[:16]


def dir_state(root: Path, rel: str) -> str:
    """A folder's listing (names), or for a "glob:TOP/**/*SUFFIX" key (gensite: the body definitions' search) the sorted
    repository paths below TOP whose name ends in SUFFIX: only the files the search can return, not every folder it walks."""
    if rel.startswith("values:"):
        return values_state(root, rel)
    if rel.startswith("glob:"):
        top, _, pat = rel[5:].partition("/**/*")
        hits = []
        for d, dirs, files in os.walk(root / top):
            dirs[:] = sorted(x for x in dirs if x not in GLOB_SKIP)
            hits += [os.path.relpath(os.path.join(d, f), root) for f in sorted(files) if f.endswith(pat)]
        return hashlib.sha256("\n".join(sorted(hits)).encode()).hexdigest()[:16]
    p = root / rel
    try:
        names = sorted(os.listdir(p))
    except OSError:
        return "missing"
    return hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ copy-on-write
_libc = None


def clone(src: Path, dst: Path):
    """dst = a copy of src, as an APFS clone when possible (no disk until one of them changes); atomic for readers of dst."""
    global _libc
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.gc-tmp")
    if tmp.exists():
        tmp.unlink()
    done = False
    if sys.platform == "darwin":
        try:
            if _libc is None:
                _libc = ctypes.CDLL("/usr/lib/libSystem.dylib", use_errno=True)
            done = _libc.clonefile(os.fsencode(src), os.fsencode(tmp), 0) == 0
        except (OSError, AttributeError):
            done = False
    if not done:
        shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def blob(sha: str) -> Path:
    return store() / "blobs" / sha[:2] / sha


def put_blob(p: Path, sha: str):
    b = blob(sha)
    if not b.exists():
        clone(p, b)


# ------------------------------------------------------------------------------------------------ entries
def newest_outputs(gid: str) -> set:
    e = next(entries(gid), None)
    return set(e["outputs"]) if e else set()


def entries(gid: str):
    d = store() / "entries" / gid
    fs = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if d.is_dir() else []
    for f in fs:
        e = cachefs.read_json(f)
        if e and e.get("version") == VERSION:
            e["_file"] = f
            yield e


def feedback(e, earlier: set, later: set) -> set:
    """Reads of entry e that are feedback, not inputs: files no earlier generator writes that this generator (its own
    outputs) or a later one writes. Their content is what the previous make left there (the chain has loops: a builder
    reading its own last output, or a manifest a later step writes); in a settled chain it follows from the real inputs,
    so a checkout with another history must not miss on them. A file an earlier generator wrote and this one edits in
    place stays an input."""
    return {r for r in e["reads"] if r not in earlier and (r in e["outputs"] or r in later)}


def why_invalid(e, root: Path, overlay=None, earlier=frozenset(), later=frozenset()):
    """None when every input entry e recorded is unchanged in the checkout at root, else the first input that changed.
    overlay {rel: sha}: files as earlier restores in the same walk would leave them (make.py status); earlier / later:
    files the generators before / after this one write (feedback reads are skipped)."""
    fb = feedback(e, earlier, later)
    if any(v == "unknown" for r, v in e["reads"].items() if r not in fb):
        return "a file it read was also written by it, outside its declared outputs (pre-run content unknown)"
    ov = overlay or {}
    for r, h in e["reads"].items():
        if r in fb:
            continue
        if (ov[r] if r in ov else file_state(root, r)) != h:
            return f"changed: {r}"
    for r, h in e.get("dirs", {}).items():
        if dir_state(root, r) != h:
            return f"listing changed: {r}/"
    for r, h in e["outputs"].items():
        if not blob(h).exists():
            return f"blob gone: {r}"
    return None


def lookup(gid: str, root: Path, command, overlay=None, earlier=frozenset(), later=frozenset()):
    """(entry, None) for the newest valid entry of generator gid with this command; else (None, why the newest is not)."""
    why = "never run through make"
    for e in entries(gid):
        if e["command"] != command:
            why = why if why != "never run through make" else "command changed"
            continue
        w = why_invalid(e, root, overlay, earlier, later)
        if w is None:
            return e, None
        if why in ("never run through make", "command changed"):
            why = w
    return None, why


def restore(e, root: Path):
    """Clone e's outputs into the checkout (files already equal are left alone); returns (cloned, unchanged, removed)."""
    n_clone = n_same = n_rm = 0
    for r, h in e["outputs"].items():
        p = root / r
        if p.is_file() and file_state(root, r) == h:
            n_same += 1
            continue
        if p.is_symlink():
            p.unlink()
        clone(blob(h), p)
        idx().known(p, h)
        n_clone += 1
    for r in e.get("deleted", []):
        p = root / r
        if p.is_file() or p.is_symlink():
            p.unlink(); n_rm += 1
    os.utime(e["_file"])                     # used: stays within the grace period
    return n_clone, n_same, n_rm


def save(gid, command, reads, dirs, outputs, deleted, root: Path, **meta) -> dict:
    for r, h in outputs.items():
        put_blob(root / r, h)
    body = dict(version=VERSION, generator=gid, command=command, reads=reads, dirs=dirs, outputs=outputs, deleted=deleted,
                at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), checkout=str(root), **meta)
    digest = hashlib.sha256(json.dumps([command, reads, dirs, outputs], sort_keys=True).encode()).hexdigest()[:20]
    f = store() / "entries" / gid / f"{digest}.json"
    cachefs.write_text(f, json.dumps(body, indent=0))
    body["_file"] = f
    return body


# ------------------------------------------------------------------------------------------------ checkouts
def checkout_id(root: Path) -> str:
    return hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]


def mark(root: Path, gid: str, e):
    cachefs.update_json(store() / "checkouts" / f"{checkout_id(root)}.json",
                        {"path": str(root.resolve()), "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                         f"gen:{gid}": e["_file"].stem})


def live_checkouts() -> list[Path]:
    out = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
    return [Path(l[9:]) for l in out.splitlines() if l.startswith("worktree ") and Path(l[9:]).is_dir()]


# ------------------------------------------------------------------------------------------------ gc
def gc(dry_run=False, grace_days=GRACE_DAYS, verbose=True):
    """Delete entries no live checkout holds and nobody made or used within grace_days, then blobs no kept entry names.
    Returns a summary dict (bytes are logical sizes: a blob that is still a clone of a checkout's file frees less)."""
    S = store()
    live = {str(p.resolve()) for p in live_checkouts()}
    held = set()
    for f in (S / "checkouts").glob("*.json"):
        d = cachefs.read_json(f, {}) or {}
        if d.get("path") in live:
            held |= {(k[4:], v) for k, v in d.items() if k.startswith("gen:")}
        elif not dry_run:
            f.unlink(missing_ok=True)
    now, keep_blobs = time.time(), set()
    n_e = n_e_del = 0
    for gdir in sorted((S / "entries").glob("*")) if (S / "entries").is_dir() else []:
        for f in gdir.glob("*.json"):
            n_e += 1
            e = cachefs.read_json(f)
            young = now - f.stat().st_mtime < grace_days * 86400
            if e and ((gdir.name, f.stem) in held or young):
                keep_blobs |= set(e.get("outputs", {}).values())
                continue
            n_e_del += 1
            if not dry_run:
                f.unlink(missing_ok=True)
    n_b = n_b_del = bytes_del = 0
    for b in (S / "blobs").glob("*/*") if (S / "blobs").is_dir() else []:
        n_b += 1
        if b.name in keep_blobs or b.name.endswith(".gc-tmp") and now - b.stat().st_mtime < 3600:
            continue
        n_b_del += 1
        bytes_del += b.stat().st_size
        if not dry_run:
            b.unlink(missing_ok=True)
    res = dict(dry_run=dry_run, grace_days=grace_days, live_checkouts=len(live), entries=n_e, entries_deleted=n_e_del,
               blobs=n_b, blobs_deleted=n_b_del, logical_bytes_deleted=bytes_del)
    if verbose:
        print(f"gen-cache gc{' (dry run)' if dry_run else ''}: {n_e_del}/{n_e} entries, {n_b_del}/{n_b} blobs "
              f"({bytes_del / 1e9:.2f} GB logical) {'would go' if dry_run else 'deleted'}; {len(live)} live checkouts, grace {grace_days:g} d")
    return res
