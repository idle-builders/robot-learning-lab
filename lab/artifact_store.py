"""Local SHA-256 file storage with byte-preserving legacy artifact paths.

This is a filesystem adapter, not a replacement for experiment provenance. Existing
reports/seals remain authoritative and unchanged. No garbage collection is provided.
Objects are private copies (APFS clones when available), never writable hardlinks.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import uuid


STORE = ".artifact-store"
SCHEMA = "robot-learning-lab.artifact-store.v1"


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _logical(path) -> Path:
    """Normalize directory aliases (macOS /var), preserving the final CAS link."""
    path = Path(path).absolute()
    return path.parent.resolve() / path.name


def _root(repo_root=None) -> Path:
    root = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
    if (root / STORE).is_symlink():
        raise ValueError("artifact store must not be a symlink")
    return root


def _relative(root: Path, path: Path) -> str:
    # Do not resolve the final component: a legacy CAS link is itself an input.
    path = _logical(path)
    rel = path.relative_to(root)
    if not rel.parts or STORE in rel.parts:
        raise ValueError("expected a repository path outside the store")
    if not path.parent.resolve().is_relative_to(root):
        raise ValueError("parent path escapes repository")
    return rel.as_posix()


def _safe_path(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError(f"unsafe manifest path: {relative}")
    p = root / rel
    _relative(root, p)
    return p


def object_path(root: Path, digest: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid SHA-256")
    path = root / STORE / "objects" / "sha256" / digest[:2] / digest[2:]
    for parent in path.parents:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("store directory symlink forbidden")
    return path


def _restore_missing(root: Path, path: Path):
    # Portable exports have no cold index and need no optional blob reader.
    if not path.is_file() and (root / "data" / "blob-store.sqlite").is_file():
        from .blob_store import ensure
        ensure(path)


def _require_local_tree(root: Path, path: Path):
    if (root / "data" / "blob-store.sqlite").is_file():
        from .blob_store import require_materialized_tree
        require_materialized_tree(path)


def validate_file(path, *, repo_root=None) -> Path:
    """Accept workspace files or exact, byte-verified CAS links; return logical path.

    Callers still enforce their own allowed logical run/folder boundary. This does
    not relax external-worker extraction rules or permit arbitrary symlink targets.
    """
    root, path = _root(repo_root), _logical(path)
    _relative(root, path)
    _restore_missing(root, path)
    if not path.is_file():
        raise ValueError(f"missing regular file: {path}")
    if path.is_symlink():
        resolved = path.resolve()
        try:
            parts = resolved.relative_to(root / STORE / "objects" / "sha256").parts
        except ValueError as error:
            raise ValueError("symlink is not an artifact-store object") from error
        if len(parts) != 2:
            raise ValueError("invalid artifact-store object path")
        digest = "".join(parts)
        expected = object_path(root, digest)
        # Require a direct link, not an arbitrary chain ending in a valid object.
        direct = Path(os.path.abspath(path.parent / os.readlink(path)))
        if direct != expected or expected.is_symlink() or sha256(expected) != digest:
            raise ValueError("invalid or corrupt artifact-store link")
    elif not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("not a regular file")
    return path


@contextmanager
def _lock(root: Path):
    (root / STORE).mkdir(exist_ok=True)
    with (root / STORE / "writer.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _sync_dir(path: Path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def clone_or_copy(source: Path, destination: Path) -> str:
    """Create a fresh independently writable file; clonefile never shares writes."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        clone = libc.clonefile
        clone.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        clone.restype = ctypes.c_int
        if clone(os.fsencode(source), os.fsencode(destination), 0) == 0:
            return "apfs_clone"
        # Unsupported/cross-device/permissions can fall back to a regular copy.
        # If clonefile created anything on failure, do not overwrite it.
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, 1024 * 1024)
    return "copy"


def _stat_identity(path: Path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def _put(root: Path, source: Path, digest: str, size: int, mode=0o444) -> Path:
    obj = object_path(root, digest)
    if obj.is_symlink():
        raise ValueError(f"object must not be a symlink: {obj}")
    if obj.exists():
        if not obj.is_file() or obj.is_symlink() or obj.stat().st_size != size or sha256(obj) != digest:
            raise ValueError(f"corrupt existing object: {obj}")
        if mode & 0o111 and stat.S_IMODE(obj.stat().st_mode) != 0o555:
            os.chmod(obj, 0o555)
        return obj
    obj.parent.mkdir(parents=True, exist_ok=True)
    tmp = obj.with_name(f".ingest-{uuid.uuid4().hex}")
    try:
        clone_or_copy(source, tmp)
        if tmp.stat().st_size != size or sha256(tmp) != digest:
            raise ValueError(f"source changed while storing: {source}")
        os.chmod(tmp, 0o555 if mode & 0o111 else 0o444)
        with tmp.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(tmp, obj)
        _sync_dir(obj.parent)
    finally:
        tmp.unlink(missing_ok=True)
    return obj


def _link(obj: Path, destination: Path, *, replace=False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not replace and (destination.exists() or destination.is_symlink()):
        raise FileExistsError(destination)
    tmp = destination.with_name(f".cas-link-{uuid.uuid4().hex}")
    try:
        tmp.symlink_to(os.path.relpath(obj, destination.parent))
        os.replace(tmp, destination)
        _sync_dir(destination.parent)
    finally:
        tmp.unlink(missing_ok=True)


def _json_write_new(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    _sync_dir(path.parent)


def _record_reference(root: Path, ref: dict):
    parts = Path(ref["path"]).parts
    run = "/".join(parts[:2]) if parts[0] == "artifacts" else parts[0]
    key = hashlib.sha256(run.encode()).hexdigest()
    log = root / STORE / "runs" / key / "references.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write(json.dumps(dict(schema=SCHEMA, run=run, **ref), sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def snapshot_file(source, destination, *, repo_root=None) -> dict:
    """Capture one input, retaining legacy read paths without another byte copy.

    A new snapshot must not embed a prior source snapshot. Reference its manifest
    or select direct producer outputs instead. Historical migration is exempt.
    """
    root = _root(repo_root)
    source, destination = _logical(source), _logical(destination)
    external = not destination.is_relative_to(root)
    source_external = not source.is_relative_to(root)
    if source_external and not external:
        raise ValueError("external input requires a private external destination")
    src_rel = str(source) if source_external else _relative(root, source)
    rel = str(destination) if external else _relative(root, destination)
    if Path(rel).parts.count("source") > 1 or (
        Path(src_rel).parts[0] == "artifacts" and "source" in Path(src_rel).parts
    ):
        raise ValueError("recursive source snapshot forbidden; reference the prior manifest")
    _restore_missing(root, source)
    if not source.is_file():
        raise ValueError(f"not a regular input: {source}")
    if source.is_symlink() and not source.resolve().is_relative_to(root / STORE / "objects"):
        raise ValueError("only verified store symlinks can be captured")
    if not source_external:
        validate_file(source, repo_root=root)
    if external:
        # Temporary test/export locations must remain portable and independent.
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        before = _stat_identity(source)
        digest, info = sha256(source), source.stat()
        destination.parent.mkdir(parents=True, exist_ok=True)
        method = clone_or_copy(source, destination)
        os.chmod(destination, (info.st_mode & 0o777) | 0o200)
        if sha256(destination) != digest or _stat_identity(source) != before:
            raise ValueError("source changed during portable snapshot")
        return dict(path=str(destination), source_path=src_rel, sha256=digest,
                    size=info.st_size, mode=stat.S_IMODE(info.st_mode),
                    mtime_ns=info.st_mtime_ns, storage="materialized_external", method=method)
    with _lock(root):
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        before = _stat_identity(source)
        digest = sha256(source)
        info = source.stat()
        obj = _put(root, source, digest, info.st_size, info.st_mode)
        if _stat_identity(source) != before:
            raise ValueError("source changed during snapshot")
        ref = dict(path=rel, source_path=src_rel, sha256=digest,
                   size=info.st_size, mode=stat.S_IMODE(info.st_mode),
                   mtime_ns=info.st_mtime_ns, captured_at=_stamp())
        # Durable manifest precedes link mutation; an interruption is repairable.
        _record_reference(root, ref)
        _link(obj, destination)
        return ref


def retain_dependency(source, *, repo_root=None) -> dict:
    """Retain an upstream tree as ONE immutable manifest, never a new source tree.

    Historical nested sources are enumerated once and stored by hash. Downstream
    reports retain the returned reference, including its original logical root.
    No upstream reports, paths, seals, or files are modified.
    """
    root, source = _root(repo_root), _logical(source)
    rel = _relative(root, source)
    if not source.is_dir() or source.is_symlink():
        raise ValueError("dependency must be a real directory")
    _require_local_tree(root, source)
    entries = []
    children = {}
    with _lock(root):
        for folder, dirs, names in os.walk(source, followlinks=False):
            for name in dirs:
                if (Path(folder) / name).is_symlink():
                    raise ValueError("dependency directory symlink forbidden")
            for name in sorted(names):
                path = validate_file(Path(folder) / name, repo_root=root)
                before = _stat_identity(path)
                info, digest = path.stat(), sha256(path)
                _put(root, path, digest, info.st_size, info.st_mode)
                if before != _stat_identity(path):
                    raise ValueError(f"dependency changed during capture: {path}")
                entries.append(dict(path=path.relative_to(root).as_posix(), sha256=digest,
                                    size=info.st_size, mode=stat.S_IMODE(info.st_mode),
                                    mtime_ns=info.st_mtime_ns))
                # Follow only the declared dependency field in known run reports.
                # Read the captured object, not a possibly changing live report.
                if path.parent == source and path.name in {"report.json", "arm-report.json", "build-report.json"}:
                    report = json.loads(object_path(root, digest).read_text())
                    declared = report.get("dependency_manifests", {})
                    # Early circuit reports used a list. Preserve their exact bytes,
                    # assign stable index labels and verify every inherited reference.
                    if isinstance(declared, list):
                        declared = {f"legacy-index-{i}": ref for i, ref in enumerate(declared)}
                    if not isinstance(declared, dict):
                        raise ValueError("dependency_manifests must be a mapping or historical list")
                    for label, ref in declared.items():
                        if verify_dependency(ref, repo_root=root)["status"] != "pass":
                            raise ValueError(f"invalid inherited dependency: {path}/{label}")
                        children[f"{path.name}:{label}"] = ref
        entries.sort(key=lambda row: row["path"])
        manifest = dict(schema=SCHEMA, purpose="immutable upstream dependency", root=rel,
                        entries=entries, preserved_symlinks=[])
        if children:
            manifest["dependencies"] = children
            _dependency_closure(root, manifest)
        payload = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        digest = hashlib.sha256(payload).hexdigest()
        dest = root / STORE / "dependencies" / f"{digest}.json"
        if dest.exists():
            if sha256(dest) != digest:
                raise ValueError("corrupt dependency manifest")
        else:
            _json_write_new(dest, manifest)
            dest.chmod(0o444)
            with dest.with_suffix(".json.sha256").open("x") as stream:
                stream.write(digest + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        return dict(schema=SCHEMA, source_root=rel,
                    manifest_path=dest.relative_to(root).as_posix(), sha256=digest,
                    file_count=len(entries), logical_bytes=sum(row["size"] for row in entries))


def _read_dependency(root: Path, ref: dict) -> dict:
    expected = ref["sha256"]
    # Construct from digest; never accept an arbitrary supplied manifest path.
    object_path(root, expected)
    manifest = root / STORE / "dependencies" / f"{expected}.json"
    if ref["manifest_path"] != manifest.relative_to(root).as_posix() or sha256(manifest) != expected:
        raise ValueError("dependency manifest identity mismatch")
    data = read_manifest(manifest)
    if data.get("root") != ref["source_root"] or len(data["entries"]) != ref["file_count"]:
        raise ValueError("dependency reference metadata mismatch")
    return data


def _dependency_closure(root: Path, manifest: dict):
    """Collect direct and transitive entries once, refusing ambiguous path content."""
    files, visited, active = {}, set(), set()
    def visit(data):
        for entry in data["entries"]:
            _safe_path(root, entry["path"])
            previous = files.get(entry["path"])
            if previous is not None and (previous["sha256"], previous["size"]) != (entry["sha256"], entry["size"]):
                raise ValueError(f"conflicting dependency bytes for path: {entry['path']}")
            files.setdefault(entry["path"], entry)
        for ref in data.get("dependencies", {}).values():
            digest = ref["sha256"]
            if digest in active:
                raise ValueError("dependency manifest cycle")
            child = _read_dependency(root, ref)
            if digest in visited:
                continue
            active.add(digest)
            visit(child)
            active.remove(digest)
            visited.add(digest)
    visit(manifest)
    return list(files.values()), len(visited)


def verify_dependency(ref: dict, *, repo_root=None) -> dict:
    """Verify the entire immutable dependency graph, without mutable source files."""
    root = _root(repo_root)
    data = _read_dependency(root, ref)
    entries, child_count = _dependency_closure(root, data)
    verified, errors = set(), []
    for entry in entries:
        key = entry["sha256"], entry["size"]
        if key in verified:
            continue
        obj = object_path(root, entry["sha256"])
        _restore_missing(root, obj)
        if not obj.is_file() or obj.is_symlink() or obj.stat().st_size != entry["size"] or sha256(obj) != entry["sha256"]:
            errors.append(dict(path=entry["path"], error="missing or corrupt dependency object"))
        verified.add(key)
    return dict(status="fail" if errors else "pass", checked=len(entries),
                dependency_manifests=1 + child_count, verified_objects=len(verified), errors=errors)


def snapshot_tree(source, destination, *, repo_root=None) -> list[dict]:
    """Snapshot a fresh nonrecursive input tree. Never follow directory symlinks."""
    root = _root(repo_root)
    source, destination = _logical(source), _logical(destination)
    external = not destination.is_relative_to(root)
    source_external = not source.is_relative_to(root)
    if source_external and not external:
        raise ValueError("external input requires a private external destination")
    src_rel = str(source) if source_external else _relative(root, source)
    dest_rel = str(destination) if external else _relative(root, destination)
    if not source.is_dir() or source.is_symlink():
        raise ValueError("expected a real source directory")
    if destination.is_relative_to(source):
        raise ValueError("snapshot destination must not be inside its source")
    if destination.exists():
        raise FileExistsError(destination)
    _require_local_tree(root, source)
    files = []
    for folder, dirs, names in os.walk(source, followlinks=False):
        for name in dirs:
            p = Path(folder) / name
            if p.is_symlink():
                raise ValueError(f"directory symlink forbidden: {p}")
            if Path(src_rel).parts[0] == "artifacts" and name == "source":
                raise ValueError("artifact dependency contains source snapshots; select direct outputs")
        files.extend(Path(folder) / name for name in names)
    # Preflight all paths before creating the destination.
    if Path(dest_rel).parts.count("source") > 1:
        raise ValueError("recursive source snapshot forbidden")
    return [snapshot_file(p, destination / p.relative_to(source), repo_root=root) for p in sorted(files)]


def plan_migration(repo_root, roots, *, output=None, min_bytes=0) -> Path:
    """Hash a retained tree without changing it. The sealed manifest is the plan."""
    root = _root(repo_root)
    entries, preserved_links, seen = [], [], set()
    for requested in roots:
        folder = Path(requested)
        if not folder.is_absolute():
            folder = root / folder
        folder = _logical(folder)
        relroot = _relative(root, folder)
        if folder.is_dir():
            _require_local_tree(root, folder)
        if folder.is_symlink() and folder.is_file():
            validate_file(folder, repo_root=root)
        elif folder.is_symlink() or not (folder.is_dir() or folder.is_file()):
            raise ValueError(f"expected real artifact directory or file: {folder}")
        if Path(relroot).parts[0] != "artifacts":
            raise ValueError("historical migration is restricted to artifacts/")
        walk = os.walk(folder, followlinks=False) if folder.is_dir() else [(folder.parent, [], [folder.name])]
        for current, dirs, names in walk:
            for name in list(dirs):
                p = Path(current) / name
                if p.is_symlink():
                    raise ValueError(f"directory symlink requires manual review: {p}")
            for name in sorted(names):
                p = Path(current) / name
                rel = _relative(root, p)
                if rel in seen:
                    continue
                seen.add(rel)
                if p.is_symlink():
                    # A later inventory of an already migrated run must remain
                    # materializable: CAS links are content refs, not opaque links.
                    if p.resolve().is_relative_to(root / STORE / "objects"):
                        validate_file(p, repo_root=root)
                    else:
                        preserved_links.append(dict(path=rel, target=os.readlink(p)))
                        continue
                info = p.stat()
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError(f"special file requires manual review: {p}")
                if info.st_size < min_bytes:
                    continue
                before = _stat_identity(p)
                digest = sha256(p)
                if before != _stat_identity(p):
                    raise ValueError(f"file changed during inventory: {p}")
                entries.append(dict(path=rel, sha256=digest, size=info.st_size,
                                    mode=stat.S_IMODE(info.st_mode), mtime_ns=info.st_mtime_ns,
                                    allocated_bytes=p.lstat().st_blocks * 512,
                                    storage_at_inventory="cas_link" if p.is_symlink() else "regular"))
    entries.sort(key=lambda row: row["path"])
    unique = {row["sha256"]: row["size"] for row in entries}
    manifest = dict(schema=SCHEMA, created_at=_stamp(), purpose="byte-preserving historical migration",
                    roots=[str(p) for p in roots], min_bytes=min_bytes,
                    entries=entries, preserved_symlinks=preserved_links,
                    before=dict(files=len(entries), logical_bytes=sum(r["size"] for r in entries),
                                allocated_bytes_sum=sum(r["allocated_bytes"] for r in entries),
                                unique_objects=len(unique), unique_bytes=sum(unique.values()),
                                repeated_logical_bytes=sum(r["size"] for r in entries)-sum(unique.values()),
                                note="st_blocks sums may double-count APFS clone extents; not physical reclaim"))
    target = Path(output) if output else root / STORE / "manifests" / uuid.uuid4().hex / "manifest.json"
    _json_write_new(target, manifest)
    with target.with_suffix(target.suffix + ".sha256").open("x") as stream:
        stream.write(sha256(target) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return target


def read_manifest(path) -> dict:
    path = Path(path)
    expected = path.with_suffix(path.suffix + ".sha256").read_text().strip()
    if sha256(path) != expected:
        raise ValueError("manifest seal mismatch")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported artifact manifest")
    return manifest


def apply_migration(repo_root, manifest_path) -> dict:
    """Idempotently replace unchanged files with verified relative object links."""
    root = _root(repo_root)
    manifest = read_manifest(manifest_path)
    result = dict(schema=SCHEMA, manifest_sha256=sha256(Path(manifest_path)),
                  started_at=_stamp(), status="running", replaced=0, already_linked=0)
    journal = Path(manifest_path).parent / f"apply-{uuid.uuid4().hex}.json"
    verified = set()
    try:
        with _lock(root):
            for entry in manifest["entries"]:
                p = _safe_path(root, entry["path"])
                obj = object_path(root, entry["sha256"])
                if obj.is_symlink():
                    raise ValueError(f"object must not be a symlink: {obj}")
                if p.is_symlink():
                    direct = Path(os.path.abspath(p.parent / os.readlink(p)))
                    if direct != obj:
                        raise ValueError(f"unexpected symlink: {p}")
                    if entry["sha256"] not in verified:
                        if not obj.is_file() or obj.stat().st_size != entry["size"] or sha256(obj) != entry["sha256"]:
                            raise ValueError(f"corrupt object: {obj}")
                        verified.add(entry["sha256"])
                    result["already_linked"] += 1
                    continue
                before = _stat_identity(p)
                if p.stat().st_size != entry["size"] or sha256(p) != entry["sha256"]:
                    raise ValueError(f"file changed after plan; left untouched: {p}")
                if entry["sha256"] not in verified:
                    _put(root, p, entry["sha256"], entry["size"], entry["mode"])
                    verified.add(entry["sha256"])
                elif entry["mode"] & 0o111 and stat.S_IMODE(obj.stat().st_mode) != 0o555:
                    os.chmod(obj, 0o555)
                if _stat_identity(p) != before:
                    raise ValueError(f"file changed during migration; left untouched: {p}")
                _link(obj, p, replace=True)
                result["replaced"] += 1
        result["status"] = "pass"
    except BaseException as error:
        result["status"] = "fail"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["finished_at"] = _stamp()
        _json_write_new(journal, result)
    return dict(result, journal=str(journal))


def verify_manifest(repo_root, manifest_path) -> dict:
    """Verify every historical pathname and every distinct expected object."""
    root = _root(repo_root)
    manifest = read_manifest(manifest_path)
    if manifest.get("purpose") == "immutable upstream dependency":
        digest = sha256(Path(manifest_path))
        return verify_dependency(dict(sha256=digest, source_root=manifest["root"],
                                      manifest_path=f"{STORE}/dependencies/{digest}.json",
                                      file_count=len(manifest["entries"])), repo_root=root)
    verified, errors, links, regular = set(), [], 0, 0
    for entry in manifest["entries"]:
        try:
            p = _safe_path(root, entry["path"])
            obj = object_path(root, entry["sha256"])
            _restore_missing(root, p)
            if p.is_symlink():
                if p.resolve() != obj.resolve() or obj.is_symlink():
                    raise ValueError("wrong CAS target")
                links += 1
                if entry["sha256"] in verified:
                    continue
            else:
                regular += 1
            if not p.is_file() or p.stat().st_size != entry["size"] or sha256(p) != entry["sha256"]:
                raise ValueError("missing or changed bytes")
            if p.is_symlink():
                verified.add(entry["sha256"])
        except (OSError, ValueError) as error:
            errors.append(dict(path=entry["path"], error=str(error)))
    for entry in manifest["preserved_symlinks"]:
        p = _safe_path(root, entry["path"])
        if not p.is_symlink() or os.readlink(p) != entry["target"]:
            errors.append(dict(path=entry["path"], error="pre-existing symlink changed"))
    return dict(status="fail" if errors else "pass", checked=len(manifest["entries"]),
                cas_links=links, regular_files=regular, verified_objects=len(verified), errors=errors)


def materialize_manifest(repo_root, manifest_path, destination, *, writable=False) -> dict:
    """Restore original relative paths to a NEW private tool/export directory."""
    root, destination = _root(repo_root), Path(destination).absolute()
    manifest = read_manifest(manifest_path)
    entries, child_count = _dependency_closure(root, manifest)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    methods = {}
    destination.mkdir(parents=True)
    for entry in entries:
        _safe_path(root, entry["path"])
        obj = object_path(root, entry["sha256"])
        _restore_missing(root, obj)
        if not obj.is_file() or obj.is_symlink() or obj.stat().st_size != entry["size"] or sha256(obj) != entry["sha256"]:
            raise ValueError(f"missing or corrupt object: {obj}")
        target = destination / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        method = clone_or_copy(obj, target)
        os.chmod(target, entry["mode"] | (0o200 if writable else 0))
        os.utime(target, ns=(entry["mtime_ns"], entry["mtime_ns"]))
        if sha256(target) != entry["sha256"]:
            raise ValueError(f"materialized bytes mismatch: {target}")
        methods[method] = methods.get(method, 0) + 1
    # Preserve links verbatim, but never traverse them during extraction.
    for entry in manifest["preserved_symlinks"]:
        _safe_path(root, entry["path"])
        target = destination / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(entry["target"])
    return dict(status="pass", files=len(entries), methods=methods, dependency_manifests=child_count,
                destination=str(destination), writable=writable,
                note="Private copies; original modes unless writable requested; not new qualification evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("roots", nargs="+", type=Path)
    plan.add_argument("--output", type=Path)
    plan.add_argument("--min-bytes", type=int, default=0)
    for name in ("apply", "verify", "materialize"):
        cmd = commands.add_parser(name)
        cmd.add_argument("manifest", type=Path)
        if name == "materialize":
            cmd.add_argument("destination", type=Path)
            cmd.add_argument("--writable", action="store_true", help="Add owner write permission for private tool work")
    args = parser.parse_args()
    if args.command == "plan":
        path = plan_migration(args.repo, args.roots, output=args.output, min_bytes=args.min_bytes)
        print(json.dumps(dict(manifest=str(path), before=read_manifest(path)["before"]), indent=2))
    elif args.command == "apply":
        print(json.dumps(apply_migration(args.repo, args.manifest), indent=2))
    elif args.command == "verify":
        result = verify_manifest(args.repo, args.manifest)
        print(json.dumps(result, indent=2))
        if result["status"] != "pass":
            raise SystemExit(1)
    else:
        print(json.dumps(materialize_manifest(args.repo, args.manifest, args.destination, writable=args.writable), indent=2))


if __name__ == "__main__":
    main()
