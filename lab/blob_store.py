"""Verified private blob packs and explicit restoration of cold workspace files.

The SQLite index is a local convenience; every pack has its own verified remote
manifest so the index can be recovered independently. Files retain their exact
logical paths and hashes. Active outputs must never be selected for archival.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import zlib
from lab.archive_config import settings as archive_settings, client as archive_client

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data/blob-store.sqlite"
BUCKET = archive_settings()["bucket"]
ENDPOINT = archive_settings()["endpoint"]
PREFIX = "robot-learning-lab/cold/v1"
ALLOWED = ("artifacts/", "studies/", "assets/", ".artifact-store/objects/sha256/")


class BlobUnavailable(RuntimeError):
    """The local path is archived, but its verified bytes could not be restored."""


def client():
    return archive_client()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def safe_path(name):
    rel = Path(name)
    if rel.is_absolute() or ".." in rel.parts or not any(
            str(rel).startswith(prefix) or str(rel) == prefix.rstrip("/") for prefix in ALLOWED):
        raise ValueError(f"Invalid cold-storage path: {name}")
    path = ROOT / rel
    if not path.parent.resolve().is_relative_to(ROOT):
        raise ValueError("Cold-storage parent escapes workspace")
    return path


def connect():
    INDEX.parent.mkdir(exist_ok=True)
    db = sqlite3.connect(INDEX, timeout=60)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS packs (
          key TEXT PRIMARY KEY, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL,
          manifest_key TEXT NOT NULL, verified_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS files (
          path TEXT PRIMARY KEY, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL,
          mtime_ns INTEGER NOT NULL, mode INTEGER NOT NULL,
          pack_key TEXT NOT NULL, offset INTEGER NOT NULL, stored_bytes INTEGER NOT NULL,
          encoding TEXT NOT NULL, evicted_at TEXT, restored_at TEXT,
          FOREIGN KEY(pack_key) REFERENCES packs(key));
    """)
    return db


def lookup(path):
    if not INDEX.exists():
        return None
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if p.is_symlink():
        p = p.resolve()
    try:
        name = p.relative_to(ROOT).as_posix()
    except ValueError:
        return None
    with sqlite3.connect(f"file:{INDEX}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT f.* FROM files f JOIN packs p ON p.key=f.pack_key WHERE f.path=?",
                         (name,)).fetchone()
    return dict(row) if row else None


def restore_entry(row, s3=None, destination=None):
    if row["encoding"] != "zlib":
        raise ValueError("Unsupported archived file encoding")
    path = safe_path(row["path"]) if destination is None else Path(destination)
    if path.exists() or path.is_symlink():
        if path.is_file() and path.stat().st_size == row["bytes"] and digest(path) == row["sha256"]:
            return path
        raise FileExistsError(f"Restore would replace different content: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    s3 = s3 or client()
    end = row["offset"] + row["stored_bytes"] - 1
    response = s3.get_object(Bucket=BUCKET, Key=row["pack_key"], Range=f"bytes={row['offset']}-{end}")
    if response.get("ContentLength") != row["stored_bytes"]:
        response["Body"].close()
        raise ValueError("Blob range length differs from manifest")
    decoder = zlib.decompressobj()
    h, size = hashlib.sha256(), 0
    partial = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".blob-restore-", delete=False) as out:
            partial = Path(out.name)
            for block in iter(lambda: response["Body"].read(1 << 20), b""):
                data = decoder.decompress(block)
                size += len(data)
                if size > row["bytes"]:
                    raise ValueError("Restored content exceeds expected length")
                h.update(data)
                out.write(data)
            tail = decoder.flush()
            h.update(tail)
            out.write(tail)
            size += len(tail)
        if not decoder.eof or decoder.unused_data or size != row["bytes"] or h.hexdigest() != row["sha256"]:
            raise ValueError(f"Restored SHA-256/size differs: {row['path']}")
        os.chmod(partial, row["mode"])
        os.utime(partial, ns=(row["mtime_ns"], row["mtime_ns"]))
        # link publishes without overwriting a file another writer just created.
        os.link(partial, path)
        return path
    finally:
        response["Body"].close()
        if partial:
            partial.unlink(missing_ok=True)


def ensure(path):
    """Materialize one missing indexed file; existing writable files are untouched."""
    path = Path(path)
    if path.is_file():
        return path
    row = lookup(path)
    if not row:
        return path
    target = safe_path(row["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ROOT / ".cache/blob-restore.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            restore_entry(row)
        except Exception as error:
            raise BlobUnavailable(f"Archived file could not be restored: {row['path']}") from error
        with connect() as db:
            db.execute("UPDATE files SET restored_at=? WHERE path=?", (stamp(), row["path"]))
    return path


def require_materialized_tree(source):
    """Prevent filesystem walks from silently omitting evicted regular files."""
    source = Path(source).absolute()
    if not INDEX.exists() or not source.is_relative_to(ROOT):
        return
    prefix = source.relative_to(ROOT).as_posix().rstrip("/") + "/"
    with sqlite3.connect(f"file:{INDEX}?mode=ro", uri=True) as db:
        rows = db.execute("SELECT path FROM files WHERE substr(path,1,?)=?", (len(prefix), prefix))
        for name, in rows:
            if not (ROOT / name).is_file():
                raise BlobUnavailable(
                    f"Restore cold inputs before capturing this tree: "
                    f".venv/bin/python -B -m lab.blob_store restore {json.dumps(prefix)}")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def verify_remote(s3, key, expected_hash, expected_bytes):
    response = s3.get_object(Bucket=BUCKET, Key=key)
    h, size = hashlib.sha256(), 0
    try:
        for block in iter(lambda: response["Body"].read(1 << 20), b""):
            h.update(block)
            size += len(block)
    finally:
        response["Body"].close()
    if size != expected_bytes or h.hexdigest() != expected_hash:
        raise ValueError(f"Remote readback differs: {key}")


def store_pack(pack, entries, s3, prune):
    from boto3.s3.transfer import TransferConfig
    pack_hash = digest(pack)
    size = pack.stat().st_size
    key = f"{PREFIX}/packs/{pack_hash}"
    # A content key is idempotent; the complete remote stream is checked on every attempt.
    s3.upload_file(str(pack), BUCKET, key, ExtraArgs={"Metadata": {"sha256": pack_hash}},
                   Config=TransferConfig(multipart_threshold=16 << 20, multipart_chunksize=16 << 20,
                                         max_concurrency=4, use_threads=True))
    verify_remote(s3, key, pack_hash, size)
    for row in entries:
        row["pack_key"] = key
    manifest = {"schema": "robot-learning-lab.cold-pack.v1", "key": key, "sha256": pack_hash,
                "bytes": size, "files": entries, "verified_at": stamp()}
    raw = (json.dumps(manifest, separators=(",", ":")) + "\n").encode()
    manifest_key = f"{PREFIX}/manifests/{pack_hash}.json"
    s3.put_object(Bucket=BUCKET, Key=manifest_key, Body=raw, ContentType="application/json")
    verify_remote(s3, manifest_key, hashlib.sha256(raw).hexdigest(), len(raw))
    # Exercise the actual range + decompression + atomic restore path for every pack.
    with tempfile.TemporaryDirectory(prefix="restore-check-", dir=pack.parent) as scratch:
        row = min(entries[1:] or entries, key=lambda e: e["stored_bytes"])
        restore_entry(row, s3, Path(scratch) / "sample")
    with connect() as db:
        db.execute("INSERT OR REPLACE INTO packs VALUES (?,?,?,?,?)",
                   (key, pack_hash, size, manifest_key, manifest["verified_at"]))
        for row in entries:
            old = db.execute("SELECT sha256 FROM files WHERE path=?", (row["path"],)).fetchone()
            if old and old[0] != row["sha256"]:
                raise ValueError(f"Indexed file changed: {row['path']}")
            db.execute("""INSERT OR IGNORE INTO files
                (path,sha256,bytes,mtime_ns,mode,pack_key,offset,stored_bytes,encoding)
                VALUES (:path,:sha256,:bytes,:mtime_ns,:mode,:pack_key,:offset,:stored_bytes,:encoding)""", row)
    removed = reclaimed = changed = 0
    if prune:
        with connect() as db:
            for row in entries:
                path = safe_path(row["path"])
                if not path.is_file() or path.is_symlink():
                    continue
                before = path.stat()
                if before.st_size != row["bytes"] or before.st_mtime_ns != row["mtime_ns"]:
                    changed += 1
                    continue
                if digest(path) != row["sha256"] or identity(path.stat()) != identity(before):
                    changed += 1
                    continue
                path.unlink()
                db.execute("UPDATE files SET evicted_at=? WHERE path=?", (stamp(), row["path"]))
                removed += 1
                reclaimed += row["bytes"]
    print(json.dumps({"pack": pack_hash[:16], "files": len(entries), "remote_bytes": size,
                      "evicted_files": removed, "evicted_bytes": reclaimed,
                      "held_changed": changed, "verified_at": manifest["verified_at"]}), flush=True)


def archive(plan_path, prune=False, workers=2):
    with gzip.open(plan_path, "rt") if str(plan_path).endswith(".gz") else open(plan_path) as f:
        plan = json.load(f)
    if plan.get("schema") != "robot-learning-lab.cold-selection.v1":
        raise ValueError("Unsupported explicit selection")
    staging = ROOT / ".cache/blob-staging"
    staging.mkdir(parents=True, exist_ok=True)
    with (staging / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        s3 = client()
        with connect() as db:
            indexed = {r[0] for r in db.execute("SELECT path FROM files")}
        with tempfile.TemporaryDirectory(dir=staging) as folder, ThreadPoolExecutor(max_workers=workers) as pool:
            folder = Path(folder)
            pending = []
            part = 0
            pack = folder / str(part)
            stream = pack.open("xb")
            entries = []
            def dispatch():
                nonlocal part, pack, stream, entries
                stream.close()
                future = pool.submit(store_pack, pack, entries, s3, prune)
                pending.append((future, pack))
                if len(pending) >= workers:
                    job, old_pack = pending.pop(0)
                    job.result()
                    old_pack.unlink()
                part += 1
                pack = folder / str(part)
                stream = pack.open("xb")
                entries = []
            try:
                for selected in plan["files"]:
                    name = selected["path"]
                    if name in indexed:
                        continue
                    path = safe_path(name)
                    if not path.is_file() or path.is_symlink():
                        continue
                    before = path.stat()
                    if before.st_size != selected["bytes"] or before.st_mtime_ns != selected["mtime_ns"]:
                        continue
                    offset = stream.tell()
                    h, encoder = hashlib.sha256(), zlib.compressobj(level=1)
                    with path.open("rb") as source:
                        for block in iter(lambda: source.read(1 << 20), b""):
                            h.update(block)
                            stream.write(encoder.compress(block))
                    stream.write(encoder.flush())
                    if identity(path.stat()) != identity(before):
                        stream.seek(offset)
                        stream.truncate()
                        continue
                    if name.startswith(".artifact-store/objects/sha256/") and h.hexdigest() != "".join(Path(name).parts[-2:]):
                        raise ValueError(f"Source CAS object hash differs from its name: {name}")
                    entries.append({"path": name, "bytes": before.st_size, "sha256": h.hexdigest(),
                                    "mtime_ns": before.st_mtime_ns, "mode": before.st_mode & 0o777,
                                    "offset": offset, "stored_bytes": stream.tell() - offset, "encoding": "zlib"})
                    if stream.tell() >= 256 << 20:
                        dispatch()
                if entries:
                    dispatch()
                for job, old_pack in pending:
                    job.result()
                    old_pack.unlink()
            finally:
                stream.close()


def recover_index():
    """Rebuild a missing local index from private, self-describing pack manifests."""
    s3 = client()
    count = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=f"{PREFIX}/manifests/"):
        for item in page.get("Contents", []):
            body = s3.get_object(Bucket=BUCKET, Key=item["Key"])["Body"]
            try:
                record = json.loads(body.read())
            finally:
                body.close()
            if (record.get("schema") != "robot-learning-lab.cold-pack.v1"
                    or record["key"] != f"{PREFIX}/packs/{record['sha256']}"
                    or not record.get("verified_at")):
                raise ValueError("Unexpected cold-pack manifest")
            with connect() as db:
                db.execute("INSERT OR IGNORE INTO packs VALUES (?,?,?,?,?)",
                           (record["key"], record["sha256"], record["bytes"],
                            item["Key"], record["verified_at"]))
                for row in record["files"]:
                    safe_path(row["path"])
                    db.execute("""INSERT OR IGNORE INTO files
                      (path,sha256,bytes,mtime_ns,mode,pack_key,offset,stored_bytes,encoding)
                      VALUES (:path,:sha256,:bytes,:mtime_ns,:mode,:pack_key,:offset,:stored_bytes,:encoding)""", row)
                    count += 1
    print(json.dumps({"recovered_manifest_files": count}))


def rows_for_restore(prefix):
    base = safe_path(prefix)
    with connect() as db:
        indexed = {r["path"]: dict(r) for r in db.execute("SELECT * FROM files")}
    selected = {name for name in indexed if name.startswith(prefix)}
    paths = base.rglob("*") if base.is_dir() else [base]
    for path in paths:
        if path.is_symlink():
            target = path.resolve()
            if target.is_relative_to(ROOT):
                name = target.relative_to(ROOT).as_posix()
                if name in indexed:
                    selected.add(name)
    return [indexed[name] for name in sorted(selected)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("archive")
    a.add_argument("plan", type=Path)
    a.add_argument("--prune", action="store_true")
    a.add_argument("--workers", type=int, default=2)
    sub.add_parser("recover-index")
    for command in ("restore", "status", "evict"):
        p = sub.add_parser(command)
        if command == "status":
            p.add_argument("prefix", nargs="?", default="")
        else:
            p.add_argument("prefix")
    args = parser.parse_args()
    if args.command == "archive":
        return archive(args.plan, args.prune, args.workers)
    if args.command == "recover-index":
        return recover_index()
    if args.command == "restore":
        rows = rows_for_restore(args.prefix)
    else:
        with connect() as db:
            rows = [dict(r) for r in db.execute("SELECT * FROM files ORDER BY path")
                    if r["path"].startswith(args.prefix)]
    if args.command == "status":
        local = [r for r in rows if safe_path(r["path"]).is_file()]
        print(json.dumps({"files": len(rows), "bytes": sum(r["bytes"] for r in rows),
                          "local_files": len(local), "local_bytes": sum(r["bytes"] for r in local)}, indent=2))
        return
    s3 = client()
    if args.command == "evict":
        with connect() as db:
            packs = {r["key"]: dict(r) for r in db.execute("SELECT * FROM packs")}
        for key in {row["pack_key"] for row in rows if safe_path(row["path"]).is_file()}:
            verify_remote(s3, key, packs[key]["sha256"], packs[key]["bytes"])
    for row in rows:
        if args.command == "restore":
            restore_entry(row, s3)
        else:
            path = safe_path(row["path"])
            if path.is_file() and not path.is_symlink() and digest(path) == row["sha256"]:
                path.unlink()
    print(json.dumps({"operation": args.command, "indexed_files": len(rows)}))


if __name__ == "__main__":
    main()
