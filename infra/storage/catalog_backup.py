#!/usr/bin/env python3
"""Back up and restore the local catalogs a fresh clone needs to reach private evidence.

`data/results.sqlite` (dashboard records), `data/blob-store.sqlite` (the path -> bucket
pack index used to restore cold files) and the other local records under `data/` (storage run
state, run records and build records that code reads) are Git-ignored. Without a current copy, a new
computer cannot restore anything offloaded since the last backup. This tool keeps that copy
in the existing private bucket and records the exact objects in `data/catalog-backup.json`.
That pointer is Git-ignored like the rest of `data/` (the repository is public); each backup
also stores it in the bucket as `robot-learning-lab/catalogs/latest.json`, which `restore`
fetches when the local pointer is missing.

    .venv/bin/python infra/storage/catalog_backup.py backup    # after publishing or offloading
    .venv/bin/python infra/storage/catalog_backup.py restore   # on a fresh clone, before the dashboard

Backups use SQLite's online backup API (consistent while other processes write), gzip, a full
read-back SHA-256 check, and never delete remote objects.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from infra.storage.archive_config import client, settings  # noqa: E402

CATALOGS = ("data/results.sqlite", "data/blob-store.sqlite")
# Everything else under data/ travels as one archive; the pointer travels as latest.json.
TRACKED = {"data/catalog-backup.json"}
RECORDS = "data/records.tar.gz"


def pack_records(target: Path) -> int:
    import tarfile
    count = 0
    with tarfile.open(target, "w:gz", compresslevel=6) as tar:
        for path in sorted((ROOT / "data").rglob("*")):
            name = path.relative_to(ROOT).as_posix()
            if (not path.is_file() or path.is_symlink() or name in TRACKED or name.startswith(CATALOGS)
                    or name == "data/private-archive.json"):
                continue
            tar.add(path, arcname=name, recursive=False)
            count += 1
    return count
POINTER = ROOT / "data/catalog-backup.json"
PREFIX = "robot-learning-lab/catalogs"
LATEST = f"{PREFIX}/latest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def backup() -> dict:
    s3, bucket = client(), settings()["bucket"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows = []
    with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as tmp:
        for name in CATALOGS:
            snapshot, packed = Path(tmp) / Path(name).name, Path(tmp) / (Path(name).name + ".gz")
            with sqlite3.connect(ROOT / name) as source, sqlite3.connect(snapshot) as target:
                source.backup(target)
            with snapshot.open("rb") as raw, gzip.open(packed, "wb", compresslevel=6) as out:
                shutil.copyfileobj(raw, out, 1 << 20)
            key = f"{PREFIX}/{stamp}/{packed.name}"
            s3.upload_file(str(packed), bucket, key)
            check = Path(tmp) / "readback.gz"
            s3.download_file(bucket, key, str(check))
            if sha256(check) != sha256(packed):
                raise RuntimeError(f"Read-back mismatch for {key}; pointer not updated")
            rows.append({"path": name, "key": key, "sha256": sha256(packed), "bytes": packed.stat().st_size,
                         "sqlite_sha256": sha256(snapshot), "sqlite_bytes": snapshot.stat().st_size})
            check.unlink()
        packed = Path(tmp) / Path(RECORDS).name
        files = pack_records(packed)
        key = f"{PREFIX}/{stamp}/{packed.name}"
        s3.upload_file(str(packed), bucket, key)
        check = Path(tmp) / "readback.gz"
        s3.download_file(bucket, key, str(check))
        if sha256(check) != sha256(packed):
            raise RuntimeError(f"Read-back mismatch for {key}; pointer not updated")
        rows.append({"path": RECORDS, "key": key, "sha256": sha256(packed), "bytes": packed.stat().st_size,
                     "files": files})
    record = {"schema": "robot-learning-lab.catalog-backup.v1", "created_at_utc": stamp,
              "bucket_setting": "data/private-archive.json or ROBOTLAB_ARCHIVE_BUCKET", "objects": rows}
    POINTER.write_text(json.dumps(record, indent=1) + "\n")
    s3.upload_file(str(POINTER), bucket, LATEST)
    return record


def restore(force: bool) -> list[str]:
    s3, bucket, done = client(), settings()["bucket"], []
    if not POINTER.exists():
        POINTER.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, LATEST, str(POINTER))
    record = json.loads(POINTER.read_text())
    for row in record["objects"]:
        if row["path"] == RECORDS:
            done += restore_records(s3, bucket, row, force)
            continue
        target = ROOT / row["path"]
        if target.exists() and not force:
            raise FileExistsError(f"{row['path']} exists; pass --force to replace it")
        with tempfile.TemporaryDirectory() as tmp:
            packed = Path(tmp) / "catalog.gz"
            s3.download_file(bucket, row["key"], str(packed))
            if sha256(packed) != row["sha256"]:
                raise RuntimeError(f"Downloaded {row['key']} does not match the recorded SHA-256")
            plain = Path(tmp) / "catalog.sqlite"
            with gzip.open(packed, "rb") as src, plain.open("wb") as out:
                shutil.copyfileobj(src, out, 1 << 20)
            if sha256(plain) != row["sqlite_sha256"]:
                raise RuntimeError(f"Decompressed {row['path']} does not match the recorded SHA-256")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(plain, target)
        done.append(row["path"])
    return done


def restore_records(s3, bucket, row, force) -> list[str]:
    import tarfile
    with tempfile.TemporaryDirectory() as tmp:
        packed = Path(tmp) / "records.tar.gz"
        s3.download_file(bucket, row["key"], str(packed))
        if sha256(packed) != row["sha256"]:
            raise RuntimeError(f"Downloaded {row['key']} does not match the recorded SHA-256")
        with tarfile.open(packed) as tar:
            members = [m for m in tar.getmembers() if m.isfile() and m.name.startswith("data/")
                       and ".." not in Path(m.name).parts and (force or not (ROOT / m.name).exists())]
            tar.extractall(ROOT, members=members, filter="data")
    return [f"{RECORDS} ({len(members)} files)"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("backup")
    restore_parser = sub.add_parser("restore")
    restore_parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    (ROOT / ".cache").mkdir(exist_ok=True)
    result = backup() if args.command == "backup" else restore(args.force)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
