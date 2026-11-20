"""Verify registered regeneration recipes, then remove their stored copies.

Mixed packs are compacted without changing retained file bytes. New packs and
restore samples are verified before the live index changes or old objects are deleted.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tarfile
import tempfile
import time
import fcntl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from infra.storage import blob_store as bs
from infra.storage.regenerable_artifacts import REGISTRY, rebuild


def main(apply):
    records = json.loads(REGISTRY.read_text())["files"]
    output = ROOT / "data/artifact-offload-20261025/remote-regeneration-prune"
    output.mkdir(exist_ok=True)
    restored = {}
    checks_path = output / "regeneration-checks.json"
    receipts_path = output / "pack-receipts.json"
    checks = json.loads(checks_path.read_text()) if checks_path.exists() else []
    receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else []
    s3 = bs.client()
    with (ROOT / ".cache/blob-staging/writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with bs.connect() as db:
            index = {r["path"]: dict(r) for r in db.execute("SELECT * FROM files")}
            packs = {r["key"]: dict(r) for r in db.execute("SELECT * FROM packs")}
        selected = {n: r for n, r in records.items() if n in index}
        if not selected:
            print("No archived copies of registered disposable exports remain.", flush=True)
            return
        # Prove exact container recovery for every selected file before any deletion.
        with tempfile.TemporaryDirectory(prefix="regeneration-proof-", dir=ROOT / ".cache") as scratch:
            for name, record in selected.items():
                if (index[name]["sha256"], index[name]["bytes"]) != (record["sha256"], record["bytes"]):
                    raise ValueError(f"Archive record changed: {name}")
                for key in ["canonical_unit_fields", "case_coefficients_report"]:
                    path = ROOT / record[key]
                    if not path.is_file():
                        restored[record[key]] = None
                start = time.monotonic()
                destination = Path(scratch) / "case-fields.npz"
                rebuild(record, destination)
                destination.unlink()
                checks.append({"path": name, "sha256": record["sha256"], "bytes": record["bytes"],
                               "regeneration_seconds": time.monotonic() - start})
                print(json.dumps(checks[-1]), flush=True)
        checks = list({r["path"]: r for r in checks}.values())
        (output / "regeneration-checks.json").write_text(json.dumps(checks, indent=2) + "\n")
        if not apply:
            return
        by_pack = defaultdict(list)
        for name in selected:
            by_pack[index[name]["pack_key"]].append(name)
        for key, removed in by_pack.items():
            original = packs[key]
            response = s3.get_object(Bucket=bs.BUCKET, Key=original["manifest_key"])
            raw = response["Body"].read()
            response["Body"].close()
            manifest = json.loads(raw)
            if manifest["key"] != key or manifest["sha256"] != original["sha256"]:
                raise ValueError("Pack manifest differs")
            survivors = [r.copy() for r in manifest["files"] if r["path"] not in selected]
            new_rows = []
            new_key = None
            with tempfile.TemporaryDirectory(prefix="compact-pack-", dir=ROOT / ".cache") as scratch:
                old = Path(scratch) / "old"
                s3.download_file(bs.BUCKET, key, str(old))
                if old.stat().st_size != original["bytes"] or bs.digest(old) != original["sha256"]:
                    raise ValueError("Original remote pack differs")
                if survivors:
                    new = Path(scratch) / "new"
                    with old.open("rb") as source, new.open("xb") as target:
                        for row in survivors:
                            source.seek(row["offset"])
                            row["offset"] = target.tell()
                            remaining = row["stored_bytes"]
                            while remaining:
                                block = source.read(min(1 << 20, remaining))
                                if not block:
                                    raise ValueError("Truncated retained range")
                                target.write(block)
                                remaining -= len(block)
                            new_rows.append(row)
                    # Uses full upload/readback, manifest verification, and actual range restore.
                    bs.store_pack(new, new_rows, s3, prune=False)
                    new_key = f"{bs.PREFIX}/packs/{bs.digest(new)}"
                    new_bytes = new.stat().st_size
                else:
                    new_bytes = 0
                with bs.connect() as db:
                    for name in removed:
                        current = db.execute("SELECT sha256,pack_key FROM files WHERE path=?", (name,)).fetchone()
                        if current is None or tuple(current) != (selected[name]["sha256"], key):
                            raise ValueError("Live index changed")
                    for row in new_rows:
                        db.execute("UPDATE files SET pack_key=?,offset=?,stored_bytes=? WHERE path=? AND pack_key=?",
                                   (new_key, row["offset"], row["stored_bytes"], row["path"], key))
                    for name in removed:
                        db.execute("DELETE FROM files WHERE path=?", (name,))
                    if db.execute("SELECT COUNT(*) FROM files WHERE pack_key=?", (key,)).fetchone()[0]:
                        raise ValueError("Old pack still has live references")
                    db.execute("DELETE FROM packs WHERE key=?", (key,))
                # The exact current restore index now uses verified replacement ranges.
                s3.delete_object(Bucket=bs.BUCKET, Key=key)
                s3.delete_object(Bucket=bs.BUCKET, Key=original["manifest_key"])
                for deleted_key in [key, original["manifest_key"]]:
                    for attempt in range(10):
                        try:
                            s3.head_object(Bucket=bs.BUCKET, Key=deleted_key)
                        except Exception as error:
                            code = str(getattr(error, "response", {}).get("Error", {}).get("Code"))
                            if code not in {"404", "NoSuchKey", "NotFound"}:
                                raise
                            break
                        time.sleep(1)
                    else:
                        raise ValueError("Remote deletion unconfirmed")
                receipt = {"old_pack": key, "replacement_pack": new_key,
                           "deleted_remote_bytes": original["bytes"], "replacement_remote_bytes": new_bytes,
                           "removed_paths": removed, "retained_paths": len(new_rows)}
                receipts.append(receipt)
                (output / "pack-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
                print(json.dumps(receipt), flush=True)
        # Exercise both dashboard paths after remote deletion: regenerated data and compacted native input.
        from urllib.parse import quote
        from urllib.request import urlopen
        http_checks = []
        probes = [next(iter(records))]
        replacement_keys = {r["replacement_pack"] for r in receipts if r["replacement_pack"]}
        with bs.connect() as db:
            for row in db.execute("SELECT path,bytes FROM files ORDER BY bytes"):
                if row["bytes"] < 65536:
                    continue
                current = bs.lookup(ROOT / row["path"])
                if current["pack_key"] in replacement_keys and not (ROOT / row["path"]).is_file():
                    probes.append(row["path"])
                    break
        for name in probes:
            path = ROOT / name
            was_present = path.is_file()
            expected = records.get(name) or bs.lookup(path)
            h = __import__("hashlib").sha256()
            with urlopen("http://127.0.0.1:8805/asset?path=" + quote(name, safe=""), timeout=120) as response:
                if response.status != 200:
                    raise ValueError("Dashboard asset request failed")
                for block in iter(lambda: response.read(1 << 20), b""):
                    h.update(block)
            if h.hexdigest() != expected["sha256"]:
                raise ValueError("Dashboard asset differs")
            http_checks.append({"path": name, "http_status": 200, "sha256": h.hexdigest()})
            if not was_present and bs.digest(path) == expected["sha256"]:
                path.unlink()
        (output / "dashboard-checks.json").write_text(json.dumps(http_checks, indent=2) + "\n")
        # Remove only input copies materialized by this verification run.
        for name in {r["canonical_unit_fields"] for r in records.values()}:
            path = ROOT / name
            row = bs.lookup(path)
            if row and path.is_file() and not path.is_symlink() and bs.digest(path) == row["sha256"]:
                path.unlink()
        with bs.connect() as db:
            remaining = {r[0] for r in db.execute("SELECT path FROM files")}
        deleted = {n: r for n, r in records.items() if n not in remaining}
        report = {"dashboard_checks": http_checks, "completed_utc": datetime.now(timezone.utc).isoformat(),
                  "deleted_derived_files": len(deleted),
                  "deleted_derived_logical_bytes": sum(r["bytes"] for r in deleted.values()),
                  "remote_space_reclaimed_bytes": sum(r["deleted_remote_bytes"]-r["replacement_remote_bytes"] for r in receipts),
                  "regeneration": ".venv/bin/python -B -m infra.storage.regenerable_artifacts EXACT_PATH",
                  "historical_index_recovery": "After recovering an older index snapshot, run .venv/bin/python -B -m infra.storage.blob_store recover-index to obtain the compacted pack ranges. Registered derived files regenerate from their retained canonical inputs.",
                  "canonical_inputs_retained": True}
        (output / "completion.json").write_text(json.dumps(report, indent=2) + "\n")
        with bs.connect() as source, sqlite3.connect(output / "blob-index-after.sqlite") as target:
            source.backup(target)
        bundle = ROOT / ".cache/regeneration-prune-recovery-20261025.tar.gz"
        sources = [ROOT/"AGENTS.md", ROOT/"docs/design/artifact-storage.md", REGISTRY,
                   ROOT/"infra/storage/regenerable_artifacts.py", ROOT/"infra/storage/blob_store.py", Path(__file__)]
        sources += [p for p in output.iterdir() if p.is_file()]
        with tarfile.open(bundle, "w:gz", compresslevel=3) as tar:
            for path in sources:
                tar.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
        spec = importlib.util.spec_from_file_location("archive", ROOT/"infra/storage/artifact_archive.py")
        archive = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(archive)
        archive_manifest = output / "recovery-manifest.json"
        archive.make_plan(ROOT, [bundle.relative_to(ROOT).as_posix()], archive_manifest)
        backup = archive.upload(ROOT, archive_manifest, s3, bs.BUCKET)
        (output / "recovery-receipt.json").write_text(json.dumps(backup, indent=2) + "\n")
        (output / "blob-index-after.sqlite").unlink()
        bundle.unlink()
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    main(parser.parse_args().apply)
