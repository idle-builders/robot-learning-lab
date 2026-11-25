"""Release exact local copies from a previously hash-verified private archive.

Usage: .venv/bin/python -B infra/storage/reclaim_archived_storage.py PLAN.json
The plan names existing blob-index entries. Each remote pack must still have its
verified SHA-256 metadata/size and a server LastModified no later than its recorded
full readback. --fresh-readback additionally downloads every selected original.
This never uploads or deletes remote objects. Restore with infra.storage.blob_store.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from infra.storage import blob_store as bs


def stamp():
    return datetime.now(timezone.utc).isoformat()


def declared_paths():
    """Keep local every artifact the body definitions declare: the body manifest and each generator's inputs and outputs
    and pinned files (robot/**/<builder>.bom.yaml, robot/body/bom/definitions.py)."""
    sys.path.insert(0, str(ROOT / "robot/body/bom"))
    import definitions
    D = definitions.load()
    paths = {D.body["manifest"]} if D.body else set()
    for g in D.generators:
        paths.update(g.get("inputs", []) + g.get("outputs", []) + g.get("in_place_edits", []) + g.get("writes_into", []))
    paths.update(x["path"] for x in D.pinned)
    return {x.rstrip("/") for x in paths if x.startswith("artifacts/")}


def verify_range(s3, row):
    """Recover and hash the entire original without retaining a scratch copy."""
    if row["encoding"] != "zlib":
        raise ValueError("Unsupported encoding")
    end = row["offset"] + row["stored_bytes"] - 1
    response = s3.get_object(Bucket=bs.BUCKET, Key=row["pack_key"],
                             Range=f"bytes={row['offset']}-{end}")
    decoder, h, size = zlib.decompressobj(), hashlib.sha256(), 0
    try:
        if (response["ResponseMetadata"]["HTTPStatusCode"] != 206
                or response["ContentLength"] != row["stored_bytes"]):
            raise ValueError("Remote range response differs")
        for block in iter(lambda: response["Body"].read(64 << 10), b""):
            data = decoder.decompress(block)
            h.update(data)
            size += len(data)
            if size > row["bytes"]:
                raise ValueError("Recovered file exceeds original size")
        tail = decoder.flush()
        h.update(tail)
        size += len(tail)
        if (not decoder.eof or decoder.unused_data or decoder.unconsumed_tail
                or size != row["bytes"] or h.hexdigest() != row["sha256"]):
            raise ValueError("Remote original SHA-256 or size differs")
        return {"bytes": size, "sha256": h.hexdigest(), "http_status": 206,
                "compressed_range_bytes": row["stored_bytes"], "zlib_eof": True}
    finally:
        response["Body"].close()


def verify_archived_pack(s3, pack):
    response = s3.head_object(Bucket=bs.BUCKET, Key=pack["key"])
    modified = response["LastModified"]
    verified = datetime.fromisoformat(pack["verified_at"])
    if (response["ContentLength"] != pack["bytes"]
            or response.get("Metadata", {}).get("sha256") != pack["sha256"]):
        raise ValueError(f"Remote pack identity differs: {pack['key']}")
    proof = {"pack_key": pack["key"], "sha256": pack["sha256"],
            "bytes": pack["bytes"], "full_readback_verified_at": pack["verified_at"],
            "server_last_modified": modified.isoformat(),
            "head_checked_at": stamp(), "etag": response["ETag"],
            "manifest_key": pack["manifest_key"],
            "basis": "Recorded complete SHA-256 readback; remote object unchanged since verification"}
    if modified > verified:
        # Some earlier server timestamps are ahead of the client readback clock.
        # Do not loosen the comparison: read and hash the whole pack again.
        bs.verify_remote(s3, pack["key"], pack["sha256"], pack["bytes"])
        proof.update(basis="Fresh complete pack SHA-256 readback",
                     fresh_full_readback_at=stamp())
    return proof


def main(plan_path, fresh_readback=False):
    plan_path = plan_path.resolve()
    plan = json.loads(plan_path.read_text())
    if plan["schema"] != "robot-learning-lab.local-cold-release.v1":
        raise ValueError("Unexpected plan schema")
    paths = [r["path"] for r in plan["files"]]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate selection")
    journal = plan_path.with_suffix(".receipt.jsonl")
    # One receipt is written after each action; reruns verify and skip completed paths.
    completed = {r["path"] for line in journal.read_text().splitlines()
                 if (r := json.loads(line)).get("action") == "released"} if journal.exists() else set()
    open_files = {line[1:] for line in subprocess.run(
        ["lsof", "-nP", "-Fn"], capture_output=True, text=True).stdout.splitlines()
                  if line.startswith("n/")}
    protected = declared_paths()
    allowed = ("artifacts/cove50-strength/", "artifacts/cove50-power-integration/",
               "artifacts/cove50-engineering-competition/")
    total = 0
    with (ROOT / ".cache/blob-staging/writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        s3 = bs.client()
        with sqlite3.connect(f"file:{bs.INDEX}?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            keys = {r["pack_key"] for r in plan["files"] if r["path"] not in completed}
            packs = [dict(r) for r in db.execute("SELECT * FROM packs") if r["key"] in keys]
            if len(packs) != len(keys):
                raise ValueError("A selected pack is no longer indexed")
            proofs, held_packs = [], {}
            proof_path = plan_path.with_suffix(".remote-packs.json")
            with ThreadPoolExecutor(max_workers=4) as pool:
                pending = {pool.submit(verify_archived_pack, s3, p): p for p in packs}
                for future in as_completed(pending):
                    pack = pending[future]
                    try:
                        proofs.append(future.result())
                    except Exception as error:
                        held_packs[pack["key"]] = f"{type(error).__name__}: {error}"
                    proof_path.write_text(json.dumps({"verified": proofs, "held": held_packs}, indent=2) + "\n")
                    print(json.dumps({"checked_packs": len(proofs) + len(held_packs),
                                      "verified_packs": len(proofs), "held_packs": len(held_packs)}), flush=True)
            pack_proofs = {p["pack_key"]: p for p in proofs}
            print(json.dumps({"unchanged_hash_verified_packs": len(proofs)}), flush=True)
            with journal.open("a", buffering=1) as receipt:
                for item in plan["files"]:
                    name = item["path"]
                    path = bs.safe_path(name)
                    if name in completed:
                        if path.exists():
                            raise ValueError(f"Completed path was rematerialized: {name}")
                        continue
                    if item["pack_key"] in held_packs:
                        receipt.write(json.dumps({"path": name, "action": "held",
                                                  "reason": held_packs[item["pack_key"]], "at": stamp()}) + "\n")
                        continue
                    if (not name.startswith(allowed) or "/hand-" in name
                            or "/internal-r33-" in name or "/selected-round-" in name
                            or any(name == p or name.startswith(p + "/") for p in protected)
                            or str(path) in open_files):
                        raise ValueError(f"Protected or open path: {name}")
                    row = db.execute("SELECT f.* FROM files f JOIN packs p ON p.key=f.pack_key "
                                     "WHERE f.path=? AND p.verified_at IS NOT NULL", (name,)).fetchone()
                    if row is None or any(row[k] != item[k] for k in
                                          ("sha256", "bytes", "pack_key", "offset", "stored_bytes")):
                        raise ValueError(f"Archive index changed: {name}")
                    if path.is_symlink() or not path.is_file():
                        raise ValueError(f"Expected regular local file: {name}")
                    before = bs.identity(path.stat())
                    if (path.stat().st_size != row["bytes"] or bs.digest(path) != row["sha256"]
                            or bs.identity(path.stat()) != before):
                        raise ValueError(f"Local bytes changed: {name}")
                    proof = (verify_range(s3, row) if fresh_readback else
                             {"pack_proof": str(proof_path.relative_to(ROOT)),
                              "pack_key": row["pack_key"], "original_sha256": row["sha256"],
                              "basis": pack_proofs[row["pack_key"]]["basis"]})
                    if bs.identity(path.stat()) != before:
                        raise ValueError(f"Local file changed during recovery: {name}")
                    # Restore references already exist in the live index and sealed plan.
                    receipt.write(json.dumps({"path": name, "action": "verified",
                                              "at": stamp(), "remote_original": proof}) + "\n")
                    path.unlink()
                    receipt.write(json.dumps({"path": name, "action": "released",
                                              "bytes": row["bytes"], "at": stamp()}) + "\n")
                    total += row["bytes"]
                    print(json.dumps({"released": name, "bytes": row["bytes"],
                                      "this_run_bytes": total}), flush=True)
    print(json.dumps({"this_run_released_bytes": total, "receipt": str(journal)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--fresh-readback", action="store_true")
    args = parser.parse_args()
    main(args.plan, args.fresh_readback)
