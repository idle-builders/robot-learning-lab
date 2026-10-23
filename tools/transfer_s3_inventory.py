"""Copy an explicit Runpod S3 inventory to Infomaniak S3, with full read-back hashes.

This never deletes source or destination objects. A JSONL journal makes reruns safe.
The inventory is a point-in-time key/size list, not a content hash: both source
and destination bytes are streamed and SHA-256 checked during transfer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import time


CHUNK = 16 * 1024 * 1024
HASH_CHUNK = 64 * 1024 * 1024
DEST_PREFIX = "robot-learning-lab/runpod-migration/v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_key(key: str) -> str:
    if not isinstance(key, str) or "\\" in key or "\x00" in key:
        raise ValueError("unsafe S3 key")
    path = PurePosixPath(key)
    if not path.parts or path.is_absolute() or ".." in path.parts or path.as_posix() != key:
        raise ValueError(f"unsafe S3 key: {key!r}")
    return key


def load_inventory(path: Path, selection: Path | None, all_objects: bool) -> tuple[dict, list[dict]]:
    inventory = json.loads(path.read_text())
    volume = inventory.get("volume_id")
    if not isinstance(volume, str) or not re.fullmatch(r"[a-z0-9]{10}", volume):
        raise ValueError("inventory must identify one Runpod volume")
    objects = inventory.get("objects")
    if not isinstance(objects, list):
        raise ValueError("inventory has no objects list")
    by_key = {}
    for row in objects:
        key = safe_key(row["key"])
        if key in by_key or type(row["size"]) is not int or row["size"] < 0:
            raise ValueError(f"invalid or repeated inventory key: {key}")
        by_key[key] = row
    if all_objects == (selection is not None):
        raise ValueError("choose exactly one of --all or --selection")
    if selection:
        selected = json.loads(selection.read_text())
        if selected.get("volume_id") != volume or not isinstance(selected.get("keys"), list):
            raise ValueError("selection must identify this volume and a keys list")
        keys = selected["keys"]
        if len(set(keys)) != len(keys) or any(key not in by_key for key in keys):
            raise ValueError("selection contains repeated or unknown keys")
        expected = selected.get("expected_sha256", {})
        if (not isinstance(expected, dict) or set(expected) - set(keys) or
                any(not isinstance(sha, str) or
                    not re.fullmatch(r"[0-9a-f]{64}", sha)
                    for sha in expected.values())):
            raise ValueError("invalid expected SHA-256 in selection")
    else:
        keys = list(by_key)
        expected = {}
    if not keys:
        raise ValueError("empty transfer selection")
    return inventory, [{**by_key[key], **({"expected_sha256": expected[key]}
                                    if key in expected else {})} for key in keys]


def client(profile: str | None, endpoint: str, region: str):
    if not endpoint.startswith("https://"):
        raise ValueError("S3 endpoint must use HTTPS")
    import boto3
    from botocore.config import Config

    return boto3.Session(profile_name=profile).client(
        "s3", endpoint_url=endpoint, region_name=region,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                      request_checksum_calculation="when_required",
                      response_checksum_validation="when_required",
                      connect_timeout=10, read_timeout=90,
                      retries={"max_attempts": 2}),
    )


def destination_key(volume: str, key: str) -> str:
    return f"{DEST_PREFIX}/volumes/{volume}/{safe_key(key)}"


def head_or_none(s3, bucket: str, key: str):
    try:
        return s3.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        if str(code) in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise


def retry(operation, description: str):
    for attempt in range(5):
        try:
            return operation()
        except Exception:
            if attempt == 4:
                raise
            print(f"retry {attempt + 1}/4 {description}", flush=True)
            time.sleep(min(2 ** attempt, 8))


def read_range(s3, bucket: str, key: str, start: int, length: int) -> bytes:
    def read():
        response = s3.get_object(Bucket=bucket, Key=key,
                                 Range=f"bytes={start}-{start + length - 1}")
        body = response["Body"]
        try:
            data = read_exact(body, length + 1)
        finally:
            body.close()
        if len(data) != length:
            raise ValueError(f"ranged read length mismatch: {key} at {start}")
        return data
    return retry(read, f"read {key} at {start}")


def stream_hash(s3, bucket: str, key: str, expected_size: int) -> str:
    digest = hashlib.sha256()
    if expected_size == 0:
        response = retry(lambda: s3.get_object(Bucket=bucket, Key=key),
                         f"read empty {key}")
        try:
            if response["Body"].read(1):
                raise ValueError(f"object size mismatch: {key}")
        finally:
            response["Body"].close()
    else:
        for start in range(0, expected_size, HASH_CHUNK):
            digest.update(read_range(s3, bucket, key, start,
                                     min(HASH_CHUNK, expected_size - start)))
    return digest.hexdigest()


def read_exact(body, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = body.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def copy_one(source, destination, source_bucket: str, destination_bucket: str,
             row: dict) -> dict:
    source_key = safe_key(row["key"])
    size = row["size"]
    target_key = destination_key(source_bucket, source_key)
    before = source.head_object(Bucket=source_bucket, Key=source_key)
    if before["ContentLength"] != size:
        raise ValueError(f"source size changed: {source_key}")
    if row.get("etag") and before.get("ETag", "").strip('"') != row["etag"]:
        raise ValueError(f"source ETag changed: {source_key}")

    existing = head_or_none(destination, destination_bucket, target_key)
    if existing is not None:
        if existing["ContentLength"] != size:
            raise ValueError(f"destination size conflict: {target_key}")
        source_hash = stream_hash(source, source_bucket, source_key, size)
        destination_hash = stream_hash(destination, destination_bucket, target_key, size)
        if destination_hash != source_hash:
            raise ValueError(f"destination hash conflict: {target_key}")
    else:
        digest = hashlib.sha256()
        received = 0
        upload_id = None
        try:
            if size <= CHUNK:
                if size:
                    data = read_range(source, source_bucket, source_key, 0, size)
                else:
                    response = source.get_object(Bucket=source_bucket, Key=source_key)
                    try:
                        data = response["Body"].read(1)
                    finally:
                        response["Body"].close()
                if len(data) != size:
                    raise ValueError(f"source read size changed: {source_key}")
                digest.update(data)
                received = len(data)
                retry(lambda: destination.put_object(Bucket=destination_bucket,
                                                    Key=target_key, Body=data),
                      f"write {target_key}")
            else:
                upload_id = destination.create_multipart_upload(
                    Bucket=destination_bucket, Key=target_key)["UploadId"]
                parts = []
                while received < size:
                    data = read_range(source, source_bucket, source_key, received,
                                      min(CHUNK, size - received))
                    number = len(parts) + 1
                    response_part = retry(
                        lambda: destination.upload_part(
                            Bucket=destination_bucket, Key=target_key,
                            UploadId=upload_id, PartNumber=number, Body=data),
                        f"write part {number} {target_key}")
                    parts.append({"ETag": response_part["ETag"], "PartNumber": number})
                    received += len(data)
                    digest.update(data)
                destination.complete_multipart_upload(
                    Bucket=destination_bucket, Key=target_key, UploadId=upload_id,
                    MultipartUpload={"Parts": parts})
                upload_id = None
            source_hash = digest.hexdigest()
        except BaseException:
            if upload_id is not None:
                try:
                    destination.abort_multipart_upload(
                        Bucket=destination_bucket, Key=target_key, UploadId=upload_id)
                except Exception:
                    pass
            raise
        if received != size:
            raise ValueError(f"source size changed during copy: {source_key}")
        destination_hash = stream_hash(destination, destination_bucket, target_key, size)
        if destination_hash != source_hash:
            raise ValueError(f"destination SHA-256 mismatch: {target_key}")

    after = source.head_object(Bucket=source_bucket, Key=source_key)
    if (after["ContentLength"] != size or
            after.get("ETag") != before.get("ETag")):
        raise ValueError(f"source changed during copy: {source_key}")
    if row.get("expected_sha256") and source_hash != row["expected_sha256"]:
        raise ValueError(f"source differs from sealed SHA-256: {source_key}")
    return {"source_volume_id": source_bucket, "source_key": source_key,
            "destination_bucket": destination_bucket, "destination_key": target_key,
            "size": size, "sha256": source_hash}


def load_journal(path: Path, volume: str, bucket: str) -> dict[str, dict]:
    completed = {}
    if not path.exists():
        return completed
    for line in path.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row["source_volume_id"] != volume or row["destination_bucket"] != bucket:
            raise ValueError("journal belongs to another source or destination")
        key = row["source_key"]
        if (row["destination_key"] != destination_key(volume, key) or
                not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])):
            raise ValueError(f"invalid journal row: {key}")
        if key in completed and completed[key] != row:
            raise ValueError(f"conflicting journal rows: {key}")
        completed[key] = row
    return completed


def transfer(inventory_path: Path, selection: Path | None, all_objects: bool,
             source, destination, destination_bucket: str,
             journal_path: Path, manifest_path: Path) -> dict:
    inventory, objects = load_inventory(inventory_path, selection, all_objects)
    source_bucket = inventory["volume_id"]
    completed = load_journal(journal_path, source_bucket, destination_bucket)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    for row in objects:
        key = row["key"]
        if key in completed:
            saved = completed[key]
            if saved["size"] != row["size"]:
                raise ValueError(f"journal size conflict: {key}")
            if row.get("expected_sha256") and saved["sha256"] != row["expected_sha256"]:
                raise ValueError(f"journal differs from sealed SHA-256: {key}")
            current_source = source.head_object(Bucket=source_bucket, Key=key)
            if (current_source["ContentLength"] != row["size"] or
                    row.get("etag") and
                    current_source.get("ETag", "").strip('"') != row["etag"]):
                raise ValueError(f"source changed since journal: {key}")
            if stream_hash(destination, destination_bucket, saved["destination_key"],
                           saved["size"]) != saved["sha256"]:
                raise ValueError(f"journal destination hash conflict: {key}")
            continue
        verified = copy_one(source, destination, source_bucket, destination_bucket, row)
        with journal_path.open("a") as file:
            file.write(json.dumps(verified, sort_keys=True) + "\n")
            file.flush()
            os.fsync(file.fileno())
        completed[key] = verified
        print(f"verified {len(completed)}/{len(objects)} {key} ({row['size']} bytes)",
              flush=True)

    manifest = {
        "schema": "robot-learning-lab.runpod-s3-migration.v1",
        "source_inventory_captured_at_utc": inventory.get("captured_at_utc"),
        "source_inventory_sha256": sha256_bytes(inventory_path.read_bytes()),
        "source_volume_id": source_bucket,
        "destination_bucket": destination_bucket,
        "objects": [completed[row["key"]] for row in objects],
    }
    encoded = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        if manifest_path.read_bytes() != encoded:
            raise ValueError("existing migration manifest differs")
    else:
        manifest_path.write_bytes(encoded)
    digest = sha256_bytes(encoded)
    key = f"{DEST_PREFIX}/manifests/sha256/{digest[:2]}/{digest[2:]}.json"
    existing = head_or_none(destination, destination_bucket, key)
    if existing is None:
        destination.put_object(Bucket=destination_bucket, Key=key, Body=encoded)
    if stream_hash(destination, destination_bucket, key, len(encoded)) != digest:
        raise ValueError("remote migration manifest hash mismatch")
    return {"objects": len(objects), "bytes": sum(row["size"] for row in objects),
            "manifest": str(manifest_path), "remote_manifest_key": key,
            "manifest_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--all", action="store_true")
    choice.add_argument("--selection", type=Path)
    parser.add_argument("--source-profile", default="runpod")
    parser.add_argument("--source-endpoint", default="https://s3api-eu-ro-1.runpod.io/")
    parser.add_argument("--source-region", default="EU-RO-1")
    parser.add_argument("--destination-profile", help="omit to use current AWS environment")
    parser.add_argument("--destination-endpoint", required=True)
    parser.add_argument("--destination-region", required=True)
    parser.add_argument("--destination-bucket", required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    source = client(args.source_profile, args.source_endpoint, args.source_region)
    destination = client(args.destination_profile, args.destination_endpoint,
                         args.destination_region)
    result = transfer(args.inventory, args.selection, args.all, source, destination,
                      args.destination_bucket, args.journal, args.manifest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
