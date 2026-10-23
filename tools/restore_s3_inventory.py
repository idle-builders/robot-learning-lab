"""Restore exact files from a verified Runpod-to-Infomaniak migration manifest.

Use the existing Infomaniak Keychain environment. A fresh destination is required;
each restored file is checked against the manifest's full SHA-256 before publish.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile

from transfer_s3_inventory import CHUNK, client, destination_key, read_range, safe_key


def restore(manifest_path: Path, destination: Path, source_keys: list[str] | None,
            s3) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "robot-learning-lab.runpod-s3-migration.v1":
        raise ValueError("unsupported migration manifest")
    bucket = manifest["destination_bucket"]
    volume = safe_key(manifest["source_volume_id"])
    objects = manifest["objects"]
    by_key = {row["source_key"]: row for row in objects}
    if len(by_key) != len(objects):
        raise ValueError("repeated source key")
    keys = source_keys if source_keys is not None else list(by_key)
    if not keys or len(set(keys)) != len(keys) or any(k not in by_key for k in keys):
        raise ValueError("empty, repeated or unknown restore key")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=destination.name + ".partial-",
                                     dir=destination.parent) as scratch:
        scratch_path = Path(scratch)
        total = 0
        for key in keys:
            row = by_key[key]
            safe_key(key)
            remote_key = safe_key(row["destination_key"])
            size = row["size"]
            expected = row["sha256"]
            if (type(size) is not int or size < 0 or
                    not isinstance(expected, str) or
                    not re.fullmatch(r"[0-9a-f]{64}", expected) or
                    row["source_volume_id"] != volume or
                    remote_key != destination_key(volume, key) or
                    row["destination_bucket"] != bucket):
                raise ValueError(f"invalid restore row: {key}")
            target = scratch_path / volume / key
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with target.open("xb") as stream:
                for start in range(0, size, CHUNK):
                    data = read_range(s3, bucket, remote_key, start,
                                      min(CHUNK, size - start))
                    stream.write(data)
                    digest.update(data)
            if size == 0:
                head = s3.head_object(Bucket=bucket, Key=remote_key)
                if head["ContentLength"] != 0:
                    raise ValueError(f"empty source has content: {key}")
            if target.stat().st_size != size or digest.hexdigest() != expected:
                raise ValueError(f"restored file differs from manifest: {key}")
            total += size
            print(f"restored {key} ({size} bytes)", flush=True)
        scratch_path.rename(destination)
    return {"files": len(keys), "bytes": total, "destination": str(destination)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--source-key", action="append")
    choice.add_argument("--all", action="store_true")
    parser.add_argument("--profile", help="omit to use current AWS environment")
    parser.add_argument("--endpoint", default="https://s3.pub2.infomaniak.cloud")
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()
    s3 = client(args.profile, args.endpoint, args.region)
    result = restore(args.manifest, args.destination, args.source_key, s3)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
