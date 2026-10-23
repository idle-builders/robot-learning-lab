"""Restore one or all archived ape run files with per-file SHA-256 checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile

import boto3
from botocore.config import Config


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--member", help="restore just one manifest path")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema") not in (
            "ape.scan20-raw-frame-archive.v1",
            "ape.scan30-raw-frame-archive.v1",
            "ape.raw-frame-archive.v1",
            "ape.run-media-archive.v1") or \
            not manifest.get("remote_verified") or manifest.get("remote_sha256") != \
            manifest.get("tar_sha256"):
        raise ValueError("Archive is not sealed and remotely verified")
    entries = {row["path"]: row for row in manifest["files"]}
    if len(entries) != manifest["file_count"]:
        raise ValueError("Duplicate or missing manifest paths")
    wanted = {args.member} if args.member else set(entries)
    if not wanted.issubset(entries):
        raise ValueError("Requested member is absent from manifest")
    root = args.destination.resolve()
    allowed_suffixes = ({".mp4", ".jpg", ".jpeg", ".json"} if
                        manifest["schema"] == "ape.run-media-archive.v1" else
                        {".png"})
    root.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", endpoint_url=manifest["endpoint"],
        region_name=manifest["region"],
        config=Config(signature_version="s3v4",
                      s3={"addressing_style": "path"},
                      request_checksum_calculation="when_required",
                      response_checksum_validation="when_required"))
    response = s3.get_object(Bucket=manifest["bucket"],
                             Key=manifest["object_key"])
    restored = set()
    body = response["Body"]
    try:
        with tarfile.open(fileobj=body, mode="r|") as archive:
            for item in archive:
                if item.name not in entries or not item.isfile():
                    raise ValueError(f"Unexpected archive member: {item.name}")
                expected = entries[item.name]
                if item.size != expected["size"]:
                    raise ValueError(f"Member size changed: {item.name}")
                if item.name not in wanted:
                    continue
                relative = Path(item.name)
                if relative.is_absolute() or ".." in relative.parts or \
                        relative.suffix.lower() not in allowed_suffixes:
                    raise ValueError(f"Unsafe member: {item.name}")
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.parent.resolve().is_relative_to(root):
                    raise ValueError(f"Path escaped destination: {item.name}")
                if target.exists():
                    if not target.is_file() or target.stat().st_size != item.size or \
                            digest_file(target) != expected["sha256"]:
                        raise ValueError(f"Existing file differs: {target}")
                else:
                    source = archive.extractfile(item)
                    if source is None:
                        raise ValueError(f"Could not read member: {item.name}")
                    digest = hashlib.sha256()
                    total = 0
                    with tempfile.NamedTemporaryFile(dir=target.parent,
                            prefix=".restore-", delete=False) as temporary:
                        partial = Path(temporary.name)
                        try:
                            while block := source.read(1 << 20):
                                digest.update(block)
                                total += len(block)
                                temporary.write(block)
                        except BaseException:
                            partial.unlink(missing_ok=True)
                            raise
                    if total != item.size or digest.hexdigest() != \
                            expected["sha256"]:
                        partial.unlink(missing_ok=True)
                        raise ValueError(f"Restored hash mismatch: {item.name}")
                    os.replace(partial, target)
                restored.add(item.name)
                if restored == wanted:
                    break
    finally:
        body.close()
    if restored != wanted:
        raise ValueError(f"Archive omitted {len(wanted - restored)} requested files")
    print(json.dumps({"restored_or_verified": len(restored),
                      "destination": str(root)}))


if __name__ == "__main__":
    main()
