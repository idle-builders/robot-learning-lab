"""Plan, upload and restore selected local artifacts through an S3-compatible store.

Use the existing Infomaniak Keychain environment for the destination. Credentials never enter a manifest.
This tool copies only explicitly selected paths; it never deletes local or remote data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile


SCHEMA = "robot-learning-lab.selected-archive.v1"
PREFIX = "robot-learning-lab/selected/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(name: str) -> Path:
    posix = PurePosixPath(name)
    if not posix.parts or posix.is_absolute() or ".." in posix.parts or posix.as_posix() != name:
        raise ValueError(f"Unsafe archive path: {name!r}")
    return Path(*posix.parts)


def selected_files(root: Path, includes: list[str]) -> list[tuple[str, Path]]:
    root = root.resolve()
    files: dict[str, Path] = {}
    for name in includes:
        relative = safe_relative(name)
        path = root / relative
        if path.is_dir():
            if path.is_symlink():
                raise ValueError(f"Directory symlink is not supported: {name}")
            for folder, dirs, names in os.walk(path, followlinks=False):
                if any((Path(folder) / child).is_symlink() for child in dirs):
                    raise ValueError(f"Directory symlink in {folder}")
                for child in names:
                    item = Path(folder) / child
                    logical = item.relative_to(root).as_posix()
                    files[logical] = item
        else:
            files[name] = path
    result = []
    for name, path in sorted(files.items()):
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Missing or out-of-root regular file: {name}")
        result.append((name, path))
    return result


def make_plan(root: Path, includes: list[str], output: Path) -> dict:
    if not includes:
        raise ValueError("At least one --include is required")
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for name, path in selected_files(root, includes):
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"File changed during inventory: {name}")
        rows.append({"path": name, "size": after.st_size, "sha256": digest})
    manifest = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def read_manifest(path: Path) -> tuple[dict, bytes]:
    encoded = path.read_bytes()
    manifest = json.loads(encoded)
    if manifest.get("schema") != SCHEMA or not isinstance(manifest.get("files"), list):
        raise ValueError("Unsupported archive manifest")
    seen = set()
    for row in manifest["files"]:
        name = row["path"]
        safe_relative(name)
        if name in seen or type(row["size"]) is not int or row["size"] < 0:
            raise ValueError(f"Invalid archive row: {name}")
        digest = row["sha256"]
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError(f"Invalid SHA-256: {name}")
        seen.add(name)
    return manifest, encoded


def object_key(digest: str) -> str:
    return f"{PREFIX}/objects/sha256/{digest[:2]}/{digest[2:]}"


def client(profile: str | None, endpoint: str, region: str):
    if profile is None and endpoint.rstrip("/") == "https://s3.pub2.infomaniak.cloud" and region == "us-east-1":
        from lab.blob_store import client as private_client
        return private_client()
    import boto3
    from botocore.config import Config

    if not endpoint.startswith("https://"):
        raise ValueError("S3 endpoint must use HTTPS")
    return boto3.Session(profile_name=profile).client(
        "s3", endpoint_url=endpoint, region_name=region,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                      request_checksum_calculation="when_required",
                      response_checksum_validation="when_required",
                      connect_timeout=10, read_timeout=120, retries={"max_attempts": 3}),
    )


def remote_hash(s3, bucket: str, key: str, expected_size: int) -> str:
    response = s3.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    digest = hashlib.sha256()
    size = 0
    try:
        while block := body.read(1024 * 1024):
            size += len(block)
            if size > expected_size:
                raise ValueError(f"Remote object exceeds expected size: {key}")
            digest.update(block)
    finally:
        body.close()
    if size != expected_size:
        raise ValueError(f"Remote object size mismatch: {key}")
    return digest.hexdigest()


def upload(root: Path, manifest_path: Path, s3, bucket: str) -> dict:
    root = root.resolve()
    manifest, encoded = read_manifest(manifest_path)
    verified: set[str] = set()
    total = 0
    for row in manifest["files"]:
        path = root / safe_relative(row["path"])
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Source missing or escaped root: {row['path']}")
        if path.stat().st_size != row["size"] or sha256_file(path) != row["sha256"]:
            raise ValueError(f"Source changed since plan: {row['path']}")
        digest = row["sha256"]
        if digest in verified:
            continue
        key = object_key(digest)
        exists = False
        try:
            head = s3.head_object(Bucket=bucket, Key=key)
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if str(code) not in {"404", "NoSuchKey", "NotFound"}:
                raise
        else:
            if head["ContentLength"] != row["size"]:
                raise ValueError(f"Existing object has wrong size: {key}")
            exists = True
        if not exists:
            s3.upload_file(str(path), bucket, key, ExtraArgs={"Metadata": {"sha256": digest}})
        if remote_hash(s3, bucket, key, row["size"]) != digest:
            raise ValueError(f"Remote SHA-256 mismatch: {key}")
        verified.add(digest)
        total += row["size"]
        print(f"verified {row['path']} ({row['size']} bytes)", flush=True)
    manifest_digest = hashlib.sha256(encoded).hexdigest()
    manifest_key = f"{PREFIX}/manifests/sha256/{manifest_digest[:2]}/{manifest_digest[2:]}.json"
    s3.put_object(Bucket=bucket, Key=manifest_key, Body=encoded,
                  Metadata={"sha256": manifest_digest})
    if remote_hash(s3, bucket, manifest_key, len(encoded)) != manifest_digest:
        raise ValueError("Remote manifest SHA-256 mismatch")
    return {"manifest_key": manifest_key, "manifest_sha256": manifest_digest,
            "files": len(manifest["files"]), "distinct_objects": len(verified),
            "distinct_bytes": total}


def restore(manifest_path: Path, destination: Path, s3, bucket: str) -> dict:
    manifest, _ = read_manifest(manifest_path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=destination.name + ".partial-",
                                     dir=destination.parent) as scratch:
        scratch_path = Path(scratch)
        for row in manifest["files"]:
            target = scratch_path / safe_relative(row["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size = 0
            response = s3.get_object(Bucket=bucket, Key=object_key(row["sha256"]))
            body = response["Body"]
            try:
                with target.open("xb") as stream:
                    while block := body.read(1024 * 1024):
                        size += len(block)
                        if size > row["size"]:
                            raise ValueError(f"Remote object too large: {row['path']}")
                        digest.update(block)
                        stream.write(block)
            finally:
                body.close()
            if size != row["size"] or digest.hexdigest() != row["sha256"]:
                raise ValueError(f"Restored file failed SHA-256: {row['path']}")
        scratch_path.rename(destination)
    return {"files": len(manifest["files"]),
            "bytes": sum(row["size"] for row in manifest["files"]),
            "destination": str(destination)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--root", type=Path, required=True)
    plan.add_argument("--include", action="append", required=True)
    plan.add_argument("--output", type=Path, required=True)
    for command in ("upload", "restore"):
        part = sub.add_parser(command)
        part.add_argument("--manifest", type=Path, required=True)
        part.add_argument("--profile", help="omit to use current AWS environment")
        part.add_argument("--endpoint", required=True)
        part.add_argument("--region", required=True)
        part.add_argument("--bucket", required=True)
        if command == "upload":
            part.add_argument("--root", type=Path, required=True)
        else:
            part.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        result = make_plan(args.root.resolve(), args.include, args.output)
        print(json.dumps({"files": len(result["files"]),
                          "bytes": sum(row["size"] for row in result["files"]),
                          "manifest": str(args.output)}, indent=2))
    else:
        s3 = client(args.profile, args.endpoint, args.region)
        result = (upload(args.root.resolve(), args.manifest, s3, args.bucket)
                  if args.command == "upload" else
                  restore(args.manifest, args.destination, s3, args.bucket))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
