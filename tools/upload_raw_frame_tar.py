"""Stream a sealed local PNG selection to the existing Infomaniak S3 archive.

The manifest contains only paths, sizes and hashes. Credentials come from the
caller's environment; this script neither prints nor stores them. It does not
delete source files. It uses fixed-length multipart parts because this S3
endpoint rejects AWS CLI's chunked stdin upload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


PART_BYTES = 32 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") not in ("ape.scan20-raw-frame-archive.v1",
                                      "ape.raw-frame-archive.v1") or \
            not manifest.get("tar_sha256") or manifest.get("remote_verified"):
        raise ValueError("Expected an unuploaded sealed raw-frame manifest")
    if not manifest["object_key"].startswith(
            "robot-learning-lab/selected/v1/raw-frames/"):
        raise ValueError("Archive key outside selected raw frames")
    rows = manifest["files"]
    if len(rows) != manifest["file_count"]:
        raise ValueError("File count changed")
    for row in rows:
        path = Path(row["path"])
        if path.is_absolute() or ".." in path.parts or path.suffix != ".png":
            raise ValueError(f"Unsafe frame path: {path}")
        if path.stat().st_size != row["size"] or \
                sha256_file(path) != row["sha256"]:
            raise ValueError(f"Source frame changed: {path}")
    client = boto3.client("s3", endpoint_url=manifest["endpoint"],
        region_name=manifest["region"],
        config=Config(signature_version="s3v4",
                      s3={"addressing_style": "path"},
                      request_checksum_calculation="when_required",
                      response_checksum_validation="when_required"))
    bucket, key = manifest["bucket"], manifest["object_key"]
    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if str(error.response["Error"]["Code"]) not in \
                ("404", "NoSuchKey", "NotFound"):
            raise
    else:
        raise FileExistsError(f"Archive object already exists: {key}")
    process = subprocess.Popen(["tar", "--no-xattrs", "-cf", "-", "--",
                                *(row["path"] for row in rows)],
                               stdout=subprocess.PIPE,
                               env={**os.environ, "COPYFILE_DISABLE": "1"})
    upload = client.create_multipart_upload(Bucket=bucket, Key=key,
        Metadata={"sha256": manifest["tar_sha256"]})
    upload_id = upload["UploadId"]
    parts = []
    digest = hashlib.sha256()
    size = 0
    try:
        assert process.stdout is not None
        number = 1
        while True:
            block = process.stdout.read(PART_BYTES)
            if not block:
                break
            digest.update(block)
            size += len(block)
            result = client.upload_part(Bucket=bucket, Key=key,
                UploadId=upload_id, PartNumber=number, Body=block,
                ContentLength=len(block))
            parts.append({"ETag": result["ETag"], "PartNumber": number})
            number += 1
        if process.wait() != 0 or digest.hexdigest() != manifest["tar_sha256"]:
            raise ValueError("Tar stream differed from sealed source hash")
        client.complete_multipart_upload(Bucket=bucket, Key=key,
            UploadId=upload_id, MultipartUpload={"Parts": parts})
    except BaseException:
        process.kill()
        process.wait()
        client.abort_multipart_upload(Bucket=bucket, Key=key,
                                      UploadId=upload_id)
        raise
    remote = client.get_object(Bucket=bucket, Key=key)
    remote_hash = hashlib.sha256()
    remote_size = 0
    body = remote["Body"]
    try:
        while block := body.read(1 << 20):
            remote_hash.update(block)
            remote_size += len(block)
    finally:
        body.close()
    if remote_hash.hexdigest() != manifest["tar_sha256"] or remote_size != size:
        raise ValueError("Remote read-back differed from uploaded tar")
    manifest["archive_bytes"] = size
    manifest["remote_verified"] = True
    manifest["remote_sha256"] = remote_hash.hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"files": len(rows), "archive_bytes": size,
                      "parts": len(parts), "remote_verified": True}))


if __name__ == "__main__":
    main()
