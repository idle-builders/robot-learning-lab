"""Copy a sealed raw-frame tar from an SSH compute worker to private S3.

The worker never receives archive credentials. The local process streams fixed
multipart parts, then reads the S3 object back and verifies its full SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


PART_BYTES = 32 * 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--ssh-port", required=True, type=int)
    parser.add_argument("--ssh-key", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema") not in (
            "ape.scan30-raw-frame-archive.v1", "ape.raw-frame-archive.v1",
            "ape.run-media-archive.v1") or \
            not manifest.get("tar_sha256") or manifest.get("remote_verified"):
        raise ValueError("Expected an unuploaded sealed raw-frame manifest")
    prefix = ("robot-learning-lab/selected/v1/media/" if manifest["schema"] ==
              "ape.run-media-archive.v1" else
              "robot-learning-lab/selected/v1/raw-frames/")
    if not manifest["object_key"].startswith(prefix):
        raise ValueError("Archive key outside expected prefix")
    tar_path = Path(manifest["remote_tar_path"])
    if not tar_path.is_absolute() or not str(tar_path).startswith(
            "/root/scan30-") or tar_path.suffix != ".tar" or \
            ".." in tar_path.parts:
        raise ValueError("Unexpected worker tar location")
    if not args.ssh_host.startswith("root@") or not 1 <= args.ssh_port <= 65535:
        raise ValueError("Unexpected task worker")
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
    process = subprocess.Popen(["ssh", "-i", str(args.ssh_key),
        "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-p", str(args.ssh_port), args.ssh_host,
        "cat -- " + shlex.quote(manifest["remote_tar_path"])],
        stdout=subprocess.PIPE)
    upload = client.create_multipart_upload(Bucket=bucket, Key=key,
        Metadata={"sha256": manifest["tar_sha256"]})
    upload_id = upload["UploadId"]
    parts = []
    digest = hashlib.sha256()
    size = 0
    try:
        assert process.stdout is not None
        number = 1
        while block := process.stdout.read(PART_BYTES):
            digest.update(block)
            size += len(block)
            result = client.upload_part(Bucket=bucket, Key=key,
                UploadId=upload_id, PartNumber=number, Body=block,
                ContentLength=len(block))
            parts.append({"ETag": result["ETag"], "PartNumber": number})
            number += 1
        if process.wait() != 0 or size != manifest["archive_bytes"] or \
                digest.hexdigest() != manifest["tar_sha256"]:
            raise ValueError("SSH tar stream differs from sealed worker tar")
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
    if remote_hash.hexdigest() != manifest["tar_sha256"] or \
            remote_size != manifest["archive_bytes"]:
        raise ValueError("S3 read-back differs from sealed worker tar")
    manifest["remote_verified"] = True
    manifest["remote_sha256"] = remote_hash.hexdigest()
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"files": manifest["file_count"],
        "archive_bytes": remote_size, "parts": len(parts),
        "remote_verified": True}))


if __name__ == "__main__":
    main()
