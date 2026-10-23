"""Stream a sealed worker archive to private S3 with AWS CLI and read it back.

Use an environment wrapper that supplies local credentials to the AWS CLI.
The worker receives no credentials and the archive never lands on local disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--ssh-port", required=True, type=int)
    parser.add_argument("--ssh-key", required=True, type=Path)
    parser.add_argument("--env-wrapper", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema") not in ("ape.raw-frame-archive.v1",
                                      "ape.run-media-archive.v1") or \
            manifest.get("remote_verified") or not manifest.get("tar_sha256"):
        raise ValueError("Expected an unuploaded sealed archive")
    prefix = ("robot-learning-lab/selected/v1/media/" if manifest["schema"] ==
              "ape.run-media-archive.v1" else
              "robot-learning-lab/selected/v1/raw-frames/")
    if not manifest["object_key"].startswith(prefix):
        raise ValueError("Archive object key outside expected prefix")
    tar_path = Path(manifest["remote_tar_path"])
    if not tar_path.is_absolute() or not str(tar_path).startswith(
            "/root/scan30-") or tar_path.suffix != ".tar" or \
            ".." in tar_path.parts:
        raise ValueError("Unexpected worker archive path")
    if not args.ssh_host.startswith("root@") or not 1 <= args.ssh_port <= 65535:
        raise ValueError("Unexpected worker address")
    if not args.env_wrapper.is_file() or not args.ssh_key.is_file():
        raise FileNotFoundError("Missing local credential wrapper or SSH key")

    dest = f"s3://{manifest['bucket']}/{manifest['object_key']}"
    aws = [str(args.env_wrapper), "env",
           "AWS_REQUEST_CHECKSUM_CALCULATION=when_required",
           "AWS_RESPONSE_CHECKSUM_VALIDATION=when_required",
           "AWS_CLI_DISABLE_PAGER=1",
           "aws", "--endpoint-url", manifest["endpoint"],
           "--region", manifest["region"]]
    head = subprocess.run(aws + ["s3api", "head-object", "--bucket",
        manifest["bucket"], "--key", manifest["object_key"],
        "--query", "ContentLength", "--output", "text"],
        capture_output=True, text=True)
    if head.returncode == 0:
        raise FileExistsError(dest)
    if "404" not in head.stderr and "Not Found" not in head.stderr:
        raise RuntimeError("Could not verify the destination is absent: " +
                           head.stderr[-300:])

    worker = subprocess.Popen(["ssh", "-i", str(args.ssh_key), "-p",
        str(args.ssh_port), args.ssh_host,
        "cat -- " + shlex.quote(str(tar_path))], stdout=subprocess.PIPE)
    upload = subprocess.Popen(aws + ["s3", "cp", "-", dest,
        "--expected-size", str(manifest["archive_bytes"]),
        "--metadata", "sha256=" + manifest["tar_sha256"],
        "--no-progress"], stdin=subprocess.PIPE)
    digest = hashlib.sha256()
    size = 0
    try:
        assert worker.stdout is not None and upload.stdin is not None
        while block := worker.stdout.read(4 << 20):
            digest.update(block)
            size += len(block)
            upload.stdin.write(block)
        upload.stdin.close()
        worker_code = worker.wait()
        upload_code = upload.wait()
        if worker_code != 0 or upload_code != 0:
            raise RuntimeError(f"Worker stream or S3 upload failed: "
                               f"worker={worker_code}, upload={upload_code}, "
                               f"stream_bytes={size}")
    finally:
        if worker.poll() is None:
            worker.kill(); worker.wait()
        if upload.poll() is None:
            upload.kill(); upload.wait()
    if size != manifest["archive_bytes"] or digest.hexdigest() != manifest[
            "tar_sha256"]:
        raise ValueError("Worker archive stream changed from sealed manifest")

    restored = subprocess.Popen(aws + ["s3", "cp", dest, "-", "--quiet"],
                                stdout=subprocess.PIPE)
    read_hash = hashlib.sha256()
    read_size = 0
    assert restored.stdout is not None
    while block := restored.stdout.read(4 << 20):
        read_hash.update(block)
        read_size += len(block)
    if restored.wait() != 0 or read_size != manifest["archive_bytes"] or \
            read_hash.hexdigest() != manifest["tar_sha256"]:
        raise ValueError("S3 read-back differs from sealed worker archive")
    manifest["remote_verified"] = True
    manifest["remote_sha256"] = read_hash.hexdigest()
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"files": manifest["file_count"],
        "archive_bytes": read_size, "remote_verified": True}))


if __name__ == "__main__":
    main()
