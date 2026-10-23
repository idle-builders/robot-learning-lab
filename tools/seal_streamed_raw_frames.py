"""Seal raw PNG frames for a deterministic S3 tar stream without local tar space."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--exclude-part", action="append", default=[])
    args = parser.parse_args()
    root = Path.cwd().resolve()
    run = (root / args.run).resolve()
    manifest = (root / args.manifest).resolve()
    if not run.is_relative_to(root) or not run.is_dir() or \
            not manifest.is_relative_to(root) or manifest.exists() or \
            not args.label or "/" in args.label:
        raise ValueError("Expected an existing in-repository run and new manifest")
    frames = sorted(frame for frame in run.rglob("*.png")
                    if not any(part in frame.relative_to(run).parts
                               for part in args.exclude_part))
    if not frames:
        raise ValueError("No PNG frames")
    files = [{"path": str(frame.relative_to(root)),
              "size": frame.stat().st_size, "sha256": digest(frame)}
             for frame in frames]
    process = subprocess.Popen(["tar", "--no-xattrs", "-cf", "-", "--",
                                *(row["path"] for row in files)],
                               stdout=subprocess.PIPE,
                               env={**os.environ, "COPYFILE_DISABLE": "1"})
    assert process.stdout is not None
    tar_hash = hashlib.sha256()
    tar_size = 0
    while block := process.stdout.read(1 << 20):
        tar_hash.update(block)
        tar_size += len(block)
    if process.wait() != 0:
        raise RuntimeError("Could not stream source tar")
    record = {"schema": "ape.raw-frame-archive.v1",
        "source_run": str(run.relative_to(root)),
        "file_count": len(files), "files": files,
        "archive_bytes": tar_size, "tar_sha256": tar_hash.hexdigest(),
        "endpoint": "https://s3.pub2.infomaniak.cloud",
        "region": "us-east-1", "bucket": "robot-learning-lab-archive",
        "object_key": "robot-learning-lab/selected/v1/raw-frames/" +
            f"{args.label}-{tar_hash.hexdigest()[:16]}.tar",
        "remote_verified": False}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"files": len(files), "bytes": tar_size,
                      "sha256": tar_hash.hexdigest()}))


if __name__ == "__main__":
    main()
