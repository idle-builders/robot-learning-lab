"""Seal a completed run's raw frames or compact media for private archiving."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path,
                        help="Run directory relative to --root")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--tar", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--label", required=True,
                        help="Archive object name prefix, no path separators")
    parser.add_argument("--media", action="store_true",
                        help="Archive JSON, JPEG and MP4 instead of raw PNG frames")
    args = parser.parse_args()
    root = args.root.resolve()
    run = (root / args.run).resolve()
    tar = args.tar.resolve()
    manifest = args.manifest.resolve()
    if not run.is_relative_to(root) or not run.is_dir() or \
            not tar.is_relative_to(root) or \
            not manifest.is_relative_to(root) or \
            "/" in args.label or not args.label:
        raise ValueError("Archive paths or label are outside the run root")
    if tar.exists() or manifest.exists():
        raise FileExistsError("Archive tar or manifest already exists")
    frames = (sorted(path for path in run.rglob("*") if path.is_file() and
                     path.suffix.lower() in (".json", ".jpg", ".jpeg", ".mp4"))
              if args.media else sorted(run.rglob("*.png")))
    if not frames:
        raise ValueError("No matching archive files")
    files = []
    with tarfile.open(tar, "w") as archive:
        for frame in frames:
            relative = frame.relative_to(root)
            files.append({"path": str(relative), "size": frame.stat().st_size,
                          "sha256": digest(frame)})
            info = archive.gettarinfo(str(frame), arcname=str(relative))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with frame.open("rb") as stream:
                archive.addfile(info, stream)
    tar_hash = digest(tar)
    record = {"schema": ("ape.run-media-archive.v1" if args.media else
                         "ape.raw-frame-archive.v1"),
        "source_run": str(run.relative_to(root)),
        "file_count": len(files), "files": files,
        "archive_bytes": tar.stat().st_size,
        "tar_sha256": tar_hash, "remote_tar_path": str(tar),
        "endpoint": "https://s3.pub2.infomaniak.cloud",
        "region": "us-east-1", "bucket": "robot-learning-lab-archive",
        "object_key": ("robot-learning-lab/selected/v1/media/" if args.media
                       else "robot-learning-lab/selected/v1/raw-frames/") +
            f"{args.label}-{tar_hash[:16]}.tar",
        "remote_verified": False}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"files": len(files), "bytes": record["archive_bytes"],
                      "sha256": tar_hash}))


if __name__ == "__main__":
    main()
