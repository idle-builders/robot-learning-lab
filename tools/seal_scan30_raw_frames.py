"""Inventory and tar one completed Scan30 run before off-worker archiving."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/real_indoor_scenes/scan30_unified_policy_20261015"
TAR = ROOT / "scan30_raw_frames.tar"
MANIFEST = ROOT / "artifacts/ape-real-room-next-captures-20261014/scan30_raw_frames_archive_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if TAR.exists() or MANIFEST.exists():
        raise FileExistsError("Sealed archive or manifest already exists")
    frames = sorted(OUT.rglob("*.png"))
    if not frames:
        raise ValueError("No raw frames to seal")
    files = []
    with tarfile.open(TAR, "w") as archive:
        for path in frames:
            relative = path.relative_to(ROOT)
            files.append({"path": str(relative), "size": path.stat().st_size,
                          "sha256": sha256(path)})
            info = archive.gettarinfo(str(path), arcname=str(relative))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as stream:
                archive.addfile(info, stream)
    tar_hash = sha256(TAR)
    manifest = {"schema": "ape.scan30-raw-frame-archive.v1",
        "source_run": str(OUT.relative_to(ROOT)), "file_count": len(files),
        "files": files, "archive_bytes": TAR.stat().st_size,
        "tar_sha256": tar_hash, "remote_tar_path": str(TAR),
        "endpoint": "https://s3.pub2.infomaniak.cloud",
        "region": "us-east-1", "bucket": "robot-learning-lab-archive",
        "object_key": "robot-learning-lab/selected/v1/raw-frames/"
            f"scan30-png-20261015-{tar_hash[:16]}.tar",
        "remote_verified": False}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"files": len(files), "bytes": TAR.stat().st_size,
                      "sha256": tar_hash}))


if __name__ == "__main__":
    main()
