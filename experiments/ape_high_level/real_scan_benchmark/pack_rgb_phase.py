"""Losslessly pack redundant RGB PNGs after checking a saved RGB-D episode."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import cv2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def video_frames(path: Path) -> int:
    video = cv2.VideoCapture(str(path))
    count = 0
    while video.read()[0]:
        count += 1
    video.release()
    return count


def pack(folder: Path, *, prefix: str = "rgb", video_name: str = "first_person.mp4",
         depth_prefix: str = "depth") -> dict:
    trace = json.loads((folder / "trace.json").read_text())
    frames = trace["frames"]
    if video_frames(folder / video_name) != frames:
        raise ValueError(f"RGB video and trace differ: {folder}")
    if len(list(folder.glob(f"{depth_prefix}_*.png"))) != frames:
        raise ValueError(f"Depth PNGs and trace differ: {folder}")
    rgb = sorted(folder.glob(f"{prefix}_*.png"))
    archive = folder / f"{prefix}_frames.tar.gz"
    manifest = folder / f"{prefix}_frames_manifest.json"
    if not rgb and archive.is_file() and manifest.is_file():
        record = json.loads(manifest.read_text())
        if record["frame_count"] != frames or record["archive_sha256"] != sha256(archive):
            raise ValueError(f"Packed RGB manifest mismatch: {folder}")
        return record
    if len(rgb) != frames or [p.name for p in rgb] != [f"{prefix}_{i:03d}.png" for i in range(frames)]:
        raise ValueError(f"RGB PNG sequence and trace differ: {folder}")
    hashes = {path.name: sha256(path) for path in rgb}
    temporary = folder / f"{prefix}_frames.tar.gz.tmp"
    with tarfile.open(temporary, "w:gz", compresslevel=9) as tar:
        for path in rgb:
            tar.add(path, arcname=path.name, recursive=False)
    with tarfile.open(temporary, "r:gz") as tar:
        members = tar.getmembers()
        if [member.name for member in members] != list(hashes):
            raise ValueError(f"Packed RGB sequence differs: {folder}")
        for member in members:
            stream = tar.extractfile(member)
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != hashes[member.name]:
                raise ValueError(f"Packed RGB frame differs: {folder}/{member.name}")
    temporary.replace(archive)
    record = {"schema": "ape.lossless-rgb-phase-pack.v1", "frame_count": frames,
              "archive": archive.name, "archive_sha256": sha256(archive),
              "frame_sha256": hashes, "video": video_name,
              "depth_pattern": f"{depth_prefix}_*.png"}
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    for path in rgb:
        path.unlink()
    return record


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    result = pack(args.folder)
    print(json.dumps({"frame_count": result["frame_count"],
                      "archive": str(args.folder / result["archive"]),
                      "archive_sha256": result["archive_sha256"]}, indent=2))
