"""Drop only hash-matched local frames after a verified archive upload."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    manifest = (root / args.manifest).resolve()
    receipt = (root / args.receipt).resolve()
    if not manifest.is_relative_to(root) or not receipt.is_relative_to(root) or \
            receipt.exists():
        raise ValueError("Manifest and new receipt must be inside the repository")
    record = json.loads(manifest.read_text())
    if record.get("schema") != "ape.raw-frame-archive.v1" or not record.get(
            "remote_verified") or record.get("remote_sha256") != record.get(
            "tar_sha256"):
        raise ValueError("Remote tar was not fully hash-verified")
    source = (root / record["source_run"]).resolve()
    files = [(root / entry["path"]).resolve() for entry in record["files"]]
    if len(files) != record["file_count"] or not all(
            path.is_relative_to(source) and path.suffix == ".png"
            for path in files):
        raise ValueError("Manifest paths leave the source run")
    for entry, path in zip(record["files"], files):
        if not path.is_file() or path.stat().st_size != entry["size"] or \
                sha(path) != entry["sha256"]:
            raise ValueError(f"Local frame changed: {path}")
    for path in files:
        path.unlink()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({
        "schema": "ape.local-raw-prune.v1",
        "manifest": str(manifest.relative_to(root)),
        "remote_object_key": record["object_key"],
        "remote_tar_sha256": record["tar_sha256"],
        "files_removed": len(files),
        "bytes_removed": sum(entry["size"] for entry in record["files"]),
        "media_and_traces_retained": True}, indent=2) + "\n")
    print(receipt.read_text())


if __name__ == "__main__":
    main()
