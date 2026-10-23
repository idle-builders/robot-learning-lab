"""Retain the ten MuSHRoom iPhone COLMAP pose listings, not the large archive."""

from __future__ import annotations

import hashlib
import json
import tarfile
import tempfile
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[3] / "assets/real_indoor_scenes/mushroom"
RECORD = "https://zenodo.org/api/records/13986996"


def main() -> None:
    record = requests.get(RECORD, timeout=30).json()
    item = next(f for f in record["files"] if f["key"] == "room_datasets_iphone_colmap.tar")
    directory = ROOT / "colmap_poses"
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5()
    with tempfile.TemporaryDirectory() as temp:
        archive_path = Path(temp) / item["key"]
        with requests.get(item["links"]["self"], stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with archive_path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        digest.update(chunk)
                        output.write(chunk)
        actual = f"md5:{digest.hexdigest()}"
        if actual != item["checksum"]:
            raise ValueError(f"COLMAP archive checksum mismatch: {actual} != {item['checksum']}")
        rooms = {}
        with tarfile.open(archive_path) as archive:
            for member in archive:
                parts = Path(member.name).parts
                if not member.isfile() or len(parts) != 8 or parts[-1] != "images.txt":
                    continue
                room = parts[1]
                source = archive.extractfile(member)
                if source is None:
                    continue
                target = directory / f"{room}_images.txt"
                target.write_bytes(source.read())
                rooms[room] = str(target)
        if len(rooms) != 10:
            raise ValueError(f"Expected ten pose files, found {sorted(rooms)}")
    (directory / "source.json").write_text(json.dumps({"record": RECORD, "archive": item["key"],
                                                       "checksum": item["checksum"], "rooms": rooms}, indent=2) + "\n")
    print(f"Verified {actual}; retained {len(rooms)} pose listings")


if __name__ == "__main__":
    main()
