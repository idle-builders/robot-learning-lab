"""Fetch a few real photographs per MuSHRoom visit without storing ten archives.

The previews are source evidence, not rendered ape-height observations.  Full
archives and their checksums are needed before a room can enter simulation.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[3] / "assets/real_indoor_scenes/mushroom"
RECORD_URL = "https://zenodo.org/api/records/10230733"
PREVIEWS_PER_ROOM = 8


def main() -> None:
    record = requests.get(RECORD_URL, timeout=30)
    record.raise_for_status()
    files = record.json()["files"]
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {"source": RECORD_URL, "status": "recorded_photos_only", "rooms": {}}
    for item in files:
        name = item["key"]
        if not name.endswith("_iphone.tar.gz"):
            continue
        room = name.removesuffix("_iphone.tar.gz")
        out_dir = ROOT / "previews" / room
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(out_dir.glob("*.jpg"))
        saved = {p.name for p in existing}
        if len(saved) < PREVIEWS_PER_ROOM:
            url = item["links"]["self"]
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                response.raw.decode_content = False
                with tarfile.open(fileobj=response.raw, mode="r|gz") as archive:
                    for member in archive:
                        if not member.isfile() or not member.name.endswith(".jpg"):
                            continue
                        if "/iphone/long_capture/images/" not in member.name:
                            continue
                        target = out_dir / Path(member.name).name
                        if target.name in saved:
                            continue
                        source = archive.extractfile(member)
                        if source is None:
                            continue
                        content = source.read()
                        if not content.startswith(b"\xff\xd8"):
                            raise ValueError(f"Invalid JPEG in {name}: {member.name}")
                        target.write_bytes(content)
                        saved.add(target.name)
                        if len(saved) >= PREVIEWS_PER_ROOM:
                            break
        manifest["rooms"][room] = {
            "archive": name,
            "archive_bytes": item["size"],
            "archive_checksum": item["checksum"],
            "preview_frames": sorted(saved),
            "archive_downloaded": (ROOT / room / name).exists(),
            "qualification": "real_photo_preview_not_simulation",
        }
        print(room, len(saved), "recorded frames", flush=True)
        (ROOT / "preview_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
