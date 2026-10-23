"""Fetch one open MuSHRoom real-room capture with verified Zenodo provenance.

Download only the iPhone capture unless reference geometry is requested. The
archive remains unexpanded until its camera coverage and room usefulness are
screened, since the local POC disk is limited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "assets/real_indoor_scenes/mushroom"
RECORDS = {"iphone": 10230733, "reference_mesh": 10222321}


def fetch(room: str, include_reference_mesh: bool) -> dict:
    destination = ASSETS / room
    destination.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    records = ["iphone", "reference_mesh"] if include_reference_mesh else ["iphone"]
    files = []
    for kind in records:
        record_id = RECORDS[kind]
        record_response = session.get(f"https://zenodo.org/api/records/{record_id}",
                                      timeout=30)
        record_response.raise_for_status()
        record = record_response.json()
        name = f"{room}_{'iphone' if kind == 'iphone' else 'mesh_pd'}.tar.gz"
        source = next((row for row in record["files"] if row["key"] == name), None)
        if source is None:
            raise ValueError(f"Room {room!r} is not in MuSHRoom {kind} record")
        path = destination / name
        digest = source["checksum"].removeprefix("md5:")
        if path.exists():
            actual = hashlib.md5(path.read_bytes()).hexdigest()
            if actual != digest:
                raise ValueError(f"Existing {path} has unexpected MD5 {actual}")
        else:
            pending = path.with_suffix(path.suffix + ".partial")
            md5 = hashlib.md5()
            total = 0
            with session.get(source["links"]["self"], stream=True,
                             timeout=(20, 90)) as response:
                response.raise_for_status()
                with pending.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        if not chunk:
                            continue
                        stream.write(chunk)
                        md5.update(chunk)
                        total += len(chunk)
            if total != source["size"] or md5.hexdigest() != digest:
                pending.unlink(missing_ok=True)
                raise ValueError(f"Incomplete or corrupt download of {name}")
            pending.replace(path)
        files.append({"kind": kind, "filename": name, "bytes": source["size"],
                      "md5": digest, "zenodo_record_id": record_id,
                      "doi": record["metadata"].get("doi"),
                      "license": record["metadata"].get("license", {}).get("id"),
                      "source_url": source["links"]["self"]})
        print(f"Verified {name}: {source['size'] / 1e6:.1f} MB", flush=True)
    manifest = {"schema": "ape.mushroom-room-source.v1", "room": room,
                "files": files, "qualification": "capture_only_not_ape_height_simulation"}
    (destination / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", required=True)
    parser.add_argument("--include-reference-mesh", action="store_true")
    args = parser.parse_args()
    print(json.dumps(fetch(args.room, args.include_reference_mesh), indent=2))
