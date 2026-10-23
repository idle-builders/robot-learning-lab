"""Fetch a licensed ScanSceneVerse GLB from its previously verified tar sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[3]
GLBS = ROOT / "assets/real_indoor_scenes/scansceneverse/glbs"
CHUNK = 4 * 1024 * 1024


def read_range(session: requests.Session, url: str, first: int,
               last: int) -> bytes:
    response = session.get(url, headers={"Range": f"bytes={first}-{last}"},
                           timeout=(15, 120))
    response.raise_for_status()
    if response.status_code != 206 or len(response.content) != last - first + 1:
        raise IOError(f"Source did not return exact byte range {first}-{last}")
    return response.content


def fetch(scene_id: str) -> Path:
    sidecar = GLBS / f"{scene_id}.source.json"
    source = json.loads(sidecar.read_text())
    if source["scene_id"] != scene_id or source["license"] not in (
            "by", "by-sa", "by-nc", "by-nc-sa", "cc0"):
        raise ValueError("Scene identity or license is not approved")
    output = GLBS / f"{scene_id}.glb"
    if output.is_file():
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        if digest != source["sha256"]:
            raise IOError(f"Existing GLB hash differs from sidecar: {scene_id}")
        return output
    url = source["dataset_url"]
    offset = int(source["tar_header_offset"])
    size = int(source["size_bytes"])
    with requests.Session() as session:
        header = read_range(session, url, offset, offset + 511)
        name = header[:100].split(b"\0")[0].decode()
        tar_size = int(header[124:136].strip(b"\0 "), 8)
        if header[257:262] != b"ustar" or name != source["tar_member"] or \
                tar_size != size:
            raise IOError(f"Tar header does not match source sidecar: {scene_id}")
        partial = output.with_suffix(".glb.part")
        written = partial.stat().st_size if partial.exists() else 0
        if written > size:
            raise IOError("Partial GLB exceeds expected source size")
        with partial.open("ab") as stream:
            while written < size:
                first = offset + 512 + written
                last = first + min(CHUNK, size - written) - 1
                block = read_range(session, url, first, last)
                stream.write(block)
                written += len(block)
                print(scene_id, written, "/", size, flush=True)
    digest = hashlib.sha256(partial.read_bytes()).hexdigest()
    if digest != source["sha256"]:
        raise IOError(f"Downloaded GLB hash differs from sidecar: {scene_id}")
    partial.replace(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_id")
    args = parser.parse_args()
    print(fetch(args.scene_id))


if __name__ == "__main__":
    main()
