"""File facts of manifest parts, the way the checks and builders find files: on disk, as a regenerable artifact, in the
local blob index, under a post-restructure name (infra/storage/path-moves.json), or only recorded as packed into the private
bucket (local storage records). Generic: no knowledge of any part. Used by robot/body/bom/build.py."""
from __future__ import annotations

import functools
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from infra.storage.blob_store import lookup as blob_lookup, moved  # noqa: E402
from infra.storage.regenerable_artifacts import lookup as regen_lookup  # noqa: E402


@functools.lru_cache(maxsize=1)
def _cold_records():
    """{path: {sha256: record file}} from the local records of files packed into the private bucket: the index-row dumps
    of the 2026-10-29 storage cleanup (data/storage-cleanup-*/lean/*indexed-selection.json.gz, rows with a pack key),
    the archive receipts (data/*archive-receipt*.json, rows with a pack key) and the selections of storage runs in state
    `cold` (data/storage-runs/*.json). data/ is Git-ignored; without it this is empty."""
    out = defaultdict(dict)

    def add(rows, src, need_pack):
        for e in rows if isinstance(rows, list) else []:
            if isinstance(e, dict) and e.get("path") and e.get("sha256") and (e.get("pack_key") or not need_pack):
                out[e["path"]][e["sha256"]] = src
    for f in sorted(ROOT.glob("data/storage-cleanup-*/lean/*indexed-selection.json.gz")):
        try:
            add(json.loads(gzip.open(f).read())["files"], str(f.relative_to(ROOT)), True)
        except (OSError, ValueError, KeyError):
            pass
    for f in sorted(ROOT.glob("data/*archive-receipt*.json")):
        try:
            d = json.loads(f.read_text())
            add(d.get("files") if isinstance(d, dict) else None, str(f.relative_to(ROOT)), True)
        except (OSError, ValueError):
            pass
    for f in sorted(ROOT.glob("data/storage-runs/*.json")):
        try:
            d = json.loads(f.read_text())
            if d.get("state") == "cold" and d.get("selection"):
                add(json.loads(gzip.open(ROOT / d["selection"]).read())["files"], str(f.relative_to(ROOT)), False)
        except (OSError, ValueError, KeyError, AttributeError):
            pass
    return out


def cold_record(path, sha256):
    """The local record that names exactly these bytes (path and the manifest's sha256) as packed into the private
    bucket, or None. Such a file is not on disk and not in data/blob-store.sqlite, so blob_store.ensure cannot restore it
    until the index is restored (`infra/storage/catalog_backup.py restore`)."""
    if not path or not sha256:
        return None
    return _cold_records().get(path, {}).get(sha256)


def resolve(path):
    """(status, path to read) the way the checks find a file (bodylib.src -> blob_store.ensure), without restoring."""
    if not path:
        return "none", None
    if (ROOT / path).is_file():
        return "disk", path
    if regen_lookup(path):
        return "regenerable", path
    if blob_lookup(path):
        return "blob", path
    new = moved(path)
    if new is not None:
        new = str(new)
        if (ROOT / new).is_file():
            return "moved_disk", new
        if regen_lookup(new):
            return "moved_regenerable", new
        if blob_lookup(new):
            return "moved_blob", new
    return "missing", path


def part_files(p):
    """Per manifest part: how its STEP and mesh resolve, bucket records of missing ones, and file flags
    (file_missing, file_in_bucket_not_indexed, mesh_only, no_geometry, mesh_npz_not_stl)."""
    st_status, st_path = resolve(p.get("native_step"))
    me_status, me_path = resolve(p.get("mesh"))
    st_cold = cold_record(p.get("native_step"), p.get("native_step_sha256")) if st_status == "missing" else None
    me_cold = cold_record(p.get("mesh"), p.get("mesh_sha256")) if me_status == "missing" else None
    flags = []
    if (st_status == "missing" and not st_cold) or (me_status == "missing" and not me_cold):
        flags.append("file_missing")
    if st_cold or me_cold:
        flags.append("file_in_bucket_not_indexed")
    if st_status in ("none", "missing") and me_status not in ("none", "missing"):
        flags.append("mesh_only")
    if st_status in ("none", "missing") and me_status in ("none", "missing"):
        flags.append("no_geometry")
    if (p.get("mesh") or "").endswith(".npz"):
        flags.append("mesh_npz_not_stl")
    return dict(name=p["name"], role=p.get("role"), owning_link=p.get("owning_link"), native_step=p.get("native_step"), step_status=st_status,
                step_resolved=st_path, mesh=p.get("mesh"), mesh_status=me_status, mesh_resolved=me_path, step_cold_record=st_cold,
                mesh_cold_record=me_cold, material=p.get("material"), flags=flags)
