"""Remove migrated local video copies after exact bucket and reference checks.

Usage: python tools/prune_result_videos.py plan OUTPUT.json
       python tools/prune_result_videos.py apply OUTPUT.json
Only video rows in the results table are eligible. Native reports and traces are untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "results.sqlite"
SEARCH = ("docs", "experiments", "tools", "robot", "skills", "studies", "README.md", "AGENTS.md")
TEXT = {".md", ".json", ".py", ".js", ".html", ".csv", ".txt"}


def direct_references(paths: set[str]) -> set[str]:
    references = set()
    pattern = re.compile(r"artifacts/[^\s\"<>)]*?\.mp4")
    for name in SEARCH:
        start = ROOT / name
        for file in ([start] if start.is_file() else start.rglob("*")):
            if file.is_file() and file.suffix in TEXT and file.stat().st_size < 5_000_000:
                try:
                    references.update(set(pattern.findall(file.read_text(errors="ignore"))) & paths)
                except OSError:
                    pass
    return references


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plan(output: Path):
    if output.exists():
        raise FileExistsError(output)
    with sqlite3.connect(DB) as db:
        rows = db.execute("""SELECT source_path, MIN(sha256), MIN(bytes),
            COUNT(*), SUM(verified_at IS NOT NULL), COUNT(DISTINCT sha256)
            FROM clips GROUP BY source_path""").fetchall()
    paths = {row[0] for row in rows}
    refs = direct_references(paths)
    eligible = []
    held = {"direct_reference": [], "not_verified": [], "changed_or_missing": []}
    for name, digest, size, uses, verified, distinct_hashes in rows:
        path = ROOT / name
        if name in refs:
            held["direct_reference"].append(name)
        elif verified != uses or distinct_hashes != 1:
            held["not_verified"].append(name)
        elif not path.is_file() or path.is_symlink() or path.stat().st_size != size or sha256(path) != digest:
            held["changed_or_missing"].append(name)
        else:
            eligible.append({"path": name, "sha256": digest, "bytes": size})
    manifest = {"schema": "robot-learning-lab.results-video-prune.v1", "eligible": eligible, "held": held}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Planned {len(eligible)} files, {sum(row['bytes'] for row in eligible):,} bytes; held {sum(map(len, held.values()))}")


def apply(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "robot-learning-lab.results-video-prune.v1":
        raise ValueError("Wrong prune manifest")
    candidates = manifest["eligible"]
    names = {row["path"] for row in candidates}
    if len(names) != len(candidates) or direct_references(names):
        raise ValueError("A selected video is now referenced or listed twice")
    with sqlite3.connect(DB) as db:
        for row in candidates:
            name = row["path"]
            if not name.startswith("artifacts/") or ".." in Path(name).parts:
                raise ValueError(f"Unsafe path: {name}")
            records = db.execute("SELECT sha256,bytes,verified_at FROM clips WHERE source_path=?", (name,)).fetchall()
            if not records or any(digest != row["sha256"] or size != row["bytes"] or not verified for digest, size, verified in records):
                raise ValueError(f"No longer verified: {name}")
            path = ROOT / name
            if not path.is_file() or path.is_symlink() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
                raise ValueError(f"Changed or missing: {name}")
        for row in candidates:
            (ROOT / row["path"]).unlink()
    print(f"Removed {len(candidates)} verified local video copies, {sum(row['bytes'] for row in candidates):,} logical bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "apply"))
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    (plan if args.command == "plan" else apply)(args.manifest)
