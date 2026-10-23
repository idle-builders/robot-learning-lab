"""Inventory and export an explicit public source selection without Git history.

plan writes an exact hash inventory; export checks those bytes and scans the new
directory with Gitleaks. Neither command pushes, uploads, deletes or stages files.
"""

from __future__ import annotations

import argparse
from collections import Counter
from fnmatch import fnmatchcase
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "publication/public-source-policy.json"
PRIVATE_ROOTS = {"data", "artifacts", ".artifact-store", ".artifact-incoming", ".cache", ".tools", ".codex", ".claude", "studies", "output"}
PRIVATE_SUFFIXES = {".pt", ".pth", ".onnx", ".sqlite", ".db", ".npz", ".npy", ".mp4", ".webm", ".step", ".stp", ".stl", ".glb", ".zip", ".gz", ".pem", ".key", ".log", ".patch", ".bundle"}


def relative(name):
    p = PurePosixPath(name)
    if not p.parts or p.is_absolute() or ".." in p.parts or p.as_posix() != name:
        raise ValueError("Unsafe source inventory path")
    return p


def allowed(name, policy):
    p = relative(name)
    if p.parts[0] in PRIVATE_ROOTS or p.suffix.lower() in PRIVATE_SUFFIXES:
        return False
    if any(part in policy["exclude_parts"] or part.startswith(".venv") for part in p.parts):
        return False
    if p.name.startswith(".env") and not p.name.endswith(".example"):
        return False
    if any(fnmatchcase(name, pattern) for pattern in policy["exclude_patterns"]):
        return False
    if name not in policy["exact_assets"] and p.suffix not in policy["extensions"] and p.name not in {".gitignore", ".env.example", "lab", "runpod-flash", "pre-push", "CMakeLists.txt"}:
        return False
    return any(fnmatchcase(name, pattern) for pattern in policy["include"])


def read_source(name):
    relative(name)
    path = ROOT / name
    # Refuse every symlink component, including links into the private CAS.
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError(f"Source is missing or contains a symlink: {name}")
    return path.read_bytes(), path.stat().st_mode & 0o777


def fingerprint(raw):
    return hashlib.sha256(raw).hexdigest()


def plan(output):
    policy = json.loads(POLICY.read_text())
    names = set(subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
    ).decode().rstrip("\0").split("\0"))
    rows, excluded = [], Counter()
    mapping = {name: name for name in names if name and allowed(name, policy)}
    mapping.update(policy["overlays"])
    for destination, source in sorted(mapping.items()):
        path = ROOT / source
        if not path.exists() or path.is_symlink() or any(p.is_symlink() for p in path.parents):
            excluded["missing_or_symlink"] += 1
            continue
        # The body map is a deliberately selected UI asset, not a research image.
        if path.stat().st_size > policy["max_file_bytes"] and source not in policy["exact_assets"]:
            excluded["oversized"] += 1
            continue
        raw, mode = read_source(source)
        if b"\0" in raw[:8192] and source not in policy["exact_assets"]:
            excluded["binary"] += 1
            continue
        rows.append({"path": destination, "source": source, "sha256": fingerprint(raw), "bytes": len(raw), "mode": mode})
    inventory = {"schema": "robot-learning-lab.public-source-inventory.v1", "policy_sha256": fingerprint(POLICY.read_bytes()), "files": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(inventory, f, indent=2)
        f.write("\n")
    print(json.dumps({"files": len(rows), "bytes": sum(r["bytes"] for r in rows), "groups": dict(Counter(r["path"].split('/')[0] for r in rows)), "omitted_candidates": dict(excluded), "inventory": str(output)}))


def export(inventory_path, destination):
    inventory = json.loads(inventory_path.read_text())
    policy = json.loads(POLICY.read_text())
    if inventory.get("schema") != "robot-learning-lab.public-source-inventory.v1" or inventory["policy_sha256"] != fingerprint(POLICY.read_bytes()):
        raise ValueError("Inventory schema or current policy differs; make a new plan")
    if not shutil.which("gitleaks"):
        raise RuntimeError("Install Gitleaks before exporting public source")
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Export destination must be new")
    seen = set()
    # Verify the entire frozen selection before creating the export.
    for row in inventory["files"]:
        name, source = row["path"], row["source"]
        relative(name)
        if name in seen or not (allowed(name, policy) or policy["overlays"].get(name) == source):
            raise ValueError("Duplicate or unapproved destination in inventory")
        if source != name and policy["overlays"].get(name) != source:
            raise ValueError("Unapproved source overlay")
        raw, _ = read_source(source)
        if fingerprint(raw) != row["sha256"] or len(raw) != row["bytes"]:
            raise ValueError(f"Source changed after inventory: {source}")
        seen.add(name)
    destination.mkdir(parents=True)
    for row in inventory["files"]:
        raw, _ = read_source(row["source"])
        if fingerprint(raw) != row["sha256"]:
            raise ValueError("Source changed during export; keep this export private")
        target = destination / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        target.chmod(0o755 if row["mode"] & 0o111 else 0o644)
    # Reports stay beside the private inventory, never inside the public tree.
    report = inventory_path.with_name(inventory_path.stem + "-gitleaks.json")
    result = subprocess.run(["gitleaks", "dir", "--redact=100", "--no-banner", "--max-archive-depth=2", "--report-format=json", f"--report-path={report}", str(destination)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Export needs secret-scan review; keep it private. Redacted report: {report}")
    print(json.dumps({"export": str(destination), "files": len(inventory["files"]), "gitleaks_findings": 0, "git_history_copied": False}))


def check():
    """Check the tracked public tree, including Git modes, without cloud access."""
    policy = json.loads(POLICY.read_text())
    entries = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=ROOT).decode().rstrip("\0").split("\0")
    count = 0
    for entry in entries:
        if not entry:
            continue
        info, name = entry.split("\t", 1)
        mode = info.split()[0]
        if mode not in {"100644", "100755"} or not (allowed(name, policy) or name in policy["overlays"]):
            raise ValueError(f"Tracked path or Git mode is outside public source policy: {name}")
        raw, _ = read_source(name)
        if name not in policy["exact_assets"] and (len(raw) > policy["max_file_bytes"] or b"\0" in raw[:8192]):
            raise ValueError(f"Unexpected binary or large source: {name}")
        count += 1
    print(json.dumps({"tracked_public_source_files": count}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("plan")
    p.add_argument("inventory", type=Path)
    p = commands.add_parser("export")
    p.add_argument("inventory", type=Path)
    p.add_argument("destination", type=Path)
    commands.add_parser("check", help="Check tracked files in the clean public repository")
    args = parser.parse_args()
    if args.command == "plan":
        plan(args.inventory)
    elif args.command == "export":
        export(args.inventory, args.destination)
    else:
        check()


if __name__ == "__main__":
    main()
