"""Snapshot completed legacy inputs for later review; not a producer attestation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def inventory(folder):
    from lab.artifact_store import validate_file
    result = {}
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            validate_file(path)
        if path.is_file():
            with path.open("rb") as stream:
                result[path.relative_to(folder).as_posix()] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    return result


def run(folders, output):
    root = Path.cwd().resolve()
    entries = {}
    for folder in folders:
        relative = folder.resolve().relative_to(root).as_posix()
        name = (
            "build-report.json"
            if (folder / "build-report.json").exists()
            else "report.json"
        )
        report = json.loads((folder / name).read_text())
        if report.get("status") not in ("pass", "fail", "partial"):
            raise ValueError("requires terminal report")
        entries[relative] = inventory(folder)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(
        scope="External snapshot captured after completion; preserves current bytes, not producer-time attestation or signature",
        captured_at=datetime.now(timezone.utc).isoformat(),
        folders=entries,
    )
    path = output / "manifest.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    (output / "manifest.sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output)
