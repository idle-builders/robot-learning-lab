"""Move files or folders and keep every catalog that names them by path consistent.

    .venv/bin/python infra/storage/move_paths.py OLD NEW [OLD NEW ...] [--dry-run]
    .venv/bin/python infra/storage/move_paths.py --map moves.json [--dry-run]

OLD and NEW are repository-relative paths of a file or folder (moves.json is a
list of [OLD, NEW] pairs). One run:

- moves what is on disk (tracked, untracked, ignored and evicted symlinks alike;
  relative symlinks are re-pointed), and moves tracked index entries like `git mv`;
- rewrites the path in the blob index (data/blob-store.sqlite), so cold files
  restore at their new path;
- rewrites the path in the dashboard catalog (data/results.sqlite): assets, clips,
  record text and every JSON payload;
- rewrites the path in tracked text files (code, docs, pointers), except files whose
  bytes are registered evidence (dashboard assets or blob-indexed files) and lines
  that cite a commit, which name a historical path on purpose.

Afterwards commit, then run `.venv/bin/python infra/storage/catalog_backup.py backup` and
commit data/catalog-backup.json. See docs/operations/git-and-bookkeeping.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "data" / "results.sqlite"
BLOBS = ROOT / "data" / "blob-store.sqlite"
TEXT_SUFFIXES = {".py", ".md", ".json", ".jsonl", ".txt", ".toml", ".yaml", ".yml", ".sh", ".js",
                 ".html", ".css", ".csv", ".xml", ".urdf", ".mjcf", ".cfg", ".ini", ""}
COMMIT_LINE = re.compile(r"(?i)(commit|git (show|checkout|worktree|log))[^\n]*\b[0-9a-f]{7,40}\b"
                         r"|\b[0-9a-f]{7,40}\b[^\n]*(commit|git (show|checkout|worktree))")


def git(*args, input=None):
    return subprocess.run(["git", *args], cwd=ROOT, input=input, capture_output=True,
                          check=True, text=isinstance(input, str) or input is None).stdout


def clean(path: str) -> str:
    rel = Path(path)
    if rel.is_absolute() or ".." in rel.parts or not path or path.startswith(".git"):
        raise ValueError(f"Not a repository-relative path: {path}")
    return rel.as_posix().rstrip("/")


def pattern(old: str) -> re.Pattern:
    # The whole path or a prefix followed by '/', never a longer name (r1 is not r10),
    # never the tail of another path (except an absolute path into this checkout).
    # A one-word name (lab, config) is also an ordinary word: only rewrite it as a path prefix.
    tail = r"(?=/)" if "/" not in old and "." not in old else r"(?![\w\-]|\.\w)"
    return re.compile(rf"(?:(?<![\w.\-/])|(?<=robot-learning-lab/)){re.escape(old)}{tail}")


def rename(text: str, moves: list[tuple[str, str, re.Pattern]]) -> str:
    for old, new, rx in moves:
        if old in text:
            text = rx.sub(new, text)
    return text


def rewrite_lines(text: str, moves) -> str:
    if not any(old in text for old, _, _ in moves):
        return text
    return "".join(line if COMMIT_LINE.search(line) else rename(line, moves)
                   for line in text.splitlines(keepends=True))


def disk_entries(old: Path) -> list[Path]:
    if old.is_symlink() or old.is_file():
        return [old]
    if old.is_dir():
        return sorted(p for p in old.rglob("*") if p.is_symlink() or p.is_file())
    return []


def plan(pairs):
    moves = []
    for old, new in pairs:
        old, new = clean(old), clean(new)
        if old == new or new.startswith(old + "/"):
            raise ValueError(f"Move into itself: {old} -> {new}")
        moves.append((old, new, pattern(old)))
    files = {}  # old relative path -> new relative path, for everything on disk / in git / indexed
    tracked = set(git("ls-files", "-z").split("\0")) - {""}
    with sqlite3.connect(BLOBS) as db:
        indexed = {row[0] for row in db.execute("SELECT path FROM files")}
        versions = db.execute("SELECT name FROM sqlite_master WHERE name='file_versions'").fetchone()
    for old, new, _ in moves:
        for path in disk_entries(ROOT / old):
            rel = path.relative_to(ROOT).as_posix()
            files.setdefault(rel, new + rel[len(old):])
        for rel in (tracked | indexed):
            if rel == old or rel.startswith(old + "/"):
                files.setdefault(rel, new + rel[len(old):])
    targets = {}
    for src, dst in files.items():
        if dst in targets and targets[dst] != src:
            raise ValueError(f"Two paths would move to {dst}: {targets[dst]} and {src}")
        targets[dst] = src
        if dst not in files and ((ROOT / dst).exists() or (ROOT / dst).is_symlink()
                                 or dst in tracked or dst in indexed):
            raise ValueError(f"Target already exists: {dst} (from {src})")
    return moves, files, tracked, indexed, bool(versions)


def move_disk(files: dict[str, str]) -> int:
    moved = 0
    for src, dst in files.items():
        a, b = ROOT / src, ROOT / dst
        if not (a.is_symlink() or a.exists()):
            continue
        b.parent.mkdir(parents=True, exist_ok=True)
        if a.is_symlink():
            target = os.readlink(a)
            if not os.path.isabs(target):
                absolute = os.path.normpath(os.path.join(a.parent, target))
                target = os.path.relpath(absolute, b.parent)
            os.symlink(target, b)
            a.unlink()
        else:
            os.rename(a, b)
        moved += 1
    for src in sorted({str(Path(s).parent) for s in files}, key=len, reverse=True):
        d = ROOT / src
        while d != ROOT and d.is_dir() and not any(d.iterdir()):
            d.rmdir()
            d = d.parent
    return moved


def move_index(files: dict[str, str], tracked: set[str]) -> int:
    entries = git("ls-files", "-s", "-z").split("\0")
    staged = {}
    for entry in entries:
        if entry:
            meta, name = entry.split("\t", 1)
            staged[name] = meta.split()
    lines = []
    for src, dst in files.items():
        if src in tracked and src in staged:
            mode, blob, _stage = staged[src]
            lines.append(f"0 {'0' * 40}\t{src}")
            lines.append(f"{mode} {blob}\t{dst}")
    if lines:
        git("update-index", "--index-info", input="\n".join(lines) + "\n")
    return len(lines) // 2


def move_blobs(moves) -> int:
    count = 0
    with sqlite3.connect(BLOBS) as db:
        tables = ["files"] + (["file_versions"] if db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='file_versions'").fetchone() else [])
        for old, new, _ in moves:
            for table in tables:
                count += db.execute(
                    f"UPDATE {table} SET path=?||substr(path,?) WHERE path=? OR substr(path,1,?)=?",
                    (new, len(old) + 1, old, len(old) + 1, old + "/")).rowcount
    return count


def json_columns(db):
    for table, key, column in (("sections", "rowid", "payload_json"), ("source_payload", "rowid", "payload_json"),
                               ("design_boms", "rowid", "payload_json"), ("design_bom_history", "rowid", "payload_json"),
                               ("results", "rowid", "provenance"),
                               ("results", "rowid", "matrix_json"), ("results", "rowid", "summary"),
                               ("results", "rowid", "observation"), ("results", "rowid", "measurements"),
                               ("results", "rowid", "limitations"), ("results", "rowid", "body"),
                               ("clips", "rowid", "body"), ("clips", "rowid", "observation")):
        if db.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone():
            yield table, key, column


def move_results(moves) -> int:
    if not RESULTS.exists():
        return 0
    changed = 0
    with sqlite3.connect(RESULTS) as db:
        for old, new, _ in moves:
            for table, column in (("assets", "path"), ("clips", "source_path"), ("clips", "report_path"),
                                  ("clips", "trace_path")):
                changed += db.execute(
                    f"UPDATE {table} SET {column}=?||substr({column},?) WHERE {column}=? OR substr({column},1,?)=?",
                    (new, len(old) + 1, old, len(old) + 1, old + "/")).rowcount
        for table, key, column in json_columns(db):
            like = " OR ".join(f"instr({column},?)>0" for _ in moves)
            rows = db.execute(f"SELECT {key},{column} FROM {table} WHERE {like}",
                              [old for old, _, _ in moves]).fetchall()
            for rowid, value in rows:
                updated = rename(value, moves)
                if updated != value:
                    db.execute(f"UPDATE {table} SET {column}=? WHERE {key}=?", (updated, rowid))
                    changed += 1
    return changed


def protected_files() -> set[str]:
    protected = set()
    if RESULTS.exists():
        with sqlite3.connect(RESULTS) as db:
            protected |= {row[0] for row in db.execute("SELECT DISTINCT path FROM assets")}
    with sqlite3.connect(BLOBS) as db:
        protected |= {row[0] for row in db.execute("SELECT path FROM files")}
    return protected


def move_text(moves, protected: set[str]) -> list[str]:
    names = git("ls-files", "-z", "--cached", "--others", "--exclude-standard").split("\0")
    edited = []
    for name in names:
        path = ROOT / name
        if (not name or name in protected or name.startswith(("data/legacy-results-source/", ".bookkeeping/"))
                or path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file() or path.is_symlink()
                or path.stat().st_size > 5_000_000):
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        updated = rewrite_lines(text, moves)
        if updated != text:
            path.write_text(updated)
            edited.append(name)
    return edited


def run(pairs, dry_run=False, quiet=False, catalogs=True, record=None):
    moves, files, tracked, indexed, _ = plan(pairs)
    report = {"moves": [[o, n] for o, n, _ in moves], "paths": len(files),
              "tracked": sum(1 for s in files if s in tracked),
              "indexed": sum(1 for s in files if s in indexed)}
    if dry_run:
        report["sample"] = list(files.items())[:20]
        return report
    protected = protected_files()
    report["disk_moved"] = move_disk(files)
    report["index_moved"] = move_index(files, tracked)
    if catalogs:
        report["blob_rows"] = move_blobs(moves)
        report["catalog_rows"] = move_results(moves)
    if record:
        # Pairs whose catalog rewrite is still due (branch work merged later): --catalogs-only.
        done = json.loads(record.read_text()) if record.exists() else []
        record.write_text(json.dumps(done + [[o, n] for o, n, _ in moves], indent=1) + "\n")
    report["text_files"] = move_text(moves, {files.get(p, p) for p in protected} | protected)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="OLD NEW pairs")
    parser.add_argument("--map", type=Path, help="JSON list of [OLD, NEW] pairs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-catalogs", action="store_true",
                        help="move files and text only (work on a branch); append the pairs to --record")
    parser.add_argument("--record", type=Path, help="JSON file collecting pairs for a later --catalogs-only run")
    parser.add_argument("--catalogs-only", action="store_true",
                        help="rewrite only the blob index and dashboard catalog for already-moved paths")
    args = parser.parse_args()
    pairs = json.loads(args.map.read_text()) if args.map else []
    if len(args.paths) % 2:
        parser.error("give OLD NEW pairs")
    pairs += list(zip(args.paths[::2], args.paths[1::2]))
    if not pairs:
        parser.error("nothing to move")
    if args.catalogs_only:
        moves = [(clean(o), clean(n), pattern(clean(o))) for o, n in pairs]
        print(json.dumps({"blob_rows": move_blobs(moves), "catalog_rows": move_results(moves),
                          "local_moved": move_disk({src: dst for o, n, _ in moves for src, dst in
                                                    ((p.relative_to(ROOT).as_posix(), n + p.relative_to(ROOT).as_posix()[len(o):])
                                                     for p in disk_entries(ROOT / o))})}, indent=2))
        return
    report = run(pairs, args.dry_run, catalogs=not args.no_catalogs, record=args.record)
    print(json.dumps(report, indent=2))
    if not args.dry_run:
        print("Next: review `git status`, commit, then run infra/storage/catalog_backup.py backup "
              "and commit data/catalog-backup.json.", file=sys.stderr)


if __name__ == "__main__":
    main()
