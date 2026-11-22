#!/usr/bin/env python3
"""Make a Git worktree that can run the design checks at once.

  python3 infra/tools/worktree.py add NAME [--from REV]

Creates ../robot-learning-lab.worktrees/NAME on a new branch NAME (from REV, default HEAD), links the main checkout's
.venv, data/ and .tools/ into it, and copies artifacts/ (Git-ignored: the manifests, STEPs and meshes the checks read) as
an APFS copy-on-write clone, which takes about 100 s and no disk space until a file changes. The content-addressed check
caches are shared with the main checkout (robot/body/checks/cachefs.py), so the first check run in the new worktree is
warm. Worktrees made by hand without these links failed with `No module named 'numpy'` or missing STEPs (2026-11-20).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

MAIN = Path(__file__).resolve().parents[2]
LINKS = (".venv", "data", ".tools")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add")
    add.add_argument("name")
    add.add_argument("--from", dest="rev", default="HEAD")
    a = ap.parse_args()
    common = Path(subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=MAIN, capture_output=True, text=True, check=True).stdout.strip())
    main_co = (MAIN / common).resolve().parent              # the main checkout, also when run from a worktree
    dest = main_co.parent / f"{main_co.name}.worktrees" / a.name
    if dest.exists():
        sys.exit(f"{dest} exists")
    subprocess.run(["git", "worktree", "add", "-b", a.name, str(dest), a.rev], cwd=main_co, check=True)
    for n in LINKS:
        if (main_co / n).exists() and not (dest / n).exists():
            (dest / n).symlink_to(Path("../..") / main_co.name / n)
    excl = (main_co / common).resolve() / "info" / "exclude" if not common.is_absolute() else common / "info" / "exclude"
    excl.parent.mkdir(parents=True, exist_ok=True)
    have = excl.read_text().splitlines() if excl.exists() else []
    with excl.open("a") as fh:                              # the links are not ignored by the directory patterns of .gitignore
        for n in LINKS:                                     # anchored: a bare `data` also ignored robot/body/checks/data
            if f"/{n}" not in have:
                fh.write(f"/{n}\n")
    t0 = time.time()
    clone = ["cp", "-c", "-R"] if sys.platform == "darwin" else ["cp", "-R", "--reflink=auto"]
    subprocess.run(clone + [str(main_co / "artifacts"), str(dest / "artifacts")], check=True)
    print(f"{dest}: branch {a.name}, links {', '.join(LINKS)}, artifacts cloned in {time.time() - t0:.0f} s; check caches shared with {main_co.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
