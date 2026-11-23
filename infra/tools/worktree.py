#!/usr/bin/env python3
"""Git worktrees that work on their own and land cleanly in main (owner 2026-11-22: "our job is just to make those
worktrees work"; what a worktree is for is the agent's choice: a task, a body section, anything).

  python3 infra/tools/worktree.py add NAME [--from REV]   # new worktree + branch NAME, ready to build and check at once
  python3 infra/tools/worktree.py list                    # every worktree: branch, ahead / behind main, uncommitted files,
                                                          # the body definitions and builders its branch changes
  python3 infra/tools/worktree.py land [NAME] [--keep]    # bring a finished worktree into main (below)
  python3 infra/tools/worktree.py remove NAME [--force]   # delete a worktree (and its branch once merged)
  python3 infra/tools/worktree.py prune [--yes]           # worktrees whose branch is merged and that hold nothing uncommitted

add: ../robot-learning-lab.worktrees/NAME on a new branch NAME; .venv, data, .tools and .artifact-store are links to the
main checkout's (artifacts/ holds 229k links into .artifact-store); artifacts/ (Git-ignored: manifests, STEPs, meshes) is
one APFS clone of the main checkout's, made in one clonefile call (13 s for 111k files; `cp -c -R` took 1083 s under
load): no disk until a file changes, and nothing the worktree writes reaches main. The check caches and the generator
cache (robot/body/bom/gencache.py) are shared, so the first build or check in the worktree reuses what any checkout made.

land, run by the worktree's own agent when its work is committed (owner 2026-11-22):
  1. refuses while the worktree has uncommitted changes (naming them);
  2. `make.py` in the worktree: every generator whose inputs changed runs there, so its outputs are in the shared cache;
  3. under main's land lock (one land at a time; a second waits), merges the branch into main (`Merge NAME: <subject>`).
     Refuses, changing nothing, when main has uncommitted edits in files the branch changes (another session's work in
     progress), and aborts a conflicting merge (merge main into the branch in the worktree, resolve, land again). Checks
     afterwards that main contains the branch;
  4. `make.py --report` in main: the selected body brought up to date from the cache (the generators this worktree ran
     are reused; only what main changed meanwhile runs), then the BOM, the bom check and the BOM findings that changed;
     scores, never a gate;
  5. removes the worktree and its branch (--keep keeps them), and runs the generator cache gc.
Main stays open for small edits (docs, Idle bookkeeping); work that runs builders or checks belongs in a worktree."""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
LINKS = (".venv", "data", ".tools", ".artifact-store")
BODY_FILES = ("robot/body/", "robot/design.json", "robot/electronics/", "robot/actuators/")


def git(*a, cwd, check=True):
    r = subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode:
        sys.exit(f"git {' '.join(a)} failed in {cwd}:\n{r.stderr.strip() or r.stdout.strip()}")
    return r.stdout.strip() if check else r


def main_checkout() -> Path:
    common = Path(git("rev-parse", "--git-common-dir", cwd=HERE.parent))
    if not common.is_absolute():
        common = (HERE.parent / common).resolve()
    return common.parent


MAIN = main_checkout()
WT_DIR = MAIN.parent / f"{MAIN.name}.worktrees"


def worktrees():
    """[(path, branch)] of the linked worktrees (not main)."""
    out, cur = [], {}
    for line in git("worktree", "list", "--porcelain", cwd=MAIN).splitlines() + [""]:
        if not line:
            if cur.get("worktree") and Path(cur["worktree"]) != MAIN:
                out.append((Path(cur["worktree"]), cur.get("branch", "").removeprefix("refs/heads/")))
            cur = {}
        else:
            k, _, v = line.partition(" ")
            cur[k] = v
    return out


def resolve(name: str | None) -> tuple[Path, str]:
    here = Path.cwd().resolve()
    for p, b in worktrees():
        if (name and (p.name == name or b == name)) or (not name and (here == p or p in here.parents)):
            return p, b
    sys.exit(f"no worktree {'named ' + name if name else 'contains ' + str(here)} (see `worktree.py list`)")


def dirty(path: Path) -> list[str]:
    out = subprocess.run(["git", "status", "--porcelain"], cwd=path, capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if l.strip()]      # not stripped: the status columns are fixed-width (XY path)


def changed_files(branch: str) -> list[str]:
    r = git("merge-base", "main", branch, cwd=MAIN, check=False)
    if r.returncode:
        return []                     # unrelated history (e.g. a branch from before the 2026-11-20 re-root)
    base = r.stdout.strip()
    return [f for f in git("diff", "--name-only", base, branch, cwd=MAIN).splitlines() if f]


# ------------------------------------------------------------------------------------------------ add
def clone_tree(src: Path, dst: Path):
    t0 = time.time()
    if sys.platform == "darwin":
        libc = ctypes.CDLL("/usr/lib/libSystem.dylib", use_errno=True)
        if libc.clonefile(os.fsencode(src), os.fsencode(dst), 0) == 0:
            return time.time() - t0, "one APFS clonefile"
    cmd = ["cp", "-c", "-R"] if sys.platform == "darwin" else ["cp", "-R", "--reflink=auto"]
    subprocess.run(cmd + [str(src), str(dst)], check=True)
    return time.time() - t0, " ".join(cmd)


def add(name, rev):
    dest = WT_DIR / name
    if dest.exists():
        sys.exit(f"{dest} exists")
    git("worktree", "add", "-b", name, str(dest), rev, cwd=MAIN)
    for n in LINKS:
        if (MAIN / n).exists() and not (dest / n).exists():
            (dest / n).symlink_to(Path("../..") / MAIN.name / n)
    excl = Path(git("rev-parse", "--git-common-dir", cwd=MAIN))
    excl = (excl if excl.is_absolute() else MAIN / excl) / "info" / "exclude"
    have = excl.read_text().splitlines() if excl.exists() else []
    if any(h in LINKS for h in have):                       # a bare line (an older copy of this script) also ignores robot/body/checks/data
        have = [h for h in have if h not in LINKS]
        excl.write_text("\n".join(have) + "\n")
    with excl.open("a") as fh:                              # the links are not ignored by the directory patterns of .gitignore
        for n in LINKS:
            if n not in have:
                fh.write(n + "\n")
    secs, how = clone_tree(MAIN / "artifacts", dest / "artifacts")
    print(f"{dest}: branch {name} from {rev}; links {', '.join(LINKS)}; artifacts cloned in {secs:.0f} s ({how}); "
          f"check and generator caches shared with {MAIN.name}")
    print("next: work and commit here; `.venv/bin/python -B robot/body/bom/make.py` brings this checkout's body up to date "
          "(reusing every unchanged output); `python3 infra/tools/worktree.py land` when done")
    others = [(p, b) for p, b in worktrees() if p != dest]
    touched = {}
    for p, b in others:
        fs = [f for f in changed_files(b) + [l[3:] for l in dirty(p)] if f.startswith(BODY_FILES)] if b else []
        if fs:
            touched[p.name] = sorted(set(fs))
    if touched:
        print("other worktrees are changing (avoid duplicate work; `worktree.py list` for more):")
        for n, fs in touched.items():
            print(f"  {n}: " + ", ".join(fs[:8]) + (f" (+{len(fs) - 8})" if len(fs) > 8 else ""))


# ------------------------------------------------------------------------------------------------ list
def listing():
    rows = []
    for p, b in worktrees():
        if not b:
            rows.append(dict(name=p.name, branch="(detached)")); continue
        if git("merge-base", "main", b, cwd=MAIN, check=False).returncode:
            rows.append(dict(name=p.name, branch=f"{b} (no common history with main: from before the 2026-11-20 re-root)")); continue
        ahead = int(git("rev-list", "--count", f"main..{b}", cwd=MAIN))
        behind = int(git("rev-list", "--count", f"{b}..main", cwd=MAIN))
        merged = git("merge-base", "--is-ancestor", b, "main", cwd=MAIN, check=False).returncode == 0
        d = dirty(p)
        age = git("log", "-1", "--format=%cr", b, cwd=MAIN)
        defs = sorted({f for f in changed_files(b) + [l[3:] for l in d] if f.endswith(".bom.yaml") or f.endswith(".py") and f.startswith("robot/body/")})
        rows.append(dict(name=p.name, branch=b, ahead=ahead, behind=behind, uncommitted=len(d), merged=merged, last_commit=age, body_files=defs))
    for r in rows:
        if "ahead" not in r:
            print(f"{r['name']:24s} {r['branch']}"); continue
        state = "merged" if r["merged"] and not r["uncommitted"] else f"+{r['ahead']}/-{r['behind']}"
        print(f"{r['name']:24s} {state:10s} uncommitted {r['uncommitted']:3d}  last commit {r['last_commit']:16s} "
              + (", ".join(Path(f).name for f in r["body_files"][:6]) + (f" (+{len(r['body_files']) - 6})" if len(r["body_files"]) > 6 else "")))
    return rows


# ------------------------------------------------------------------------------------------------ land / remove
def run(cmd, cwd):
    print(f"$ {' '.join(cmd)}  (in {cwd.name})", flush=True)
    return subprocess.run(cmd, cwd=cwd).returncode


def remove(path: Path, branch: str, force=False):
    d = dirty(path)
    if d and not force:
        sys.exit(f"{path.name} has uncommitted changes ({len(d)}): {', '.join(l[3:] for l in d[:8])}; commit them or pass --force")
    git("worktree", "remove", *(["--force"] if force else []), str(path), cwd=MAIN)
    if branch and git("merge-base", "--is-ancestor", branch, "main", cwd=MAIN, check=False).returncode == 0:
        git("branch", "-d", branch, cwd=MAIN)
        print(f"removed {path.name} and its merged branch {branch}")
    else:
        print(f"removed {path.name}; branch {branch} kept (not merged into main)")


def gc():
    """The generator cache gc (one store for every checkout), with main's make.py, or this copy's before it is in main."""
    make = MAIN / "robot/body/bom/make.py"
    if not make.exists():
        make = HERE.parents[2] / "robot/body/bom/make.py"
    return run([str(MAIN / ".venv/bin/python"), "-B", str(make), "gc"], MAIN)


def land(name, keep):
    path, branch = resolve(name)
    d = dirty(path)
    if d:
        sys.exit(f"{path.name}: commit (or remove) these first, land takes committed work only:\n  " + "\n  ".join(d[:30]))
    if git("rev-parse", "--abbrev-ref", "HEAD", cwd=MAIN) != "main":
        sys.exit(f"the main checkout is not on branch main; nothing done")
    print(f"== 1/4 bring {path.name}'s body up to date (its generator runs go into the shared cache)")
    if run([".venv/bin/python", "-B", "robot/body/bom/make.py"], path):
        sys.exit("make failed in the worktree (log above); fix and land again. Nothing was merged.")
    lockf = MAIN / ".cache/land.lock"
    lockf.parent.mkdir(parents=True, exist_ok=True)
    with open(lockf, "a+") as lk:
        try:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lk.seek(0)
            print(f"waiting for another land ({lk.read().strip()}) ...", flush=True)
            fcntl.flock(lk, fcntl.LOCK_EX)
        lk.seek(0); lk.truncate(); lk.write(f"{branch} pid {os.getpid()} since {time.strftime('%H:%M:%S')}\n"); lk.flush()
        print(f"== 2/4 merge {branch} into main")
        mine = set(changed_files(branch))
        theirs = {l[3:].split(" -> ")[-1] for l in dirty(MAIN)}
        clash = sorted(mine & theirs)
        if clash:
            sys.exit("main has uncommitted edits (another session's work in progress) in files this branch changes; nothing "
                     "merged. Wait until they are committed, then land again:\n  " + "\n  ".join(clash))
        subj = git("log", "-1", "--format=%s", branch, cwd=MAIN)
        r = git("merge", "--no-ff", "-m", f"Merge {branch}: {subj}", branch, cwd=MAIN, check=False)
        if r.returncode:
            conflicts = git("diff", "--name-only", "--diff-filter=U", cwd=MAIN, check=False).stdout.split()
            git("merge", "--abort", cwd=MAIN, check=False)
            sys.exit(f"merge of {branch} failed; main is unchanged. {('Conflicts: ' + ', '.join(conflicts)) if conflicts else r.stderr.strip()}\n"
                     f"In the worktree: `git merge main`, resolve, commit, then land again.")
        if git("merge-base", "--is-ancestor", branch, "HEAD", cwd=MAIN, check=False).returncode:
            sys.exit(f"main does not contain {branch} after the merge; check `git log` in main")
        print(git("log", "-1", "--format=%h %s", cwd=MAIN))
        print("== 3/4 bring main's body up to date, BOM, bom check")
        rc = run([".venv/bin/python", "-B", "robot/body/bom/make.py", "--report"], MAIN)
    if rc:
        print(f"make --report in main returned {rc} (log above): the merge stands; fix the failing generator in a new worktree")
    print("== 4/4 clean up")
    if keep:
        print(f"kept {path.name} (--keep)")
    else:
        remove(path, branch)
    gc()
    return rc


def prune(yes):
    rows = [r for r in listing() if r.get("merged") and not r.get("uncommitted")]
    if not rows:
        print("nothing to prune"); return 0
    print(f"merged and clean: {', '.join(r['name'] for r in rows)}")
    if not yes:
        print("rerun with --yes to remove them"); return 0
    for r in rows:
        remove(WT_DIR / r["name"], r["branch"])
    return gc()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a_add = sub.add_parser("add"); a_add.add_argument("name"); a_add.add_argument("--from", dest="rev", default="main")
    sub.add_parser("list")
    a_land = sub.add_parser("land"); a_land.add_argument("name", nargs="?"); a_land.add_argument("--keep", action="store_true")
    a_rm = sub.add_parser("remove"); a_rm.add_argument("name"); a_rm.add_argument("--force", action="store_true")
    a_pr = sub.add_parser("prune"); a_pr.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if a.cmd == "add":
        add(a.name, a.rev)
    elif a.cmd == "list":
        listing()
    elif a.cmd == "land":
        return land(a.name, a.keep)
    elif a.cmd == "remove":
        p, b = resolve(a.name)
        remove(p, b, a.force)
        gc()
    elif a.cmd == "prune":
        return prune(a.yes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
