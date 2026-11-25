#!/usr/bin/env python3
"""Git worktrees that work on their own and land cleanly in main (owner 2026-11-22: "our job is just to make those
worktrees work"; what a worktree is for is the agent's choice: a task, a body section, anything).

  python3 infra/tools/worktree.py add NAME [--from REV]   # new worktree + branch NAME, ready to build and check at once
  python3 infra/tools/worktree.py list                    # every worktree: branch, ahead / behind main, uncommitted files,
                                                          # the body definitions and builders its branch changes
  python3 infra/tools/worktree.py land [NAME] [--keep]    # bring a finished worktree into main (below)
  python3 infra/tools/worktree.py remove NAME [--force]   # delete a worktree (and its branch once merged)
  python3 infra/tools/worktree.py prune [--yes]           # worktrees whose branch is merged and that hold nothing uncommitted
  python3 infra/tools/worktree.py adopt NAME PATH...      # move main's uncommitted edits in PATHs into a new worktree NAME

add: ../robot-learning-lab.worktrees/NAME on a new branch NAME; .venv, data, .tools and .artifact-store are links to the
main checkout's (artifacts/ holds 229k links into .artifact-store); artifacts/ (Git-ignored: manifests, STEPs, meshes) is
one APFS clone of the main checkout's, made in one clonefile call (13 s for 111k files; `cp -c -R` took 1083 s under
load): no disk until a file changes, and nothing the worktree writes reaches main. The check caches and the generator
cache (robot/body/bom/gencache.py) are shared, so the first build or check in the worktree reuses what any checkout made.

land, run by the worktree's own agent when its work is committed: merges the branch into main (`Merge NAME: <subject>`;
a conflict leaves main unchanged: `git merge main` in the worktree, resolve, land again), removes the worktree and its
branch (--keep keeps them) and runs the generator cache gc. Seconds; it builds nothing (owner 2026-11-24: generated files
are built on demand with `robot/body/bom/make.py TARGET` by whoever needs them, and cached for every checkout).
Main stays open for small edits (docs, Idle bookkeeping); work that runs builders or checks belongs in a worktree.

adopt (owner 2026-11-23: "push other sessions into worktrees"): work in progress that sits uncommitted in main blocks
everyone (merges refuse, builders in main read it). adopt makes worktree NAME from main's HEAD, carries the uncommitted
changes of the given files / folders there (tracked edits as a patch, untracked files copied), keeps a backup in
main's .bookkeeping/worktree-adopt/<time>-NAME/ (the patch and the untracked files), and only then restores those paths
in main (tracked files to HEAD, untracked ones removed). The session that owned the edits continues in the worktree."""
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
BODY_FILES = ("robot/body/", "robot/electronics/", "robot/actuators/")


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
        for n in LINKS:                                     # anchored at the root: a bare `data` would ignore robot/body/checks/data
            if "/" + n not in have:
                fh.write("/" + n + "\n")
    secs, how = clone_tree(MAIN / "artifacts", dest / "artifacts")
    print(f"{dest}: branch {name} from {rev}; links {', '.join(LINKS)}; artifacts cloned in {secs:.0f} s ({how}); "
          f"check and generator caches shared with {MAIN.name}")
    print("next: work and commit here; build what you need when you need it, `.venv/bin/python -B robot/body/bom/make.py "
          "TARGET` (a generator, `body`, a path or a part; `status TARGET` shows what is out of date; cached outputs are "
          "reused); `python3 infra/tools/worktree.py land` when done (merge only, builds nothing)")
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
    """Merge the worktree's committed branch into main and clean up; nothing is built (owner 2026-11-24: generated files are
    built on demand, `robot/body/bom/make.py TARGET`, and cached; the scripts are the source of truth). Before, land ran
    the whole generator chain in the worktree and again in main under the land lock: 8.7 h of worktree makes, 2.7 h of
    main rebuilds and 2.75 h of lock queueing over 2026-11-22..24."""
    t0 = time.time()
    path, branch = resolve(name)
    d = dirty(path)
    if d:
        sys.exit(f"{path.name}: commit (or remove) these first, land takes committed work only:\n  " + "\n  ".join(d[:30]))
    lockf = MAIN / ".cache/land.lock"                       # two merges into main at once would race on its index
    lockf.parent.mkdir(parents=True, exist_ok=True)
    with open(lockf, "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        subj = (git("log", "-1", "--no-merges", "--format=%s", f"main..{branch}", cwd=MAIN)
                or git("log", "-1", "--format=%s", branch, cwd=MAIN))
        r = git("merge", "--no-ff", "-m", f"Merge {branch}: {subj}", branch, cwd=MAIN, check=False)
        if r.returncode:
            git("merge", "--abort", cwd=MAIN, check=False)
            sys.exit(f"merge of {branch} into main did not go through; main is unchanged:\n{(r.stderr or r.stdout).strip()}\n"
                     f"In the worktree: `git merge main`, resolve, commit, then land again.")
    print(git("log", "-1", "--format=%h %s", cwd=MAIN))
    if keep:
        print(f"kept {path.name} (--keep)")
    else:
        remove(path, branch)
    gc()
    print(f"landed in {time.time() - t0:.0f} s. Nothing was built: generated files are made on demand "
          f"(`.venv/bin/python -B robot/body/bom/make.py status body` shows what is out of date, `make.py body` builds it)")
    return 0


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


def adopt(name, paths):
    """Move main's uncommitted changes in paths into a new worktree NAME (see the module docstring)."""
    paths = [p.rstrip("/") for p in paths]
    rows = [l for l in dirty(MAIN) if any(l[3:].split(" -> ")[-1] == p or l[3:].startswith(p + "/") for p in paths)]
    if not rows:
        sys.exit(f"main has no uncommitted changes in {', '.join(paths)}")
    tracked = sorted({l[3:].split(" -> ")[-1] for l in rows if not l.startswith("??")})
    untracked = []
    for l in rows:
        if l.startswith("??"):
            q = MAIN / l[3:].rstrip("/")
            untracked += [str(f.relative_to(MAIN)) for f in ([q] if q.is_file() else sorted(x for x in q.rglob("*") if x.is_file()))]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    bk = MAIN / ".bookkeeping" / "worktree-adopt" / f"{stamp}-{name}"
    bk.mkdir(parents=True, exist_ok=True)
    patch = subprocess.run(["git", "diff", "--binary", "HEAD", "--", *tracked], cwd=MAIN, capture_output=True).stdout if tracked else b""
    (bk / "tracked.patch").write_bytes(patch)
    import shutil
    for f in untracked:
        (bk / "untracked" / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(MAIN / f, bk / "untracked" / f)
    add(name, "HEAD")
    dest = WT_DIR / name
    if patch:
        r = subprocess.run(["git", "apply", "--binary", "-"], cwd=dest, input=patch, capture_output=True)
        if r.returncode:
            sys.exit(f"the patch did not apply in {dest} ({r.stderr.decode().strip()}); main is unchanged, backup in {bk}")
    for f in untracked:
        (dest / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(MAIN / f, dest / f)
    moved = subprocess.run(["git", "diff", "--binary", "HEAD", "--", *tracked], cwd=dest, capture_output=True).stdout if tracked else b""
    if moved != patch:
        sys.exit(f"{dest} does not hold the same changes as main; main is unchanged, backup in {bk}")
    in_head = set(git("ls-tree", "-r", "--name-only", "HEAD", "--", *tracked, cwd=MAIN).splitlines()) if tracked else set()
    if in_head:
        git("checkout", "HEAD", "--", *sorted(in_head), cwd=MAIN)
    for f in sorted(set(tracked) - in_head):               # staged as new: not in HEAD, so drop from the index and the disk
        git("rm", "-q", "-f", "--", f, cwd=MAIN)
    for f in untracked:
        (MAIN / f).unlink()
    print(f"moved {len(tracked)} edited and {len(untracked)} untracked file(s) from main into {dest} (branch {name}); "
          f"main restored for them; backup {bk.relative_to(MAIN)}")
    print(f"the session that made them continues in {dest}: commit there, then `python3 infra/tools/worktree.py land`")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a_add = sub.add_parser("add"); a_add.add_argument("name"); a_add.add_argument("--from", dest="rev", default="main")
    sub.add_parser("list")
    a_land = sub.add_parser("land"); a_land.add_argument("name", nargs="?"); a_land.add_argument("--keep", action="store_true")
    a_rm = sub.add_parser("remove"); a_rm.add_argument("name"); a_rm.add_argument("--force", action="store_true")
    a_pr = sub.add_parser("prune"); a_pr.add_argument("--yes", action="store_true")
    a_ad = sub.add_parser("adopt"); a_ad.add_argument("name"); a_ad.add_argument("paths", nargs="+")
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
    elif a.cmd == "adopt":
        adopt(a.name, a.paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
