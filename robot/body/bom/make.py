"""Bring this checkout's body up to date from the declared generators, reusing every output whose inputs have not changed.

  .venv/bin/python -B robot/body/bom/make.py status          # per generator: reused / runs, and why (seconds, runs nothing)
  .venv/bin/python -B robot/body/bom/make.py                 # restore or run every generator in order, in this checkout
  .venv/bin/python -B robot/body/bom/make.py --report        # ... then the BOM, the bom check, and the findings that changed
  .venv/bin/python -B robot/body/bom/make.py --run ID[,ID]   # also run these section builders (rebuild: false) here
  .venv/bin/python -B robot/body/bom/make.py gc [--dry-run]  # delete cache entries and blobs nobody needs (gencache.py)

Works the same in the main checkout and in any worktree (infra/tools/worktree.py): the generators come from the body
definitions (robot/**/<builder>.bom.yaml; order from their inputs, outputs and `after`, robot/body/bom/definitions.py).
For each one make looks for a cache entry whose recorded inputs all still hash the same here (robot/body/bom/gencache.py);
if there is one, its outputs are cloned into place (copy-on-write, seconds), whoever made them; otherwise the generator runs
here under infra/tools/memguard.py with its declared resources, the files it reads and writes are recorded
(robot/body/bom/gensite/), and its outputs go into the cache for every other checkout. A generator whose run fails stops
make (exit code of the run); nothing after it runs.

Section builders (`rebuild: false`) are not run unless named with --run; their outputs are taken as they are, and a cache
entry of one (made by a --run elsewhere, e.g. in the worktree that changed it) is restored like any other.

Reads that point at a problem in the definitions are printed and kept on the entry: a file written by a generator that runs
later (an input the order cannot honour), and outputs written outside the generator's declared outputs. One make per
checkout at a time (lock .cache/make.lock)."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import definitions as DEF  # noqa: E402
import gencache as GC  # noqa: E402

GENSITE = HERE / "gensite"
DEFAULT_RES = dict(limit_gb=6, timeout_s=900)
# blob-store indexes only say where a missing file is restored from (the file then read is recorded itself); as inputs they
# would invalidate every entry whenever any session uploads anything (same rule as checks/resultcache.py)
IGNORE_READS = ("infra/tools/memguard.py", "data/private-archive.json", "data/regenerable-artifacts.json", "data/blob-store")
TMP_SUFFIX = (".gc-tmp", ".tmp", ".part")
WRITABLE = "artifacts/robot/body"      # where generators write; snapshotted around every run


def generators(run_sections=()):
    """[(generator, managed)] in run order: section builders (rebuild false, not presentation) first, then the rebuild
    generators in their derived order. managed = make may run it."""
    D = DEF.load()
    for f in D.findings:
        if f["kind"].startswith("generator"):
            print(f"definition finding: {f['kind']} {f['file']} {f['entry']}: {f['what']}")
    order = D.generator_order()
    sections = [g for g in D.generators if not g.get("rebuild", True) and not g.get("presentation")]
    out = [(g, g["id"] in run_sections) for g in sections] + [(D.generator(i), True) for i in order]
    return out, D


def roots_of(g):
    return [r for r in g.get("outputs", []) + g.get("in_place_edits", []) + g.get("writes_into", [])]


def scan(roots):
    """{rel: (size, mtime_ns, ino)} of the files under the given repository-relative paths."""
    out = {}
    for r in roots:
        p = ROOT / r
        if p.is_file():
            st = p.stat(); out[r] = (st.st_size, st.st_mtime_ns, st.st_ino)
        elif p.is_dir():
            for d, _, files in os.walk(p):
                for f in files:
                    q = Path(d) / f
                    if q.name.endswith(TMP_SUFFIX) or q.name == ".DS_Store" or q.is_symlink():
                        continue
                    try:
                        st = q.stat()
                    except OSError:
                        continue
                    out[str(q.relative_to(ROOT))] = (st.st_size, st.st_mtime_ns, st.st_ino)
    return out


def covers(root, path):
    return path == root or path.startswith(root.rstrip("/") + "/")


def run_one(g, log_dir: Path):
    """Run generator g here with read / write recording; returns (rc, reads, dirs, outputs, deleted, notes, seconds, peak)."""
    roots = roots_of(g)
    # the whole writable body area, not only the declared outputs: builders also write files they do not declare (some
    # from OpenCascade, invisible to Python), and a file a run reads and then rewrites needs its content from before
    before = scan(roots + [WRITABLE])
    pre_sha = {r: GC.file_state(ROOT, r) for r in before}
    GC.idx().flush()
    res = {**DEFAULT_RES, **(g.get("resources") or {})}
    rec = Path(tempfile.mkdtemp(prefix=f"gen-{g['id']}-", dir=GC.store()))
    cmd = ["python3", "infra/tools/memguard.py", "--limit-gb", str(res["limit_gb"]), "--timeout-s", str(res["timeout_s"]), "--"] + g["command"]
    if shutil.which("caffeinate"):
        cmd = ["caffeinate", "-i", "-s"] + cmd
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "GENCACHE_LOG": str(rec), "GENCACHE_ROOT": str(ROOT),
           "PYTHONPATH": str(GENSITE) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}
    log = log_dir / f"{g['id']}.log"
    t0 = time.time()
    with open(log, "w") as fh:
        fh.write("$ " + " ".join(cmd) + "\n"); fh.flush()
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    secs = time.time() - t0
    text = log.read_text(errors="replace")
    peak = [float(x) for x in re.findall(r"memguard: peak ([0-9.]+) GB", text)]
    R, W, Dl = set(), set(), set()
    for f in rec.glob("*.json"):
        d = GC.cachefs.read_json(f, {}) or {}
        R |= set(d.get("reads", [])); W |= set(d.get("writes", [])); Dl |= set(d.get("dirs", []))
    shutil.rmtree(rec, ignore_errors=True)
    if rc != 0:
        return rc, None, None, None, None, [], secs, peak[-1] if peak else None
    after = scan(roots + [WRITABLE])
    outputs, notes = {}, []
    for r, st in after.items():
        if before.get(r) != st:                           # written by this run, also when with the same content: an entry
            outputs[r] = GC.file_state(ROOT, r)            # names every file the run writes, so a restore leaves none stale
    for r in sorted(W):                                   # Python writes elsewhere under artifacts/
        if r in outputs or r in after or not (ROOT / r).is_file() or r.endswith(TMP_SUFFIX) or not r.startswith("artifacts/"):
            continue
        outputs[r] = GC.file_state(ROOT, r)
    undeclared = sorted(r for r in outputs if not any(covers(x, r) for x in roots))
    if undeclared:
        notes.append(f"{len(undeclared)} outputs outside the declared outputs, e.g. {', '.join(undeclared[:3])}")
    deleted = sorted(r for r in set(before) - set(after) if any(covers(x, r) for x in roots))
    reads = {}
    reports = set(g.get("report_files", []))
    for r in sorted(R):
        if r.startswith(IGNORE_READS) or r in reports:
            continue
        if r in pre_sha:
            reads[r] = pre_sha[r]                          # its content before the run (it may have rewritten it)
        elif r in outputs and (r.startswith(WRITABLE + "/") or any(covers(x, r) for x in roots)):
            reads[r] = "missing"                           # a file it created itself: before the run there was none
        elif r in outputs:
            reads[r] = "unknown"; notes.append(f"read and rewrote a file outside {WRITABLE}: {r}")
        else:
            reads[r] = GC.file_state(ROOT, r)
    dirs = {r: GC.dir_state(ROOT, r) for r in sorted(Dl) if r.startswith(("glob:", "values:")) or data_dir(r) and not any(covers(x, r) or covers(r, x) for x in roots)}
    return 0, reads, dirs, outputs, deleted, notes, secs, peak[-1] if peak else None


def data_dir(r):
    """A folder listing that can change a result: not a Python code folder the import system lists (the modules it then
    loads are recorded as reads themselves)."""
    p = ROOT / r
    if "__path_hook__" in r or not p.is_dir():
        return False
    try:
        return not any(n.endswith(".py") for n in os.listdir(p))
    except OSError:
        return False


def late_reads(reads, pos, plan):
    """Files this run read that a generator later in the order writes (by the outputs its newest cache entry recorded):
    inputs the declared order cannot honour, so this generator reads what the previous make left there and the chain only
    settles after another pass. Grouped per writer."""
    by = {}
    for j in range(pos + 1, len(plan)):
        h = plan[j][0]
        e = next(GC.entries(h["id"]), None)
        outs = set(e["outputs"]) if e else set()
        hit = sorted(r for r in reads if r in outs)
        if hit:
            by[h["id"]] = hit
    return [f"reads {len(v)} file(s) that {k} (runs later) writes, e.g. {', '.join(v[:2])}" for k, v in by.items()]


def lock():
    p = ROOT / ".cache/make.lock"
    p.parent.mkdir(parents=True, exist_ok=True)
    fh = open(p, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.seek(0)
        sys.exit(f"make.py: another make holds {p} in this checkout ({fh.read().strip() or 'pid unknown'}); wait for it to finish")
    fh.seek(0); fh.truncate(); fh.write(f"pid {os.getpid()} since {time.strftime('%H:%M:%S')}\n"); fh.flush()
    return fh


MAX_PASSES = 3


def make(status_only=False, run_sections=(), only=None, upto=None, quiet=False, rerun=()):
    plan, D = generators(run_sections)
    log_dir = ROOT / ".cache/work/make-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary, upstream_missed, t_all = [], False, time.time()
    overlay = {}                    # status: files as the restores before this generator would leave them
    outs = {g["id"]: GC.newest_outputs(g["id"]) for g, _ in plan}   # who writes what (newest entries; feedback reads)
    for pos, (g, managed) in enumerate(plan):
        gid = g["id"]
        if only and gid not in only:
            continue
        t0 = time.time()
        earlier = set().union(*(outs[h["id"]] for h, _ in plan[:pos])) if pos else set()
        later = set().union(*(outs[h["id"]] for h, _ in plan[pos + 1:])) if pos + 1 < len(plan) else set()
        e, why = GC.lookup(gid, ROOT, g["command"], overlay if status_only else None, earlier, later)
        if gid in rerun and not status_only:
            e, why = None, "--rerun"
        if e is not None:
            if status_only:
                overlay.update(e["outputs"]); overlay.update({r: "missing" for r in e.get("deleted", [])})
            if status_only and not quiet:
                print(f"  reuse  {gid:34s} made {e['at'][:16]} in {Path(e['checkout']).name}" + (" (after an upstream rerun: may change)" if upstream_missed else ""))
            else:
                c, s, rm = GC.restore(e, ROOT)
                GC.mark(ROOT, gid, e)
                print(f"  reused {gid:34s} {len(e['outputs'])} outputs ({c} cloned, {s} already equal{f', {rm} removed' if rm else ''}) "
                      f"made {e['at'][:16]} in {Path(e['checkout']).name}, {time.time() - t0:.1f} s")
            summary.append(dict(id=gid, action="reused", seconds=round(time.time() - t0, 2)))
            continue
        if not managed:
            if not quiet and (status_only or why != "never run through make"):
                print(f"  as-is  {gid:34s} section builder (rebuild false): outputs taken as they are ({why}; --run {gid} reruns it)")
            summary.append(dict(id=gid, action="as-is", why=why))
            continue
        if status_only:
            if not quiet:
                print(f"  RUNS   {gid:34s} {why}")
            upstream_missed = True
            summary.append(dict(id=gid, action="runs", why=why))
            continue
        print(f"  run    {gid:34s} {why} ...", flush=True)
        rc, reads, dirs, outputs, deleted, notes, secs, peak = run_one(g, log_dir)
        if rc != 0:
            print(f"  FAILED {gid}: return code {rc} after {secs:.0f} s; log {(log_dir / (gid + '.log')).relative_to(ROOT)}")
            print("\n".join("        " + l for l in (log_dir / f"{gid}.log").read_text(errors="replace").splitlines()[-12:]))
            summary.append(dict(id=gid, action="failed", rc=rc, seconds=round(secs, 1)))
            return rc, summary
        notes += late_reads({r: h for r, h in reads.items() if r not in outputs}, pos, plan)
        e = GC.save(gid, g["command"], reads, dirs, outputs, deleted, ROOT, file=g["file"], seconds=round(secs, 1), peak_gb=peak, notes=notes)
        outs[gid] = set(outputs)
        GC.mark(ROOT, gid, e)
        print(f"  ran    {gid:34s} {secs:.0f} s, peak {peak} GB, {len(reads)} inputs read, {len(outputs)} outputs stored")
        for n in notes:
            print(f"         finding: {n}")
        summary.append(dict(id=gid, action="ran", seconds=round(secs, 1), notes=notes))
        if upto and gid == upto:
            break
    GC.idx().flush()
    if quiet:
        return 0, summary
    n = {k: sum(s["action"] == k for s in summary) for k in ("reused", "ran", "runs", "as-is")}
    print(f"make{' status' if status_only else ''}: {n['reused']} reused, {n['ran'] if not status_only else n['runs']} "
          f"{'ran' if not status_only else 'would run'}, {n['as-is']} section builders as they are; {time.time() - t_all:.0f} s; "
          f"selected body {json.loads((ROOT / 'robot/design.json').read_text())['body_manifest']}")
    return 0, summary


# ------------------------------------------------------------------------------------------------ report
def bom_findings():
    f = ROOT / ".cache/work/bom/findings.json"
    d = GC.cachefs.read_json(f, None)
    if d is None:
        return None
    rows = d if isinstance(d, list) else d.get("findings", [])
    return {f"{r.get('kind')}|{r.get('file')}|{r.get('entry')}|{r.get('what')}" for r in rows}


def report():
    """BOM + bom check on the selected body; prints the BOM findings that appeared or went away since the last BOM here."""
    before = bom_findings()
    py = ".venv/bin/python"
    r1 = subprocess.run([py, "-B", "robot/body/bom/build.py"], cwd=ROOT, capture_output=True, text=True)
    print(f"bom build: rc {r1.returncode}; " + " ".join(r1.stdout.strip().splitlines()[-2:])[:300])
    r2 = subprocess.run(["python3", "infra/tools/memguard.py", "--limit-gb", "4", "--timeout-s", "900", "--", py, "-B",
                         "robot/body/checks/run.py", "--checks", "bom", "--label", "bom"], cwd=ROOT, capture_output=True, text=True)
    print(f"bom check: rc {r2.returncode}; report .cache/work/cad-speed/checks/bom/report.json")
    after = bom_findings()
    if before is not None and after is not None:
        new, gone = sorted(after - before), sorted(before - after)
        print(f"BOM findings: {len(after)} now, {len(new)} new, {len(gone)} gone since the previous BOM in this checkout")
        for x in new[:25]:
            print("  + " + x.replace("|", "  "))
        for x in gone[:25]:
            print("  - " + x.replace("|", "  "))
    elif after is not None:
        print(f"BOM findings: {len(after)} (no earlier BOM in this checkout to compare)")
    return r1.returncode or r2.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", nargs="?", default="make", choices=["make", "status", "gc"])
    ap.add_argument("--run", default="", help="section builders (rebuild false) to run here too, comma-separated ids")
    ap.add_argument("--only", default="", help="only these generator ids (assumes the others are current)")
    ap.add_argument("--upto", help="stop after this generator")
    ap.add_argument("--rerun", default="", help="run these generator ids even when an entry could be reused (a fresh entry "
                    "after make or gensite changed what is recorded); the ones after them are reused when its outputs match")
    ap.add_argument("--report", action="store_true", help="after make: BOM, bom check, changed findings")
    ap.add_argument("--dry-run", action="store_true", help="gc: list only")
    ap.add_argument("--grace-days", type=float, default=GC.GRACE_DAYS)
    a = ap.parse_args()
    if a.action == "gc":
        GC.gc(dry_run=a.dry_run, grace_days=a.grace_days)
        return 0
    only = set(x for x in a.only.split(",") if x) or None
    runs = tuple(x for x in a.run.split(",") if x)
    if a.action == "status":
        rc, _ = make(status_only=True, run_sections=runs, only=only)
        return rc
    _lk = lock()
    rc, summary = make(run_sections=runs, only=only, upto=a.upto, rerun=set(x for x in a.rerun.split(",") if x))
    # the chain has loops (generators that read what a later one rewrites, see the findings): a pass that ran something may
    # leave earlier generators out of date; repeat until a pass runs nothing, at most MAX_PASSES
    n = 1
    while rc == 0 and not only and not a.upto and any(s["action"] == "ran" for s in summary) and n < MAX_PASSES:
        _, st = make(status_only=True, run_sections=runs, quiet=True)
        if not any(s["action"] == "runs" for s in st):
            break
        n += 1
        print(f"pass {n}: {', '.join(s['id'] for s in st if s['action'] == 'runs')} read files the previous pass changed")
        rc, summary = make(run_sections=runs)
    if rc == 0 and n > 1:
        _, st = make(status_only=True, run_sections=runs, quiet=True)
        left = [s["id"] for s in st if s["action"] == "runs"]
        print(f"settled after {n} passes" if not left else
              f"NOT settled after {n} passes: {', '.join(left)} still see changed inputs (a builder that writes different "
              f"content from the same inputs, or a loop in the chain); see their findings")
    if rc == 0 and a.report:
        rc = report()
    return rc


if __name__ == "__main__":
    sys.exit(main())
