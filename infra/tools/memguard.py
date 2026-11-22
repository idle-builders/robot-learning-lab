#!/usr/bin/env python3
"""Run a local command and stop it before it exhausts the machine's memory.

Usage: python3 infra/tools/memguard.py [--limit-gb 12] [--timeout-s SECONDS] -- <command ...>

Polls the resident memory of the command's whole process tree every second
and terminates the tree if it exceeds the limit. Prints the peak on exit.
Use it for local CPU work that loads many meshes or traces (CAD fit sweeps,
FEM, trace scoring); a 2026-11-08 run had one python process grow to about
169 GB and push the 32 GB laptop into an out-of-memory event.
--timeout-s also stops the tree after that many seconds of wall time (exit 124),
so a hung gmsh / OCC call fails fast with a message instead of running for hours.

Parallel budget (owner 2026-11-20, docs/operations/working-rules.md): before starting, memguard waits for a slot so
that at most MEMGUARD_MAX_JOBS (4) guarded jobs run on the machine and the resident memory they actually use now plus
the new job's --limit-gb is at most MEMGUARD_BUDGET_GB (24) (owner 2026-11-22: count real memory, not the summed caps;
a light check at 1.3 GB had waited 136 s behind caps that its neighbours never reached). Running and waiting jobs register in ~/.cache/memguard (claimed under a file lock, dead pids
ignored); memguard processes of older copies of this script that do not register are counted from `ps` with their own
--limit-gb. Shell wrappers whose command line merely contains "memguard.py" are never counted. --no-wait skips the
wait (for a job that must start now); the timeout clock starts when the job starts, not while it waits.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

SLOTS = Path.home() / ".cache" / "memguard"
MAX_JOBS = int(os.environ.get("MEMGUARD_MAX_JOBS", "4"))
BUDGET_GB = float(os.environ.get("MEMGUARD_BUDGET_GB", "24"))


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _running_jobs() -> list[tuple[int, float, float]]:
    """(pid, cap GB, resident GB of its whole process tree now) of the guarded jobs running: registered running slots +
    unregistered memguard python processes (older copies of this script)."""
    reg = {}
    for f in SLOTS.glob("*.json"):
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if _alive(int(d["pid"])):
            reg[int(d["pid"])] = d
        else:
            f.unlink(missing_ok=True)
    jobs = {pid: float(d["limit_gb"]) for pid, d in reg.items() if d.get("state") == "running"}
    out = subprocess.run(["ps", "-Ao", "pid=,command="], capture_output=True, text=True).stdout
    for line in out.splitlines():
        pid_s, _, cmd = line.strip().partition(" ")
        argv = cmd.split()
        # a python interpreter running memguard.py (not a shell whose -c string mentions it)
        if len(argv) < 2 or "python" not in os.path.basename(argv[0]).lower() or not argv[1].endswith("memguard.py"):
            continue
        pid = int(pid_s)
        if pid == os.getpid() or pid in reg:
            continue
        m = re.search(r"--limit-gb\s+([0-9.]+)", cmd)
        jobs[pid] = float(m.group(1)) if m else 12.0
    kids, rss = _ps_tree()
    return [(pid, cap, _tree_kb(pid, kids, rss) / 1048576) for pid, cap in jobs.items()]


def _claim_slot(limit_gb: float, args: list[str]) -> Path | None:
    """Wait until the job fits the budget, then register it as running. None when the registry is unusable."""
    try:
        SLOTS.mkdir(parents=True, exist_ok=True)
        lock = open(SLOTS / ".lock", "w")
    except OSError as e:
        print(f"memguard: no slot registry ({e}); starting without waiting", file=sys.stderr)
        return None
    me = SLOTS / f"{os.getpid()}.json"
    t0, said = time.time(), 0.0
    while True:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            jobs = _running_jobs()
            used = sum(r for _, _, r in jobs)
            if len(jobs) < MAX_JOBS and used + limit_gb <= BUDGET_GB:
                me.write_text(json.dumps(dict(pid=os.getpid(), limit_gb=limit_gb, state="running", since=time.time(),
                                              cwd=os.getcwd(), cmd=" ".join(args[:6]))))
                if said:
                    print(f"memguard: slot after {time.time() - t0:.0f} s wait", file=sys.stderr)
                return me
            me.write_text(json.dumps(dict(pid=os.getpid(), limit_gb=limit_gb, state="waiting", since=t0)))
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
        if time.time() - said > 300:
            print(f"memguard: waiting for a slot ({len(jobs)} jobs running using {used:.1f} GB, caps {sum(c for _, c, _ in jobs):.0f} GB; "
                  f"budget {MAX_JOBS} jobs / {BUDGET_GB:.0f} GB used + this cap {limit_gb:g})", file=sys.stderr)
            said = time.time()
        time.sleep(5)


def _ps_tree() -> tuple[dict, dict]:
    out = subprocess.run(["ps", "-Ao", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
    kids, rss = {}, {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        pid, ppid, r = map(int, parts)
        kids.setdefault(ppid, []).append(pid)
        rss[pid] = r
    return kids, rss


def _tree_kb(root: int, kids: dict, rss: dict) -> int:
    total, stack = 0, [root]
    while stack:
        p = stack.pop()
        total += rss.get(p, 0)
        stack.extend(kids.get(p, []))
    return total


def tree_rss_kb(root: int) -> int:
    return _tree_kb(root, *_ps_tree())


def main() -> int:
    args = sys.argv[1:]
    limit_gb = 12.0
    timeout_s = None
    wait = True
    while args[:1] in (["--limit-gb"], ["--timeout-s"], ["--no-wait"]):
        if args[0] == "--no-wait":
            wait, args = False, args[1:]
            continue
        if args[0] == "--limit-gb":
            limit_gb = float(args[1])
        else:
            timeout_s = float(args[1])
        args = args[2:]
    if args[:1] == ["--"]:
        args = args[1:]
    if not args:
        print(__doc__)
        return 2
    slot = _claim_slot(limit_gb, args) if wait else None
    try:
        return _run(args, limit_gb, timeout_s)
    finally:
        if slot is not None:
            slot.unlink(missing_ok=True)


def _run(args: list[str], limit_gb: float, timeout_s: float | None) -> int:
    t0 = time.time()
    env = dict(os.environ)
    if timeout_s is not None:       # the job may stop itself cleanly before this (robot/body/checks/run.py does, 20 s ahead)
        env["MEMGUARD_DEADLINE"] = f"{t0 + timeout_s:.1f}"
    proc = subprocess.Popen(args, start_new_session=True, env=env)
    peak = 0
    while proc.poll() is None:
        kb = tree_rss_kb(proc.pid)
        peak = max(peak, kb)
        over_time = timeout_s is not None and time.time() - t0 > timeout_s
        if kb > limit_gb * 1024 * 1024 or over_time:
            why = (f"{time.time() - t0:.0f} s > {timeout_s:.0f} s timeout" if over_time
                   else f"{kb / 1048576:.1f} GB > {limit_gb} GB limit")
            print(f"memguard: {why}, stopping {' '.join(args[:4])}", file=sys.stderr)
            os.killpg(proc.pid, signal.SIGTERM)
            time.sleep(3)
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            print(f"memguard: peak {peak / 1048576:.2f} GB", file=sys.stderr)
            return 124 if over_time else 137
        time.sleep(1)
    print(f"memguard: peak {peak / 1048576:.2f} GB", file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
