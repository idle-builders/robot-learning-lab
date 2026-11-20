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
"""
import os
import signal
import subprocess
import sys
import time


def tree_rss_kb(root: int) -> int:
    out = subprocess.run(["ps", "-Ao", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
    kids, rss = {}, {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        pid, ppid, r = map(int, parts)
        kids.setdefault(ppid, []).append(pid)
        rss[pid] = r
    total, stack = 0, [root]
    while stack:
        p = stack.pop()
        total += rss.get(p, 0)
        stack.extend(kids.get(p, []))
    return total


def main() -> int:
    args = sys.argv[1:]
    limit_gb = 12.0
    timeout_s = None
    while args[:1] in (["--limit-gb"], ["--timeout-s"]):
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
    proc = subprocess.Popen(args, start_new_session=True)
    peak = 0
    t0 = time.time()
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
