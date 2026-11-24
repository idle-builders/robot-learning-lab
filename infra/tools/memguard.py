#!/usr/bin/env python3
"""Run a local command; start it when the guarded jobs together are expected to fit in memory. Best effort.

Usage: python3 infra/tools/memguard.py [--timeout-s SECONDS] [--max-gb GB] [--limit-gb GUESS] [--no-wait] -- <command ...>
       python3 infra/tools/memguard.py --status [-- <command ...>]

Nobody states how much memory a job needs (owner 2026-11-24: "the agents cannot really know and the hardcoded values are
random"), jobs are not budgeted at their peak ("over a set of multiple jobs it will even out"), and it stays simple ("this
is no brutal memguard just a best effort"; the Mac has 32 GB and swaps fast):

* Measure. A job's memory is the physical footprint of its process tree, sampled every second (macOS: phys_footprint
  from proc_pid_rusage, what Activity Monitor and `top` MEM show, compressed pages included; elsewhere the resident set).
  Resident memory alone missed compressed pages: on 2026-11-24 a hip-section job read 2 GB resident at a 15 GB footprint.
* Remember. Each command keeps two weighted numbers in ~/.cache/memguard/profiles.json: its mean footprint over a run and
  its peak, under three keys from its command line: the exact command (paths relative to the working directory, --label
  dropped), the script with its subcommand and --checks value, and the script alone. A finished run is folded in as a
  weighted running average (1/n for the first runs, then MEMGUARD_WEIGHT 0.25, so it follows a command whose memory
  changes); no samples are kept. Summaries unused for 60 days are dropped.
* Admit. A job counts with the summary under its most specific key that has one (a running job's mean at least its
  footprint now); a command never seen with its --limit-gb, else MEMGUARD_DEFAULT_GB (6), until it has run once. It starts
  when the jobs' means + room for one of them at its peak (the largest peak - mean) is at most MEMGUARD_BUDGET_GB (24;
  owner 2026-11-22: 8 of the 32 GB stay for macOS and the apps) and at most what the jobs use now + what the machine has
  available now - MEMGUARD_FLOOR_GB (2) (macOS kern.memorystatus_level x RAM; Linux MemAvailable). A job may start ahead
  of the longest-waiting one only if that one still fits after it. With no guarded job running a job starts at once.
  Peaks that happen to coincide can go over; macOS compresses and swaps, and the next jobs wait.
* Stop. Only a runaway: the tree past --max-gb (default MEMGUARD_JOB_MAX_GB = 24; the 2026-11-08 run grew to 169 GB),
  exit 137; and --timeout-s, seconds of awake time (sleep not counted), exit 124, so a hung gmsh / OCC call fails fast.

--limit-gb is accepted from older callers; it is only the first guess for a command without history, never a cap.
Running and waiting jobs register in ~/.cache/memguard (claimed under a file lock, dead pids ignored); memguard
processes of older copies of this script count flat at their --limit-gb. Shell wrappers whose command line merely
contains "memguard.py" are never counted. `--status` prints the machine, the running jobs with their summaries and, given
a command, its summary and whether it would start now, without running anything.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

SLOTS = Path(os.environ.get("MEMGUARD_DIR") or Path.home() / ".cache" / "memguard")   # MEMGUARD_DIR: a separate registry (tests)
PROFILES = SLOTS / "profiles.json"
BUDGET_GB = float(os.environ.get("MEMGUARD_BUDGET_GB", "24"))
FLOOR_GB = float(os.environ.get("MEMGUARD_FLOOR_GB", "2"))
DEFAULT_GB = float(os.environ.get("MEMGUARD_DEFAULT_GB", "6"))
JOB_MAX_GB = float(os.environ.get("MEMGUARD_JOB_MAX_GB", str(BUDGET_GB)))
WEIGHT = float(os.environ.get("MEMGUARD_WEIGHT", "0.25"))   # weight of the newest run once a summary has 4 runs (1/n before)
FORGET_DAYS = 60                     # a summary not updated for this long is dropped
GIB = 2 ** 30


# ------------------------------------------------------------------------------------------------ measuring
_LIBPROC = _LIBC = None
if sys.platform == "darwin":
    try:
        _LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib")
        _LIBC = ctypes.CDLL(ctypes.util.find_library("c"))
    except OSError:
        _LIBPROC = _LIBC = None


def _footprint(pid: int) -> tuple[float, float] | None:
    """(physical footprint now, its lifetime maximum) of one process in GiB, macOS; None when not readable.
    rusage_info_v4 as uint64 words: [9] ri_phys_footprint, [30] ri_lifetime_max_phys_footprint."""
    if _LIBPROC is None:
        return None
    buf = (ctypes.c_uint64 * 48)()
    if _LIBPROC.proc_pid_rusage(pid, 4, ctypes.byref(buf)) != 0:
        return None
    return buf[9] / GIB, buf[30] / GIB


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


def _tree_pids(root: int, kids: dict) -> list[int]:
    out, stack = [], [root]
    while stack:
        p = stack.pop()
        out.append(p)
        stack.extend(kids.get(p, []))
    return out


def _tree_kb(root: int, kids: dict, rss: dict) -> int:
    """Resident set of a tree in KiB (kept for callers that sample resident memory)."""
    return sum(rss.get(p, 0) for p in _tree_pids(root, kids))


def tree_rss_kb(root: int) -> int:
    return _tree_kb(root, *_ps_tree())


def tree_gb(root: int, tree=None) -> tuple[float, float]:
    """(footprint of the process tree now, largest lifetime footprint of any one process in it) in GiB. The second catches
    a spike of a single process between two samples. A process whose footprint cannot be read counts with its resident set."""
    kids, rss = tree or _ps_tree()
    now = top = 0.0
    for p in _tree_pids(root, kids):
        f = _footprint(p)
        if f is None:
            g = rss.get(p, 0) / 1048576
            now += g; top = max(top, g)
        else:
            now += f[0]; top = max(top, f[1])
    return now, top


def _sysctl_int(name: str) -> int | None:
    if _LIBC is None:
        return None
    buf = (ctypes.c_uint8 * 8)()
    n = ctypes.c_size_t(8)
    if _LIBC.sysctlbyname(name.encode(), buf, ctypes.byref(n), None, 0) != 0:
        return None
    return int.from_bytes(bytes(buf)[:n.value], sys.byteorder)


def machine() -> dict:
    """total / available GiB and pressure (1 normal, 2 warning, 4 critical; 0 unknown) of the machine now."""
    if _LIBC is not None:
        total = (_sysctl_int("hw.memsize") or 0) / GIB
        level = _sysctl_int("kern.memorystatus_level")                 # % of memory the kernel counts as available
        return dict(total_gb=total, available_gb=total * (level if level is not None else 100) / 100,
                    pressure=_sysctl_int("kern.memorystatus_vm_pressure_level") or 0)
    try:
        m = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
        return dict(total_gb=m["MemTotal"] / 1048576, available_gb=m["MemAvailable"] / 1048576, pressure=0)
    except (OSError, KeyError, ValueError, IndexError):
        return dict(total_gb=0.0, available_gb=float("inf"), pressure=0)


# ------------------------------------------------------------------------------------------------ history
def _rel(tok: str, cwd: str) -> str:
    if tok.startswith("/") and tok.startswith(cwd.rstrip("/") + "/"):
        return os.path.relpath(tok, cwd)
    return tok


def command_keys(args: list[str], cwd: str | None = None) -> list[str]:
    """[exact command, script + subcommand + --checks, script] of a command line, the same in every checkout."""
    cwd = cwd or os.getcwd()
    t = list(args)
    while t and (os.path.basename(t[0]) in ("caffeinate", "env", "nice", "nohup", "time") or re.match(r"^\w+=", t[0])
                 or (t[0].startswith("-") and len(t) > 1)):
        t = t[1:]
    if t and re.match(r"python[0-9.]*$", os.path.basename(t[0]), re.I):
        t = t[1:]
        while t and t[0].startswith("-") and t[0] != "-m":
            t = t[1:]
        if t[:1] == ["-m"]:
            t = t[1:]
    if not t:
        return ["", "", ""]
    script = _rel(t[0], cwd)
    rest, i = [], 1
    while i < len(t):
        if t[i] == "--label" and i + 1 < len(t):
            i += 2
            continue
        rest.append(re.sub(r"\d{6,}", "#", _rel(t[i], cwd)))
        i += 1
    sub = rest[0] if rest and not rest[0].startswith("-") and "/" not in rest[0] and "." not in rest[0] else ""
    checks = rest[rest.index("--checks") + 1] if "--checks" in rest[:-1] else ""
    k2 = " ".join(x for x in (script, sub, f"--checks {checks}" if checks else "") if x)
    return [" ".join([script] + rest), k2, script]


def _profiles() -> dict:
    try:
        return json.loads(PROFILES.read_text())
    except (OSError, ValueError):
        return {}


def summary_for(keys: list[str]) -> tuple[dict | None, str]:
    """(summary under the most specific key that has one, what it is)."""
    prof = _profiles()
    names = ("this exact command", "this script and subcommand", "this script")
    for lvl, k in enumerate(keys):
        if k and k in prof:
            n = prof[k]["n"]
            return prof[k], f"{n} earlier run{'s' if n > 1 else ''} of {names[lvl]} `{k[:90]}`"
    return None, ""


def _update(old: dict | None, mean: float, peak: float) -> dict:
    """One finished run folded into a command's summary by a weighted running average: the newest run weighs 1/n while
    there are fewer than 1/WEIGHT runs, then WEIGHT, so the summary follows a command whose memory changes."""
    if not old:
        return dict(n=1, t=round(time.time()), mean=round(mean, 3), peak=round(peak, 3))
    n = old["n"] + 1
    w = max(1.0 / n, WEIGHT)
    return dict(n=n, t=round(time.time()), mean=round((1 - w) * old["mean"] + w * mean, 3),
                peak=round((1 - w) * old["peak"] + w * peak, 3))


def _remember(keys: list[str], mean: float, peak: float):
    """Fold a finished run (its mean footprint while it ran, its peak) into the summaries of its keys."""
    try:
        SLOTS.mkdir(parents=True, exist_ok=True)
        with open(SLOTS / ".lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            prof = _profiles()
            for k in dict.fromkeys(x for x in keys if x):
                prof[k] = _update(prof.get(k), mean, peak)
            old = time.time() - FORGET_DAYS * 86400
            prof = {k: v for k, v in prof.items() if v.get("t", 0) >= old}
            tmp = PROFILES.with_suffix(".tmp"); tmp.write_text(json.dumps(prof, separators=(",", ":"))); tmp.replace(PROFILES)
    except OSError:
        pass


# ------------------------------------------------------------------------------------------------ expecting
class Forecast:
    """What a job is expected to use: its mean footprint over a run and its usual peak (a running job: its mean at least
    its footprint now)."""

    def __init__(self, keys: list[str] | None, guess: float | None, now_gb: float | None = None, flat: float | None = None):
        self.sm, self.basis = summary_for(keys) if keys and flat is None else (None, "")
        if flat is not None:
            self.mean = self.peak = flat
            self.basis = f"older memguard, its cap {flat:.1f} GB"
        elif self.sm is None:
            self.mean = self.peak = guess if guess is not None else DEFAULT_GB
            self.basis = (f"no earlier run; the caller's --limit-gb {self.mean:.1f} GB as a first guess" if guess is not None
                          else f"no earlier run; the default first guess {self.mean:.1f} GB (MEMGUARD_DEFAULT_GB)")
        else:
            self.mean, self.peak = self.sm["mean"], max(self.sm["peak"], self.sm["mean"])
        if now_gb is not None and now_gb > self.mean:
            self.mean, self.peak = now_gb, max(self.peak, now_gb)

    def describe(self) -> str:
        if self.sm is None:
            return self.basis
        return f"{self.basis}: mean {self.mean:.2f} GB, usual peak {self.peak:.2f} GB"


# ------------------------------------------------------------------------------------------------ registry
def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _slots() -> dict[int, dict]:
    reg = {}
    for f in SLOTS.glob("*.json"):
        try:
            d = json.loads(f.read_text())
            pid = int(d["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if _alive(pid):
            reg[pid] = d
        else:
            f.unlink(missing_ok=True)
    return reg


def running_jobs() -> list[dict]:
    """Guarded jobs running now: pid, cmd, now (footprint), fc (what it is expected to use)."""
    reg = _slots()
    tree = _ps_tree()
    jobs = []
    for pid, d in reg.items():
        if d.get("state") != "running":
            continue
        if "keys" in d:
            now = float(d.get("now_gb", 0.0))
            fc = Forecast(d["keys"], d.get("guess_gb"), now)
        else:                                              # older memguard: flat at its cap
            now = tree_gb(pid, tree)[0]
            fc = Forecast(None, None, flat=max(float(d.get("limit_gb", 12.0)), now))
        jobs.append(dict(pid=pid, cmd=d.get("cmd", ""), now=now, fc=fc))
    out = "" if os.environ.get("MEMGUARD_DIR") else subprocess.run(["ps", "-Ao", "pid=,command="], capture_output=True, text=True).stdout
    for line in out.splitlines():
        pid_s, _, cmd = line.strip().partition(" ")
        argv = cmd.split()
        # a python interpreter running memguard.py (not a shell whose -c string mentions it)
        if len(argv) < 2 or "python" not in os.path.basename(argv[0]).lower() or not argv[1].endswith("memguard.py"):
            continue
        pid = int(pid_s)
        if pid == os.getpid() or pid in reg or "--status" in argv:
            continue
        m = re.search(r"--limit-gb\s+([0-9.]+)", cmd)
        now = tree_gb(pid, tree)[0]
        jobs.append(dict(pid=pid, cmd=" ".join(argv[2:8]), now=now,
                         fc=Forecast(None, None, flat=max(float(m.group(1)) if m else 12.0, now))))
    return jobs


def _oldest_waiter(since: float) -> Forecast | None:
    """Forecast of the live job that has waited longest, if it started waiting before `since` (this job's own start)."""
    best = None
    for pid, d in _slots().items():
        if pid == os.getpid() or d.get("state") != "waiting" or float(d.get("since", since)) >= since:
            continue
        if best is None or float(d["since"]) < best[0]:
            best = (float(d["since"]), d)
    if not best:
        return None
    d = best[1]
    return Forecast(d.get("keys"), d.get("guess_gb")) if "keys" in d else Forecast(None, None, flat=float(d.get("limit_gb", DEFAULT_GB)))


def fits(new: Forecast, jobs: list[dict], mach: dict, ahead: Forecast | None = None) -> tuple[bool, str]:
    """Whether the new job is expected to fit with the running ones, and the numbers: the jobs' means + room for one of
    them at its usual peak (jobs are not all at their peak at once) within the budget and what the machine has free."""
    fcs = [j["fc"] for j in jobs] + [new] + ([ahead] if ahead else [])
    means = sum(f.mean for f in fcs)
    spike = max(f.peak - f.mean for f in fcs)
    have_now = sum(j["now"] for j in jobs)
    cap = min(BUDGET_GB, have_now + mach["available_gb"] - FLOOR_GB)
    why = (f"{len(jobs)} guarded jobs use {have_now:.1f} GB now; with this job"
           + (" and the longest-waiting one" if ahead else "") +
           f" expected {means + spike:.1f} GB (means {means:.1f} + one spike {spike:.1f}) against {cap:.1f} GB (budget "
           f"{BUDGET_GB:g}, machine available {mach['available_gb']:.1f} of {mach['total_gb']:.0f} GB, floor {FLOOR_GB:g})")
    return (not jobs) or means + spike <= cap, why


def _write_slot(slot: Path, **kw):
    try:
        d = json.loads(slot.read_text()); d.update(kw)
        tmp = slot.with_suffix(".tmp"); tmp.write_text(json.dumps(d)); tmp.replace(slot)
    except (OSError, ValueError):
        pass


def _claim_slot(new: Forecast, keys: list[str], guess: float | None, args: list[str]) -> Path | None:
    """Wait until the job is expected to fit, then register it as running. None when the registry is unusable."""
    try:
        SLOTS.mkdir(parents=True, exist_ok=True)
        lock = open(SLOTS / ".lock", "w")
    except OSError as e:
        print(f"memguard: no slot registry ({e}); starting without waiting", file=sys.stderr)
        return None
    me = SLOTS / f"{os.getpid()}.json"
    # limit_gb: what older copies of this script read as the job's share
    base = dict(pid=os.getpid(), keys=keys, guess_gb=guess, limit_gb=round(new.mean, 2), cwd=os.getcwd(),
                cmd=" ".join(args[:6]))
    t0, said = time.time(), 0.0
    while True:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            jobs = running_jobs()
            mach = machine()
            ok, why = fits(new, jobs, mach, _oldest_waiter(t0))
            if ok:
                if not jobs and mach["available_gb"] - new.peak < FLOOR_GB:
                    print(f"memguard: no other guarded job is running, starting although the machine is short: {why}",
                          file=sys.stderr)
                me.write_text(json.dumps(dict(base, state="running", since=time.time(), now_gb=0.0, peak_gb=0.0)))
                if said:
                    print(f"memguard: slot after {time.time() - t0:.0f} s wait", file=sys.stderr)
                return me
            me.write_text(json.dumps(dict(base, state="waiting", since=t0)))
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
        if time.time() - said > 120:
            print(f"memguard: waiting for memory: {why}", file=sys.stderr)
            said = time.time()
        time.sleep(3)


# ------------------------------------------------------------------------------------------------ running
def awake_clock() -> float:
    """Seconds the machine has been awake, the same in every process: CLOCK_UPTIME_RAW on macOS (stops during sleep),
    CLOCK_MONOTONIC on Linux. Not Python's monotonic clock: in Python 3.9 it starts at 0 in each process, so a deadline
    handed to a child in another Python would be wrong (measured 2026-11-23: 3.9 read 0.006, 3.12 read 646751)."""
    for c in ("CLOCK_UPTIME_RAW", "CLOCK_MONOTONIC"):
        if hasattr(time, c):
            return time.clock_gettime(getattr(time, c))
    return time.time()


def _run(args, max_gb, timeout_s, slot):
    """Run the job; (return code, mean footprint while it ran, peak GiB)."""
    # The timeout counts time the machine is awake: awake_clock() stops while the laptop sleeps (2026-11-23: a 3 min job
    # stopped at "6437 s" after a hibernation).
    t0, m0 = time.time(), awake_clock()
    env = dict(os.environ)
    if timeout_s is not None:       # the job may stop itself cleanly before this (robot/body/checks/run.py does, 20 s ahead)
        env["MEMGUARD_DEADLINE"] = f"{t0 + timeout_s:.1f}"            # wall clock, for older readers
        env["MEMGUARD_DEADLINE_AWAKE"] = f"{m0 + timeout_s:.3f}"       # awake time: compare with awake_clock() (same in every process)
    proc = subprocess.Popen(args, start_new_session=True, env=env)
    total, n, peak, told = 0.0, 0, 0.0, 0.0
    why = None
    while proc.poll() is None:
        now, top = tree_gb(proc.pid)
        peak = max(peak, now, top)
        total += now; n += 1
        if slot is not None and time.time() - told > 2:     # the other memguards read the footprint from here
            _write_slot(slot, now_gb=round(now, 3), peak_gb=round(peak, 3))
            told = time.time()
        awake = awake_clock() - m0
        if timeout_s is not None and awake > timeout_s:
            slept = (time.time() - t0) - awake
            why, rc = (f"{awake:.0f} s awake > {timeout_s:.0f} s timeout"
                       + (f" (and {slept:.0f} s asleep, not counted)" if slept > 30 else "")), 124
        elif now > max_gb:
            why, rc = f"{now:.1f} GB > {max_gb:g} GB per-job maximum (runaway)", 137
        if why:
            print(f"memguard: {why}, stopping {' '.join(args[:4])}", file=sys.stderr)
            os.killpg(proc.pid, signal.SIGTERM)
            time.sleep(3)
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            print(f"memguard: peak {peak:.2f} GB footprint", file=sys.stderr)
            return rc, (total / n if n else 0.0), peak
        time.sleep(1)
    slept = (time.time() - t0) - (awake_clock() - m0)
    print(f"memguard: peak {peak:.2f} GB footprint"
          + (f"; the machine slept {slept:.0f} s during the job (not counted in the timeout)" if slept > 30 else ""), file=sys.stderr)
    return proc.returncode, (total / n if n else 0.0), peak


def status(args: list[str]) -> int:
    mach = machine()
    jobs = running_jobs()
    print(f"machine: {mach['available_gb']:.1f} of {mach['total_gb']:.0f} GB available, pressure level {mach['pressure']}")
    for j in jobs:
        print(f"  pid {j['pid']:>6} footprint {j['now']:5.2f} GB  {j['cmd'][:70]}\n           {j['fc'].describe()}")
    if args:
        new = Forecast(command_keys(args), None)
        ok, why = fits(new, jobs, mach, _oldest_waiter(time.time()))
        print(f"this command: {new.describe()}\n{'would start now' if ok else 'would wait'}: {why}")
    return 0


def main() -> int:
    args = sys.argv[1:]
    guess, timeout_s, max_gb, wait, show = None, None, JOB_MAX_GB, True, False
    while args[:1] in (["--limit-gb"], ["--timeout-s"], ["--max-gb"], ["--no-wait"], ["--status"]):
        if args[0] == "--no-wait":
            wait, args = False, args[1:]
            continue
        if args[0] == "--status":
            show, args = True, args[1:]
            continue
        v = float(args[1])
        if args[0] == "--limit-gb":
            guess = v
        elif args[0] == "--max-gb":
            max_gb = v
        else:
            timeout_s = v
        args = args[2:]
    if args[:1] == ["--"]:
        args = args[1:]
    if show:
        return status(args)
    if not args:
        print(__doc__)
        return 2
    keys = command_keys(args)
    new = Forecast(keys, guess)
    print(f"memguard: expecting {new.describe()}", file=sys.stderr)
    slot = _claim_slot(new, keys, guess, args) if wait else None
    try:
        rc, mean, peak = _run(args, max_gb, timeout_s, slot)
        if peak > 0:
            _remember(keys, mean, peak)
        return rc
    finally:
        if slot is not None:
            slot.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
