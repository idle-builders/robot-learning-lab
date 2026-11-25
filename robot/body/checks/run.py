"""Shared design check for the printed body. One report per run, a worst-first summary on stdout. No pass/fail.

  python3 infra/tools/memguard.py --timeout-s 900 -- .venv/bin/python -B robot/body/checks/run.py \
      [--manifest PATH  (default: the declared body manifest, bom_config.body_manifest())] \
      [--parts GLOB[,GLOB]] [--section Hip] [--link GLOB] [--role ROLE] [--changed-since REV|last-run] \
      [--checks clearance,gaps,mass,press,render,geometry,fit,static_audit,range_sweep,collision_sweep,shell_offset,fem,fem_setup,print_setup,screws,driver_access] [--fem-variant NAME] [--poses key|library|all|fit] [--exact-press] [--process KEY=VALUE ...] [--label NAME]

Scope selectors combine with AND; none = every non-hardware part. Checks: none named = all that apply to the scope.
Scoped parts are checked against their neighbours (close over the pose set or touching at standing), not the whole body.
Report: .cache/work/cad-speed/checks/<label>/report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

# The orchestrating parent imports nothing heavy (no numpy, trimesh, manifest, solids): it spawns child processes that do the
# work and only merges their JSON, so its memory stays small while a heavy check runs. Children import the package lazily.
ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / ".cache/work/cad-speed"
GIB = 2 ** 30
# Checks that need more than ~1 GiB: each runs in its own child process (fem: one child per scoped part with a setup).
# The other (light) checks share one child.
HEAVY = ("fit", "range_sweep", "collision_sweep", "fem", "screws")



def awake_clock() -> float:
    """Seconds the machine has been awake, the same in every process (as infra/tools/memguard.py awake_clock)."""
    for c in ("CLOCK_UPTIME_RAW", "CLOCK_MONOTONIC"):
        if hasattr(time, c):
            return time.clock_gettime(getattr(time, c))
    return time.time()

def gib_of_maxrss(ru):
    """GiB (memguard's unit) of a rusage max RSS: bytes on macOS, KiB on Linux."""
    return round(ru.ru_maxrss / (1 if sys.platform == "darwin" else 1024) / GIB, 2)


def tree_rss_gib(root):
    """(whole process tree of root, root alone) resident set in GiB, as memguard counts it."""
    out = subprocess.run(["ps", "-Ao", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
    kids, rss = {}, {}
    for ln in out.splitlines():
        try:
            pid, ppid, r = map(int, ln.split())
        except ValueError:
            continue
        kids.setdefault(ppid, []).append(pid); rss[pid] = r
    todo, tot = [root], 0
    while todo:
        p = todo.pop(); tot += rss.get(p, 0); todo += kids.get(p, [])
    return tot * 1024 / GIB, rss.get(root, 0) * 1024 / GIB


_REF: dict = {}


def _ref_volume(ctx, n):
    """(volume mm3, source) the screening solid of part n is compared with. The manifest volume_mm3 is the exact STEP
    volume only for parts that have a native STEP; a mesh-only shell was stretched / morphed after it was recorded (the
    ape-proportions morph shrank R_upper_arm_hard_cover by 28 %), so its recorded volume is stale. For those the
    reference is the signed volume of the part's own mesh when that is a closed manifold, else the recorded volume."""
    if n in _REF:
        return _REF[n]
    _REF[n] = _ref_volume_uncached(ctx, n)
    return _REF[n]


def _ref_volume_uncached(ctx, n):
    p = ctx.body.parts[n]
    rec = p.get("volume_mm3")
    from checks import core
    if p.get("native_step") and not core.BL.src(p["native_step"]).exists():
        p = {**p, "native_step": None}
    big = p.get("native_step") and core._mb(p["native_step"]) > core.FINE_STEP_MAX_MB
    big_bought = p.get("role") not in core.STEP_ROLES and big
    # a STEP up to FINE_STEP_MAX_MB is the reference for every part (scoped or neighbour): the recorded volume_mm3 goes
    # stale when a builder changes the part (the stage-1 chassis sleeves read -7 % against 643.45 mm3 recorded before sleeves
    # were removed; the STEP's 31 solids give 603.6, the mesh 598.6), and native_solid_count is often not recorded
    if p.get("native_step") and not p.get("placement_offset_mm") and ((n in ctx.scope and not big_bought) or not big):
        # big STEPs (battery cells, supplier boards, over FINE_STEP_MAX_MB) cost minutes to tessellate just for this
        # comparison: they use the recorded volume or their own closed mesh below
        # several STEP solids overlap (a servo: 6 solids, summed 41257 mm3, fused 32983): the recorded / summed volume
        # double-counts the overlap, so the reference is the fused volume of all solids
        try:
            m = core.step_mesh(core.BL.src(p["native_step"]), p.get("native_step_offset_mm"), tol=0.1, ang=0.5)
            return float(abs(m.volume)), "native STEP, all solids fused"
        except Exception:  # noqa: BLE001
            pass
    if p.get("native_step") and rec:
        return rec, "native STEP volume"
    full = ctx.body.mesh(n)
    if full.volume < 0:
        full = full.copy(); full.invert()
    if len(full.faces) and core.kit.mf(full).status().name == "NoError":
        return float(full.volume), "own closed mesh signed volume"
    return rec, "recorded volume_mm3 (may predate a morph / stretch)"


def _vdiff(ctx, n, m):
    """Screening-solid volume vs its reference volume (_ref_volume) of part n, in percent (None when none)."""
    ref, _ = _ref_volume(ctx, n)
    return round(100.0 * (abs(float(m.volume)) - ref) / ref, 1) if ref else None


# Checks over the whole manifest: their result does not come from the scoped parts one by one, so a failure is not split by part.
WHOLE_MANIFEST = ("bom", "connectivity", "render")


def _why(exc):
    """One line for a check failure: exception, message and the innermost frame inside the repository."""
    tb = [fr for fr in traceback.extract_tb(exc.__traceback__) if str(ROOT) in fr.filename and "/.venv/" not in fr.filename]
    where = f" ({Path(tb[-1].filename).name}:{tb[-1].lineno} {tb[-1].name})" if tb else ""
    return f"{type(exc).__name__}: {str(exc)[:300]}{where}"


def cpu_seconds():
    """CPU seconds (user + system) of this process and its finished subprocesses (mount_check, render)."""
    import resource
    a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


def _run_one(ctx, fn, scope):
    """fn(ctx) with ctx.scope = scope (the neighbour list follows the scope); (result, None) or (None, failure line)."""
    keep, nb = ctx.scope, ctx._nb
    ctx.scope, ctx._nb = list(scope), (nb if list(scope) == list(keep) else None)
    try:
        return fn(ctx), None
    except Exception as exc:  # noqa: BLE001
        return None, _why(exc)
    finally:
        ctx.scope, ctx._nb = keep, nb


def isolate(ctx, name, fn, scope, why):
    """The check failed on `scope` with `why`. Find the parts it fails on by bisection (a part fails when the check fails on it
    alone), then run it once on the others, so one broken input (a STEP that resolves nowhere, an empty B-rep) costs that
    part's row, not every part's. Returns (result or None, {part: failure line})."""
    bad = {}

    def probe(sub, w):
        if len(sub) == 1:
            bad[sub[0]] = w
            return
        h = len(sub) // 2
        for half in (sub[:h], sub[h:]):
            r, w2 = _run_one(ctx, fn, half)
            if w2:
                probe(half, w2)
    probe(list(scope), why)
    good = [n for n in scope if n not in bad]
    if not good:
        return None, bad
    res, w = _run_one(ctx, fn, good)
    if w:                                   # fails only in combination: not a single part's input
        return None, dict(bad, **{"(combination of the other parts)": w})
    return res, bad


def run_checks_here(ctx, checks, names, out_file, cache=None):
    """Child mode: run the named checks in this process; the result file is rewritten after each check. A check that raises
    is retried without the parts it fails on (isolate): their failures are listed in `failed_parts`, the rest is measured."""
    import faulthandler
    import resource
    import signal
    from checks.core import T
    stack_file = Path(str(out_file) + ".stack")
    sf = open(stack_file, "w")
    # memguard's SIGTERM (timeout / memory cap) or the parent's deadline: the stack of the stalled call goes to the report
    faulthandler.register(signal.SIGTERM, file=sf, all_threads=False, chain=True)
    faulthandler.register(signal.SIGUSR1, file=sf, all_threads=False, chain=False)
    if os.environ.get("CHECKS_STACK_DUMP_S"):        # diagnosing a stall: the child prints its Python stack every N s
        faulthandler.dump_traceback_later(float(os.environ["CHECKS_STACK_DUMP_S"]), repeat=True, file=sys.stderr)
    out = dict(checks={}, solids={}, other_solids={}, seconds={})

    def dump():
        out["solids"] = {n: dict(how=h, volume_mm3=round(float(m.volume), 1), faces=len(m.faces),
                                 manifest_volume_mm3=ctx.body.parts[n].get("volume_mm3"), reference_volume_mm3=_ref_volume(ctx, n)[0],
                                 reference=_ref_volume(ctx, n)[1], volume_diff_pct=_vdiff(ctx, n, m))
                         for n, (m, h) in ctx.sol.items() if n in ctx.scope}
        out["other_solids"] = {n: h for n, (_, h) in ctx.sol.items() if n not in ctx.scope}
        # screening solid vs the part's recorded volume, every part used (a voxel-filled open mesh can be +-50 % off)
        # housing-solid servo cases leave their internals out on purpose (smaller than the full mesh by design): not listed
        out["solid_volume_off_over_5pct"] = {n: dict(how=h, diff_pct=d) for n, (m, h) in ctx.sol.items()
                                             if "housing solids" not in h and (d := _vdiff(ctx, n, m)) is not None and abs(d) > 5.0}
        out["seconds"] = dict(T)
        tmp = Path(str(out_file) + ".part")
        tmp.write_text(json.dumps(out, default=float))
        tmp.replace(out_file)                   # never a half-written file for the parent to read
    dump()                                      # an empty result file: the parent can tell a crash in setup from one in a check
    for name in names:
        t0, c0 = time.time(), cpu_seconds()
        print(f"[{name}] started on {len(ctx.scope)} parts", file=sys.stderr, flush=True)
        res, why = _run_one(ctx, checks[name]["run"], ctx.scope)
        bad = {}
        if why:
            print(f"[{name}] failed: {why}", file=sys.stderr, flush=True)
            if len(ctx.scope) > 1 and name not in WHOLE_MANIFEST:
                print(f"[{name}] finding the parts it fails on ...", file=sys.stderr, flush=True)
                res, bad = isolate(ctx, name, checks[name]["run"], ctx.scope, why)
            if res is None:
                res = dict(failed=why if not bad else f"failed on every part tried ({len(bad)})", per_part={}, worst_cases=[],
                           bad_events={}, evidence=[], notes="")
        if bad:
            res["failed_parts"] = bad
            res.setdefault("bad_events", {})["parts_failed"] = len(bad)
            print(f"[{name}] failed on {len(bad)} part(s): " + "; ".join(f"{n}: {w}" for n, w in list(bad.items())[:3]), file=sys.stderr, flush=True)
        res["seconds"] = round(time.time() - t0, 1)
        res["cpu_s"] = round(cpu_seconds() - c0, 1)     # far below seconds = the process waited (starved, blocked), not computed
        res["peak_gib"] = gib_of_maxrss(resource.getrusage(resource.RUSAGE_SELF))
        out["checks"][name] = res
        if cache is not None:
            from checks import resultcache as RC
            if RC.enabled() and RC.storable(res):
                try:
                    k = RC.key(cache["manifest"], name, ctx.scope, cache["options"])
                    RC.store(k, name, ctx.scope, RC.snapshot(checks[name]["run"]), RC.sets_now(), res, cache["label"])
                except Exception as exc:  # noqa: BLE001  a result that cannot be stored is only not cached
                    print(f"[{name}] result not cached: {_why(exc)}", file=sys.stderr, flush=True)
        dump()
        print(f"[{name}] done in {res['seconds']} s", file=sys.stderr, flush=True)
    from checks.core import save_sha
    save_sha()
    log = os.environ.get("CHECKS_DEPS_LOG")
    if log:
        shutil.rmtree(log, ignore_errors=True)    # this child's subprocess read logs, merged into its stored results
    sf.close()
    if stack_file.stat().st_size == 0:
        stack_file.unlink()


SETUP_TIMERS = ("solids (scope)", "solids (others)", "neighbours")     # every child builds its own solids and neighbour list


def recordable(man):
    """A manifest whose reads the result cache records (resultcache.rel_or_none): inside the repository and not in the
    checks' own cache / report folders (eval copies live there); a scratch manifest elsewhere under .cache/work is."""
    from checks import resultcache
    return resultcache.rel_or_none(man) is not None


def man_rel(man):
    """The manifest as the report names it: repository-relative, or absolute when it lies outside the repository (a baseline
    copy in a session scratchpad; refused before 2026-11-22). Its part files are repository-relative either way."""
    return str(man.relative_to(ROOT)) if man.is_relative_to(ROOT) else str(man)


def label_of(a, split):
    """The run's label, lower case: macOS paths ignore case, so 'triage-C' and 'triage-c' were one folder and one run's
    report silently replaced the other's (2026-11-22)."""
    lab = a.label or (a.section or (split(a.parts) or ["all"])[0])
    return lab.replace(" ", "-").replace("/", "")[:60].lower()


def child_ctx(a, man, split):
    """Heavy side, children only: the Ctx of this run and the registered checks."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from checks import registry
    from checks.core import WORK as W
    from checks.ctx import Ctx
    checks = registry.load_all()
    out_dir = W / "checks" / label_of(a, split)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(man, [], a.poses, a.search_mm, a.exact_press, a.render_timeout, out_dir, a.include_shells)
    ctx.fem_variant, ctx.fem_mesh_h, ctx.fem_rigid_load = a.fem_variant, a.fem_mesh_h, a.fem_rigid_load
    ctx.joints, ctx.step_deg = split(a.joints), a.step_deg
    ctx.fit_pair_budget_s = a.fit_pair_budget_s
    ctx.held = {k: float(v) for k, v in (w.split("=") for w in a.hold)}
    from checks.process import parse_override
    ctx.process_override = parse_override(a.process)
    return ctx, checks, out_dir


# Named sets for --checks (mixable with check names). The light iteration and the candidate set follow working-rules.md;
# `all` = every check that applies, as no --checks does.
PRESETS = {
    "light": "geometry,mass,gaps,press,clearance",
    "print": "print_setup,overhang,walls,bed,small_features,holes,edges,elephant_foot",
    "fasteners": "screws,driver_access,fastener_motion,press,holes",     # not "screws": that is a check
    "motion": "clearance,range_sweep,collision_sweep,fastener_motion",
    "standing": "gaps,static_audit,fit_class,shell_offset,heat",
    "candidate": "geometry,mass,gaps,static_audit,press,clearance,fit_class,print_setup,overhang,walls,bed,small_features,holes,"
                 "edges,elephant_foot,screws,driver_access,fastener_motion,heat,shell_offset,fem_setup",
}


def expand_checks(want, checks):
    """Check names with presets expanded, in the registry's order; exits naming the unknown ones and the presets."""
    out = []
    for w in want:
        if w == "all":
            out += list(checks)
        else:
            out += PRESETS[w].split(",") if w in PRESETS else [w]
    unknown = [c for c in out if c not in checks]
    if unknown:
        sys.exit(f"unknown checks {unknown}; known {list(checks)}; presets " + ", ".join(f"{k} ({v})" for k, v in PRESETS.items()) + ", all")
    return [c for c in checks if c in out]


WHOLE_TIMED = ("bom", "connectivity", "render", "assembly")     # time does not grow with the number of scoped parts


def estimate(run, n_parts, poses):
    """Rough seconds per check for this run from earlier reports on this machine (this checkout's and the main checkout's):
    the median seconds per scoped part (per run for the whole-manifest checks) of earlier finished results, same pose set where
    there are any. {check: dict(s, p90_s, runs)}; checks never run before are left out."""
    dirs = {WORK / "checks"}
    try:
        common = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirs.add((ROOT / common).resolve().parent / ".cache/work/cad-speed/checks")
    except OSError:
        pass
    samples = {}
    for d in dirs:
        for f in d.glob("*/report.json"):
            try:
                r = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            k = max(1, len(r.get("scope", {}).get("parts", [])))
            for c, v in r.get("checks", {}).items():
                if c in run and v.get("seconds") and not v.get("failed") and not v.get("cached"):
                    samples.setdefault(c, []).append((r.get("pose_set") == poses, v["seconds"] / (1 if c in WHOLE_TIMED else k)))
    out = {}
    for c, xs in samples.items():
        same = [x for m, x in xs if m] or [x for _, x in xs]
        same.sort()
        mult = 1 if c in WHOLE_TIMED else n_parts
        out[c] = dict(s=round(same[len(same) // 2] * mult, 1), p90_s=round(same[min(len(same) - 1, int(0.9 * len(same)))] * mult, 1), runs=len(same))
    return out


def manifest_freshness(man):
    """Signs that the checked manifest is older than what it is built from (2026-11-22: the selected body lacked 20 parts and
    named old torso STEPs because the generators after the roll-flip manifest had not been rerun, and nothing said so).
    1. recorded sources: every (path, sha256) the generators wrote into the manifest's metadata (`source_manifest` +
       `source_manifest_sha256`, `{path, sha256}` ...) whose file has changed since;
    2. declared upstream: the manifests the body definitions' generators chain into this one (`manifest_out` of the
       generators whose outputs this one's generator reads, transitively), by file time, when one was written after it.
    A list of findings (measurements: what changed, when); empty when nothing points at a stale manifest."""
    from checks.core import sha
    out = []
    m = json.loads(man.read_text())

    def walk(d, where):
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, str) and (k.endswith("manifest") or k in ("path", "source", "source_body")):
                    h = d.get(f"{k}_sha256") or (d.get("sha256") if k == "path" else None)
                    p = ROOT / v
                    if h and p.is_file():
                        now = sha(p, raw=True)          # the bytes the generator hashed
                        if now != h:
                            out.append(dict(kind="recorded source changed", field=f"{where}{k}", path=v, recorded_sha256=h[:12], now_sha256=now[:12],
                                            changed_at=time.strftime("%m-%d %H:%M", time.localtime(p.stat().st_mtime))))
                elif isinstance(v, (dict, list)):
                    walk(v, f"{where}{k}.")
        elif isinstance(d, list):
            for x in d:
                walk(x, where)
    walk({k: v for k, v in m.items() if k not in ("parts", "joints", "controls", "bom")}, "")
    try:
        sys.path.insert(0, str(ROOT / "robot/body/bom"))
        import definitions as DEF
        D = DEF.load()
        rel = str(man.relative_to(ROOT)) if man.is_relative_to(ROOT) else str(man)
        by_out = {}
        for g in D.generators:
            for o in [g.get("manifest_out")] + g.get("in_place_edits", []):
                if o:
                    by_out.setdefault(o, []).append(g)
        seen, todo, t_man = set(), [rel], man.stat().st_mtime
        while todo:
            o = todo.pop()
            for g in by_out.get(o, []):
                for inp in g.get("inputs", []):
                    if inp in by_out and inp not in seen:
                        seen.add(inp); todo.append(inp)
        for up in sorted(seen):
            p = ROOT / up
            if p.is_file() and p.stat().st_mtime > t_man + 1.0:
                out.append(dict(kind="upstream manifest written later", path=up, by_generators=sorted({g["id"] for g in by_out[up]}),
                                later_by_s=round(p.stat().st_mtime - t_man), written_at=time.strftime("%m-%d %H:%M", time.localtime(p.stat().st_mtime))))
    except Exception as exc:  # noqa: BLE001  the definitions are another session's; a broken one must not stop a check run
        out.append(dict(kind="declared upstream not read", error=_why(exc)))
    try:                                         # generators are built on demand (owner 2026-11-24): say which are out of date
        if man.is_relative_to(ROOT):
            import make as MK                    # robot/body/bom/make.py (on sys.path above)
            rel = str(man.relative_to(ROOT))
            st = MK.stale([rel])
            if st:
                out.append(dict(kind="generators out of date", path=rel, generators=[g for g, _ in st],
                                why={g: w for g, w in st}, command=f".venv/bin/python -B robot/body/bom/make.py {rel}"))
    except SystemExit:
        pass                                     # no generator makes this manifest (a scratch copy): nothing to say
    except Exception as exc:  # noqa: BLE001
        out.append(dict(kind="generator status not read", error=_why(exc)))
    return out


def make_plan(a, man, split):
    """Plan child: resolve the scope, which checks apply, the FEM parts, the last-run keys; write them to a JSON file."""
    ctx, checks, out_dir = child_ctx(a, man, split)
    from checks import scope as S
    from checks import fem_core as FC
    from checks.core import save_sha, sha
    names, desc = S.select(ctx.body, man, split(a.parts), a.section, split(a.link), split(a.role), a.changed_since, ctx.neighbours_of)
    if not names:
        if a.changed_since and S.select(ctx.body, man, split(a.parts), a.section, split(a.link), split(a.role))[0]:
            print(f"run.py: nothing changed since {a.changed_since} in scope {({k: v for k, v in desc.items() if k != 'selected_because'})}: "
                  "no part to check (not an error)", file=sys.stderr)
            Path(a._plan).write_text(json.dumps(dict(nothing_changed=True)))
            return
        sys.exit(f"scope {desc} selects no part of {a.manifest}")
    ctx.scope = names
    want = expand_checks(split(a.checks), checks)
    todo = [n for n in (want or checks) if n in checks]
    run = [n for n in todo if checks[n]["applies"](ctx)]
    nbs = list(ctx.neighbours())
    plan = dict(manifest=man_rel(man), manifest_sha256=sha(man), names=names, desc=desc, todo=todo, run=run,
               not_applicable=[n for n in todo if n not in run], settings={n: ctx.settings(n) for n in names},
               definitions={n: dict(file=ctx.definition(n)[0], entry=ctx.definition(n)[1], settings_source=ctx.settings_source.get(n, {})) for n in ctx.body.parts},
               out_dir=str(out_dir),
               fem_parts=[n for n in names if FC.merged_setup(ctx, n, a.fem_variant)] if "fem" in run else [],
               keys={n: S.part_key(ctx.body.parts[n]) for n in names + nbs}, last_file=str(S.LAST / (man.stem + ".json")))
    plan["estimate"] = estimate(run, len(names), a.poses)
    plan["manifest_freshness"] = manifest_freshness(man)
    plan["cached"], plan["fem_cached"], plan["cache_misses"] = {}, {}, {}
    from checks import resultcache as RC
    if RC.enabled() and not a.no_cache and recordable(man):   # outside the repository its reads are not recorded
        mrel, opts = man_rel(man), RC.options_of(a)
        for n in run:
            if n == "fem":
                for p in plan["fem_parts"]:
                    res, e = RC.lookup(RC.key(mrel, "fem", [p], opts))
                    if res is not None:
                        plan["fem_cached"][p] = dict(result=res, at=e["at"], label=e["label"])
                continue
            res, e = RC.lookup(RC.key(mrel, n, names, opts))
            if res is not None:
                plan["cached"][n] = dict(result=res, at=e["at"], label=e["label"], inputs=len(e["deps"]))
            else:
                plan["cache_misses"][n] = e
    save_sha()
    Path(a._plan).write_text(json.dumps(plan, default=float))


def merge_fem(parts_res):
    """One fem result from per-part children: {part: result or failure text}."""
    out = dict(per_part={}, worst_cases=[], bad_events={}, evidence=[], notes=[], failed_parts={})
    for n, r in parts_res.items():
        if r.get("failed"):
            out["failed_parts"][n] = r["failed"]
            continue
        out["per_part"].update(r["per_part"]); out["worst_cases"] += r["worst_cases"]; out["evidence"] += r["evidence"]
        for k, v in r["bad_events"].items():
            out["bad_events"][k] = out["bad_events"].get(k, 0) + v
        out["notes"].append(r["notes"])
    out["worst_cases"].sort(key=lambda c: -c["value"])
    out["bad_events"]["parts_failed"] = len(out["failed_parts"])
    out["notes"] = " || ".join(dict.fromkeys(out["notes"]))
    if len(out["failed_parts"]) == len(parts_res):
        out["failed"] = "; ".join(f"{n}: {w}" for n, w in out["failed_parts"].items())
    if not out["failed_parts"]:
        del out["failed_parts"]
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", help="default: the declared body manifest (`body:` in robot/body/complete_body.bom.yaml)")
    ap.add_argument("--parts", help="comma-separated part names or globs")
    ap.add_argument("--section")
    ap.add_argument("--link", help="comma-separated owning-link names or globs")
    ap.add_argument("--role", help="comma-separated part roles")
    ap.add_argument("--changed-since", help="Git revision, or 'last-run' (file sha256 since the previous run on this manifest)")
    ap.add_argument("--checks", help="comma-separated; default all that apply")
    ap.add_argument("--poses", default="key", choices=("key", "library", "all", "fit"))
    ap.add_argument("--exact-press", action="store_true")
    ap.add_argument("--joints", help="comma-separated joint names for range_sweep / collision_sweep (default: joints that move the scoped parts)")
    ap.add_argument("--step-deg", type=float, help="joint sweep step (range_sweep 2.5, collision_sweep 5)")
    ap.add_argument("--hold", action="append", default=[], help="JOINT=deg held during range_sweep")
    ap.add_argument("--fit-pair-budget-s", type=float, default=120.0, help="fit: seconds of exact B-rep work per part pair; a pair over it "
                    "is reported as timed out with the poses it reached")
    ap.add_argument("--search-mm", type=float, default=5.0)
    ap.add_argument("--render-timeout", type=float, default=240.0)
    ap.add_argument("--include-shells", action="store_true", help="also screen body shells as neighbours (heavy)")
    ap.add_argument("--fem-variant", help="named variant of the per-part FEM setups (e.g. studied_axes)")
    ap.add_argument("--fem-mesh-h", type=float, help="mesh size h (mm) replacing mesh_h_mm of every FEM setup (mesh convergence)")
    ap.add_argument("--fem-rigid-load", action="store_true", help="make the load patch of every FEM setup rigid (comparison run)")
    ap.add_argument("--process", action="append", default=[], help="KEY=VALUE replacing the print setup of every printed part in scope "
                    "(cross-evaluation, not stored): nozzle_mm, layer_height_mm, line_width_mm, material, printer, up, plate_yaw_deg")
    ap.add_argument("--label")
    ap.add_argument("--no-cache", action="store_true", help="run every check even when its inputs are unchanged since a stored result "
                    "(checks/resultcache.py; CHECKS_RESULT_CACHE=0 does the same)")
    ap.add_argument("--_scope", help=argparse.SUPPRESS)   # child mode: exact scoped part names
    ap.add_argument("--_out", help=argparse.SUPPRESS)     # child mode: result file
    ap.add_argument("--_plan", help=argparse.SUPPRESS)    # child mode: write the plan (scope, applicable checks) to this file
    ap.add_argument("--list-checks", action="store_true")
    a = ap.parse_args(argv)
    t_all, m_all = time.time(), awake_clock()        # awake_clock stops while the laptop sleeps: slept_s = the difference
    split = lambda s: [x for x in (s or "").split(",") if x]  # noqa: E731
    if a.list_checks:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from checks import registry
        for n, c in registry.load_all().items():
            print(f"{n:10s} {c['doc']}")
        return
    if not a.manifest:
        sys.path.insert(0, str(ROOT / "robot/body/bom"))
        import bom_config
        a.manifest = bom_config.body_manifest()
        argv += ["--manifest", a.manifest]   # children get it explicitly
    man = (ROOT / a.manifest).resolve()
    if a._plan:
        return make_plan(a, man, split)
    if a._out:                           # child: run the named checks on the exact scope
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from checks import resultcache as RC
        RC.install()                     # record every input from here on, before any check module is imported
        ctx, checks, _ = child_ctx(a, man, split)
        ctx.scope = split(a._scope)
        run_checks_here(ctx, checks, split(a.checks), a._out,
                        cache=dict(manifest=man_rel(man), options=RC.options_of(a), label=label_of(a, split)) if recordable(man) else None)
        return
    me = [sys.executable, "-B", str(Path(__file__).resolve())]
    label = label_of(a, split)
    out_dir = WORK / "checks" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = hold_label(out_dir, label)
    prev = out_dir / "report.json"
    if prev.exists():                                     # a previous run's report under this label must not pass for this run's
        try:
            p = json.loads(prev.read_text())
            print(f"run.py: label {label!r} replaces the report of {time.strftime('%m-%d %H:%M', time.localtime(prev.stat().st_mtime))} "
                  f"(scope {p.get('scope', {}).get('selectors')}, checks {', '.join(p.get('checks', {}))})", file=sys.stderr, flush=True)
        except (OSError, ValueError):
            pass
        prev.unlink(missing_ok=True)
    for old in out_dir.glob("child-*"):                    # leftovers of a killed run under this label (the lock says none runs)
        old.unlink(missing_ok=True)
    keep_awake()
    peaks = dict(tree=0.0, parent=0.0)
    # memguard exports its deadline: stop 20 s before it, so the report and the summary are written and say where it stood
    # in awake time when memguard gives it (MEMGUARD_DEADLINE_AWAKE, infra/tools/memguard.py awake_clock): a sleeping
    # laptop does not run the clock out; an older memguard gives wall time only
    aw = bool(os.environ.get("MEMGUARD_DEADLINE_AWAKE"))
    deadline = float(os.environ.get("MEMGUARD_DEADLINE_AWAKE" if aw else "MEMGUARD_DEADLINE", "0") or 0) - 20.0
    clock = awake_clock if aw else time.time
    stop = dict(why=None, child=None)

    def on_term(sig, _frm):
        stop["why"] = stop["why"] or f"stopped by signal {signal.Signals(sig).name} (memguard timeout or memory cap, or a kill)"
        raise StopRun()
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    def spawn(extra, tmp):
        """Run a child of this script; (exit code, its own peak GiB, tree peak GiB seen while it ran). Never raises except StopRun."""
        tmp.unlink(missing_ok=True)
        p = subprocess.Popen(me + argv + extra)
        stop["child"] = p
        tree = 0.0
        while True:
            pid, status, ru = os.wait4(p.pid, os.WNOHANG)
            if pid:
                break
            if deadline > 0 and clock() > deadline and not stop["why"]:
                stop["why"] = "stopped 20 s before the memguard timeout"
                raise StopRun()
            t, par = tree_rss_gib(os.getpid())
            tree, peaks["tree"], peaks["parent"] = max(tree, t), max(peaks["tree"], t), max(peaks["parent"], par)
            time.sleep(0.5)
        stop["child"] = None
        return os.waitstatus_to_exitcode(status), gib_of_maxrss(ru), round(tree, 2)

    plan_f = out_dir / f"plan-{os.getpid()}.json"
    try:
        code, _, _ = spawn(["--_plan", str(plan_f)], plan_f)
    except StopRun:
        end_child(stop["child"], None)[0]
        sys.exit(f"run.py: {stop['why']} while planning the scope")
    if code or not plan_f.exists():
        sys.exit(code or 1)
    pl = json.loads(plan_f.read_text())
    plan_f.unlink()
    if pl.get("nothing_changed"):
        return
    names, run = pl["names"], pl["run"]
    fresh = pl.get("manifest_freshness") or []
    for x in fresh:
        if x["kind"] == "recorded source changed":
            print(f"manifest freshness: {x['field']} = {x['path']} has changed since this manifest was built "
                  f"(recorded {x['recorded_sha256']}, now {x['now_sha256']}, file of {x['changed_at']})", file=sys.stderr, flush=True)
        elif x["kind"] == "upstream manifest written later":
            print(f"manifest freshness: upstream {x['path']} ({', '.join(x['by_generators'])}) was written {x['later_by_s'] // 60} min "
                  f"after the checked manifest ({x['written_at']}): the checked body may lack its changes; rerun the generators after it "
                  "(robot/body/bom/rebuild_chain.py --dry-run lists the order)", file=sys.stderr, flush=True)
        elif x["kind"] == "generators out of date":
            print(f"OUT OF DATE: {len(x['generators'])} generator(s) behind {x['path']} have changed inputs, so this checks an older "
                  f"body: {', '.join(f'{g} ({w})' for g, w in list(x['why'].items())[:6])}{' ...' if len(x['why']) > 6 else ''}; "
                  f"build it with `{x['command']}` (cached outputs are reused)", file=sys.stderr, flush=True)
    cached = {n: c for n, c in pl.get("cached", {}).items() if n in run}
    if cached:
        print(f"cached (inputs unchanged): " + ", ".join(f"{n} (from {c['at'][11:16]}, {c['inputs']} inputs)" for n, c in cached.items())
              + (f"; rerun: {', '.join(f'{n} ({w})' for n, w in pl.get('cache_misses', {}).items())}" if pl.get("cache_misses") else ""),
              file=sys.stderr, flush=True)
    fem_cached = pl.get("fem_cached", {})
    est = {n: e for n, e in pl.get("estimate", {}).items() if n not in pl.get("cached", {})}
    tot, p90 = sum(e["s"] for e in est.values()), sum(e["p90_s"] for e in est.values())
    heavy = [n for n in run if n in HEAVY]
    nohist = [n for n in run if n not in est and n not in cached]
    if run and len(cached) == len(run) and "fem" not in run:
        cost = "all cached"
    else:
        cost = f"earlier runs suggest about {tot:.0f} s (p90 {p90:.0f} s)" + (f", no history for {', '.join(nohist)}" if nohist else "")
    msg = f"plan: {len(names)} part(s), {len(run)} check(s){' (heavy: ' + ', '.join(heavy) + ')' if heavy else ''}; " + cost
    if deadline > 0:
        left = deadline - clock()
        msg += f"; {left:.0f} s left before memguard stops it"
        if p90 > left:
            acc, late = 0.0, []
            for n in [x for x in run if x not in HEAVY] + heavy:      # the order the groups run in
                acc += est.get(n, {}).get("p90_s", 0.0)
                if acc > left:
                    late.append(n)
            msg += f" (may not reach: {', '.join(late)})"
    if not split(a.checks):
        msg += "; no --checks given = every check that applies (presets: " + ", ".join(PRESETS) + ", all)"
    print(msg, file=sys.stderr, flush=True)
    # complete stays false until the last write: a killed or still running run's report reads as unfinished
    rep = dict(complete=False, label=label, manifest=pl["manifest"], manifest_sha256=pl["manifest_sha256"], scope=dict(selectors=pl["desc"], parts=names),
               pose_set=a.poses, process_override=a.process, checks={}, not_applicable=pl["not_applicable"], solids={}, other_solids={}, part_settings=pl["settings"])
    rep["manifest_freshness"] = pl.get("manifest_freshness") or []
    if pl.get("checks_left_out"):
        rep["checks_left_out"] = pl["checks_left_out"]
    f = out_dir / "report.json"
    secs, setup_secs, n_children = {}, {}, [0]      # setup timers are repeated in every child, so they are summed apart from the check timers

    defs = pl.get("definitions", {})
    rep["definitions"] = {n: d for n, d in defs.items() if n in set(names)}

    def write(running=None):
        rep["checks"] = {n: rep["checks"][n] for n in pl["todo"] if n in rep["checks"]}
        for r in rep["checks"].values():          # every finding names the definition file and entry of its part
            for c in r.get("worst_cases", []) if isinstance(r, dict) else []:
                d = defs.get(c.get("part")) or {}
                if d.get("file") and "definition" not in c:
                    c["definition"] = f"{d['file']} ({d['entry']})"
        rep["seconds"] = dict(secs, total=round(time.time() - t_all, 1))
        slept = (time.time() - t_all) - (awake_clock() - m_all)
        rep["slept_s"] = round(slept, 1) if slept > 5 else 0.0      # check / step seconds include it: a long step with slept_s > 0 is sleep, not work
        rep["setup_seconds_summed_over_children"] = dict(setup_secs, children=n_children[0])
        rep["peak_tree_gib"] = round(peaks["tree"], 2)
        rep["peak_parent_gib"] = round(peaks["parent"], 2)
        if running:
            rep["incomplete_running"] = running
        else:
            rep.pop("incomplete_running", None)
        tmp = f.with_suffix(".json.part")
        tmp.write_text(json.dumps(rep, indent=1, default=float) + "\n")
        tmp.replace(f)

    def absorb(got):
        rep["solids"].update(got["solids"]); rep["other_solids"].update(got["other_solids"]); rep.setdefault("solid_volume_off_over_5pct", {}).update(got.get("solid_volume_off_over_5pct", {}))
        n_children[0] += 1
        for k, v in got["seconds"].items():
            d = setup_secs if k in SETUP_TIMERS else secs
            d[k] = round(d.get(k, 0.0) + v, 2)

    def read_child(tmp):
        try:
            return json.loads(tmp.read_text()) if tmp.exists() else None
        except ValueError:
            return None

    def child(grp, scope, tag):
        """Run checks grp on scope in one child; {check: result}, failures as `failed` results. When the child dies (a crash in
        native code, a memory cap of its own), the check it was on is marked failed and the group's remaining checks run in a
        new child, so one crash costs one check."""
        out, todo = {}, list(grp)
        while todo:
            tmp = out_dir / f"child-{tag}-{os.getpid()}.json"
            try:
                code, own, tree = spawn(["--_scope", ",".join(scope), "--_out", str(tmp), "--checks", ",".join(todo)], tmp)
            except StopRun:
                got = read_child(tmp) or dict(checks={}, solids={}, other_solids={}, seconds={})
                absorb(got); out.update(got["checks"])
                on = next((n for n in todo if n not in got["checks"]), None)
                stack, cpu = end_child(stop["child"], Path(str(tmp) + ".stack"))
                if on:
                    done_cpu = sum(v.get("cpu_s") or 0.0 for v in got["checks"].values())
                    out[on] = dict(failed=f"{stop['why']} after {round(time.time() - t_run[0], 1)} s in this check", stopped_in=stack,
                                   seconds=round(time.time() - t_run[0], 1), cpu_s=None if cpu is None else round(max(0.0, cpu - done_cpu), 1),
                                   per_part={}, worst_cases=[], bad_events={}, evidence=[], notes="")
                rep["checks"].update(out)
                raise
            got = read_child(tmp)
            tmp_seen = got is not None
            got = got or dict(checks={}, solids={}, other_solids={}, seconds={})
            for x in (tmp, Path(str(tmp) + ".part")):
                x.unlink(missing_ok=True)
            stack_f = Path(str(tmp) + ".stack")
            stack = stack_f.read_text()[-3000:] if stack_f.exists() else None
            stack_f.unlink(missing_ok=True)
            absorb(got)
            for n in todo:
                if n in got["checks"]:
                    got["checks"][n].setdefault("peak_gib", own)
                    got["checks"][n]["tree_peak_gib"] = tree
                    out[n] = got["checks"][n]
            left = [n for n in todo if n not in got["checks"]]
            if left and not tmp_seen:          # died before its first check (manifest, imports, solids): the same for every check
                for n in left:
                    out[n] = dict(failed=f"child process ended with exit status {code} before its first check (setup)", stopped_in=stack,
                                  seconds=None, peak_gib=own, tree_peak_gib=tree, per_part={}, worst_cases=[], bad_events={}, evidence=[], notes="")
                break
            if left:
                out[left[0]] = dict(failed=f"child process ended with exit status {code} (peak {own} GiB) during this check", stopped_in=stack,
                                    seconds=None, peak_gib=own, tree_peak_gib=tree, per_part={}, worst_cases=[], bad_events={}, evidence=[], notes="")
                print(f"[{left[0]}] child process ended with exit status {code}; {len(left) - 1} check(s) of the group go on in a new one",
                      file=sys.stderr, flush=True)
            todo = left[1:]
        return out

    t_run = [time.time()]
    # light group first, then heavies in order; every finished check is written to the report at once
    for n, c in cached.items():                 # results whose inputs are unchanged: not run again
        rep["checks"][n] = dict(c["result"], cached=dict(from_run_at=c["at"], label=c["label"], inputs=c["inputs"]))
    todo_run = [n for n in run if n not in cached]
    groups = [[n for n in todo_run if n not in HEAVY]] + [[n] for n in todo_run if n in HEAVY]
    try:
        for grp in [g for g in groups if g]:
            write(running=grp)
            t_run[0] = time.time()
            if grp == ["fem"]:
                t0, per = time.time(), {}
                for n in pl["fem_parts"]:
                    if n in fem_cached:
                        per[n] = dict(fem_cached[n]["result"], cached=dict(from_run_at=fem_cached[n]["at"], label=fem_cached[n]["label"]))
                        print(f"[fem] {n}: cached (inputs unchanged since {fem_cached[n]['at']})", file=sys.stderr, flush=True)
                    else:
                        per[n] = child(["fem"], [n], f"fem-{n}")["fem"]
                    m = merge_fem(per)
                    m.update(seconds=round(time.time() - t0, 1), peak_gib=max(r.get("peak_gib") or 0 for r in per.values()),
                             tree_peak_gib=max(r.get("tree_peak_gib") or 0 for r in per.values()))
                    rep["checks"]["fem"] = m
                    write(running=grp)
            else:
                rep["checks"].update(child(grp, names, grp[0]))
            write()
    except StopRun:
        rep["stopped"] = stop["why"]
        rep["not_started"] = [n for n in run if n not in rep["checks"]]
        write(running=[n for n in run if n not in rep["checks"] or rep["checks"][n].get("stopped_in") is not None])
        summary(rep)
        print(f"\n{f.relative_to(ROOT)}")
        sys.exit(124)
    failed = {n: r["failed"] for n, r in rep["checks"].items() if r.get("failed")}
    bad_parts = set()
    for n, r in rep["checks"].items():
        if r.get("failed_parts"):
            failed[n] = r["failed_parts"]
            bad_parts |= set(r["failed_parts"])
    if failed:
        rep["failed_checks"] = failed
    if not any(r.get("failed") for r in rep["checks"].values()):
        # a run with a failed check does not mark its parts as seen; parts a check failed on stay unseen too
        lf = Path(pl["last_file"])
        prev = json.loads(lf.read_text()) if lf.exists() else {}
        prev.update({k: v for k, v in pl["keys"].items() if k not in bad_parts})
        lf.parent.mkdir(parents=True, exist_ok=True)
        lf.write_text(json.dumps(prev))
    rep["complete"] = True
    write()
    summary(rep)
    print(f"\n{f.relative_to(ROOT)}")


class StopRun(Exception):
    """The run is stopped (memguard deadline, SIGTERM, Ctrl-C): write what finished and the summary, then exit 124."""


def end_child(p, stack_file):
    """Stop a running child: ask for its Python stack (SIGUSR1, written by faulthandler), then end it. (stack text or None,
    CPU seconds the child used in all, or None when it was already reaped)."""
    import signal
    stack, cpu = None, None
    if p is not None:
        if p.poll() is None:
            try:
                p.send_signal(signal.SIGUSR1)
                time.sleep(0.7)
            except OSError:
                pass
            p.kill()
        try:
            _, _, ru = os.wait4(p.pid, 0)
            cpu = ru.ru_utime + ru.ru_stime
        except ChildProcessError:
            pass
    if stack_file is not None and stack_file.exists():
        stack = stack_file.read_text()[-3000:] or None
    return stack, cpu


def hold_label(out_dir, label):
    """An exclusive lock on the label's folder for this run: a second run under the same label (another session, the same
    part without --label) stops at once and names the run that holds it, instead of deleting that run's report."""
    import fcntl
    lf = open(out_dir / ".lock", "a+")
    try:
        fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lf.seek(0)
        sys.exit(f"run.py: label {label!r} is in use by another run ({lf.read().strip() or 'unknown'}); "
                 "wait for it or pass another --label")
    lf.seek(0); lf.truncate()
    lf.write(f"pid {os.getpid()} since {time.strftime('%H:%M:%S')}, cwd {os.getcwd()}: {' '.join(sys.argv[1:])[:200]}")
    lf.flush()
    return lf


def keep_awake():
    """macOS: hold an idle-sleep assertion while this run lives (caffeinate exits with this process). The laptop slept twice
    during the 2026-11-22 whole-body runs and the part each job was on carried 800-1000 s (docs/research/design-checks.md)."""
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def summary(rep, top=6):
    sc = rep["scope"]
    hows = {**{n: v.get("how", "") for n, v in rep.get("solids", {}).items()}, **rep.get("other_solids", {})}
    approximate = {n: ("voxel fill of an open mesh" if "voxel" in h else "convex hull of an open mesh")
                   for n, h in hows.items() if "voxel" in h or "hull" in h}
    print(f"manifest {rep['manifest']} ({rep['manifest_sha256'][:12]}), scope {sc['selectors']}: {len(sc['parts'])} parts"
          + (f" ({', '.join(sc['parts'][:6])}{' ...' if len(sc['parts']) > 6 else ''})"))
    if rep.get("slept_s"):
        print(f"  the laptop slept {rep['slept_s']:.0f} s during this run: check and step seconds include it (memguard's timeout does not)")
    odd = sorted(n for n, h in rep["other_solids"].items() if "hull" in h)
    if odd:
        print(f"  neighbours screened as convex hulls (over-state overlap): {len(odd)}")
    off = rep.get("solid_volume_off_over_5pct") or {}
    if off:
        print(f"  screening solids more than 5 % off the recorded part volume (overlaps / distances with them are approximate): "
              + ", ".join(f"{n} {v['diff_pct']:+.0f} % ({v['how']})" for n, v in sorted(off.items())))
    # screening solids not made as asked (core.screen_solid / _step_how name it in the solid's description)
    import re
    coarse = {n: m.group(1) for n, h in hows.items() if (m := re.search(r"\[tolerance (\S+ mm, coarser than the \S+ mm asked)", h))}
    if coarse:
        print(f"  STEP tessellated coarser than asked: {len(coarse)}: " + ", ".join(f"{n} ({v})" for n, v in sorted(coarse.items())[:8])
              + (" ..." if len(coarse) > 8 else ""))
    nomesh = sorted(n for n, h in hows.items() if "resolves nowhere: this STEP" in h)
    if nomesh:
        print(f"  mesh file missing, screened by the STEP alone (may not match the manifest's mesh): {', '.join(nomesh)}")
    moved = {n: m.group(1) for n, h in hows.items() if (m := re.search(r"\[STEP read from (\S+), not", h))}
    if moved:
        print(f"  STEP read from another path than the manifest names: " + ", ".join(f"{n} ({v})" for n, v in sorted(moved.items())))
    unused = sorted(n for n, h in hows.items() if "STEP not used" in h)
    if unused:
        print(f"  STEP not used as the screening solid (why in the solid's description): {len(unused)}: {', '.join(unused[:8])}"
              + (" ..." if len(unused) > 8 else ""))
    for name, r in rep["checks"].items():
        waited = (r.get("seconds") or 0) > 30 and r.get("cpu_s") is not None and r["cpu_s"] < 0.5 * r["seconds"]
        cpu_note = f" (CPU {r['cpu_s']} s of {r['seconds']} s: the process waited, starved or blocked, rather than computed)" if waited else ""
        if r.get("failed"):
            print(f"[{name}] FAILED: {r['failed']}{cpu_note}")
            if r.get("stopped_in"):
                import re
                frames = re.findall(r'File "(.+?)", line (\d+) in (\S+)', r["stopped_in"])   # faulthandler: innermost first
                print("  stopped in: " + " <- ".join(f"{fn} ({Path(p).name}:{ln})" for p, ln, fn in frames[:5]))
            continue
        by_why = {}
        for n, why in (r.get("failed_parts") or {}).items():
            by_why.setdefault(why, []).append(n)
        for why, ns in by_why.items():
            print(f"[{name}] FAILED for {len(ns)} part(s) ({', '.join(ns[:4])}{' ...' if len(ns) > 4 else ''}): {why}")
        bad = ", ".join(f"{k} {v}" for k, v in r["bad_events"].items())
        ca = r.get("cached")
        how = (f"cached: inputs unchanged since {ca['from_run_at']} (label {ca['label']}, ran {r['seconds']} s)" if ca
               else f"{r['seconds']} s, peak {r.get('peak_gib', '?')} GiB (tree {r.get('tree_peak_gib', '?')})")
        print(f"[{name}] {how}{cpu_note}; counts: {bad or '-'}")
        # standing 0 mm gaps without overlap are contact by design (shell and sleeve seats): counted, kept in the report only
        standing0 = [c for c in r["worst_cases"] if c["unit"].endswith("distance") and c.get("pose") == "standing" and c["value"] == 0
                    and not c.get("overlap_mm3")]
        context = [c for c in r["worst_cases"] if c.get("context")]       # per-part measurements without a finding (bed margin of a part that fits ...)
        shown = [c for c in r["worst_cases"] if c not in standing0 and c not in context]
        if context:
            print(f"  ({len(context)} context lines without a finding not listed; see report)")
        if standing0:
            print(f"  ({len(standing0)} standing 0 mm gap pairs without overlap not listed; see report)")
        for c in shown[:top]:
            loc = f" at {c['location_mm']}" if c.get("location_mm") else ""
            mate = f" x {c['mating_part']}" if c.get("mating_part") else ""
            pose = f" @ {c['pose']}" if c.get("pose") else ""
            # an overlap or distance against a voxel-filled / convex-hull screening solid is approximate: say so on the line
            approx = [f"{n}: {approximate[n]}" for n in (c.get("part"), c.get("mating_part")) if n in approximate]
            note = f"  [approximate: {'; '.join(approx)}]" if approx and str(c.get("unit", "")).startswith(("mm3", "mm min", "mm distance")) else ""
            print(f"  {c['part']}{mate}: {c['value']:g} {c['unit']} ({c['what']}){pose}{loc}{note}")
        for ev in r["evidence"]:
            print(f"  evidence: {ev}")
    if rep["not_applicable"]:
        print("not applicable to this scope:", ", ".join(rep["not_applicable"]))
    if not rep.get("complete", True) or rep.get("incomplete_running"):
        print("RUN DID NOT FINISH" + (f" ({rep['stopped']})" if rep.get("stopped") else "") + "; was running:", ", ".join(rep.get("incomplete_running") or ["?"])
              + (f"; not started: {', '.join(rep['not_started'])}" if rep.get("not_started") else ""))
    print(f"memory: tree peak {rep.get('peak_tree_gib')} GiB, orchestrating parent {rep.get('peak_parent_gib')} GiB (memguard cap 6)")
    print("seconds:", rep["seconds"])
    su = rep.get("setup_seconds_summed_over_children")
    if su:
        print(f"setup seconds, summed over {su['children']} children (each child repeats them): { {k: v for k, v in su.items() if k != 'children'} }")


if __name__ == "__main__":
    main()
