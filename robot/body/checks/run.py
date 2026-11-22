"""Shared design check for the printed body. One report per run, a worst-first summary on stdout. No pass/fail.

  python3 infra/tools/memguard.py --limit-gb 6 --timeout-s 900 -- .venv/bin/python -B robot/body/checks/run.py \
      [--manifest PATH  (default: body_manifest in robot/design.json)] \
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
    big_bought = p.get("role") not in core.STEP_ROLES and p.get("native_step") and core._mb(p["native_step"]) > core.FINE_STEP_MAX_MB
    if n in ctx.scope and p.get("native_step") and not p.get("placement_offset_mm") and (p.get("native_solid_count", 1) > 1 or not rec) and not big_bought:
        # scoped parts only: a neighbour's fused multi-solid STEP (battery cells, supplier boards) costs minutes to
        # tessellate just for this comparison; neighbours use their own closed mesh below
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


def run_checks_here(ctx, checks, names, out_file):
    """Child mode: run the named checks in this process; the result file is rewritten after each check."""
    import resource
    from checks.core import T
    if os.environ.get("CHECKS_STACK_DUMP_S"):        # diagnosing a stall: the child prints its Python stack every N s
        import faulthandler
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
        Path(out_file).write_text(json.dumps(out, default=float))
    for name in names:
        t0 = time.time()
        try:
            res = checks[name]["run"](ctx)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            res = dict(failed=f"{type(exc).__name__}: {exc}")
        res["seconds"] = round(time.time() - t0, 1)
        res["peak_gib"] = gib_of_maxrss(resource.getrusage(resource.RUSAGE_SELF))
        out["checks"][name] = res
        dump()


SETUP_TIMERS = ("solids (scope)", "solids (others)", "neighbours")     # every child builds its own solids and neighbour list


def child_ctx(a, man, split):
    """Heavy side, children only: the Ctx of this run and the registered checks."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from checks import registry
    from checks.core import WORK as W
    from checks.ctx import Ctx
    checks = registry.load_all()
    label = a.label or (a.section or (split(a.parts) or ["all"])[0]).replace(" ", "-").replace("/", "")[:40]
    out_dir = W / "checks" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(man, [], a.poses, a.search_mm, a.exact_press, a.render_timeout, out_dir, a.include_shells)
    ctx.fem_variant, ctx.fem_mesh_h, ctx.fem_rigid_load = a.fem_variant, a.fem_mesh_h, a.fem_rigid_load
    ctx.joints, ctx.step_deg = split(a.joints), a.step_deg
    ctx.fit_pair_budget_s = a.fit_pair_budget_s
    ctx.held = {k: float(v) for k, v in (w.split("=") for w in a.hold)}
    from checks.process import parse_override
    ctx.process_override = parse_override(a.process)
    return ctx, checks, out_dir


def make_plan(a, man, split):
    """Plan child: resolve the scope, which checks apply, the FEM parts, the last-run keys; write them to a JSON file."""
    ctx, checks, out_dir = child_ctx(a, man, split)
    from checks import scope as S
    from checks import fem_core as FC
    from checks.core import save_sha, sha
    names, desc = S.select(ctx.body, man, split(a.parts), a.section, split(a.link), split(a.role), a.changed_since, ctx.neighbours_of)
    if not names:
        sys.exit(f"scope {desc} selects no part of {a.manifest}")
    ctx.scope = names
    want = split(a.checks)
    unknown = [c for c in want if c not in checks]
    if unknown:
        sys.exit(f"unknown checks {unknown}; known {list(checks)}")
    todo = [n for n in (want or checks) if n in checks]
    run = [n for n in todo if checks[n]["applies"](ctx)]
    nbs = list(ctx.neighbours())
    plan = dict(manifest=str(man.relative_to(ROOT)), manifest_sha256=sha(man), names=names, desc=desc, todo=todo, run=run,
               not_applicable=[n for n in todo if n not in run], settings={n: ctx.settings(n) for n in names},
               definitions={n: dict(file=ctx.definition(n)[0], entry=ctx.definition(n)[1], settings_source=ctx.settings_source.get(n, {})) for n in ctx.body.parts},
               out_dir=str(out_dir),
               fem_parts=[n for n in names if FC.merged_setup(ctx, n, a.fem_variant)] if "fem" in run else [],
               keys={n: S.part_key(ctx.body.parts[n]) for n in names + nbs}, last_file=str(S.LAST / (man.stem + ".json")))
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
    ap.add_argument("--manifest", help="default: body_manifest in robot/design.json (the selected body)")
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
    ap.add_argument("--fem-variant", help="named variant of the per-part FEM setups (e.g. legacy_coordinates)")
    ap.add_argument("--fem-mesh-h", type=float, help="mesh size h (mm) replacing mesh_h_mm of every FEM setup (mesh convergence)")
    ap.add_argument("--fem-rigid-load", action="store_true", help="make the load patch of every FEM setup rigid (comparison run)")
    ap.add_argument("--process", action="append", default=[], help="KEY=VALUE replacing the print setup of every printed part in scope "
                    "(cross-evaluation, not stored): nozzle_mm, layer_height_mm, line_width_mm, material, printer, up, plate_yaw_deg")
    ap.add_argument("--label")
    ap.add_argument("--_scope", help=argparse.SUPPRESS)   # child mode: exact scoped part names
    ap.add_argument("--_out", help=argparse.SUPPRESS)     # child mode: result file
    ap.add_argument("--_plan", help=argparse.SUPPRESS)    # child mode: write the plan (scope, applicable checks) to this file
    ap.add_argument("--list-checks", action="store_true")
    a = ap.parse_args(argv)
    t_all = time.time()
    split = lambda s: [x for x in (s or "").split(",") if x]  # noqa: E731
    if a.list_checks:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from checks import registry
        for n, c in registry.load_all().items():
            print(f"{n:10s} {c['doc']}")
        return
    if not a.manifest:
        a.manifest = json.loads((ROOT / "robot/design.json").read_text())["body_manifest"]
        argv += ["--manifest", a.manifest]   # children get it explicitly
    man = (ROOT / a.manifest).resolve()
    if a._plan:
        return make_plan(a, man, split)
    if a._out:                           # child: run the named checks on the exact scope
        ctx, checks, _ = child_ctx(a, man, split)
        ctx.scope = split(a._scope)
        run_checks_here(ctx, checks, split(a.checks), a._out)
        return
    me = [sys.executable, "-B", str(Path(__file__).resolve())]
    label = a.label or (a.section or (split(a.parts) or ["all"])[0]).replace(" ", "-").replace("/", "")[:40]
    out_dir = WORK / "checks" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").unlink(missing_ok=True)     # a previous run's report under this label must not pass for this run's
    peaks = dict(tree=0.0, parent=0.0)

    def spawn(extra, tmp):
        """Run a child of this script; (exit code, its own peak GiB, tree peak GiB seen while it ran). Never raises."""
        tmp.unlink(missing_ok=True)
        p = subprocess.Popen(me + argv + extra)
        tree = 0.0
        while True:
            pid, status, ru = os.wait4(p.pid, os.WNOHANG)
            if pid:
                break
            t, par = tree_rss_gib(os.getpid())
            tree, peaks["tree"], peaks["parent"] = max(tree, t), max(peaks["tree"], t), max(peaks["parent"], par)
            time.sleep(0.5)
        return os.waitstatus_to_exitcode(status), gib_of_maxrss(ru), round(tree, 2)

    plan_f = out_dir / "plan.json"
    code, _, _ = spawn(["--_plan", str(plan_f)], plan_f)
    if code or not plan_f.exists():
        sys.exit(code or 1)
    pl = json.loads(plan_f.read_text())
    plan_f.unlink()
    names, run = pl["names"], pl["run"]
    # complete stays false until the last write: a killed or still running run's report reads as unfinished
    rep = dict(complete=False, manifest=pl["manifest"], manifest_sha256=pl["manifest_sha256"], scope=dict(selectors=pl["desc"], parts=names),
               pose_set=a.poses, process_override=a.process, checks={}, not_applicable=pl["not_applicable"], solids={}, other_solids={}, part_settings=pl["settings"])
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
        rep["setup_seconds_summed_over_children"] = dict(setup_secs, children=n_children[0])
        rep["peak_tree_gib"] = round(peaks["tree"], 2)
        rep["peak_parent_gib"] = round(peaks["parent"], 2)
        if running:
            rep["incomplete_running"] = running
        else:
            rep.pop("incomplete_running", None)
        f.write_text(json.dumps(rep, indent=1, default=float) + "\n")

    def absorb(got):
        rep["solids"].update(got["solids"]); rep["other_solids"].update(got["other_solids"]); rep.setdefault("solid_volume_off_over_5pct", {}).update(got.get("solid_volume_off_over_5pct", {}))
        n_children[0] += 1
        for k, v in got["seconds"].items():
            d = setup_secs if k in SETUP_TIMERS else secs
            d[k] = round(d.get(k, 0.0) + v, 2)

    def child(grp, scope, tag):
        """Run checks grp on scope in one child; {check: result}, failures as `failed` results."""
        tmp = out_dir / f"child-{tag}.json"
        code, own, tree = spawn(["--_scope", ",".join(scope), "--_out", str(tmp), "--checks", ",".join(grp)], tmp)
        got = json.loads(tmp.read_text()) if tmp.exists() else dict(checks={}, solids={}, other_solids={}, seconds={})
        tmp.unlink(missing_ok=True)
        absorb(got)
        for n in grp:
            if n not in got["checks"]:
                got["checks"][n] = dict(failed=f"child process ended with exit status {code} (peak {own} GiB) before finishing",
                                        seconds=None, peak_gib=own, per_part={}, worst_cases=[], bad_events={}, evidence=[], notes="")
            got["checks"][n].setdefault("peak_gib", own)
            got["checks"][n]["tree_peak_gib"] = tree
        return got["checks"]

    # light group first, then heavies in order; every finished check is written to the report at once
    groups = [[n for n in run if n not in HEAVY]] + [[n] for n in run if n in HEAVY]
    for grp in [g for g in groups if g]:
        write(running=grp)
        if grp == ["fem"]:
            t0, per = time.time(), {}
            for n in pl["fem_parts"]:
                per[n] = child(["fem"], [n], f"fem-{n}")["fem"]
                m = merge_fem(per)
                m.update(seconds=round(time.time() - t0, 1), peak_gib=max(r["peak_gib"] for r in per.values()),
                         tree_peak_gib=max(r["tree_peak_gib"] for r in per.values()))
                rep["checks"]["fem"] = m
                write(running=grp)
        else:
            rep["checks"].update(child(grp, names, grp[0]))
        write()
    failed = {n: r["failed"] for n, r in rep["checks"].items() if r.get("failed")}
    for n, r in rep["checks"].items():
        if r.get("failed_parts"):
            failed[n] = r["failed_parts"]
    if failed:
        rep["failed_checks"] = failed
    else:   # a run with a failed check does not mark its parts as seen
        lf = Path(pl["last_file"])
        prev = json.loads(lf.read_text()) if lf.exists() else {}
        prev.update(pl["keys"])
        lf.parent.mkdir(parents=True, exist_ok=True)
        lf.write_text(json.dumps(prev))
    rep["complete"] = True
    write()
    summary(rep)
    print(f"\n{f.relative_to(ROOT)}")


def summary(rep, top=6):
    sc = rep["scope"]
    print(f"manifest {rep['manifest']} ({rep['manifest_sha256'][:12]}), scope {sc['selectors']}: {len(sc['parts'])} parts"
          + (f" ({', '.join(sc['parts'][:6])}{' ...' if len(sc['parts']) > 6 else ''})"))
    odd = sorted(n for n, h in rep["other_solids"].items() if "hull" in h)
    if odd:
        print(f"  neighbours screened as convex hulls (over-state overlap): {len(odd)}")
    off = rep.get("solid_volume_off_over_5pct") or {}
    if off:
        print(f"  screening solids more than 5 % off the recorded part volume (overlaps / distances with them are approximate): "
              + ", ".join(f"{n} {v['diff_pct']:+.0f} % ({v['how']})" for n, v in sorted(off.items())))
    for name, r in rep["checks"].items():
        if r.get("failed"):
            print(f"[{name}] FAILED: {r['failed']}")
            continue
        for n, why in (r.get("failed_parts") or {}).items():
            print(f"[{name}] FAILED for {n}: {why}")
        bad = ", ".join(f"{k} {v}" for k, v in r["bad_events"].items())
        print(f"[{name}] {r['seconds']} s, peak {r.get('peak_gib', '?')} GiB (tree {r.get('tree_peak_gib', '?')}); counts: {bad or '-'}")
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
            print(f"  {c['part']}{mate}: {c['value']:g} {c['unit']} ({c['what']}){pose}{loc}")
        for ev in r["evidence"]:
            print(f"  evidence: {ev}")
    if rep["not_applicable"]:
        print("not applicable to this scope:", ", ".join(rep["not_applicable"]))
    if not rep.get("complete", True) or rep.get("incomplete_running"):
        print("RUN DID NOT FINISH; was running:", ", ".join(rep.get("incomplete_running") or ["?"]))
    print(f"memory: tree peak {rep.get('peak_tree_gib')} GiB, orchestrating parent {rep.get('peak_parent_gib')} GiB (memguard cap 6)")
    print("seconds:", rep["seconds"])
    su = rep.get("setup_seconds_summed_over_children")
    if su:
        print(f"setup seconds, summed over {su['children']} children (each child repeats them): { {k: v for k, v in su.items() if k != 'children'} }")


if __name__ == "__main__":
    main()
