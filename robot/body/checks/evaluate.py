"""Evaluate the design check tool itself (robot/body/checks/run.py) on the selected robot body: how fast it answers the
questions sessions actually ask, whether it finishes when inputs are broken, and whether its numbers match ground truth.

  .venv/bin/python -B robot/body/checks/evaluate.py [--only ID,ID] [--group speed|robust|truth] [--list] [--keep]

Every scenario runs the real run.py under memguard on the real selected manifest or on a copy of it with one injected
fault (a part moved into its neighbour, a STEP deleted or truncated, a mesh missing ...). The copies and their part files
live in .cache/work/cad-speed/eval/<id>/ and are deleted afterwards unless --keep. Ground truth for injected
interferences is computed here independently of run.py: exact OpenCascade common volume / distance of the native STEP
solids at the standing pose. Results: .cache/work/cad-speed/eval/results.json and a table on stdout: per scenario wall
seconds (memguard slot wait apart), exit code, report complete, checks failed, and the scenario's own measurements
(e.g. overlap reported vs truth). Measurements only; the reader judges them."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / ".cache/work/cad-speed/eval"
REPORTS = ROOT / ".cache/work/cad-speed/checks"
PY = str(ROOT / ".venv/bin/python")
RUN = str(ROOT / "robot/body/checks/run.py")
MEMGUARD = str(ROOT / "infra/tools/memguard.py")


def selected_manifest() -> Path:
    return (ROOT / json.loads((ROOT / "robot/design.json").read_text())["body_manifest"]).resolve()


# ------------------------------------------------------------------------------------------------ manifest mutations
def copy_manifest(sid, edit):
    """A copy of the selected manifest with edit(manifest_dict, dir) applied, written to EVAL/sid/manifest-<sid>.json."""
    d = EVAL / sid
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    man = json.loads(selected_manifest().read_text())
    edit(man, d)
    f = d / f"manifest-{sid}.json"
    f.write_text(json.dumps(man))
    return f


def part(man, name):
    return next(p for p in man["parts"] if p["name"] == name)


def rel(p: Path) -> str:
    return str(p.resolve().relative_to(ROOT))


def translate_part(man, d, name, vec):
    """Regenerate part `name` moved by vec (mm, manifest frame): its STEP and mesh rewritten into d (new sha256, as a
    designer's rebuild would), bounds moved. Returns the new STEP path."""
    import numpy as np
    import trimesh
    from build123d import Location, export_step, import_step
    p = part(man, name)
    from build123d import Compound
    sh = import_step(str(ROOT / p["native_step"]))
    off = p.get("native_step_offset_mm") or [0, 0, 0]
    loc = Location(tuple(float(a + b) for a, b in zip(off, vec)))
    sh = Compound(children=[s.moved(loc) for s in sh.solids()])     # a moved imported compound will not export; its solids do
    p.pop("native_step_offset_mm", None)
    st = d / f"{name}.step"
    export_step(sh, str(st))
    m = trimesh.load(ROOT / p["mesh"], force="mesh")
    m.apply_translation(vec)
    ms = d / f"{name}.stl"
    m.export(ms)
    p["native_step"], p["mesh"] = rel(st), rel(ms)
    p.pop("native_step_sha256", None); p.pop("mesh_sha256", None)
    p["bounds_mm"] = (np.asarray(p["bounds_mm"], float) + np.asarray(vec, float)).tolist()
    return st


def truth_pair(man_path, a, b):
    """Exact OCC (common volume mm3, min distance mm) of parts a and b at the standing pose, from their native STEPs."""
    sys.path.insert(0, str(ROOT / "robot/body"))
    from checks import brep as L
    from checks import core
    body = core.load_body(Path(man_path))
    from build123d import Compound, Location, import_step
    sh = {}
    for n in (a, b):
        p = body.parts[n]
        s = import_step(str(ROOT / p["native_step"])).moved(Location(tuple(float(v) for v in (p.get("native_step_offset_mm") or (0, 0, 0)))))
        T = body.link_T(p["owning_link"], body.standing)
        sh[n] = L.transform_shape(Compound(children=list(s.solids())), T)
    return round(L.common_volume_mm3(sh[a], sh[b]), 2), round(L.min_distance_mm(sh[a], sh[b]), 3)


# ------------------------------------------------------------------------------------------------ running run.py
def run_tool(label, args, manifest=None, limit_gb=4, timeout_s=900, env=None, no_wait=False, cache=False):
    """run.py under memguard; dict(rc, wall_s, wait_s, exec_s, report, log). cache=False passes --no-cache: the scenarios
    measure what the checks compute; the cache scenarios turn it on."""
    cmd = ["python3", MEMGUARD] + (["--no-wait"] if no_wait else []) + ["--limit-gb", str(limit_gb), "--timeout-s", str(timeout_s), "--", PY, "-B", RUN, "--label", label]
    if manifest:
        cmd += ["--manifest", rel(Path(manifest))]
    cmd += args + ([] if cache else ["--no-cache"])
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    wall = time.time() - t0
    log = p.stdout + p.stderr
    m = re.search(r"memguard: slot after (\d+) s wait", log)
    wait = float(m.group(1)) if m else 0.0
    f = REPORTS / label / "report.json"
    rep = None
    if f.exists():
        try:
            rep = json.loads(f.read_text())
        except ValueError:
            rep = "unreadable"
    return dict(rc=p.returncode, wall_s=round(wall, 1), wait_s=wait, exec_s=round(wall - wait, 1), report=rep, log=log)


def basics(r):
    rep = r["report"] if isinstance(r["report"], dict) else {}
    chk = rep.get("checks", {})
    return dict(rc=r["rc"], exec_s=r["exec_s"], wait_s=r["wait_s"], complete=rep.get("complete"),
                parts=len(rep.get("scope", {}).get("parts", [])), checks_run=len(chk),
                checks_failed=sorted(n for n, c in chk.items() if c.get("failed")),
                parts_failed={n: sorted(c["failed_parts"]) for n, c in chk.items() if c.get("failed_parts")},
                check_seconds={n: c.get("seconds") for n, c in chk.items()},
                peak_tree_gib=rep.get("peak_tree_gib"))


def find_pair(rep, check, a, b, key):
    """per_part[a][b][key] of a check in the report (None when absent)."""
    try:
        return rep["checks"][check]["per_part"][a][b][key]
    except (KeyError, TypeError):
        return None


def log_tail(r, n=6):
    return [ln for ln in r["log"].splitlines() if ln.strip() and "waiting for a slot" not in ln][-n:]


# ------------------------------------------------------------------------------------------------ scenarios
THIGH = "R_thigh_roll_flip"           # the most iterated leg connector (roll-flip hip)
THIGH_NB = "R_hip_pitch_to_roll_roll_flip"   # its neighbour on the next link up, 1.45 mm away at standing
ARM = "R_shoulder_pitch_to_roll_combined"
LIGHT = "geometry,mass,gaps,press,clearance"   # the light iteration of working-rules.md
SCENARIOS = {}


def scenario(sid, group, purpose):
    def deco(fn):
        SCENARIOS[sid] = dict(fn=fn, group=group, purpose=purpose)
        return fn
    return deco


@scenario("light-part", "speed", "light iteration on the thigh connector (working-rules: about a minute)")
def s_light_part(sid):
    r = run_tool(sid, ["--parts", THIGH, "--checks", LIGHT])
    return dict(basics(r), tail=log_tail(r))


@scenario("light-part-repeat", "truth", "the same light run twice: every measurement identical (determinism)")
def s_repeat(sid):
    a = run_tool(sid + "-a", ["--parts", THIGH, "--checks", LIGHT])
    b = run_tool(sid + "-b", ["--parts", THIGH, "--checks", LIGHT])
    TIMING = {"seconds", "peak_gib", "tree_peak_gib", "exact_seconds"}

    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k not in TIMING}
        return [strip(v) for v in x] if isinstance(x, list) else x

    def meas(rep):
        return {n: strip(c.get("per_part")) for n, c in rep.get("checks", {}).items()} if isinstance(rep, dict) else None
    ma, mb = meas(a["report"]), meas(b["report"])
    differ = sorted(n for n in (ma or {}) if (ma or {}).get(n) != (mb or {}).get(n))
    return dict(basics(b), first_exec_s=a["exec_s"], identical_measurements=ma == mb, checks_differing=differ)


@scenario("light-arm", "speed", "light iteration on an arm connector (arm region poses)")
def s_light_arm(sid):
    r = run_tool(sid, ["--parts", ARM, "--checks", LIGHT])
    return dict(basics(r), tail=log_tail(r))


@scenario("candidate-part", "speed", "before showing a candidate: light checks at --poses library")
def s_candidate(sid):
    r = run_tool(sid, ["--parts", THIGH, "--checks", LIGHT, "--poses", "library"])
    return dict(basics(r), tail=log_tail(r))


@scenario("default-checks-part", "speed", "an agent names a part and no --checks (every check that applies)")
def s_default(sid):
    r = run_tool(sid, ["--parts", THIGH], timeout_s=1800, limit_gb=6)
    return dict(basics(r), tail=log_tail(r))


@scenario("section-hip-light", "speed", "section scope: the Hip with the light per-part checks")
def s_hip(sid):
    r = run_tool(sid, ["--section", "Hip", "--checks", "geometry,mass,gaps,static_audit,press,print_setup"], timeout_s=1800, limit_gb=6)
    return dict(basics(r), tail=log_tail(r))


@scenario("whole-body-light", "speed", "unscoped whole body, per-part light checks (BOM triage and release runs)")
def s_whole(sid):
    r = run_tool(sid, ["--checks", "geometry,mass,gaps,static_audit,print_setup,fem_setup"], timeout_s=2400, limit_gb=6)
    return dict(basics(r), tail=log_tail(r))


@scenario("supplier-board-scope", "speed", "a bought board with a 44 MB supplier STEP in scope (the stall class of T-c4sy9d)")
def s_board(sid):
    r = run_tool(sid, ["--parts", "ROCK4D_official", "--checks", "geometry,mass,gaps,static_audit,clearance"], timeout_s=1200)
    return dict(basics(r), tail=log_tail(r))


@scenario("label-case", "robust", "labels that differ only in case (macOS paths ignore case): neither report silently replaces the other")
def s_case(sid):
    a = run_tool("Label-Case", ["--parts", THIGH, "--checks", "mass"])
    b = run_tool("label-case", ["--parts", THIGH_NB, "--checks", "mass"])
    ra = a["report"] if isinstance(a["report"], dict) else {}
    fa = REPORTS / "Label-Case" / "report.json"
    now = json.loads(fa.read_text()) if fa.exists() else {}
    return dict(rc=[a["rc"], b["rc"]], first_run_parts=ra.get("scope", {}).get("parts"),
                first_label_report_now_holds=now.get("scope", {}).get("parts"),
                second_run_says_it_replaced=[ln for ln in b["log"].splitlines() if "replaces the report" in ln][:1],
                tails=[log_tail(a, 1), log_tail(b, 1)])


@scenario("worktree-cold-vs-shared", "speed", "a new worktree session's first light + screws run: empty cache vs the shared store")
def s_cold(sid):
    import os
    cold = EVAL / sid / "empty-cache"
    shutil.rmtree(cold, ignore_errors=True); cold.mkdir(parents=True)
    args = ["--parts", THIGH, "--checks", LIGHT + ",screws"]
    c = run_tool(sid + "-cold", args, env=dict(os.environ, LAB_SHARED_CACHE=str(cold)), timeout_s=1200)
    w = run_tool(sid + "-shared", args, timeout_s=1200)
    return dict(cold=basics(c), shared=basics(w))


TIMING = {"seconds", "peak_gib", "tree_peak_gib", "exact_seconds", "cached", "mesh_seconds"}


def strip_timing(x):
    if isinstance(x, dict):
        return {k: strip_timing(v) for k, v in x.items() if k not in TIMING}
    return [strip_timing(v) for v in x] if isinstance(x, list) else x


def results_of(rep):
    """{check: result without timing / cache fields} of a report."""
    return {n: strip_timing(c) for n, c in (rep or {}).get("checks", {}).items()} if isinstance(rep, dict) else {}


@scenario("cache-repeat", "speed", "result cache: the light run again with nothing changed (computed run, then cached run): seconds, same results")
def s_cache_repeat(sid):
    a = run_tool(sid + "-computed", ["--parts", THIGH, "--checks", LIGHT])
    b = run_tool(sid + "-cached", ["--parts", THIGH, "--checks", LIGHT], cache=True)
    cached = sorted(n for n, c in (b["report"] or {}).get("checks", {}).items() if c.get("cached"))
    return dict(computed_exec_s=a["exec_s"], cached_exec_s=b["exec_s"], checks_cached=cached, of=len((b["report"] or {}).get("checks", {})),
                results_identical=results_of(a["report"]) == results_of(b["report"]), rc=[a["rc"], b["rc"]],
                valid=a["rc"] == 0 and b["rc"] == 0 and bool(results_of(a["report"])))


@scenario("cache-repeat-heavy", "speed", "result cache on the expensive checks of one part (fem, screws, fit_class): computed, then cached")
def s_cache_heavy(sid):
    args = ["--parts", THIGH, "--checks", "fem,screws,fit_class,driver_access"]
    a = run_tool(sid + "-computed", args, limit_gb=6, timeout_s=1800)
    b = run_tool(sid + "-cached", args, limit_gb=6, timeout_s=1800, cache=True)
    rb = (b["report"] or {}).get("checks", {})
    fem_cached = [p for p, r in ((rb.get("fem") or {}).get("per_part") or {}).items()]
    return dict(computed_exec_s=a["exec_s"], cached_exec_s=b["exec_s"], checks_cached=sorted(n for n, c in rb.items() if c.get("cached")),
                fem_parts=fem_cached, results_identical=results_of(a["report"]) == results_of(b["report"]),
                differing=sorted(n for n in results_of(a["report"]) if results_of(a["report"]).get(n) != results_of(b["report"]).get(n)), rc=[a["rc"], b["rc"]])


def _shift_in_place(name, vec):
    """Rewrite the selected manifest's STEP and mesh of part `name` in place, moved by vec (mm): the manifest stays byte
    for byte the same, only the files' content changes. Returns {path: original bytes} to restore."""
    import numpy as np
    import trimesh
    from build123d import Compound, Location, export_step, import_step
    if (ROOT / ".git").is_dir():                       # the main checkout: other sessions read these files right now
        raise RuntimeError("this scenario rewrites artifact files in place: run it in a worktree "
                           "(python3 infra/tools/worktree.py add NAME), not in the main checkout")
    sys.path.insert(0, str(ROOT))
    from infra.storage.blob_store import ensure        # the file the tool reads (a moved path resolves to its new name)
    p = part(json.loads(selected_manifest().read_text()), name)
    keep = {}
    if p.get("native_step"):
        f = Path(ensure(ROOT / p["native_step"])); keep[f] = f.read_bytes()
        sh = import_step(str(f))
        export_step(Compound(children=[x.moved(Location(tuple(vec))) for x in sh.solids()]), str(f))
    f = Path(ensure(ROOT / p["mesh"])); keep[f] = f.read_bytes()
    if f.suffix == ".npz":
        z = dict(np.load(f)); z["vertices"] = np.asarray(z["vertices"], float) + np.asarray(vec, float)
        with open(f, "wb") as fh:
            np.savez(fh, **z)
    else:
        m = trimesh.load(f, force="mesh"); m.apply_translation(vec); m.export(f)
    return keep


@scenario("cache-stale-hunt", "truth", "result cache never serves a stale result: a neighbour's (and the part's own) STEP + mesh rewritten "
          "in place 1.5 mm closer, manifest unchanged; the cached run must equal a --no-cache run, check by check")
def s_stale(sid):
    import numpy as np
    m = json.loads(selected_manifest().read_text())
    targets = [THIGH_NB, "R_knee_ST3025_case", "R_thigh_shell_front", THIGH]
    run_tool(sid + "-prime", ["--parts", THIGH, "--checks", LIGHT + ",fit_class"], cache=True)     # store results of the unchanged body
    out = {}
    ca = np.asarray(part(m, THIGH)["bounds_mm"]).mean(0)
    for t in targets:
        cb = np.asarray(part(m, t)["bounds_mm"]).mean(0)
        u = (ca - cb) / (np.linalg.norm(ca - cb) or 1.0) if t != THIGH else np.array([0.0, 0.0, -1.0])
        keep = _shift_in_place(t, (1.5 * u).round(3).tolist())
        try:
            c = run_tool(f"{sid}-cached", ["--parts", THIGH, "--checks", LIGHT + ",fit_class"], cache=True)
            n = run_tool(f"{sid}-fresh", ["--parts", THIGH, "--checks", LIGHT + ",fit_class"])
        finally:
            for f, b in keep.items():
                f.write_bytes(b)
        rc_, rn = results_of(c["report"]), results_of(n["report"])
        out[t] = dict(cached=sorted(k for k, v in (c["report"] or {}).get("checks", {}).items() if v.get("cached")),
                      rerun=sorted(k for k, v in (c["report"] or {}).get("checks", {}).items() if not v.get("cached")),
                      stale=sorted(k for k in rn if rc_.get(k) != rn.get(k)), cached_exec_s=c["exec_s"], fresh_exec_s=n["exec_s"],
                      rc=[c["rc"], n["rc"]], checks_compared=len(rn), valid=c["rc"] == 0 and n["rc"] == 0 and len(rn) > 0)
    back = run_tool(f"{sid}-restored", ["--parts", THIGH, "--checks", LIGHT + ",fit_class"], cache=True)
    return dict(per_mutated_part=out, stale_results_total=sum(len(v["stale"]) for v in out.values()),
                after_restore_cached=sorted(k for k, v in (back["report"] or {}).get("checks", {}).items() if v.get("cached")))


@scenario("render-mesh-edit", "truth", "the six-side drawing after the thigh connector's mesh is rebuilt in place 15 mm aside "
          "(manifest unchanged): pixels that change, and whether a stored render is served")
def s_render(sid):
    import numpy as np
    from PIL import Image
    args = ["--parts", THIGH, "--checks", "render"]
    a = run_tool(sid + "-before", args)
    pa = np.asarray(Image.open(REPORTS / f"{sid}-before" / "case-all-sides.png").convert("L"), float)
    import random
    dx = round(15.0 + random.random(), 4)                     # a shift no earlier run made: no stored result can match it
    keep = _shift_in_place(THIGH, [dx, 0.0, 0.0])
    try:
        c = run_tool(sid + "-cached", args, cache=True)          # must not serve the render stored before the edit
        b = run_tool(sid + "-after", args)
        pb = np.asarray(Image.open(REPORTS / f"{sid}-after" / "case-all-sides.png").convert("L"), float)
    finally:
        for f, x in keep.items():
            f.write_bytes(x)
    return dict(shift_mm=dx, pixels_changed=int((np.abs(pa - pb) > 8).sum()) if pa.shape == pb.shape else "size differs", of=int(pa.size),
                served_from_cache_after_edit=bool(((c["report"] or {}).get("checks", {}).get("render") or {}).get("cached")),
                rc=[a["rc"], c["rc"], b["rc"]])


@scenario("memguard-burst", "speed", "six light runs started at once (six parts): memguard without a job count, admitted by memory")
def s_burst(sid):
    import threading
    parts = [THIGH, THIGH_NB, ARM, "R_shin_combined", "R_foot_monocoque_combined", "combined__stage1_center_torso_spine_chassis"]
    res, peak = {}, [0.0]
    stop = threading.Event()

    def sample():                          # resident memory of every memguard job on the machine (this and other sessions)
        sys.path.insert(0, str(ROOT / "infra/tools"))
        import memguard as MG
        while not stop.is_set():
            kids, rss = MG._ps_tree()
            out = subprocess.run(["ps", "-Ao", "pid=,command="], capture_output=True, text=True).stdout
            tot = sum(MG._tree_kb(int(l.split()[0]), kids, rss) for l in out.splitlines()
                      if len(l.split()) > 2 and l.split()[2].endswith("memguard.py") and "python" in l.split()[1].lower())
            peak[0] = max(peak[0], tot / 1048576)
            time.sleep(1)
    smp = threading.Thread(target=sample); smp.start()
    t0 = time.time()

    def go(i, p):
        res[i] = run_tool(f"{sid}-{i}", ["--parts", p, "--checks", LIGHT])
    th = [threading.Thread(target=go, args=(i, p)) for i, p in enumerate(parts)]
    [x.start() for x in th]; [x.join() for x in th]
    stop.set(); smp.join()
    return dict(wall_all_s=round(time.time() - t0, 1), waits_s=[res[i]["wait_s"] for i in range(len(parts))],
                exec_s=[res[i]["exec_s"] for i in range(len(parts))], rc=[res[i]["rc"] for i in range(len(parts))],
                peak_resident_all_guarded_jobs_gb=round(peak[0], 2))


@scenario("changed-since-noop", "speed", "--changed-since last-run straight after a run of the same part: what it selects, how fast")
def s_noop(sid):
    run_tool(sid + "-prime", ["--parts", THIGH, "--checks", "mass"])
    r = run_tool(sid, ["--parts", THIGH, "--changed-since", "last-run", "--checks", "mass"])
    rep = r["report"] if isinstance(r["report"], dict) else {}
    return dict(basics(r), selected=rep.get("scope", {}).get("parts"), tail=log_tail(r, 3))


@scenario("inject-overlap", "truth", "thigh connector moved 3 mm into its hip neighbour: gaps / static_audit / clearance vs exact truth")
def s_overlap(sid):
    import numpy as np
    sys.path.insert(0, str(ROOT / "robot/body"))
    man0 = selected_manifest()
    v0, d0 = truth_pair(man0, THIGH, THIGH_NB)
    # direction: from the thigh's box centre toward the neighbour's, in the manifest frame
    m = json.loads(man0.read_text())
    ca = np.asarray(part(m, THIGH)["bounds_mm"]).mean(0); cb = np.asarray(part(m, THIGH_NB)["bounds_mm"]).mean(0)
    u = (cb - ca) / np.linalg.norm(cb - ca)
    vec = (3.0 * u).round(3).tolist()
    f = copy_manifest(sid, lambda man, d: translate_part(man, d, THIGH, vec))
    v1, d1 = truth_pair(f, THIGH, THIGH_NB)
    r = run_tool(sid, ["--parts", THIGH, "--checks", "gaps,static_audit,clearance,fit_class"], manifest=f)
    rep = r["report"] if isinstance(r["report"], dict) else {}
    return dict(basics(r), moved_mm=vec, truth_before=dict(overlap_mm3=v0, distance_mm=d0), truth_after=dict(overlap_mm3=v1, distance_mm=d1),
                gaps_overlap_mm3=find_pair(rep, "gaps", THIGH, THIGH_NB, "overlap_mm3"),
                gaps_distance_mm=find_pair(rep, "gaps", THIGH, THIGH_NB, "distance_mm"),
                static_audit_overlap_mm3=find_pair(rep, "static_audit", THIGH, THIGH_NB, "overlap_mm3"),
                clearance_standing_overlap_mm3=find_pair(rep, "clearance", THIGH, THIGH_NB, "standing_overlap_mm3"),
                fit_class=(rep.get("checks", {}).get("fit_class", {}).get("per_part", {}).get(THIGH, {}) or {}).get(THIGH_NB))


@scenario("inject-step-edit", "truth", "a STEP rebuilt in place (same path, new content): the next run measures the new geometry, not a cache")
def s_cache(sid):
    sys.path.insert(0, str(ROOT / "robot/body"))
    holder = {}

    def edit(man, d):
        holder["step"] = translate_part(man, d, THIGH, [0.0, 0.0, 0.0])
    f = copy_manifest(sid, edit)
    a = run_tool(sid + "-a", ["--parts", THIGH, "--checks", "gaps,mass"], manifest=f)
    # rebuild the same file 2 mm down (a designer regenerating the part in place)
    man = json.loads(f.read_text())
    from build123d import Location, export_step, import_step
    import trimesh
    p = part(man, THIGH)
    from build123d import Compound
    export_step(Compound(children=[s.moved(Location((0, 0, -2.0))) for s in import_step(str(holder["step"])).solids()]), str(holder["step"]))
    mm = trimesh.load(ROOT / p["mesh"], force="mesh"); mm.apply_translation([0, 0, -2.0]); mm.export(ROOT / p["mesh"])
    p["bounds_mm"] = [[x, y, z - 2.0] for x, y, z in p["bounds_mm"]]
    f.write_text(json.dumps(man))
    b = run_tool(sid + "-b", ["--parts", THIGH, "--checks", "gaps,mass"], manifest=f)
    ga = (a["report"] or {}).get("checks", {}).get("gaps", {}).get("per_part", {}).get(THIGH, {})
    gb = (b["report"] or {}).get("checks", {}).get("gaps", {}).get("per_part", {}).get(THIGH, {})
    key = lambda g: (g["distance_mm"], round(g["overlap_mm3"], 1))   # noqa: E731
    changed = {n: (key(ga[n]), key(gb[n])) for n in set(ga) & set(gb) if key(ga[n]) != key(gb[n])}
    same = {n: key(ga[n]) for n in set(ga) & set(gb) if key(ga[n]) == key(gb[n])}
    return dict(basics(b), pairs_measured=len(gb), pairs_changed=len(changed), pairs_appearing=sorted(set(gb) - set(ga)),
                pairs_vanishing=sorted(set(ga) - set(gb)), changed=changed, unchanged=same)


def _missing(field, which):
    def edit(man, d):
        p = part(man, which)
        p[field] = rel(d / f"{which}-missing{Path(p[field]).suffix}")
    return edit


@scenario("fault-missing-step-scoped", "robust", "the scoped part's native STEP is missing on disk: the run finishes, the other checks measure")
def s_miss_step(sid):
    f = copy_manifest(sid, _missing("native_step", THIGH))
    r = run_tool(sid, ["--parts", THIGH, "--checks", LIGHT], manifest=f)
    return dict(basics(r), tail=log_tail(r))


@scenario("fault-missing-step-neighbour", "robust", "a neighbour's native STEP is missing: the scoped part is still measured against it")
def s_miss_nb(sid):
    f = copy_manifest(sid, _missing("native_step", THIGH_NB))
    r = run_tool(sid, ["--parts", THIGH, "--checks", LIGHT], manifest=f)
    rep = r["report"] if isinstance(r["report"], dict) else {}
    return dict(basics(r), gaps_neighbour_measured=find_pair(rep, "gaps", THIGH, THIGH_NB, "distance_mm"), tail=log_tail(r))


@scenario("fault-missing-mesh-neighbour", "robust", "a neighbour's mesh file is missing: the run finishes and names it")
def s_miss_mesh(sid):
    f = copy_manifest(sid, _missing("mesh", THIGH_NB))
    r = run_tool(sid, ["--parts", THIGH, "--checks", LIGHT], manifest=f)
    return dict(basics(r), tail=log_tail(r))


@scenario("fault-corrupt-step", "robust", "the scoped part's STEP is truncated (half the file): the run finishes and names it")
def s_corrupt(sid):
    def edit(man, d):
        p = part(man, THIGH)
        src = (ROOT / p["native_step"]).read_bytes()
        out = d / f"{THIGH}-truncated.step"
        out.write_bytes(src[: len(src) // 2])
        p["native_step"] = rel(out)
    f = copy_manifest(sid, edit)
    r = run_tool(sid, ["--parts", THIGH, "--checks", LIGHT], manifest=f)
    return dict(basics(r), tail=log_tail(r))


@scenario("fault-one-bad-part-in-section", "robust", "one part of a 2-part scope has a missing STEP: the other part's results are all there")
def s_isolation(sid):
    f = copy_manifest(sid, _missing("native_step", THIGH))
    r = run_tool(sid, ["--parts", f"{THIGH},{THIGH_NB}", "--checks", "geometry,mass,gaps,press,walls,holes,overhang"], manifest=f)
    rep = r["report"] if isinstance(r["report"], dict) else {}
    got = {n: THIGH_NB in (c.get("per_part") or {}) for n, c in rep.get("checks", {}).items()}
    return dict(basics(r), healthy_part_measured_by_check=got, tail=log_tail(r))


@scenario("fault-timeout", "robust", "memguard stops a long run (fit) after 40 s: report says unfinished, names the check, keeps what finished")
def s_timeout(sid):
    r = run_tool(sid, ["--parts", THIGH, "--checks", "mass,gaps,fit", "--poses", "fit"], timeout_s=40)
    rep = r["report"] if isinstance(r["report"], dict) else {}
    return dict(basics(r), incomplete_running=rep.get("incomplete_running"), finished=sorted(rep.get("checks", {})), tail=log_tail(r, 4))


@scenario("fault-bad-args", "robust", "typos an agent makes: unknown part, unknown check, unknown section: fast, clear message")
def s_bad(sid):
    out = {}
    for k, args in dict(part=["--parts", "R_thigh_rollflip"], check=["--parts", THIGH, "--checks", "gap"],
                        section=["--section", "Hips"]).items():
        r = run_tool(f"{sid}-{k}", args, timeout_s=120)
        out[k] = dict(rc=r["rc"], exec_s=r["exec_s"], message=log_tail(r, 2))
    return out


@scenario("concurrent-same-label", "robust", "two sessions run the same label at once: both get a complete, uncorrupted report")
def s_concurrent(sid):
    import threading
    res = {}
    lock = REPORTS / sid / ".lock"
    lock.unlink(missing_ok=True)
    def go(k, parts, **kw):
        res[k] = run_tool(sid, ["--parts", parts, "--checks", "mass,gaps,clearance"], **kw)
    a = threading.Thread(target=go, args=("a", THIGH)); a.start()
    t0 = time.time()                    # B starts while A holds the label (A's lock file names it), past memguard's queue
    while time.time() - t0 < 900 and not (lock.exists() and lock.read_text().strip()):
        time.sleep(0.5)
    b = threading.Thread(target=go, args=("b", THIGH_NB), kwargs=dict(no_wait=True)); b.start()
    a.join(); b.join()
    rep = json.loads((REPORTS / sid / "report.json").read_text()) if (REPORTS / sid / "report.json").exists() else None
    return dict(rc=[res["a"]["rc"], res["b"]["rc"]], final_report_complete=(rep or {}).get("complete"),
                final_report_parts=(rep or {}).get("scope", {}).get("parts"), tails=[log_tail(res["a"], 2), log_tail(res["b"], 2)])


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only")
    ap.add_argument("--group")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep the mutated manifests and part files")
    a = ap.parse_args()
    if a.list:
        for k, s in SCENARIOS.items():
            print(f"{k:32s} {s['group']:7s} {s['purpose']}")
        return
    ids = [k for k in SCENARIOS if (not a.only or k in a.only.split(",")) and (not a.group or SCENARIOS[k]["group"] == a.group)]
    EVAL.mkdir(parents=True, exist_ok=True)
    res_f = EVAL / "results.json"
    allres = json.loads(res_f.read_text()) if res_f.exists() else {}
    for k in ids:
        t0 = time.time()
        print(f"== {k}: {SCENARIOS[k]['purpose']}", flush=True)
        try:
            out = SCENARIOS[k]["fn"](k)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            out = dict(harness_error=f"{type(exc).__name__}: {exc}")
        out["scenario_wall_s"] = round(time.time() - t0, 1)
        out["at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        out["commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        allres[k] = out
        res_f.write_text(json.dumps(allres, indent=1, default=str))
        print(json.dumps(out, indent=1, default=str), flush=True)
        if not a.keep:
            shutil.rmtree(EVAL / k, ignore_errors=True)
    print(f"\n{res_f.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
