"""evaluate.py scenarios of the fem check (T-r922sr): released load members, progress output, steel bore collars.

Setups that do not exist in the body definitions (a load member split in two, a released member on no hole) are written
as a temporary variant into the part's definition in place (_variant_in_place: refused in the main checkout, restored
afterwards) and run with --fem-variant; everything else runs the stored setups on the selected body."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import threading
import time

from evaluate import MEMGUARD, PY, REPORTS, ROOT, RUN, basics, log_tail, run_tool, scenario, selected_manifest

ARM1 = "R_shoulder_pitch_to_roll_combined"        # arm stage 1: hand-envelope load cases, blue-U idler released about the roll axis
SLEEVES1 = ARM1 + "_steel_sleeves"
CARRIER = "combined__head_sensor_compute_carrier"   # head carrier: bore-collar regions, 20 sleeves of three sizes


@contextlib.contextmanager
def _variant_in_place(part_name, name, setup):
    """The part's body definition with one more fem variant `name` (setup = the keys it replaces), restored on exit."""
    import yaml
    if (ROOT / ".git").is_dir():
        raise RuntimeError("this scenario rewrites a body definition in place: run it in a worktree, not in the main checkout")
    sys.path.insert(0, str(ROOT / "robot/body/bom"))
    import definitions as DEF
    names = [p["name"] for p in json.loads(selected_manifest().read_text())["parts"]]
    e = DEF.load().claims(names)[part_name]
    f = ROOT / e["file"]
    keep = f.read_bytes()
    doc = yaml.safe_load(keep)
    doc["parts"][e["index"]]["fem"].setdefault("variants", {})[name] = setup
    try:
        f.write_text(yaml.safe_dump(doc, sort_keys=False, width=200))
        yield
    finally:
        f.write_bytes(keep)


def _base_variant(part_name, variant):
    """A stored variant's keys (dict) from the part's definition."""
    sys.path.insert(0, str(ROOT / "robot/body/bom"))
    import definitions as DEF
    names = [p["name"] for p in json.loads(selected_manifest().read_text())["parts"]]
    return json.loads(json.dumps(DEF.load().claims(names)[part_name]["fem"]["variants"][variant]))


def _holes(mates, **kw):
    return dict(holes=dict({"of": SLEEVES1, "mate": mates, "gap_mm": 2.0, "r_max_mm": 5.0, "rim_mm": 2.0}, **kw))


def _fem_row(rep, part_name):
    """Compact measurements of one part's fem result: per layer normal / material / case p99, strength ratio, displacement
    (envelope cases: max p99, min ratio, max displacement over the wrenches), released rotation, failure."""
    fem = ((rep or {}).get("checks") or {}).get("fem") or {}
    p = (fem.get("per_part") or {}).get(part_name)
    out = dict(failed=fem.get("failed"), failed_parts=fem.get("failed_parts"))
    if not p:
        return out
    rows = {}
    for ln, bm in p["by_layer_normal"].items():
        for m, cs in bm.items():
            for c, r in cs.items():
                env = r.get("envelope")
                rows[f"{c} / {m} / {ln}"] = (dict(p99_max_MPa=env["p99_vm_MPa"]["max"], ratio_min=env["p99_strength_ratio"]["min"],
                                                  disp_max_mm=env["max_displacement_mm"]["max"]) if env else
                                             dict(p99_MPa=r["p99_vm_MPa"], ratio=r["p99_strength_ratio"], disp_mm=r["max_displacement_mm"]))
    digest = hashlib.sha256(json.dumps(p["by_layer_normal"], sort_keys=True).encode()).hexdigest()[:16]
    return dict(out, rows=rows, released=p.get("released_rotations", p.get("released_rotation")), nodes=p.get("nodes"), regions=p.get("regions_elements"), bore_collars=p.get("bore_collars"),
                reaction_plus_wrench_pct_max=p.get("reaction_plus_wrench_pct_max"), fem_seconds=p.get("seconds"), by_layer_normal_sha=digest)


@scenario("fem-release-split", "truth", "arm stage 1, blue-U idler released about the roll axis: the stored one-member release vs the same "
          "idler sleeves written as two released members (idler screws 0+1 and 2+3): identical numbers expected")
def s_release_split(sid):
    base = _base_variant(ARM1, "shoulder_u_convergence")
    split = dict(base, load=[base["load"][0],
                             dict(_holes(["R_shoulder_u_roll_idler_screw_0", "R_shoulder_u_roll_idler_screw_1"]), rigid=True, free_rotation_about_joint="R_shoulder_roll"),
                             dict(_holes(["R_shoulder_u_roll_idler_screw_2", "R_shoulder_u_roll_idler_screw_3"]), rigid=True, free_rotation_about_joint="R_shoulder_roll")])
    args = ["--parts", ARM1, "--checks", "fem,fem_setup"]
    a = run_tool(sid + "-stored", args + ["--fem-variant", "shoulder_u_convergence"], limit_gb=6, timeout_s=3600)
    with _variant_in_place(ARM1, "eval_release_split", split):
        b = run_tool(sid + "-split", args + ["--fem-variant", "eval_release_split"], limit_gb=6, timeout_s=3600)
    ra, rb = _fem_row(a["report"], ARM1), _fem_row(b["report"], ARM1)
    return dict(stored=dict(basics(a), fem=ra), split=dict(basics(b), fem=rb), identical=ra.get("rows") == rb.get("rows") and bool(ra.get("rows")))


@scenario("fem-release-empty", "robust", "arm stage 1 with a released load member on no hole (mate = the elbow idler, far away): "
          "the fem result names the part, the variant and the member instead of a singular matrix")
def s_release_empty(sid):
    base = _base_variant(ARM1, "shoulder_u_convergence")
    bad = dict(base, load=[base["load"][0], dict(_holes(["upper__R_elbow_C047_rear_idler"]), rigid=True, free_rotation_about_joint="R_shoulder_roll")])
    with _variant_in_place(ARM1, "eval_release_empty", bad):
        r = run_tool(sid, ["--parts", ARM1, "--checks", "fem", "--fem-variant", "eval_release_empty"], limit_gb=6, timeout_s=3600)
    return dict(basics(r), fem=_fem_row(r["report"], ARM1), tail=log_tail(r, 4))


def _run_timed(label, args, limit_gb=6, timeout_s=3600):
    """run.py under memguard with every stderr line timestamped: (rc, [(seconds since start, line)], report)."""
    cmd = ["python3", MEMGUARD, "--limit-gb", str(limit_gb), "--timeout-s", str(timeout_s), "--", PY, "-B", RUN, "--label", label] + args + ["--no-cache"]
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    lines = []
    out = threading.Thread(target=p.stdout.read)
    out.start()
    for ln in p.stderr:
        lines.append((round(time.time() - t0, 1), ln.rstrip()))
    p.wait(); out.join()
    f = REPORTS / label / "report.json"
    return p.returncode, lines, (json.loads(f.read_text()) if f.exists() else None)


@scenario("fem-progress-envelope", "speed", "arm stage 1 stored setup (3 hand-envelope cases x 3 materials, 8310 wrenches each): stderr lines "
          "while fem runs and the longest silence between them")
def s_progress(sid):
    rc, lines, rep = _run_timed(sid, ["--parts", ARM1, "--checks", "fem"])
    t = [s for s, ln in lines if ln.startswith("[fem]")]
    t_all = [s for s, ln in lines]
    a, b = (min(t), max(t)) if t else (None, None)
    inside = [s for s in t_all if a is not None and a <= s <= b]
    gaps = [y - x for x, y in zip(inside, inside[1:])]
    return dict(rc=rc, complete=(rep or {}).get("complete"), fem_lines=len(t), fem_span_s=round(b - a, 1) if t else None,
                longest_silence_in_fem_s=round(max(gaps), 1) if gaps else None, first_lines=[ln for _, ln in lines if ln.startswith("[fem]")][:12],
                fem=_fem_row(rep, ARM1))


def _steel_volume(part_name, h):
    """Volume (mm3) and element count of the steel-region elements of the part's stored setup at mesh size h, from the cached mesh
    the run used (fem_core.get_mesh with the same setup), and the region summary the mesh carries."""
    sys.path.insert(0, str(ROOT / "robot/body"))
    from checks import fem_core as FC
    from checks.ctx import Ctx
    ctx = Ctx(selected_manifest(), [part_name])
    ctx.fem_mesh_h = h
    s = FC.merged_setup(ctx, part_name)
    m = FC.get_mesh(ctx, part_name, s)
    st = m["mat_ids"] == 2
    return dict(steel_mm3=round(float(m["vol"][st].sum()), 1), steel_elements=int(st.sum()), tets=int(len(m["tet"])), regions=m.get("region_elements"))


@scenario("fem-collar-mesh", "truth", "head carrier, steel bore collars at mesh h 2.2 (stored) and 1.6: steel volume and strength ratios")
def s_collar(sid):
    out = {}
    for h in (2.2, 1.6):
        k = f"h{h:g}".replace(".", "p")
        r = run_tool(f"{sid}-{k}", ["--parts", CARRIER, "--checks", "fem,fem_setup", "--fem-mesh-h", f"{h:g}"], limit_gb=8, timeout_s=5400)
        out[k] = dict(basics(r), fem=_fem_row(r["report"], CARRIER), collar=_steel_volume(CARRIER, h))
    return out


@scenario("fem-collar-vs-sleeve", "truth", "head carrier at the stored mesh: steel bore collars (stored) vs the steel sleeves meshed themselves "
          "(bonded companions, no collar regions; a temporary variant). 2026-11-22: the sleeves do not join the part (bores 0.005 mm over the "
          "sleeve OD), max displacement ~5e9 mm, so this measures the part without steel")
def s_collar_vs_sleeve(sid):
    v = dict(bonded=[{"part": CARRIER + "_steel_sleeves", "material": "Steel 304 sleeve"}], regions=[])
    with _variant_in_place(CARRIER, "eval_sleeves_bonded", v):
        r = run_tool(sid, ["--parts", CARRIER, "--checks", "fem,fem_setup", "--fem-variant", "eval_sleeves_bonded"], limit_gb=8, timeout_s=5400,
                     env=dict(os.environ, CHECKS_FEM_SOLVER="spsolve"))     # AMG-CG: no convergence in 2000 iterations on this mesh (2026-11-22)
    return dict(basics(r), fem=_fem_row(r["report"], CARRIER), solver="spsolve", tail=log_tail(r, 4))
