"""The checks know no design: the same body with every part renamed to an opaque id (P0001 ...), its definitions and part
settings renamed alike (globs expanded to the renamed parts they match), must give the same measurements as the real body
once the names are mapped back. A difference is a place where a check still reads meaning from a part's name.
Also: a setting changed in a copy of the definitions steers the check that reads it (LAB_DEFINITIONS_ROOT)."""
from __future__ import annotations

import fnmatch
import json
import os
import shutil
from pathlib import Path

from evaluate import EVAL, LIGHT, ROOT, THIGH, results_of, run_tool, scenario, selected_manifest

GLOB = set("*?[")


def _mapping(names):
    return {n: f"P{i:04d}" for i, n in enumerate(sorted(names))}


def _rename(x, M, names, key=None):
    """x with every exact part name mapped and every part glob expanded to the mapped names it matches."""
    if isinstance(x, dict):
        return {(M.get(k, k) if isinstance(k, str) else k): _rename(v, M, names, k) for k, v in x.items()}
    if isinstance(x, list):
        out = []
        for v in x:
            r = _rename(v, M, names, key)
            out += r if isinstance(v, str) and isinstance(r, list) else [r]
        return out
    if isinstance(x, str):
        if x in M:
            return M[x]
        if GLOB & set(x):
            hit = [M[n] for n in names if fnmatch.fnmatchcase(n, x)]
            if hit:
                return hit if (key in ("match", "part", "screw", "with", "base", "include", "exclude", "parts") or len(hit) > 1) else hit[0]
        return x
    return x


def renamed_body(sid, joints=False):
    """(manifest path, definitions dir, part_settings path, checks data dir or None, name map) of the renamed copy. With
    joints=True the joints and links get opaque names too (J0001, L0001), in the manifest, the definitions and a copy of
    the checks' data tables (LAB_CHECKS_DATA)."""
    import yaml
    d = EVAL / sid
    shutil.rmtree(d, ignore_errors=True)
    (d / "definitions").mkdir(parents=True)
    man = json.loads(selected_manifest().read_text())
    names = [p["name"] for p in man["parts"]]
    M = _mapping(names)
    if joints:
        jn = sorted(j["name"] for j in man["joints"])
        ln = sorted({p["owning_link"] for p in man["parts"]} | {j["parent"] for j in man["joints"]} | {j["child"] for j in man["joints"]})
        M.update({n: f"J{i:04d}" for i, n in enumerate(jn)})
        M.update({n: f"L{i:04d}" for i, n in enumerate(ln) if n not in M})
    rman = _rename(man, M, names)
    for p, q in zip(man["parts"], rman["parts"]):          # file paths keep their real names: they are where the files are
        for k in ("native_step", "mesh", "native_step_unmodified"):
            if k in p:
                q[k] = p[k]
    mf = d / "manifest.json"
    mf.write_text(json.dumps(rman))
    import sys
    sys.path.insert(0, str(ROOT / "robot/body/bom"))
    import definitions as DEF
    for i, f in enumerate(DEF.discover(root=ROOT)):
        doc = yaml.safe_load(f.read_text())
        out = d / "definitions" / f"{i:03d}-{f.name}"
        out.write_text(yaml.safe_dump(_rename(doc, M, names), sort_keys=False, width=200))
    ps = json.loads((ROOT / "robot/body/checks/data/part_settings.json").read_text())
    psf = d / "part_settings.json"
    psf.write_text(json.dumps(_rename(ps, M, names)))
    data = None
    if joints:
        data = d / "data"
        shutil.copytree(ROOT / "robot/body/checks/data", data)
        for f in data.glob("*.json"):
            f.write_text(json.dumps(_rename(json.loads(f.read_text()), M, names), indent=1))
    return mf, d / "definitions", psf, data, M


def _unmap_text(x, R):
    for opaque, real in R.items():
        if opaque in x:
            x = x.replace(opaque, real)
    if " x " in x and x.count(" x ") == 1:              # a pair key "A x B": the pair, whichever name sorts first
        a, b = x.split(" x ")
        if a.strip() and b.strip() and " " not in a + b:
            x = " x ".join(sorted((a, b)))
    return x


def _unmap(x, R):
    if isinstance(x, dict):
        return {(_unmap_text(k, R) if isinstance(k, str) else k): _unmap(v, R) for k, v in x.items()}
    if isinstance(x, list):
        return [_unmap(v, R) for v in x]
    if isinstance(x, str):
        return _unmap_text(x, R)
    return x


def _norm_pairs(x):
    """Real-side pair keys in the same sorted order as the unmapped ones."""
    if isinstance(x, dict):
        return {(_unmap_text(k, {}) if isinstance(k, str) else k): _norm_pairs(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_norm_pairs(v) for v in x]
    return _unmap_text(x, {}) if isinstance(x, str) else x


def _diff(a, b, path="", out=None, cap=12):
    out = [] if out is None else out
    if len(out) >= cap:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k in ("definition", "definitions", "evidence", "notes", "seconds", "cpu_s", "peak_gib", "tree_peak_gib", "cached",
                     "mesh_cached", "mesh_seconds"):        # run / cache state, not measurements
                continue
            _diff(a.get(k), b.get(k), f"{path}/{k}", out, cap)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (u, v) in enumerate(zip(a, b)):
            _diff(u, v, f"{path}[{i}]", out, cap)
    elif a != b:
        out.append(f"{path}: real {str(a)[:80]} | renamed {str(b)[:80]}")
    return out


def compare_renamed(sid, scope, checks, timeout_s=1500, limit_gb=6, joints=False, extra=()):
    real = run_tool(sid + "-real", ["--parts", ",".join(scope), "--checks", checks, *extra], timeout_s=timeout_s, limit_gb=limit_gb)
    mf, defs, psf, data, M = renamed_body(sid, joints)
    env = dict(os.environ, LAB_DEFINITIONS_ROOT=str(defs), LAB_PART_SETTINGS=str(psf), **({"LAB_CHECKS_DATA": str(data)} if data else {}))
    ren = run_tool(sid + "-renamed", ["--parts", ",".join(M[n] for n in scope), "--checks", checks, *extra], manifest=mf, env=env,
                   timeout_s=timeout_s, limit_gb=limit_gb)
    R = {v: k for k, v in M.items()}
    R = dict(sorted(R.items(), key=lambda t: -len(t[0])))       # P0012 before P001 when unmapping text
    a, b = _norm_pairs(results_of(real["report"])), _unmap(results_of(ren["report"]), R)
    per = {c: _diff(a.get(c), b.get(c)) for c in sorted(set(a) | set(b))}
    return dict(rc=[real["rc"], ren["rc"]], checks=len(a), identical=sorted(c for c, d in per.items() if not d),
                differing={c: d for c, d in per.items() if d})


@scenario("agnostic-renamed-light", "truth", "light checks on the thigh connector: real body vs the same body with every part "
          "renamed to an opaque id (definitions and settings renamed alike)")
def s_renamed_light(sid):
    return compare_renamed(sid, [THIGH], LIGHT)


@scenario("agnostic-renamed-candidate", "truth", "every per-part and standing check (candidate preset) on the thigh connector: "
          "real vs renamed body")
def s_renamed_candidate(sid):
    return compare_renamed(sid, [THIGH], "candidate")


LEFT = "L_thigh_roll_flip"
ARM = "R_shoulder_pitch_to_roll_combined"
TORSO = "combined__stage1_center_torso_spine_chassis"


@scenario("agnostic-renamed-joints-motion", "truth", "motion checks (clearance, range_sweep, collision_sweep) on a right and a left leg "
          "part: real body vs the body with parts, joints and links all renamed to opaque ids")
def s_renamed_motion(sid):
    return compare_renamed(sid, [THIGH, LEFT], "clearance,range_sweep,collision_sweep", joints=True, timeout_s=2400, limit_gb=8)


@scenario("agnostic-renamed-joints-fasteners", "truth", "screw checks (screws, driver_access, fastener_motion, connectivity) on the thigh "
          "connector and the torso chassis: real vs fully renamed body")
def s_renamed_fasteners(sid):
    return compare_renamed(sid, [THIGH, TORSO], "screws,driver_access,fastener_motion,connectivity", joints=True, timeout_s=2400, limit_gb=8)


@scenario("agnostic-renamed-joints-parts", "truth", "per-part and print checks (candidate preset) on a leg, an arm and a torso part: "
          "real vs fully renamed body")
def s_renamed_parts(sid):
    return compare_renamed(sid, [THIGH, ARM, TORSO], "candidate", joints=True, timeout_s=2400, limit_gb=8)


@scenario("agnostic-renamed-joints-fem", "truth", "fem + fem_setup on the thigh connector: real vs fully renamed body")
def s_renamed_fem(sid):
    return compare_renamed(sid, [THIGH], "fem,fem_setup", joints=True, timeout_s=2400, limit_gb=8)


@scenario("steer-definition-setting", "truth", "a setting changed only in a copy of the definitions steers the check that reads it: the "
          "thigh connector's material set to PETG-CF in the copy changes mass and heat, nothing else")
def s_steer(sid):
    import yaml
    import sys
    sys.path.insert(0, str(ROOT / "robot/body/bom"))
    import definitions as DEF
    d = EVAL / sid / "definitions"
    shutil.rmtree(EVAL / sid, ignore_errors=True); d.mkdir(parents=True)
    names = [p["name"] for p in json.loads(selected_manifest().read_text())["parts"]]
    e = DEF.load().claims(names)[THIGH]
    for i, f in enumerate(DEF.discover(root=ROOT)):
        doc = yaml.safe_load(f.read_text())
        if str(f.relative_to(ROOT)) == e["file"]:
            ent = doc["parts"][e["index"]]
            proc = ent.setdefault("process", {})
            before = proc.get("material")
            proc["material"] = "Bambu PETG-CF" if before != "Bambu PETG-CF" else "Bambu PAHT-CF"
        (d / f"{i:03d}-{f.name}").write_text(yaml.safe_dump(doc, sort_keys=False, width=200))
    args = ["--parts", THIGH, "--checks", "mass,heat,print_setup,geometry,gaps"]
    a = run_tool(sid + "-as-is", args)
    b = run_tool(sid + "-steered", args, env=dict(os.environ, LAB_DEFINITIONS_ROOT=str(d)))
    ra, rb = results_of(a["report"]), results_of(b["report"])
    return dict(rc=[a["rc"], b["rc"]], material=[before, proc["material"]], changed=sorted(c for c in ra if ra.get(c) != rb.get(c)),
                unchanged=sorted(c for c in ra if ra.get(c) == rb.get(c)))
