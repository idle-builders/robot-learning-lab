"""slicer: each scoped printed part sliced headless by Bambu Studio in its print direction with the system profiles of its
printer, nozzle and material (printers.json / materials.json `bambu_studio*`; support on), so the slicer's own verdicts
on the geometry become measurements: return code and error text, slicer warnings, print time, filament mass, layers,
support time and support filament (summed from the G-code's support sections), bridge time.

The profiles are flattened from their inheritance chains and their sha256 recorded with the slicer version; the part
mesh goes to the run's work folder (derived, recreated by running this). The part's own layer height (process setting)
replaces the profile's. Measurements and counts only."""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import time

import numpy as np

from checks import process as P
from checks.core import ROOT, tick
from checks.registry import register

TIMEOUT_S = 300
FILAMENT_D_MM = 1.75


def _targets(ctx):
    return [n for n in ctx.scope if P.is_printed(ctx, n) and P.effective(ctx, n)["up"] is not None]


def flat_profile(base, kind, name):
    chain = []
    d = json.loads((base / kind / f"{name}.json").read_text()); chain.append(d)
    while d.get("inherits"):
        d = json.loads((base / kind / f"{d['inherits']}.json").read_text()); chain.append(d)
    out = {}
    for c in reversed(chain):
        out.update(c)
    out.pop("inherits", None); out.update(name=name, **{"from": "system"})
    return out


def support_grams(gcode, density):
    """Net filament (g) extruded inside '; FEATURE: Support...' sections (support and support interface)."""
    e_sup, feat, absolute, last = 0.0, "", False, 0.0
    for ln in gcode.splitlines():
        if ln.startswith("; FEATURE:"):
            feat = ln[10:].strip()
        elif ln.startswith("M82"):
            absolute = True
        elif ln.startswith("M83"):
            absolute = False
        elif ln.startswith(("G1", "G0", "G2", "G3")) and " E" in ln and "Support" in feat:
            m = re.search(r" E(-?[\d.]+)", ln)
            if m:
                v = float(m.group(1))
                de = v - last if absolute else v
                last = v if absolute else last
                e_sup += de                # net: retractions cancel their un-retractions (sum of all features = the G-code total)
    return e_sup * math.pi * FILAMENT_D_MM ** 2 / 4 * density * 1e-3


def slice_part(ctx, n, out_dir):
    t0 = time.time()
    e = P.effective(ctx, n)
    mats, prns = P.tables()
    bs = (prns.get(e["printer"] or "") or {}).get("bambu_studio")
    mat = mats.get(e["material"] or "") or {}
    noz = f"{e['nozzle_mm']:g}" if e["nozzle_mm"] else None
    fil = (mat.get("bambu_studio_filament") or {}).get(noz)
    if not bs or not noz or not fil or noz not in bs["process"]:
        return dict(skipped=f"no Bambu Studio profile for printer {e['printer']}, nozzle {noz}, material {e['material']}")
    binary, base = ROOT / bs["binary"], ROOT / bs["profiles"]
    if not binary.exists():
        return dict(skipped=f"slicer not installed at {bs['binary']} (restore_native_tool.py bambu, then copy the app out of the DMG)")
    wd = out_dir / "slicer" / n
    wd.mkdir(parents=True, exist_ok=True)
    names = dict(machine=bs["machine"].format(nozzle=noz), process=bs["process"][noz], filament=fil)
    prof, hashes = {}, {}
    for kind, nm in names.items():
        d = flat_profile(base, kind, nm)
        if kind in ("machine", "process"):
            d.update(curr_bed_type=bs["bed_type"], default_bed_type=bs["bed_type"])
        if kind == "process":
            d.update(bs.get("overrides", {}))
            if e["source"]["layer_height_mm"] in ("part setting", "seeded", "run override"):
                d["layer_height"] = f"{e['layer_height_mm']:g}"
        f = wd / f"{kind}.json"; f.write_text(json.dumps(d, indent=1))
        prof[kind] = f; hashes[kind] = hashlib.sha256(f.read_bytes()).hexdigest()
    P.print_mesh(ctx, n, e).export(wd / "part.stl")
    cmd = [str(binary), "--datadir", str(wd / "data"), "--load-settings", f"{prof['machine']};{prof['process']}", "--load-filaments", str(prof["filament"]),
           "--slice", "0", "--arrange", "1", "--orient", "0", "--outputdir", str(wd / "out"), "--export-3mf", "out.3mf", str(wd / "part.stl")]
    try:
        cp = subprocess.run(cmd, cwd=wd, capture_output=True, text=True, timeout=TIMEOUT_S)
        log, code = cp.stdout + cp.stderr, cp.returncode
    except subprocess.TimeoutExpired:
        return dict(profiles=names, failed=f"slicer timed out after {TIMEOUT_S} s")
    (wd / "cli.log").write_text(log)
    warn = sorted({re.sub(r"^\[[^\]]*\] \[[^\]]*\] ", "", ln).strip() for ln in log.splitlines()
                   if re.search(r"\[(warning|error)\]", ln) and "parent path" not in ln})
    r = dict(profiles=names, profile_sha256=hashes, slicer=bs["version"], exit_code=code, log_warnings_errors=warn[:20])
    rf = wd / "out" / "result.json"
    if rf.exists():
        res = json.loads(rf.read_text())
        r.update(return_code=res.get("return_code"), error_string=res.get("error_string"))
        if res.get("sliced_plates"):
            pl = res["sliced_plates"][0]
            ft = pl.get("feature_type_times", {})
            r.update(print_time_s=round(pl.get("total_predication", 0.0), 1), filament_g=round(sum(f.get("total_used_g", 0.0) for f in pl.get("filaments", [])), 2),
                     slicer_warning=pl.get("warning_message") or None, support_time_s=round(sum(v for k, v in ft.items() if "Support" in k), 1),
                     bridge_time_s=round(ft.get("Bridge", 0.0), 1), feature_times_s={k: round(v, 1) for k, v in ft.items()})
    g = wd / "out" / "plate_1.gcode"
    if g.exists():
        txt = g.read_text(errors="ignore")
        m = re.search(r"; total layer number: (\d+)", txt)
        r["layers"] = int(m.group(1)) if m else None
        r["support_filament_g"] = round(support_grams(txt, mat.get("density_g_cm3", 1.0)), 2)
    r["seconds"] = round(time.time() - t0, 1)
    tick("slicer", t0)
    return r


@register("slicer", lambda ctx: bool(_targets(ctx)))
def run(ctx):
    """Bambu Studio headless per printed part in its print direction: errors and warnings, print time, filament and support grams, layers."""
    out_dir = ctx.out_dir or (ROOT / ".cache/work/cad-speed/checks/slicer")
    per, cases = {}, []
    ev = dict(parts=0, sliced=0, slicer_errors=0, slicer_warnings=0, skipped=0)
    for n in _targets(ctx):
        r = slice_part(ctx, n, out_dir)
        per[n] = r
        ev["parts"] += 1
        if r.get("skipped"):
            ev["skipped"] += 1; continue
        ok = r.get("return_code") == 0
        ev["sliced"] += ok
        ev["slicer_errors"] += (not ok)
        ev["slicer_warnings"] += bool(r.get("slicer_warning")) + len([w for w in r.get("log_warnings_errors", []) if "error" in w.lower() or "warn" in w.lower()])
        if not ok:
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=None, value=float(r.get("return_code") or r.get("exit_code") or -1),
                              unit="slicer return code", what=f"{r.get('error_string') or r.get('failed')}; {'; '.join(r.get('log_warnings_errors', [])[:2])}"))
            continue
        cases.append(dict(part=n, mating_part=None, pose=None, location_mm=None, value=r.get("support_filament_g", 0.0), unit="g support filament",
                          what=f"{r['filament_g']} g total, {round(r['print_time_s'] / 60, 1)} min, {r.get('layers')} layers, support {r['support_time_s']} s, "
                               f"bridges {r['bridge_time_s']} s ({r['profiles']['process']}, {r['profiles']['filament']})"
                               + (f"; slicer warning: {r['slicer_warning']}" if r.get("slicer_warning") else "")))
    cases.sort(key=lambda c: -c["value"])
    return dict(per_part=per, worst_cases=cases, bad_events=ev, evidence=[],
                notes="Bambu Studio CLI (--slice 0 --orient 0) on the part's print-frame mesh; system profiles flattened (sha256 per run), bed type and "
                      "overrides from printers.json; support filament = net E inside '; FEATURE: Support*' sections (all features together match the G-code's total weight) x filament area x density")
