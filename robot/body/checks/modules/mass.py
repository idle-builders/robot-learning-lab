"""mass: scoped part volume from its solid x the density of its declared material (data/materials.json) vs the manifest mass; owning link
mass / COM. Raw function moved from leg/fast_check.py."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from checks.core import ROOT, R, kit, screen_solid, step_mesh, tick
from checks.registry import register

LEGACY_MATERIAL = "legacy_paht_cf_fast_check"   # data/materials.json: reproduces the old fast_check mass numbers; only the labelled comparison


def legacy_density():
    """g/mm3 of the legacy fast_check material (data/materials.json LEGACY_MATERIAL), for the labelled comparison only."""
    import json
    from checks.core import DATA
    return json.loads((DATA / "materials.json").read_text())[LEGACY_MATERIAL]["density_g_cm3"] * 1e-3


def declared_density(n, p):
    """g/mm3 of part n's declared material (its body definition's process.material, robot/body/bom/bom_config.density_kg_mm3 over
    data/materials.json); None when no definition declares one."""
    try:
        sys.path.insert(0, str(ROOT / "robot/body/bom"))
        import bom_config
        return bom_config.density_kg_mm3(n) * 1e3
    except (KeyError, ImportError):
        return None


def check_mass(body, changed, sol, density=None):
    """density(name, part) -> g/mm3 or None; default = the density of the part's declared material (declared_density), None for a part
    without one (no legacy default)."""
    if density is None:
        density = declared_density
    t0 = time.time()
    rows = {}
    for n in changed:
        p = body.parts[n]; m, how = sol[n]
        rho = density(n, p)
        rows[n] = dict(volume_mm3=round(float(m.volume), 1), solid=how, manifest_mass_g=round((p.get("mass_kg") or 0) * 1000, 2),
                       mass_from_volume_g=round(float(m.volume) * rho, 2) if rho else None,
                       centroid_mm=np.round(m.center_mass, 2).tolist())
    links = {}
    for lk in {body.parts[n]["owning_link"] for n in changed}:
        M, S = 0.0, np.zeros(3)
        miss = []
        for n, p in body.parts.items():
            if p["owning_link"] != lk:
                continue
            mk = p.get("mass_kg")
            if not mk:
                miss.append(n); continue
            if n in changed:                # scoped parts: solid centroid; every other part its bounds centre, whatever solids
                c = sol[n][0].center_mass   # other checks happened to load in this process (the result must not depend on them)
            else:
                c = np.array(p["bounds_mm"], float).mean(0)
            M += mk * 1000; S += mk * 1000 * np.asarray(c)
        links[lk] = dict(mass_g=round(M, 2), com_neutral_mm=np.round(S / M, 2).tolist() if M else None, parts_without_mass=len(miss),
                         note="changed parts: solid centroid; others: bounds centre (mass check run_mass gives inertia)")
    tick("mass", t0)
    return dict(parts=rows, links=links)


@register("mass", lambda ctx: bool(ctx.scope))
def run(ctx):
    """Solid volume x density of the part's declared material vs manifest mass; also the labelled legacy fast_check density; owning link mass
    and COM. Printed parts only (process.is_printed: process.printed, else the definition's kind, else the role table); the density is the
    one data/materials.json gives the part's resolved material (process.effective: run override, else the definition's process block,
    else part_settings.json), named per part in density_source."""
    import json
    from checks import process as P
    from checks.core import DATA
    mats = json.loads((DATA / "materials.json").read_text())
    for n in ctx.scope:
        ctx.solid(n)
    printed = {n for n in ctx.scope if P.is_printed(ctx, n)}
    eff = {n: P.effective(ctx, n) for n in printed}      # the resolved print setup: run override, else the part's definition / settings

    def dens(n, p):
        m = eff[n]["material"] if n in printed else None
        return mats[m]["density_g_cm3"] * 1e-3 if m and (mats.get(m) or {}).get("density_g_cm3") else None
    raw = check_mass(ctx.body, ctx.scope, ctx.sol, dens)
    cases = []
    for n in list(raw["parts"]):
        if n not in printed:
            del raw["parts"][n]   # volume x density is for printed parts only
    legacy = legacy_density()
    for n, r in raw["parts"].items():
        m, how = eff[n]["material"], eff[n]["source"]["material"]
        where = (ctx.settings_source.get(n) or {}).get("process", "part_settings.json") if how in ("part setting", "seeded") else how
        r["material"] = m
        r["density_g_cm3"] = (mats.get(m) or {}).get("density_g_cm3") if m else None
        r["density_source"] = dict(table="data/materials.json density_g_cm3" if r["density_g_cm3"] else
                                   (f"none: material {m!r} has no density in data/materials.json" if m else "none: no material declared"),
                                   material_from=how, definition=where)      # definition: where the material is declared
        r["mass_legacy_density_g"] = round(float(ctx.sol[n][0].volume) * legacy, 2)
        if r["mass_from_volume_g"] is not None and r["manifest_mass_g"]:
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=r["centroid_mm"], unit="g (volume - manifest)",
                              value=round(r["mass_from_volume_g"] - r["manifest_mass_g"], 2), what=f"mass from solid volume ({r['material']}) vs manifest"))
    cases.sort(key=lambda c: -abs(c["value"]))
    return dict(per_part=raw["parts"], per_link=raw["links"], worst_cases=cases,
                bad_events=dict(printed_parts_without_material=sum(1 for r in raw["parts"].values() if r["material"] is None),
                                printed_parts_without_density=sum(1 for r in raw["parts"].values() if r["density_g_cm3"] is None)),
                evidence=[], notes="density from data/materials.json by the part's resolved material (checks/process.py effective: --process override, else the "
                                   "part's body definition process block, else part_settings.json; density_source per part); mass_legacy_density_g = the labelled "
                                   f"legacy comparison only ({LEGACY_MATERIAL}, {legacy * 1e3:g} g/cm3, the old fast_check value)")
