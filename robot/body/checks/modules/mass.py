"""mass: scoped part volume from its solid x density vs the manifest mass; owning link mass / COM. Raw function moved from leg/fast_check.py."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from checks.core import ROOT, R, kit, screen_solid, step_mesh, tick
from checks.registry import register

LEGACY_DENSITY = 1.22e-3     # g/mm3: the legacy fast_check value (material 'legacy_paht_cf_fast_check'), kept only as the named comparison


def check_mass(body, changed, sol, density=None):
    """density(name, part) -> g/mm3 or None; default = the legacy fast_check density for the printed parts (process.relations:
    definition kind, else role table), None for the others."""
    t0 = time.time()
    rows = {}
    if density is None:
        from checks import process as P
        rel = P.relations(body)
        density = lambda n, p: LEGACY_DENSITY if rel.is_printed(n) else None  # noqa: E731
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
    """Solid volume x density of the part's material (setting) vs manifest mass; also the legacy fast_check density; owning link mass and COM.
    Printed parts only (process.is_printed: process.printed, else the definition's kind, else the role table); the density is the
    one data/materials.json gives the part's material (definition process block, else part_settings.json)."""
    import json
    from checks import process as P
    from checks.core import DATA
    mats = json.loads((DATA / "materials.json").read_text())
    for n in ctx.scope:
        ctx.solid(n)
    printed = {n for n in ctx.scope if P.is_printed(ctx, n)}

    def dens(n, p):
        if n not in printed:
            return None
        m = ctx.settings(n)["process"].get("material")
        return mats[m]["density_g_cm3"] * 1e-3 if m else None
    raw = check_mass(ctx.body, ctx.scope, ctx.sol, dens)
    cases = []
    for n in list(raw["parts"]):
        if n not in printed:
            del raw["parts"][n]   # volume x density is for printed parts only
    for n, r in raw["parts"].items():
        r["material"] = ctx.settings(n)["process"]["material"]
        r["mass_legacy_density_g"] = round(float(ctx.sol[n][0].volume) * LEGACY_DENSITY, 2)
        if r["mass_from_volume_g"] is not None and r["manifest_mass_g"]:
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=r["centroid_mm"], unit="g (volume - manifest)",
                              value=round(r["mass_from_volume_g"] - r["manifest_mass_g"], 2), what=f"mass from solid volume ({r['material']}) vs manifest"))
    cases.sort(key=lambda c: -abs(c["value"]))
    return dict(per_part=raw["parts"], per_link=raw["links"], worst_cases=cases,
                bad_events=dict(printed_parts_without_material=sum(1 for r in raw["parts"].values() if r["material"] is None)),
                evidence=[], notes="density from data/materials.json by the part's material setting; mass_legacy_density_g = fast_check's 1.22 g/cm3")
