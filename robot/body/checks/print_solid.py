"""The solid the print checks (walls, bed, small_features, overhang, slicer) measure a part on.

The run's screening solid (ctx.solid, core.screen_solid), except for a mesh-only part whose closed mesh the screening
decimated (to 40k faces, within 1.5 % of its volume): the print checks read that part's full closed mesh
(core.full_closed_mesh). A decimation moves the surface detail they measure: the shin back cover read walls 0.037 vs
0.35 mm, 3 vs 10-12 bed islands and 125 vs ~400 Z steps on identical geometry. Neighbour screening keeps the decimated
solid. Same frame and same print transform as process.print_mesh."""
from __future__ import annotations

import numpy as np

from checks import core
from checks import process as P


def solid(ctx, n):
    """(trimesh, how) the print checks measure part n on, built once per run."""
    cache = ctx.__dict__.setdefault("_print_solids", {})
    if n not in cache:
        m, how = ctx.solid(n)
        if how.startswith("closed CAD mesh (decimated"):
            full = core.full_closed_mesh(ctx.body, n)
            if full is not None and len(full.faces) > len(m.faces):
                how = (f"closed CAD mesh, full ({len(full.faces)} faces; the neighbour screening solid is decimated to {len(m.faces)})"
                       + how.split(" of its volume)", 1)[1])
                m = full
        cache[n] = (m, how)
    return cache[n]


def print_mesh(ctx, n, eff=None, with_transform=False):
    """process.print_mesh on the print checks' solid: up -> +Z, plate_yaw applied, lowest point at z = 0."""
    eff = eff or P.effective(ctx, n)
    if eff["up"] is None:
        return (None, None) if with_transform else None
    m = solid(ctx, n)[0].copy()
    T = np.eye(4); T[:3, :3] = P.rotation(eff["up"], eff["plate_yaw_deg"])
    m.apply_transform(T)
    dz = -float(m.bounds[0][2])
    m.apply_translation([0.0, 0.0, dz])
    T[2, 3] = dz
    return (m, T) if with_transform else m
