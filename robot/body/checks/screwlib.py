"""Screw helpers shared by the screw checks (screws, driver_access, connectivity): which screws touch a scope, the ring
points around a shank, the driver data (data/tools.json) and the driver path obstruction (a cylinder against solids)."""
from __future__ import annotations

import json
import re
import sys

import numpy as np

from checks import bodylib as BL
from checks.core import DATA, ROOT, kit

RING_MM = 0.8
NEAR_MM = 3.0       # selection ring: mount_check probes holes up to r_nominal + 3 mm around the axis, so a screw floating in a big hole still belongs to its part


FIRST_MIN_MM3 = 0.5      # an obstruction below this volume (a tessellation sliver at the head) is not the first obstruction

def mount_check():
    sys.path.insert(0, str(ROOT / "robot/body/fastening"))
    import mount_check as MC
    return MC


def screen(ctx, n):
    """(probe mesh, how, manifold) of part n as a neighbour is screened (coarse), cached on ctx apart from ctx.solid().
    Selections and graphs over the whole manifest use this, never ctx.solid(): a scoped part gets the fine 0.03 mm STEP
    tessellation there, and an unscoped run scopes every part (supplier STEPs up to 78 MB; T-c4sy9d).
    The probe mesh is what the screw points are tested against: the screening mesh when it is watertight, else the
    manifold's own mesh (a decimated closed CAD mesh can leave the decimation open: trimesh's ray-parity contains()
    then answers differently on repeated calls, measured on microphone_6049_v23 and foot_ADC_6494, 2026-11-20)."""
    import time
    from checks.core import screen_solid, tick
    c = ctx.__dict__.setdefault("_conn_solids", {})
    if n not in c:
        t0 = time.time()
        m, how = screen_solid(ctx.body, n, set())
        f = kit.mf(m)
        if not m.is_watertight:
            if not f.is_empty():
                m, how = kit.tm(f), how + ", probed as its manifold"
            else:
                how += ", not watertight (screw probe approximate)"
        c[n] = (m, how, f)
        tick("coarse solids", t0)
    return c[n]


def solid_nodes(ctx):
    """Non-hardware parts plus the steel sleeves (they are parts of the printed part they sit in)."""
    kind = mount_check().kind
    return [n for n, p in ctx.body.parts.items()
            if kind(p) is None and (not BL.is_fastener(n, p) or p.get("role") == "compression_limit_sleeves")]


def ring_points(g, ring_mm=RING_MM):
    """Points on rings just outside the shank (r_shank + ring_mm) along its length and under the head, in the screw's
    own (manifest neutral) frame. g = mount_check.screw_geometry."""
    d, h = np.asarray(g["_d"]), np.asarray(g["_head"])
    u = np.cross(d, [1, 0, 0] if abs(d[0]) < 0.9 else [0, 1, 0]); u /= np.linalg.norm(u); w = np.cross(d, u)
    ang = np.radians(np.arange(0, 360, 45))
    ring = np.array([np.cos(a) * u + np.sin(a) * w for a in ang])
    ts = np.linspace(g["head_height_mm"] + 0.2, g["length_mm"] - 0.2, 8)
    pts = [h + t * d + (g["r_shank_mm"] + ring_mm) * r for t in ts for r in ring]
    pts += [h + (g["head_height_mm"] + 0.3) * d + (g["r_shank_mm"] + ring_mm + 0.5 * max(0.0, g["r_head_mm"] - g["r_shank_mm"] - ring_mm)) * r for r in ring]
    return np.array(pts)


def standing_T(ctx):
    q0 = ctx.body.standing
    return {lk: ctx.body.link_T(lk, q0) for lk in {p["owning_link"] for p in ctx.body.parts.values()}}


def _corners(p):
    b = np.array(p["bounds_mm"], float)
    return np.array([[b[i][0], b[j][1], b[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])


def world_bounds(ctx, T, n):
    p = ctx.body.parts[n]
    Tl = T[p["owning_link"]]
    w = _corners(p) @ Tl[:3, :3].T + Tl[:3, 3]
    return np.array([w.min(0), w.max(0)])


def screws_of_scope(ctx, T=None, ring_mm=NEAR_MM):
    """{screw: dict(geometry, touched scoped parts, link transform)} of the screws that touch a scoped part or its steel sleeves:
    >= 2 points of a ring r_shank + ring_mm (default 3 mm) around the shank, along it and under the head, lie inside that
    part's solid at standing (the connectivity graph uses 0.8 mm; 3 mm is the window in which mount_check still reads a
    hole as the screw's). A selection, not a measurement: mount_check's own stack says which parts a screw really passes."""
    MC = mount_check()
    T = T or standing_T(ctx)
    parts = ctx.body.parts
    targets = {}
    for n in ctx.scope:
        targets[n] = n
        s = f"{n}_steel_sleeves"
        if s in parts:
            targets[s] = n
    WB = {t: world_bounds(ctx, T, t) for t in targets}
    out = {}
    for k, p in parts.items():
        if MC.kind(p) != "screw":
            continue
        try:
            g = MC.screw_geometry(BL.load_mesh(p["mesh"]))
        except Exception:  # noqa: BLE001
            continue
        pts = ring_points(g, ring_mm)
        Ts = T[p["owning_link"]]
        pw = pts @ Ts[:3, :3].T + Ts[:3, 3]
        lo, hi = pw.min(0) - 0.5, pw.max(0) + 0.5
        touched = []
        for t, owner in targets.items():
            if np.any(WB[t][0] > hi) or np.any(WB[t][1] < lo):
                continue
            Tn = T[parts[t]["owning_link"]]
            try:
                inside = screen(ctx, t)[0].contains((pw - Tn[:3, 3]) @ Tn[:3, :3])
            except Exception:  # noqa: BLE001
                continue
            if inside.sum() >= 2 and owner not in touched:
                touched.append(owner)
        if touched:
            out[k] = dict(geometry=g, touched=touched, world=Ts)
    return out


# ------------------------------------------------------------------------------------------------ tools
def tools():
    return json.loads((DATA / "tools.json").read_text())


def head_type(name, g, thread, data=None, head_standard=None):
    """Head type of a screw (data/tools.json head_types): the type whose standards list its head standard (screw_plan or
    body definition, mount_check.screw_spec); without one, from its name, thread family and mesh."""
    if head_standard:
        data = data or tools()
        ht = next((k for k, v in data["head_types"].items() if head_standard in v.get("standards", [])), None)
        if ht:
            return ht
    if re.search(r"_CS\d+|countersunk", name, re.I):
        return "countersunk"
    if thread and thread.startswith("PA"):
        return "pan_phillips"
    if "NBK" in name:
        return "flange_button"
    d = {"M2": 2.0, "M2.5": 2.5, "M3": 3.0, "M4": 4.0}.get(thread, 2 * g["r_shank_mm"])
    return "button" if g["head_height_mm"] < 0.7 * d else "socket_cap"


def tool_for(head, thread, data=None):
    """(drive size, blade diameter mm, drive kind, source text) of the driver for a head type and thread family."""
    data = data or tools()
    kind = data["head_types"][head]["drive"]
    t = data["tools"][kind]
    sizes = t["by_head_type_and_thread"].get(head, {})
    size = sizes.get(thread)
    if size is None and thread and thread.startswith("PA"):
        size = sizes.get("M" + thread[2:])    # a thread-forming PA screw with a metric head standard: the drive of that head and diameter
    if size is None:
        return None
    if kind == "hex_key":
        blade = t["blade_dia_mm"]["up_to_af_2.0"] if size <= 2.0 else t["blade_dia_mm"]["above_af_2.0"]
    else:
        blade = t["blade_dia_mm"][size]
    return size, blade, kind, f"{t['size_source']}; blade: {t['blade_source']}"


def exit_distance(box, p0, d):
    """Distance along the unit direction d from p0 (inside or on the box [lo, hi]) to the box's far face."""
    lo, hi = box
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(d > 1e-9, (hi - p0) / d, np.where(d < -1e-9, (lo - p0) / d, np.inf))
    return float(max(0.0, t.min()))


def cylinder(p0, d, radius, length, segments=48):
    """Manifold cylinder starting at p0 going along the unit vector d."""
    import manifold3d as M
    import trimesh
    d = np.asarray(d, float) / np.linalg.norm(d)
    Rm = trimesh.geometry.align_vectors([0, 0, 1], d)[:3, :3]
    return M.Manifold.cylinder(float(length), float(radius), float(radius), segments).transform(np.c_[Rm, np.asarray(p0, float)])


def driver_path(head, out, radius, length, solids, start=0.05):
    """Obstruction of a tool path: a cylinder of `radius` from `start` mm beyond the head top `head` along the unit vector
    `out` (away from the screw), `length` long, against solids {name: manifold in the same frame}. Returns
    dict(obstruction_mm3, blocked_share_pct, first_part, first_distance_mm (None = free), by_part [{part, mm3, from_mm}])."""
    out = np.asarray(out, float) / np.linalg.norm(out)
    p0 = np.asarray(head, float) + start * out
    cyl = cylinder(p0, out, radius, length)
    total = float(cyl.volume())
    rows = []
    for n, m in solids.items():
        try:
            inter = cyl ^ m
            v = float(inter.volume())
        except Exception:  # noqa: BLE001
            continue
        if v <= 1e-6:
            continue
        verts = np.asarray(inter.to_mesh().vert_properties)[:, :3]
        rows.append(dict(part=n, mm3=round(v, 4), from_mm=round(float(((verts - head) @ out).min()), 3)))
    rows.sort(key=lambda r: r["from_mm"])
    vol = float(sum(r["mm3"] for r in rows))
    real = [r for r in rows if r["mm3"] >= FIRST_MIN_MM3]     # the first obstruction ignores slivers (tessellation contact)
    return dict(obstruction_mm3=round(vol, 3), blocked_share_pct=round(100.0 * vol / total, 2) if total else 0.0, path_volume_mm3=round(total, 2),
                first_part=real[0]["part"] if real else None, first_distance_mm=real[0]["from_mm"] if real else None,
                first_min_mm3=FIRST_MIN_MM3, slivers_ignored=len(rows) - len(real),
                by_part=sorted(rows, key=lambda r: -r["mm3"]))
