"""Mountability of every screw connection in an assembly manifest (owner request 2026-11-17, Idle T-gckc6p).

For each screw in the manifest it measures, along the screw's own axis, the stack it actually passes through in the
CAD, and reports continuous values only (mm, deg, counts); nothing here accepts or rejects a connection:

  screw       axis, head position, head radius / height, shank radius and length from the screw mesh (a body of
              revolution: the axis is the distinct principal direction, the head is the wide end)
  radial      at every 0.25 mm along the axis, 24 rays outward from the axis into each nearby part: is the axis inside
              that part's material (solid), or in a hole (fitted hole radius, hole centre offset from the screw axis)
  axial       rays parallel to the axis on rings just outside the shank: each part's material segments along the
              axis, so every pair of consecutive layers (head -> washer -> sleeve / connector -> actuator ...) has
              a gap (mm, > 0 air, < 0 interpenetration) over the clamp zone
  terminal    where the thread bites: the part whose hole is smaller than the shank over the shank length
              (engagement mm), a nut, or, for standard parts (actuator cases, horns) without modelled tapped holes,
              the nearest tapped hole of that part's interface definition (robot/actuators/interfaces/)
  sleeve      the steel compression sleeve the screw passes through in a printed part (offset from the axis, its
              bore and how much of the head bears on it)

Then the reverse view: every sleeve without a screw through it, every standard-part mount hole without a screw.

  python3 infra/tools/memguard.py --limit-gb 6 --timeout-s 900 -- .venv/bin/python -B \
      robot/body/fastening/mount_check.py [--manifest PATH] [--only REGEX] [--out PATH]

Output: .cache/work/mount-check/<manifest stem>/report.json (derived; recreate by running this).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "robot/body/motor-refit"))
sys.path.insert(0, str(HERE))
import fit_check as F  # noqa: E402  (Body, load_mesh, src)
import interfaces as I  # noqa: E402  (standard-part instances and their defined holes)

DEFAULT_MANIFEST = ROOT / json.loads((ROOT / "robot/design.json").read_text())["body_manifest"]
HARDWARE_ROLES = {"fastener", "fastener_proxy", "nominal_fastener", "nominal_hardware", "standard_fastener_reference",
                  "standard_nominal", "spacer"}   # spacer: the B0196 / B0410 / compute M2 / M2.5 F-F standoffs
SCREW_RE = re.compile(r"(?:^|_)(M\d(?:p\d)?|B?PA\d)x(\d+(?:p\d+)?)|_CS\d+|countersunk", re.I)
NOMINAL_RE = re.compile(r"(?:^|_)(M\d(?:p\d)?|B?PA\d)(?:x(\d+(?:p\d+)?))?", re.I)
DT = 0.25            # radial probe step along the axis (mm)
N_DIR = 24           # radial rays per depth
R_MAX = 8.0          # radial probe range (mm)
RECESS_MM = 0.5      # a station this much wider than the narrowest is a pocket on the screw axis (nut trap, counterbore), not the bore
N_ANG = 16           # axial rays per ring
SEARCH = 5.0         # parts whose box comes within this of the screw's box are probed (mm)
RAY_CLEAR_MM = 0.1   # axial layer rays keep this far from the bore / recess wall radius of every part in the clamp chain


# ------------------------------------------------------------------------------------------------ classification
_DEFS: list | None = None
_SPEC: dict = {}


def definition_spec(name: str) -> dict:
    """The hardware spec (kind, thread, family, length_mm, standard ...) that the body definition claiming part `name`
    declares (robot/**/<builder>.bom.yaml via robot/body/bom/definitions.py, first claim wins as in its claims()); {}
    when no definition claims it as hardware or the definitions cannot be read."""
    global _DEFS
    if name in _SPEC:
        return _SPEC[name]
    if _DEFS is None:
        try:
            sys.path.insert(0, str(ROOT / "robot/body/bom"))
            import definitions as DEF
            _DEFS = [(DEF.names_of(e["match"]), e) for e in DEF.load().entries]
        except Exception:  # noqa: BLE001  (no definitions readable: the screw_plan and the name alone)
            _DEFS = []
    import fnmatch
    e = next((e for pats, e in _DEFS if any(fnmatch.fnmatchcase(name, p) for p in pats)), None)
    _SPEC[name] = (e.get("spec") or {}) if e and e.get("kind") == "hardware" else {}
    return _SPEC[name]


def kind(p) -> str | None:
    """screw / washer / nut / spacer / other_hardware / None (not hardware). Hardware by role; which hardware: the kind
    its body definition declares, else a screw_plan (a builder placed it as a screw), else the name (a thread token, with
    or without length, or a `screw` token make a screw)."""
    if p["role"] not in HARDWARE_ROLES:
        return None
    sk = definition_spec(p["name"]).get("kind")
    if sk:
        return sk if sk in ("screw", "washer", "nut", "spacer") else "other_hardware"
    if p.get("screw_plan"):
        return "screw"
    n = p["name"].lower()
    if "washer" in n:
        return "washer"
    if "nut" in n and "nbk" not in n:
        return "nut"
    if "spacer" in n or "standoff" in n:
        return "spacer"
    if SCREW_RE.search(p["name"]) or nominal(p["name"]).get("thread") or re.search(r"(^|_)screw(_|\d|$)", n):
        return "screw"
    return "other_hardware"


def nominal(name: str) -> dict:
    """Thread family and nominal length from a name such as R_knee_ST3025_PA2x8_0 or compute_M2p5_CS30_1 (the name
    token only; screw_spec() says what the screw is)."""
    m = NOMINAL_RE.search(name)
    if not m:
        return {}
    g = m.group(1).upper()
    fam = g.replace("P", ".") if g.startswith("M") else g.lstrip("B")
    d = dict(thread=fam)
    if m.group(2):
        d["nominal_length_mm"] = float(m.group(2).replace("p", "."))
    return d


def screw_spec(p) -> dict:
    """Thread family (M2 / M2.5 / M3 / PA2 ...), nominal length and head standard of a screw part: its screw_plan (written by
    the builder that placed it), else the spec of its body definition, else the name token (names can carry stale
    lengths). spec_source says which; name_length_mm is the name's length where it differs."""
    pl = p.get("screw_plan") or {}
    sp = definition_spec(p["name"])
    nm = nominal(p["name"])
    if pl.get("thread_label"):
        out = dict(thread=pl["thread_label"], nominal_length_mm=pl.get("length_mm"), head_standard=pl.get("head_standard"), spec_source="screw_plan")
    elif sp.get("kind") == "screw" and sp.get("thread"):
        th = ("PA" + sp["thread"].lstrip("M")) if sp.get("family") == "PA" else sp["thread"]
        out = dict(thread=th, nominal_length_mm=sp.get("length_mm"), head_standard=sp.get("standard"), spec_source="definition")
    elif nm:
        return dict(nm, spec_source="name")
    else:
        return {}
    if out["nominal_length_mm"] is None:
        out.pop("nominal_length_mm")
    if nm.get("nominal_length_mm") is not None and nm["nominal_length_mm"] != out.get("nominal_length_mm"):
        out["name_length_mm"] = nm["nominal_length_mm"]
    return out


# ------------------------------------------------------------------------------------------------ screw geometry
def revolution_axis(m: trimesh.Trimesh):
    """Axis of a body of revolution: the principal direction of the surface's second moment whose eigenvalue stands
    apart from the other two (deterministic: area-weighted triangle moments)."""
    tri = m.triangles
    w = m.area_faces
    if w.sum() <= 0:
        pts = m.vertices; c = pts.mean(0); C = np.cov((pts - c).T)
    else:
        cen = tri.mean(1)
        c = (cen * w[:, None]).sum(0) / w.sum()
        # exact second moment of each triangle about c: (sum over vertex pairs) / 12 * area
        v = tri - c
        S = np.einsum("ni,nj->nij", v.sum(1), v.sum(1)) + np.einsum("nki,nkj->nij", v, v)
        C = (S * w[:, None, None]).sum(0) / (12.0 * w.sum())
    ev, V = np.linalg.eigh(C)
    pair = [abs(ev[1] - ev[2]), abs(ev[0] - ev[2]), abs(ev[0] - ev[1])]
    k = int(np.argmin(pair))
    return V[:, k] / np.linalg.norm(V[:, k]), c


def screw_geometry(m: trimesh.Trimesh) -> dict:
    d, c = revolution_axis(m)
    v = m.vertices
    t = (v - c) @ d
    r = np.linalg.norm((v - c) - np.outer(t, d), axis=1)
    # recentre the axis on the vertices (the centroid can sit off-axis for a head with a slot / hex socket)
    lo, hi = t.min(), t.max()
    L = hi - lo
    end_lo = r[t <= lo + 0.2 * L].max()
    end_hi = r[t >= hi - 0.2 * L].max()
    if end_hi > end_lo + 1e-6:          # the head is at the high end: point d from the head to the tip
        d, t = -d, -t
        lo, hi = -hi, -lo
    head = c + lo * d
    s = t - lo                           # 0 at the head top, L at the tip
    r_head = float(r[s <= 0.35 * L].max())
    # head height: last station (from the head) whose radius is still close to the head radius
    st = np.unique(np.round(s, 2))
    rad = np.array([r[np.abs(s - x) < 0.011].max() for x in st])
    wide = st[rad >= 0.85 * r_head]
    head_h = float(wide.max()) if len(wide) else 0.0
    shank = rad[(st > head_h + 0.05) & (st < L - 0.05)]
    r_shank = float(np.median(shank)) if len(shank) else float(rad[-1])
    return dict(head_mm=head.round(3).tolist(), axis=d.round(4).tolist(), length_mm=round(float(L), 3),
                r_head_mm=round(r_head, 3), head_height_mm=round(head_h, 3), r_shank_mm=round(r_shank, 3),
                _head=head, _d=d)


def basis(d):
    u = np.cross(d, [1.0, 0.0, 0.0] if abs(d[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    return u, np.cross(d, u)


def rays_for(mesh):
    try:
        return trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    except Exception:  # noqa: BLE001
        return trimesh.ray.ray_triangle.RayMeshIntersector(mesh)


# ------------------------------------------------------------------------------------------------ probes
FIT_TOL_MM = 0.02     # a ray lies on the fitted bore circle within this, or within FIT_TOL_REL x its radius (a bore tessellated
FIT_TOL_REL = 0.02    # as a 12-gon deviates +-0.017 r; ROCK4D's stray wedges stand 0.1-0.25 mm into its 1.35 mm bores)


def _circle(p):
    A = np.c_[2 * p, np.ones(len(p))]
    sol, *_ = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None)
    cx, cy = sol[:2]
    return cx, cy, math.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0))


def robust_circle(p, tol=FIT_TOL_MM, rel=FIT_TOL_REL, min_share=1 / 3):
    """Circle through the ray hits p (2D, mm) of one depth that most hits lie on: the circumcircle of every triple of hits
    is a candidate, the one with the most hits within max(tol, rel x r) wins (then the smaller rms), refitted by least
    squares to those hits. Stray triangle wedges inside a mount hole and rays escaping through gaps of an open supplier
    mesh (ROCK4D: 1.1-1.2 mm wedges and 3-24 mm escapes around a 1.35 mm bore) no longer move the centre.
    Returns (cx, cy, r, hits on the circle), or None when fewer than min_share of N_DIR hits lie on one circle."""
    import itertools
    n = len(p)
    if n < 3 or n < N_DIR * min_share:
        return None
    tri = np.array(list(itertools.combinations(range(n), 3)))
    A, B, C = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
    dd = 2 * (A[:, 0] * (B[:, 1] - C[:, 1]) + B[:, 0] * (C[:, 1] - A[:, 1]) + C[:, 0] * (A[:, 1] - B[:, 1]))
    ok = np.abs(dd) > 1e-9
    a2, b2, c2 = (A ** 2).sum(1), (B ** 2).sum(1), (C ** 2).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ux = (a2 * (B[:, 1] - C[:, 1]) + b2 * (C[:, 1] - A[:, 1]) + c2 * (A[:, 1] - B[:, 1])) / dd
        uy = (a2 * (C[:, 0] - B[:, 0]) + b2 * (A[:, 0] - C[:, 0]) + c2 * (B[:, 0] - A[:, 0])) / dd
    rr = np.hypot(A[:, 0] - ux, A[:, 1] - uy)
    ok &= rr < R_MAX
    if not ok.any():
        return None
    ux, uy, rr = ux[ok], uy[ok], rr[ok]
    res = np.abs(np.hypot(p[None, :, 0] - ux[:, None], p[None, :, 1] - uy[:, None]) - rr[:, None])
    inl = res <= np.maximum(tol, rel * rr)[:, None]
    cnt = inl.sum(1)
    rms = np.sqrt((np.where(inl, res, 0.0) ** 2).sum(1) / np.maximum(cnt, 1))
    best = int(np.lexsort((rms, -cnt))[0])
    keep = inl[best]
    if keep.sum() < N_DIR * min_share:
        return None
    cx, cy, r = _circle(p[keep])
    return cx, cy, r, int(keep.sum())


def radial_probe(mesh, rays, head, d, t_lo, t_hi):
    """Per depth: state (hole / pocket / solid / partial / none); for a hole the fitted bore radius, its centre offset from
    the axis and the number of rays on the fitted circle (robust_circle); a pocket is a wide opening with no round bore."""
    u, w = basis(d)
    ang = np.linspace(0, 2 * np.pi, N_DIR, endpoint=False)
    dirs = np.outer(np.cos(ang), u) + np.outer(np.sin(ang), w)
    ts = np.arange(t_lo, t_hi + 1e-9, DT)
    org = np.repeat(head[None] + ts[:, None] * d[None], N_DIR, 0)
    dr = np.tile(dirs, (len(ts), 1))
    loc, idx, tri = rays.intersects_location(org, dr, multiple_hits=False)
    dist = np.full(len(org), np.nan)
    sign = np.zeros(len(org))
    if len(idx):
        dd = np.linalg.norm(loc - org[idx], axis=1)
        ok = dd <= R_MAX
        dist[idx[ok]] = dd[ok]
        sign[idx[ok]] = np.sign(np.einsum("ij,ij->i", mesh.face_normals[tri[ok]], dr[idx[ok]]))
    dist = dist.reshape(len(ts), N_DIR); sign = sign.reshape(len(ts), N_DIR)
    out = []
    for i, t in enumerate(ts):
        hit = ~np.isnan(dist[i])
        enter = hit & (sign[i] < 0)
        exitn = hit & (sign[i] > 0)
        row = dict(t=round(float(t), 3), n_enter=int(enter.sum()), n_exit=int(exitn.sum()))
        if exitn.sum() > N_DIR * 0.5:
            row["state"] = "solid"
        elif enter.sum() >= N_DIR * 0.5:
            p = dist[i, enter, None] * (np.cos(ang[enter])[:, None] * [1, 0] + np.sin(ang[enter])[:, None] * [0, 1])
            fit = robust_circle(p)
            row["r_min"] = float(dist[i, enter].min())
            if fit is None:              # no round bore at this depth: a hex nut trap, a slot, a pocket wall seen through gaps
                row["state"] = "pocket"
            else:
                cx, cy, rad, n_fit = fit
                row.update(state="hole", hole_r=rad, offset=math.hypot(cx, cy), n_fit=n_fit, offset_vec=(cx * u + cy * w))
        else:
            row["state"] = "none" if hit.sum() == 0 else "partial"
            if enter.any():
                row["r_min"] = float(dist[i, enter].min())
        out.append(row)
    return out


def axial_segments(mesh, rays, head, d, radii, t_lo, t_hi):
    """Material segments [t_in, t_out] along rays parallel to the axis: {radius: [list per ray]}."""
    u, w = basis(d)
    ang = np.linspace(0, 2 * np.pi, N_ANG, endpoint=False)
    res = {}
    for r in radii:
        off = r * (np.outer(np.cos(ang), u) + np.outer(np.sin(ang), w))
        org = head[None] + off + t_lo * d[None]
        dr = np.repeat(d[None], N_ANG, 0)
        loc, idx, tri = rays.intersects_location(org, dr, multiple_hits=True)
        segs = [[] for _ in range(N_ANG)]
        if len(idx):
            tt = ((loc - org[idx]) @ d) + t_lo
            sg = np.einsum("ij,j->i", mesh.face_normals[tri], d)
            for k in range(N_ANG):
                sel = idx == k
                order = np.argsort(tt[sel])
                hs = list(zip(tt[sel][order], sg[sel][order]))
                ded = []
                for h in hs:                     # embree reports a hit on each triangle at a shared edge
                    if not ded or abs(h[0] - ded[-1][0]) > 1e-4 or np.sign(h[1]) != np.sign(ded[-1][1]):
                        ded.append(h)
                cur = None
                for tv, sv in ded:
                    if sv < 0 and cur is None:
                        cur = tv
                    elif sv > 0:
                        start = cur if cur is not None else -np.inf  # ray started inside material: open start
                        if tv > t_lo:
                            segs[k].append((start, tv))
                        cur = None
                if cur is not None:
                    segs[k].append((cur, np.inf))                   # no exit found: open end
                segs[k] = [s for s in segs[k] if s[1] > t_lo and s[0] < t_hi and s[1] - s[0] > 1e-3]
        res[r] = segs
    return res




# ------------------------------------------------------------------------------------------------ tool paths
class BodyScene:
    """One ray scene of all probed parts (a single embree scene; hundreds of per-part scenes crash embree)."""

    def __init__(self, names, meshes):
        vs, fs, owner, off = [], [], [], 0
        for i, n in enumerate(names):
            m = meshes[n]
            vs.append(m.vertices); fs.append(m.faces + off); owner.append(np.full(len(m.faces), i)); off += len(m.vertices)
        self.names = list(names)
        self.mesh = trimesh.Trimesh(np.vstack(vs), np.vstack(fs), process=False)
        self.owner = np.concatenate(owner)
        self.rays = rays_for(self.mesh)


def path_obstruction(head, d, radius, length, scene, exclude, n_ring=3, n_ang=12):
    """Rays from the head top outward (-d) on a disc of the given radius: share of rays hitting a part within
    `length` (%), and the nearest hit (mm, None = free over `length`), ignoring the parts in `exclude`."""
    u, w = basis(d)
    pts = [head]
    for r in np.linspace(radius / n_ring, radius, n_ring):
        for a in np.linspace(0, 2 * np.pi, n_ang, endpoint=False):
            pts.append(head + r * (math.cos(a) * u + math.sin(a) * w))
    org = np.array(pts) - 0.02 * d
    dr = np.repeat(-d[None], len(org), 0)
    first = np.full(len(org), np.inf)
    who = [None] * len(org)
    loc, idx, tri = scene.rays.intersects_location(org, dr, multiple_hits=True)
    if len(idx):
        dist = np.linalg.norm(loc - org[idx], axis=1)
        for i, dd, f in zip(idx, dist, tri):
            n = scene.names[scene.owner[f]]
            if n in exclude or dd > length:
                continue
            if dd < first[i]:
                first[i] = dd; who[i] = n
    hit = first <= length
    near = float(first.min()) if hit.any() else None
    return dict(blocked_pct=round(100.0 * hit.mean(), 1), nearest_mm=None if near is None else round(near, 2),
                nearest_part=who[int(np.argmin(first))] if near is not None else None)


# ------------------------------------------------------------------------------------------------ one screw
PRINTED = ("printed_structural_connector", "printed_structural_carrier", "body_shell")
DRIVER_L = 40.0     # mm of straight access behind the head for a screwdriver / hex key


def analyse_screw(name, body, cands, meshes, rayss, kinds, iface_holes, fam, scenes):
    p = body.parts[name]
    nom = screw_spec(p)
    g = screw_geometry(meshes[name])
    head, d, L = g.pop("_head"), g.pop("_d"), g["length_mm"]
    rh, hh = g["r_head_mm"], g["head_height_mm"]
    rs = fam.get(nom.get("thread"), {}).get("major_dia_mm", 2 * g["r_shank_mm"]) / 2.0
    t_lo, t_hi = -1.0, L + 2.0
    shank = (hh, L)
    parts = {}
    for c in cands:
        rad = radial_probe(meshes[c], rayss[c], head, d, t_lo, t_hi)
        on_shank = [x for x in rad if shank[0] + 0.1 <= x["t"] <= shank[1] - 0.3]
        holes = [x for x in on_shank if x["state"] == "hole" and x["hole_r"] < rs + 3.0]
        pockets = [x for x in on_shank if x["state"] == "pocket" and x["r_min"] < rs + 3.0]
        solid = [x for x in on_shank if x["state"] == "solid"]
        if not holes and not solid and not pockets:
            continue
        thread = [x for x in holes if x["hole_r"] < rs - 0.05]
        clear = [x for x in holes if x["hole_r"] >= rs - 0.05]
        span = holes + pockets + solid
        pr = dict(role=body.parts[c]["role"], hardware=kinds.get(c),
                  t_range_mm=[round(min(x["t"] for x in span), 2), round(max(x["t"] for x in span) + DT, 2)],
                  solid_on_axis_mm=round(len(solid) * DT, 2), thread_overlap_mm=round(len(thread) * DT, 2),
                  clearance_len_mm=round(len(clear) * DT, 2))
        recess = []
        if holes:
            # the bore is the narrowest stretch; stations wider by more than RECESS_MM are a pocket on the same axis (nut trap,
            # counterbore, head recess): their circle fit says nothing about where the bore sits, so they are reported apart
            rmin = min(x["hole_r"] for x in holes)
            bore = [x for x in holes if x["hole_r"] <= rmin + RECESS_MM]
            recess = [x for x in holes if x["hole_r"] > rmin + RECESS_MM]
            hr = np.array([x["hole_r"] for x in bore]); of = np.array([x["offset"] for x in bore])
            pr.update(hole_dia_mm=round(2 * float(np.median(hr)), 3), axis_offset_mm=round(float(np.median(of)), 3),
                      axis_offset_max_mm=round(float(of.max()), 3), radial_clearance_mm=round(float(np.median(hr)) - rs, 3),
                      bore_stations=len(bore), bore_rays_on_circle_min=min(x["n_fit"] for x in bore))
            if recess:
                pr.update(recess_dia_mm=round(2 * float(np.median([x["hole_r"] for x in recess])), 3))
            if len(bore) >= 4 and bore[-1]["t"] - bore[0]["t"] > 0.4:
                tt = np.array([x["t"] for x in bore]); ov = np.array([x["offset_vec"] for x in bore])
                slope = np.polyfit(tt, ov, 1)[0]                 # least-squares drift of the bore centre per mm of depth (3 components)
                pr["axis_tilt_deg"] = round(math.degrees(math.atan(float(np.linalg.norm(slope)))), 2)
        if recess or pockets:          # pockets: wide openings with no round bore (hex nut traps); r_min is their narrowest
            pr["recess_len_mm"] = round((len(recess) + len(pockets)) * DT, 2)
            if pockets:
                pr["pocket_len_mm"] = round(len(pockets) * DT, 2)
                pr["pocket_r_min_mm"] = round(min(x["r_min"] for x in pockets), 3)
        parts[c] = pr
    stack = sorted(parts, key=lambda c: parts[c]["t_range_mm"][0])

    # terminal: defined tapped hole of a standard part > nut > modelled thread in another part
    ih = I.match_screw(iface_holes, head, d, shank) if iface_holes else None
    nuts = [c for c in stack if parts[c]["hardware"] == "nut"]
    thr = [(parts[c]["thread_overlap_mm"], c) for c in stack if parts[c]["thread_overlap_mm"] > 0 and parts[c]["hardware"] != "washer"]
    if ih:
        term = dict(kind="defined_tapped_hole", part=ih["part"], engagement_mm=ih["engagement_mm"],
                    reference_engagement_mm=ih["reference_engagement_mm"], axis_offset_mm=ih["axis_offset_mm"],
                    angle_deg=ih["angle_deg"], hole=ih)
        if ih["part"] in parts:
            term["modelled_thread_overlap_mm"] = parts[ih["part"]]["thread_overlap_mm"]
    elif nuts:
        c = nuts[0]
        term = dict(kind="nut", part=c, engagement_mm=parts[c]["clearance_len_mm"] + parts[c]["thread_overlap_mm"],
                    axis_offset_mm=parts[c].get("axis_offset_mm"))
    elif thr:
        e, c = max(thr)
        mat = "plastic" if body.parts[c]["role"] in PRINTED else "unknown"
        term = dict(kind="thread_in_part", part=c, engagement_mm=e, axis_offset_mm=parts[c].get("axis_offset_mm"),
                    reference_engagement_mm=round(2.0 * 2 * rs, 2) if mat == "plastic" else None,
                    motor_part_without_definition=body.parts[c]["role"] in I.MOTOR_ROLES)
    else:
        term = None
    tpart = term["part"] if term else None

    # clamp chain: every part between the head and the terminal that the screw passes through with clearance
    t_term = parts[tpart]["t_range_mm"][0] if tpart in parts else (term["hole"]["opening_depth_along_screw_mm"] if term and term.get("hole") else L)
    clamped = [c for c in stack if c != tpart and parts[c]["t_range_mm"][0] < t_term - 0.05 and parts[c]["hardware"] != "nut"]
    walls = {c: parts[c]["solid_on_axis_mm"] for c in stack if parts[c]["solid_on_axis_mm"] > 0 and c != tpart}

    # axial layers: head -> washer -> sleeve / printed part -> ... -> terminal. Levels = chain parts ordered along the
    # screw, a sleeve sharing the level of the printed part it sits in; each part's segments count only inside its own
    # window along the axis (where the radial probe found it, +-1 mm), so far-away faces of large parts never pair up.
    chain = [name] + clamped + ([tpart] if tpart else [])
    win = {name: (-1.0, hh + 1.0)}
    for c in clamped:
        win[c] = (parts[c]["t_range_mm"][0] - 1.0, parts[c]["t_range_mm"][1] + 1.0)
    if tpart:
        t0 = parts[tpart]["t_range_mm"][0] if tpart in parts else t_term
        win[tpart] = (t0 - 1.0, t0 + 1.0)
    levels, level_of = [], {}
    for c in sorted(clamped, key=lambda c: parts[c]["t_range_mm"][0]):
        host = next((lv for lv in levels if parts[c]["role"] == "compression_limit_sleeves" and any(
            body.parts[h]["role"] in PRINTED and parts[h]["t_range_mm"][0] - 0.3 <= parts[c]["t_range_mm"][0]
            and parts[c]["t_range_mm"][1] <= parts[h]["t_range_mm"][1] + 0.3 for h in lv)), None)
        if host is None:
            host = []; levels.append(host)
        host.append(c)
    for i, lv in enumerate(levels):
        for c in lv:
            level_of[c] = i + 1
    level_of[name] = 0
    if tpart:
        level_of[tpart] = len(levels) + 1
    radii = sorted({round(x, 2) for x in (rs + 0.3, rs + 0.7, (rs + rh) / 2 + 0.2, rh - 0.1, rh + 0.6, rh + 1.5) if x > rs + 0.2})
    # no ray within RAY_CLEAR_MM of a chain part's bore or recess wall: a ray grazing the bore a sleeve sits in reads the
    # sleeve's outer skin as 0-0.44 mm segments (M2.5 ISO 4762: (rs + rh) / 2 + 0.2 = 1.95 against a 3.96 mm bore); moved
    # inward first, then outward
    wall_r = [v / 2 for c in chain if c in parts for v in (parts[c].get("hole_dia_mm"), parts[c].get("recess_dia_mm")) if v]
    radii = sorted({round(next((y for y in (x, x - RAY_CLEAR_MM, x + RAY_CLEAR_MM, x - 2 * RAY_CLEAR_MM, x + 2 * RAY_CLEAR_MM)
                                if y > rs + 0.2 and all(abs(y - wr) >= RAY_CLEAR_MM - 1e-9 for wr in wall_r)), x), 2) for x in radii})
    seg = {c: axial_segments(meshes[c], rayss[c], head, d, radii, t_lo - 2.0, t_hi) for c in chain if c in meshes}
    contacts, overlaps = {}, {}
    for r in radii:
        for k in range(N_ANG):
            evs = sorted((s0, s1, c) for c in seg for (s0, s1) in seg[c][r][k]
                         if s1 > win[c][0] and s0 < win[c][1])
            for i, (a0, a1, ca) in enumerate(evs):
                seen = set()
                for (b0, b1, cb) in evs[i + 1:]:
                    if ca == cb or cb in seen:
                        continue
                    seen.add(cb)
                    if (np.isfinite(a0) and np.isfinite(a1) and np.isfinite(b0) and np.isfinite(b1)   # closed segments only
                            and b0 < a1 - 1e-3 and (ca, cb) != (name, tpart)):
                        overlaps.setdefault((ca, cb), []).append(-(min(a1, b1) - max(a0, b0)))   # -overlap length
                    # a layer pair is the upper part's segment that ends last among those starting before the lower part: an
                    # earlier segment of the same part (a component on a board, the far face of a flange) is not the face that bears
                    if (level_of[cb] == level_of[ca] + 1 and np.isfinite(a1) and np.isfinite(b0)
                            and a1 >= max((e1 for e0, e1, ce in evs if ce == ca and e0 < b0 and np.isfinite(e1)), default=a1) - 1e-6):
                        contacts.setdefault((ca, cb), []).append(b0 - a1)
    pairs = []
    for (a, b), v in contacts.items():
        gaps = np.array(v)
        pairs.append(dict(upper=a, lower=b, interface=True, min_gap_mm=round(float(gaps.min()), 3),
                          max_gap_mm=round(float(gaps.max()), 3), median_gap_mm=round(float(np.median(gaps)), 3),
                          rays=len(v), touching_pct=round(100.0 * float((np.abs(gaps) <= 0.05).mean()), 1)))
    for (a, b), v in overlaps.items():
        if (a, b) in contacts:
            continue
        gaps = np.array(v)
        servo = I.MOTOR_ROLES
        pairs.append(dict(upper=a, lower=b, interface=False, min_gap_mm=round(float(gaps.min()), 3), rays=len(v),
                          same_servo=body.parts[a]["role"] in servo and body.parts[b]["role"] in servo))   # spline fit
    # a level is a part with the sleeves pressed into it: the stack clamps where any of them bears on the next level, so each
    # interface also carries the level's gap (the smallest of its parts' gaps) and whether it is the part that bears (a sleeve
    # end over a counterbore reads +3.7 mm while the plate around it sits on the fan face)
    lv_gap = {}
    for x in pairs:
        if x["interface"]:
            k = (level_of[x["upper"]], level_of[x["lower"]])
            lv_gap[k] = min(lv_gap.get(k, np.inf), x["min_gap_mm"])
    for x in pairs:
        if x["interface"]:
            g_ = lv_gap[(level_of[x["upper"]], level_of[x["lower"]])]
            x.update(level_gap_mm=g_, bears=x["min_gap_mm"] == g_)
    pairs.sort(key=lambda x: (level_of.get(x["upper"], 99), level_of.get(x["lower"], 99)))
    seat = [x for x in pairs if x["upper"] == name and x["interface"]]
    seat = min(seat, key=lambda x: abs(x["min_gap_mm"])) if seat else None

    # sleeves: every clamped printed part should carry one where the screw passes
    sleeves = []
    for c in clamped:
        if body.parts[c]["role"] not in PRINTED:
            continue
        t0, t1 = parts[c]["t_range_mm"]
        sl = [s for s in stack if parts[s]["role"] == "compression_limit_sleeves"
              and parts[s]["t_range_mm"][0] < t1 - 0.1 and parts[s]["t_range_mm"][1] > t0 + 0.1]
        row = dict(printed_part=c, printed_role=body.parts[c]["role"], printed_hole_dia_mm=parts[c].get("hole_dia_mm"),
                   clamped_len_mm=round(t1 - t0, 2), sleeve=sl[0] if sl else None)
        if sl:
            s = parts[sl[0]]
            row.update(sleeve_bore_mm=s.get("hole_dia_mm"), sleeve_axis_offset_mm=s.get("axis_offset_mm"),
                       sleeve_len_mm=round(s["t_range_mm"][1] - s["t_range_mm"][0], 2),
                       sleeve_od_mm=parts[c].get("hole_dia_mm"))
            if s.get("hole_dia_mm") and parts[c].get("hole_dia_mm") and seat and seat["lower"] in (sl[0], c):
                row["head_bearing_ring_mm"] = round(min(rh, parts[c]["hole_dia_mm"] / 2) - s["hole_dia_mm"] / 2, 3)
        sleeves.append(row)

    excl = set(chain) | {name}
    ins = path_obstruction(head, d, rh + 0.1, L, scenes["no_shells"], excl)
    drv = path_obstruction(head, d, max(rh, 2.0) + 0.5, DRIVER_L, scenes["all"], excl)
    drv_ns = path_obstruction(head, d, max(rh, 2.0) + 0.5, DRIVER_L, scenes["no_shells"], excl)
    thread_match = None
    if term and term.get("hole"):
        hole_thr, scr = term["hole"]["thread"], nom.get("thread")
        thread_match = dict(screw=scr, hole=hole_thr, terminal_material=term["hole"]["terminal_material"],
                            same_family=(scr == hole_thr) or (scr == "PA2" and hole_thr == "M2"
                                                                and term["hole"]["terminal_material"] == "plastic"))
    clamped_holes = [dict(part=c, hole_dia_mm=parts[c].get("hole_dia_mm"), axis_offset_mm=parts[c].get("axis_offset_mm"),
                          radial_clearance_mm=parts[c].get("radial_clearance_mm"), axis_tilt_deg=parts[c].get("axis_tilt_deg"))
                     for c in clamped]
    return dict(name=name, role=p["role"], owning_link=p.get("owning_link"), **nom, **g, r_nominal_mm=rs,
                stack=stack, parts=parts, terminal=term, clamped=clamped, clamped_holes=clamped_holes,
                contacts=pairs, head_seat=seat, solid_wall_on_axis_mm=walls, sleeves=sleeves, thread_match=thread_match,
                insertion_path=ins, driver_path=drv, driver_path_without_shells=drv_ns)


# ------------------------------------------------------------------------------------------------ findings
REF = dict(offset_mm=0.10, gap_mm=0.05, tilt_deg=1.0)


def findings(r) -> list[dict]:
    """Measured deviations of one screw from the owner's intent (2026-11-18), each with its value and reference."""
    f = []
    add = lambda kind, value, ref, note="": f.append(dict(kind=kind, value=value, reference=ref, note=note))  # noqa: E731
    t = r["terminal"]
    if t is None:
        add("no_terminal", None, "a defined tapped hole, a nut or modelled thread", "the screw bites into nothing")
    else:
        if t.get("motor_part_without_definition"):
            add("motor_part_without_definition", t["engagement_mm"], "a defined tapped hole",
                f"{t['part']}: servo part ends the screw but has no official definition; engagement only from the printed mesh")
        if t.get("axis_offset_mm") is not None and t["axis_offset_mm"] > REF["offset_mm"]:
            add("terminal_off_axis_mm", t["axis_offset_mm"], REF["offset_mm"], t["part"])
        if t.get("angle_deg") is not None and t["angle_deg"] > REF["tilt_deg"]:
            add("terminal_tilt_deg", t["angle_deg"], REF["tilt_deg"], t["part"])
        if t.get("reference_engagement_mm") and t["engagement_mm"] < t["reference_engagement_mm"] - 1e-6:
            add("engagement_mm", t["engagement_mm"], t["reference_engagement_mm"], t["part"])
        h = t.get("hole")
        if h and h["tip_short_of_hole_bottom_mm"] < -0.05:
            add("tip_past_hole_bottom_mm", -h["tip_short_of_hole_bottom_mm"], 0.0, f"{t['part']}: screw longer than the tapped depth")
    if r["thread_match"] and not r["thread_match"]["same_family"]:
        tm = r["thread_match"]
        add("thread_family", f"{tm['screw']} into {tm['hole']} ({tm['terminal_material']})", "same family", "")
    for c, v in r["solid_wall_on_axis_mm"].items():
        add("shank_in_solid_mm", v, 0.0, c)
    for h in r["clamped_holes"]:
        if h["axis_offset_mm"] is not None and h["axis_offset_mm"] > REF["offset_mm"]:
            add("clamped_hole_off_axis_mm", h["axis_offset_mm"], REF["offset_mm"], h["part"])
        if h["radial_clearance_mm"] is not None and h["radial_clearance_mm"] < 0:
            add("clamped_hole_too_small_mm", h["radial_clearance_mm"], ">= 0", h["part"])
    s = r["head_seat"]
    if s is None:
        add("head_seat", None, "head on a layer", "no layer found under the head within 3 mm")
    elif abs(s["min_gap_mm"]) > REF["gap_mm"]:
        add("head_seat_gap_mm", s["min_gap_mm"], f"0 +- {REF['gap_mm']}", s["lower"])
    for c in r["contacts"]:
        if c["upper"] == r["name"] and c["interface"]:
            continue                                   # the head seat, above
        if c["min_gap_mm"] < -REF["gap_mm"] and not c.get("same_servo"):
            add("layers_interpenetrate_mm", c["min_gap_mm"], f"0 +- {REF['gap_mm']}", f"{c['upper']} / {c['lower']}")
        elif c["interface"] and c.get("bears", True) and c.get("level_gap_mm", c["min_gap_mm"]) > REF["gap_mm"]:
            add("layers_gap_mm", c["min_gap_mm"], f"0 +- {REF['gap_mm']}", f"{c['upper']} / {c['lower']}")   # once per level pair: its bearing part
    for sl in r["sleeves"]:
        if sl["sleeve"] is None:
            add("clamped_printed_hole_without_sleeve", sl["clamped_len_mm"], "a steel sleeve", sl["printed_part"])
        else:
            if sl.get("sleeve_axis_offset_mm") and sl["sleeve_axis_offset_mm"] > REF["offset_mm"]:
                add("sleeve_off_axis_mm", sl["sleeve_axis_offset_mm"], REF["offset_mm"], sl["sleeve"])
            if sl.get("head_bearing_ring_mm") is not None and sl["head_bearing_ring_mm"] < 0.3:
                add("head_bearing_on_sleeve_mm", sl["head_bearing_ring_mm"], ">= 0.3 (head on the steel end)", sl["sleeve"])
    if r["insertion_path"]["blocked_pct"] > 0:
        add("insertion_path_blocked_pct", r["insertion_path"]["blocked_pct"], 0.0,
            f"{r['insertion_path']['nearest_part']} at {r['insertion_path']['nearest_mm']} mm (shells off)")
    if r["driver_path_without_shells"]["blocked_pct"] > 0:
        add("driver_path_blocked_pct", r["driver_path_without_shells"]["blocked_pct"], 0.0,
            f"{r['driver_path_without_shells']['nearest_part']} at {r['driver_path_without_shells']['nearest_mm']} mm (shells off)")
    return f


# ------------------------------------------------------------------------------------------------ whole manifest
def box_near(b, lo, hi, m):
    return not (np.any(np.asarray(b[1]) < np.asarray(lo) - m) or np.any(np.asarray(b[0]) > np.asarray(hi) + m))


def tubes(mesh):
    """Each connected component of a sleeve mesh as (centre, unit axis, length, outer dia, inner dia)."""
    out = []
    for comp in mesh.split(only_watertight=False):
        if len(comp.vertices) < 8:
            continue
        a, c = revolution_axis(comp)
        v = comp.vertices - c
        t = v @ a
        r = np.linalg.norm(v - np.outer(t, a), axis=1)
        out.append((c + 0.5 * (t.max() + t.min()) * a, a, float(t.max() - t.min()), float(2 * r.max()), float(2 * r.min())))
    merged = []                           # a tube whose wall faces are not stitched splits into an inner and an outer skin
    for o in sorted(out, key=lambda o: -o[3]):
        if any(np.linalg.norm(o[0] - m[0]) < 0.3 and abs(abs(float(o[1] @ m[1])) - 1) < 1e-3 for m in merged):
            continue
        merged.append(o)
    return merged


def run(manifest_path: Path, only: str | None, out: Path, part: str | None = None, chunk: int = 30):
    """Whole manifest: screws in chunks, each in its own process (embree leaks native memory per scene), then the
    reverse view and the per-connector ranking in this process. `part` = 'a:b' runs one chunk (internal)."""
    import subprocess
    if part is None and not only:
        man = json.loads(manifest_path.read_text())
        body = F.Body(man)
        n = sum(1 for nm, p in body.parts.items() if kind(p) == "screw")
        rows = []
        for a in range(0, n, chunk):
            po = out.parent / f"part-{a}.json"
            cmd = [sys.executable, "-B", str(Path(__file__)), "--manifest", str(manifest_path), "--part", f"{a}:{a + chunk}",
                   "--out", str(po)]
            subprocess.run(cmd, check=True)
            rows += json.loads(po.read_text())["rows"]
            po.unlink()
        return finish(manifest_path, man, body, rows, out)
    return run_screws(manifest_path, only, out, part)


def finish(manifest_path, man, body, rows, out):
    t0 = time.time()
    meshes = {}

    def mesh(n):
        if n not in meshes:
            meshes[n] = body.mesh(n)
        return meshes[n]
    iface = I.instances(body, man)
    rep = dict(manifest=str(manifest_path.relative_to(ROOT)), definitions=str(I.DEFS.relative_to(ROOT)),
               screws=len(rows), references=REF, instances=iface["instances"],
               undefined_parts=iface["summary"]["undefined"], rows=rows, reverse=reverse_view(body, rows, iface, mesh),
               by_connector=by_connector(rows, body))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"{len(rows)} screws -> {out.relative_to(ROOT)} (reverse view {round(time.time() - t0)} s)")
    return rep


def run_screws(manifest_path: Path, only: str | None, out: Path, part: str | None):
    t0 = time.time()
    man = json.loads(manifest_path.read_text())
    body = F.Body(man)
    defs = json.loads(I.DEFS.read_text())
    fam = defs["screw_families"]
    kinds = {n: kind(p) for n, p in body.parts.items()}
    screws = sorted((n for n, k in kinds.items() if k == "screw" and (not only or re.search(only, n))),
                    key=lambda n: (body.parts[n].get("owning_link") or "", n))     # neighbours share caches
    if part:
        a, b = map(int, part.split(":"))
        screws = screws[a:b]
    skip_roles = {"battery_cell", "battery_insulator", "battery_heatsink_fin", "authored_harness", "adhesive_allocation"}
    meshes = {}

    class LazyRays(dict):
        def __missing__(self, n):
            self[n] = rays_for(meshes[n])
            return self[n]
    rayss = LazyRays()

    def mesh(n):
        if n not in meshes:
            meshes[n] = body.mesh(n)
        return meshes[n]

    iface = I.instances(body, man)
    reach = DRIVER_L + 2.0
    tool = set()
    for s_ in screws:
        b_ = body.parts[s_].get("bounds_mm")
        if not b_:
            continue
        tool |= {n for n, p in body.parts.items() if n != s_ and p["role"] not in skip_roles and kinds.get(n) != "screw"
                 and p.get("bounds_mm") and box_near(p["bounds_mm"], b_[0], b_[1], reach)}
    tool = sorted(tool)
    for c in tool:
        mesh(c)
    scenes = {"all": BodyScene(tool, meshes),             # one access-ray scene per chunk (embree leaks per scene)
              "no_shells": BodyScene([n for n in tool if body.parts[n]["role"] != "body_shell"], meshes)}
    print(f"registered {iface['summary']['instances']} standard-part instances ({round(time.time() - t0)} s)")
    rows = []
    for s in screws:
        m = mesh(s)
        lo, hi = m.bounds
        cands = [n for n, p in body.parts.items() if n != s and kinds.get(n) != "screw" and p["role"] not in skip_roles
                 and p.get("bounds_mm") and box_near(p["bounds_mm"], lo, hi, SEARCH)]
        for c in cands:
            mesh(c)
        cands = [c for c in cands if box_near(meshes[c].bounds, lo, hi, SEARCH)]
        try:
            r = analyse_screw(s, body, cands, meshes, rayss, kinds, iface["holes"], fam, scenes)
            r["findings"] = findings(r)
            rows.append(r)
            if len(rows) % 20 == 0:
                print(f"  {len(rows)}/{len(screws)} screws, {round(time.time() - t0)} s, {len(meshes)} meshes held", flush=True)
        except Exception as e:  # noqa: BLE001
            rows.append(dict(name=s, error=repr(e), findings=[dict(kind="check_error", value=repr(e))]))
    rep = dict(manifest=str(manifest_path.relative_to(ROOT)), definitions=str(I.DEFS.relative_to(ROOT)),
               screws=len(rows), seconds=round(time.time() - t0, 1), references=REF,
               instances=iface["instances"], undefined_parts=iface["summary"]["undefined"],
               rows=rows, by_connector=by_connector(rows, body))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"{len(rows)} screws in {rep['seconds']} s -> {out.relative_to(ROOT)}")
    return rep


def by_connector(rows, body):
    """Screws grouped by the printed part they clamp (or, for screws clamping no printed part, by their terminal)."""
    groups = {}
    for r in rows:
        if "error" in r:
            key = "(check error)"
        else:
            pr = [c for c in r["clamped"] if body.parts[c]["role"] in PRINTED]
            key = pr[0] if pr else (r["terminal"]["part"] if r["terminal"] else "(no part)")
        g = groups.setdefault(key, dict(part=key, screws=0, screws_with_findings=0, findings=0, kinds={}, worst={}))
        g["screws"] += 1
        fs = r.get("findings", [])
        g["findings"] += len(fs)
        g["screws_with_findings"] += 1 if fs else 0
        for f in fs:
            g["kinds"][f["kind"]] = g["kinds"].get(f["kind"], 0) + 1
            v = f.get("value")
            if f["kind"] == "engagement_mm" and isinstance(v, (int, float)) and isinstance(f.get("reference"), (int, float)):
                sh = f["reference"] - v                # shortfall against the official reference engagement
                w = g["worst"].get("engagement_shortfall_mm")
                if w is None or sh > w["value"]:
                    g["worst"]["engagement_shortfall_mm"] = dict(value=round(sh, 3), screw=r["name"], note=f.get("note"))
            if isinstance(v, (int, float)):
                w = g["worst"].get(f["kind"])
                if w is None or abs(v) > abs(w["value"]):
                    g["worst"][f["kind"]] = dict(value=v, screw=r["name"], note=f.get("note"))
    for g in groups.values():                      # continuous ranking measures; the counts sit beside them
        w = g["worst"]
        g["worst_engagement_shortfall_mm"] = round(max(0.0, w["engagement_shortfall_mm"]["value"]), 3) if "engagement_shortfall_mm" in w else 0.0
        offs = [w[k]["value"] for k in ("terminal_off_axis_mm", "clamped_hole_off_axis_mm") if k in w]
        g["worst_offset_mm"] = round(max(offs), 3) if offs else 0.0
    return sorted(groups.values(), key=lambda g: (-g["worst_engagement_shortfall_mm"], -g["worst_offset_mm"],
                                                  -g["screws_with_findings"], -g["findings"]))


def reverse_view(body, rows, iface, mesh):
    """Sleeves with no screw through them; defined mount holes of each standard part and which screws use them."""
    axes = [(np.array(r["head_mm"]), np.array(r["axis"]), r["length_mm"], r["name"]) for r in rows if "error" not in r]
    sleeves = []
    for n, p in body.parts.items():
        if p["role"] != "compression_limit_sleeves":
            continue
        for i, (c, a, length, od, idd) in enumerate(tubes(mesh(n))):
            best = None
            for h, d, L, sn in axes:
                if abs(float(a @ d)) < math.cos(math.radians(10)):
                    continue
                t = float((c - h) @ d)
                if t < -length or t > L + length:
                    continue
                off = float(np.linalg.norm((c - h) - t * d))
                if best is None or off < best[0]:
                    best = (off, sn)
            sleeves.append(dict(part=n, index=i, centre_mm=np.round(c, 2).tolist(), axis=np.round(a, 3).tolist(),
                                length_mm=round(length, 2), od_mm=round(od, 2), id_mm=round(idd, 2),
                                nearest_screw=best[1] if best and best[0] < 1.5 else None,
                                nearest_screw_axis_offset_mm=round(best[0], 3) if best else None))
    used = {}
    for r in rows:
        h = ((r.get("terminal") or {}).get("hole") or {})
        if h:
            used.setdefault(h["hole_id"], []).append(r["name"])
    per = {}
    for h in iface["holes"]:
        k = (h["part"], h["label"].rsplit(".", 1)[0] if not h["centre"] else h["label"])
        g = per.setdefault(k, dict(part=h["part"], standard=h["standard"], face=k[1], holes=0, used=0, screws=[]))
        g["holes"] += 1
        if h["hole_id"] in used:
            g["used"] += 1
            g["screws"] += used[h["hole_id"]]
    return dict(sleeves=sleeves, sleeves_without_screw=sum(1 for s in sleeves if s["nearest_screw"] is None),
                interface_faces=sorted(per.values(), key=lambda g: (g["part"], g["face"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--only", default=None, help="regex on screw names")
    ap.add_argument("--out", default=None)
    ap.add_argument("--part", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    mp = Path(a.manifest).resolve()
    out = Path(a.out).resolve() if a.out else ROOT / ".cache/work/mount-check" / mp.stem / ("report.json" if not a.only else "report-only.json")
    run(mp, a.only, out, a.part)
    if not a.only and not a.part:
        import summarise
        print(f"summary -> {summarise.write_summary(out).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
