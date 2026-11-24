"""One FEM path for every per-part setup: one mesher (gmsh P1 tets, fem_lib.mesh_compound_steps), one solver
(unit-wrench basis at the load patch, support patches clamped; maths moved from motor-refit/fem_refit.solve with the
material table injected from data/materials.json), one material table, patches that follow the part.

A setup is data: the `fem` block of the part's body definition (robot/**/<builder>.bom.yaml), else data/part_settings.json
part_settings[part]["fem"] (ctx.settings: definition over part_settings.json and its `_defaults`, run overrides):

  mesh_h_mm        2.2
  bonded           [{"part": "<sleeve part>", "material": "<name in data/materials.json>"}]   companions meshed with the part (bonded);
                   the first distinct material is mesh id 2, a second distinct one mesh id 0; solve reads these names
  origin_mm        [x, y, z] | {"joint": J | "joints": [A, B] (midpoint), "offset_mm": [dx, dy, dz]} | "load_centroid"   body point the wrenches are given about (manifest frame);
                   the joint form follows the joint when the part moves; load_centroid = the load patch's nodal-weight centroid
  open_difference  text: known difference to the old study (echoed into the fem and fem_setup reports)
  supports         {"name": PATCH, ...} clamped (all nodes of every support patch fixed)
  load             PATCH                 the wrench is distributed over it (minimum-weight, 6 components)
  load_cases       {"name": {"wrench_N_Nmm": [Fx, Fy, Fz, Mx, My, Mz]} | {"stall": {"joint": J, "torque_Nmm": T}} | {"hand_envelope": {...}} (checks/arm_loads.py: many wrenches)
                              | {"transfer": {"F_N": [..], "M_Nm": [..], "scale": s, "from_mm": [x, y, z], "lever": l}}}   transfer = a force / moment given at from_mm,
                   scaled, moment at the origin M*s + l * (from - origin) x F*s (neck-head head wrench)
  materials        [names in data/materials.json]   default: the part's process.material
  regions          [{"material": name, "where": REGION, "only_polymer": bool}]   elements taking another material (by centroid; bore_collars: see there), in order, later ones overwrite earlier
                   (an Al seat with only_polymer keeps the steel collars set before it); distinct materials of bonded + regions: at most two
                   REGION = {"bore_collars": {"of": sleeve part, "r_extra_mm": r, "end_extra_mm": e}}   per sleeve of the sleeve part: within r of the part's bore it is
                                                                                                 pressed into, e beyond its ends; elements by their share inside, up to the
                                                                                                 collar's volume (collar_elements: the steel volume does not follow the mesh).
                                                                                                 Each sleeve takes the material its sleeve set's definition declares
                                                                                                 (`contains` item: spec.material, else a spec.standard / spec.label naming
                                                                                                 one, matched to data/materials.json by the entry name or its `matches`
                                                                                                 words, e.g. 'brass knurled heat-set insert' -> Brass CuZn39Pb3; several
                                                                                                 items: the one with the nearest spec.outer_dia_mm); a sleeve its definition
                                                                                                 does not name keeps the region's material and is counted as assumed
                                                                                                 (report: bore_collars.by_material, assumed_sleeves, declared_quantity)
                          | {"contact_band": {"parts": [mating parts], "dist_mm": d}}                  within d of the named mating parts' surfaces (follows them)
                          | {"box": ...}                                                               axis-aligned slabs (see point_pred)
  variants         {"name": {...keys above replaced...}}   e.g. the old hand-typed patches, run with --fem-variant

  PATCH = {"contact": [mating part names], "gap_mm": g}     surface of the part within g of the named parts (follows them)
        | {"holes": {"mate": [mating part names], "gap_mm": g, "r_max_mm": 5.0, "rim_mm": 2.0}}
                                                            the part's declared screw holes for each mating part, with a rim of rim_mm around each hole face (the part's
                                                            surface within r + rim of its axis, along the face + 0.5 mm); follows the part, the mating parts and the screws.
                                                            Hole faces (select_holes): the part's concave cylinder faces (r_min_mm (default 0) .. r_max_mm) around the axis
                                                            of a screw that joins the part (or its pressed-in sleeves) and the mate (joining_screws: rings around the
                                                            shank inside both; a mate that is a screw joins when it passes the part); a screw whose holes are all wider
                                                            than r_max_mm keeps its narrowest (reported as bore_above_r_max). A mate that no screw in the manifest joins
                                                            falls back to the part's bores within g of the mate: concave faces of one radius on one axis spanning at
                                                            least "min_arc_deg" (default 270) in all. A round or fillet no screw passes through is never a hole (a slot's round ends
                                                            around the screw are).
                                                            "screws": [screw part names] takes the holes around those screws instead (mate and g unused): a fixed,
                                                            declared set of sites, e.g. to compare geometries at the same screws.
                                                            "of": <sleeve part> takes the hole faces from that companion instead of the part (the pressed-in sleeves
                                                            sit exactly at the bolt sites; their faces need not be concave): the part's surface within the sleeve radius
                                                            + rim, along the sleeve + 0.5 mm.
                                                            "axis_within_deg": {"axis": [x, y, z], "deg": d} keeps only the holes pointing within d degrees of the axis
                                                            (sign ignored): separates sleeves of two servos that both lie close to one mating part.
                                                            The report (patch info hole_selection, per mate): mode, joining screws, faces per screw, the faces taken and
                                                            the faces left out with the reason
        | {"axis_band": {...}}                              surface within r_mm of a joint axis on one side of a mating part (clamp cheeks)
        | {"box": {"y": [min|null, max|null], ...}}         axis-aligned half-spaces / slabs
        | [PATCH, PATCH]                                    union
  Any PATCH dict can carry "within": {"centre_mm": POINT, "radius_mm": r}: only its surface within r of the point is kept (POINT = [x, y, z] or
  {"joint": J | "joints": [A, B], "offset_mm": [..]}). With origin_mm a POINT instead of load_centroid, this holds the load site and the wrench origin
  fixed, so two geometries (or a rim / gap change) are compared at one load site; put it in a variant to run it as --fem-variant.
  Every patch is reported per site (patch info sites / site_count): the connected pieces of the selected surface, largest first, each with its
  centroid, area, nodes and triangles, so a patch over two bolt sites does not read as one centroid between them. An empty load or support patch
  stops the part with a ValueError naming the part, the variant, the patch and what each mate gave (fem), as fem_setup counts it.
Meshing: gmsh on the STEP as built; on failure the STEP with same-domain faces merged (clean()); on a second failure one retry at 0.85 x mesh_h_mm
(report mesh_retry: requested and used h, each failure; the numbers are then at the used h). The error of a last failure names the gmsh surface /
curve with its bounding box (from gmsh's log: the entity its error names, else the last one it was meshing).
A material of the table can be {"fem": {"factor_of_part_material": f}}: f times the part's own Hooke matrix (the old torso sleeve-zone stiffness, named option).
Any PATCH dict can carry "rigid": true. A member of a rigid load union can carry "free_rotation_about_joint": J: its nodes follow the rigid motion plus a free,
unloaded rotation about joint J's axis (a servo idler bearing: forces and off-axis moments, no torque about the axis). Several members may: the members naming
one joint are one bearing (one released rotation), each further joint adds its own; a released member that selects no node stops with a message naming it. Load: all nodes of the patch follow one rigid-body motion (translation + rotation about origin_mm, master-slave elimination with a 6-DOF reference node
at the origin, direct solve) and the wrench acts on that reference node: the load patch cannot bend between its sites. Support: the nodes are fixed, which already is a fixed rigid body (the flag changes nothing).
Run option ctx.fem_rigid_load (run.py --fem-rigid-load) makes the stored load patch rigid for a comparison run. Run option: ctx.fem_mesh_h (run.py --fem-mesh-h) replaces mesh_h_mm of every setup (mesh convergence).
The layer normal is the part's decided print direction (process.up, the print setup, sign irrelevant for the orthotropic axis), the manifest frame is the frame of all coordinates.
A part without a print direction falls back to process.layer_normal_axes and says so. process.layer_normal_axes stays in the data as provenance: it is the input of the comparison
variant --fem-variant studied_axes (the old studied axes, run as several layer-normal variants), which replaces no other key of the setup.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from checks import bodylib as BL
from checks import cachefs
from checks import fem_lib as FL
from checks.core import CACHE, DATA, ROOT, WORK, R, sha
from checks.fem_lib import hooke, materials  # noqa: F401

AX = {"x": 0, "y": 1, "z": 2}
STUDIED = "studied_axes"      # --fem-variant: layer normals = the stored process.layer_normal_axes instead of the decided print direction


def LL():
    """The FEM library (mesher, patch weights, elasticity assembly, quantiles): checks/fem_lib.py."""
    return FL


# ------------------------------------------------------------------------------------------------ setups
def merged_setup(ctx, part, variant=None):
    """fem setup of a part (None when it has none): ctx.settings(part)["fem"] (its definition, else part_settings.json); a variant
    replaces the keys it names."""
    full = ctx.settings(part).get("fem")
    if not full:
        return None
    s = {k: v for k, v in full.items() if k != "variants"}
    if variant and variant != STUDIED:
        v = full.get("variants", {}).get(variant)
        if v is None:
            return None
        s.update(v)
    if getattr(ctx, "fem_mesh_h", None):
        s["mesh_h_mm"] = float(ctx.fem_mesh_h)
    if getattr(ctx, "fem_rigid_load", False) and not is_rigid(s["load"]):     # comparison run: the stored load patch made rigid
        s["load"] = [{**x, "rigid": True} for x in (s["load"] if isinstance(s["load"], list) else [s["load"]])]
    if s.get("regions"):       # bore collars: the materials the sleeve set's definition declares (brass inserts, ...) take mesh labels too
        s["regions"] = [dict(r, _declared_materials=sorted({m for _, m, _ in declared_sleeve_items(ctx, r["where"]["bore_collars"]["of"]) if m}))
                        if "bore_collars" in r.get("where", {}) else r for r in s["regions"]]
    return s


def material_of_text(text):
    """The data/materials.json entry a declared material text names: the entry's own name, else the entry one of whose `matches` words the text
    contains (case ignored; the longest matching word wins), else None."""
    if not text:
        return None
    mats = materials()
    if text in mats:
        return text
    t = str(text).lower()
    hits = [(len(w), k) for k, v in mats.items() if isinstance(v, dict) for w in v.get("matches", []) if w.lower() in t]
    return max(hits)[1] if hits else None


def declared_sleeve_items(ctx, sleeve_part):
    """[(item, material entry | None, source)] of the bought items the sleeve set's body definition declares inside it (`contains`, the items for
    this part): the material from spec.material, else from a spec field that names one (standard, label: 'brass knurled heat-set insert')."""
    e = (getattr(ctx.rel, "claims", None) or {}).get(sleeve_part)       # the definition entry that claims the sleeve set (checks.relations)
    out = []
    for it in (e or {}).get("contains") or []:
        if it.get("part") and it["part"] != sleeve_part:
            continue
        sp = it.get("spec") or {}
        m = src = None
        for k in ("material", "standard", "label"):
            m = material_of_text(sp.get(k))
            if m:
                src = f"definition {it.get('line')}: spec.{k} {sp.get(k)!r}"
                break
        out.append((it, m, src))
    return out


def sleeve_site_materials(ctx, sleeve_part, sites, default):
    """Per collar site (sleeve_bores) its material and source: from the sleeve set's definition when its items name one material, or per site the
    item whose spec outer_dia_mm is nearest the bore when they name several; else `default` (the region's material), counted as assumed."""
    items = declared_sleeve_items(ctx, sleeve_part)
    named = [(it, m, s) for it, m, s in items if m]
    if not items or not named:
        why = "no contains in the sleeve set's definition" if not items else "its definition's items name no material"
        return [(default, f"assumed: region material ({why})") for _ in sites], dict(declared_items=len(items), declared_quantity=sum(int(it.get("quantity") or 0) for it, _, _ in items))
    out = []
    if len({m for _, m, _ in named}) == 1 and len(named) == len(items):
        out = [(named[0][1], named[0][2]) for _ in sites]
    else:
        for c in sites:
            cand = [(abs(float((it.get("spec") or {}).get("outer_dia_mm")) - 2 * c["r"]), m, s) for it, m, s in items if (it.get("spec") or {}).get("outer_dia_mm")]
            best = min(cand, key=lambda x: x[0]) if cand else None
            out.append((best[1], best[2] + " (nearest declared outer diameter)") if best and best[1] else (default, "assumed: region material (no declared item matches this sleeve)"))
    return out, dict(declared_items=len(items), declared_quantity=sum(int(it.get("quantity") or 0) for it, _, _ in items))


def is_rigid(spec):
    """True when a PATCH (or any member of a union) carries "rigid": true."""
    return any(isinstance(x, dict) and x.get("rigid") for x in (spec if isinstance(spec, list) else [spec]))


def layer_direction(ctx, part, variant=None):
    """Layer-normal axes of a part's FEM and where they come from.
    Decided run: the axis of process.up (print setup, sign dropped); a part without a print direction, or one that is not along an axis,
    falls back to its stored layer_normal_axes (`fallback` says why). Variant 'studied_axes': the stored axes.
    Returns dict(axes [0..2], up, up_source, stored [letters], source, fallback, disagrees: print axis not among the stored axes)."""
    from checks import process as PR
    eff = PR.effective(ctx, part)
    stored = list(ctx.settings(part)["process"].get("layer_normal_axes") or [])
    up, usrc = eff["up"], eff["source"]["up"]
    v = PR.up_vector(up)
    axis = None
    if v is not None:
        k = int(np.argmax(np.abs(v)))
        axis = "xyz"[k] if abs(v[k]) > 1 - 1e-6 else None
    out = dict(up=up, up_source=usrc, seeded=usrc == "seeded", stored=stored, fallback=None, disagrees=bool(axis and stored and axis not in stored),
               print_axis=axis)
    if variant == STUDIED:
        out.update(source="stored layer_normal_axes (comparison variant studied_axes)", axes=[AX[a] for a in stored])
    elif axis:
        out.update(source=f"print direction {up} ({usrc})", axes=[AX[axis]])
    else:
        out["fallback"] = "no print direction" if up is None else f"print direction {up} is not along an axis"
        out.update(source="stored layer_normal_axes (fallback: " + out["fallback"] + ")", axes=[AX[a] for a in stored])
    return out


def layer_normals(ctx, part, variant=None):
    return layer_direction(ctx, part, variant)["axes"]


def setup_materials(ctx, part, setup):
    mats = setup.get("materials") or ([ctx.settings(part)["process"].get("material")] if ctx.settings(part)["process"].get("material") else [])
    return [m for m in mats if m]


# ------------------------------------------------------------------------------------------------ geometry in the manifest frame
def frame_step(ctx, name):
    """Native STEP of a manifest part in the manifest frame (its native_step_offset_mm applied once and cached)."""
    p = ctx.parts[name]
    src = BL.src(p["native_step"])
    off = p.get("native_step_offset_mm")
    if not off or not any(abs(v) > 1e-9 for v in off):
        return src
    dst = WORK / "fem-steps" / f"{sha(src)[:16]}-{'_'.join(f'{v:g}' for v in off)}.step"
    if not dst.exists():
        from build123d import Compound, Location, export_step, import_step
        dst.parent.mkdir(parents=True, exist_ok=True)
        sh = import_step(str(src)).moved(Location(tuple(float(v) for v in off)))
        export_step(Compound(children=list(sh.solids())), str(dst))
    return dst


def cleaned_step(src):
    """The STEP with same-domain faces merged (build123d clean), cached by sha256; the fallback when gmsh cannot mesh the original."""
    dst = WORK / "fem-steps" / f"{sha(src)[:16]}-clean.step"
    if not dst.exists():
        from build123d import Compound, export_step, import_step
        dst.parent.mkdir(parents=True, exist_ok=True)
        export_step(Compound(children=list(import_step(str(src)).clean().solids())), str(dst))
    return dst


LABELS = ("steel", "metal", "metal3", "metal4")          # mesh labels of the bonded / region materials, in order of first use
LABEL_CODE = {"metal": 0, "polymer": 1, "steel": 2, "metal3": 3, "metal4": 4}


def _label_map(setup):
    """{material name: mesh label}: the first distinct bonded / region material is mesh id 2 ('steel'), a second one id 0 ('metal'), further ones
    ids 3 and 4; region materials include the materials a bore-collar region's sleeve set declares (merged_setup)."""
    labels = {}
    for m in [b["material"] for b in setup.get("bonded", [])] + [x for r in setup.get("regions", []) for x in [r["material"]] + r.get("_declared_materials", [])]:
        if m not in labels:
            if len(labels) >= len(LABELS):
                raise ValueError(f"more than {len(LABELS)} distinct bonded / region materials: {sorted(set(labels) | {m})}")
            labels[m] = LABELS[len(labels)]
    return labels


def bonded_slots(setup):
    """[(part, mesh label, material name)] of the bonded companions (meshed with the part)."""
    lm = _label_map(setup)
    return [(b["part"], lm[b["material"]], b["material"]) for b in setup.get("bonded", [])]


def slot_materials(setup):
    """{mesh label: material name} for solve (bonded companions and regions)."""
    return {lab: m for m, lab in _label_map(setup).items()}


def point_pred(spec):
    """Predicate over points (n, 3) for a box patch (also used for material regions): {"box": {"y": [min|null, max|null], ...}}."""
    bx = spec["box"]

    def pred(c):
        c = np.asarray(c); ok = np.ones(len(c), bool)
        for k, (lo, hi) in bx.items():
            ok &= (c[:, AX[k]] >= (-1e9 if lo is None else lo)) & (c[:, AX[k]] <= (1e9 if hi is None else hi))
        return ok
    return pred


def _coaxial(a, b, tol=0.1):
    """Two cylinder faces on one axis line (parallel axes, axis lines within tol mm)."""
    if abs(abs(float(a["axis"] @ b["axis"])) - 1.0) > 1e-3:
        return False
    d = b["centre"] - a["centre"]
    return float(np.linalg.norm(d - (d @ a["axis"]) * a["axis"])) <= tol


def _span(h, ref):
    """[lo, hi] of a cylinder face along the axis of ref, measured from ref's centre."""
    a = float((h["centre"] - ref["centre"]) @ ref["axis"])
    return a - h["half_len"], a + h["half_len"]


def sleeve_bores(ctx, part, sleeves, r_max=5.0, r_tol=0.3):
    """One collar site per sleeve of the companion part `sleeves`: the part's own bore it is pressed into, i.e. the part's cylindrical faces coaxial
    with the sleeve's outer cylinder, radius within r_tol of it, along the stretch where they overlap the sleeve. A sleeve = the sleeve part's coaxial,
    overlapping cylinder faces (inner bore, outer surface); its outer radius is the largest. A sleeve without such a bore keeps its outer cylinder.
    Returns [dict(r, centre, axis, half_len, source="bore" | "sleeve OD")]."""
    tubes = []
    for h in sorted(hole_faces(ctx, sleeves, r_max), key=lambda x: -x["r"]):
        t = next((t for t in tubes if _coaxial(t[0], h) and _span(h, t[0])[0] <= _span(t[0], t[0])[1] + 0.01 and _span(h, t[0])[1] >= _span(t[0], t[0])[0] - 0.01), None)
        if t is None:
            tubes.append([h])
        elif abs(h["r"] - t[0]["r"]) < 1e-3:
            t.append(h)                                # the outer cylinder split into several faces
    bores = hole_faces(ctx, part, r_max + r_tol)
    out = []
    for t in tubes:
        od = t[0]
        lo, hi = min(_span(h, od)[0] for h in t), max(_span(h, od)[1] for h in t)
        seat = [b for b in bores if abs(b["r"] - od["r"]) <= r_tol and _coaxial(od, b) and _span(b, od)[1] > lo and _span(b, od)[0] < hi]
        src, r = "sleeve OD", od["r"]
        if seat:
            src, r = "bore", max(b["r"] for b in seat)
            lo, hi = max(lo, min(_span(b, od)[0] for b in seat)), min(hi, max(_span(b, od)[1] for b in seat))
        out.append(dict(r=r, centre=od["centre"] + 0.5 * (lo + hi) * od["axis"], axis=od["axis"], half_len=0.5 * (hi - lo), source=src))
    return out


_SAMPLES = np.random.default_rng(0).dirichlet(np.ones(4), 64)    # fixed barycentric sample points of a tet (share of an element inside a region)


def collar_elements(ctx, mesh, part, spec, default=None):
    """Elements that take the metal of the bore collars of the sleeve part spec["of"]: per sleeve, the collar is the part's material within r_extra_mm of
    the bore it is pressed into (sleeve_bores) and end_extra_mm beyond the sleeve's ends. Its volume inside the part is the sum over elements of element
    volume x the share of 64 sample points inside it, which does not depend on the mesh; the elements with the largest share are taken until their volume
    is closest to it (selection by centroid made a 0.85 mm collar about one element thick, so its steel volume followed the mesh size).
    Each sleeve takes the material its definition declares (sleeve_site_materials: brass heat-set inserts, ...), else `default` (assumed).
    Returns ({material: element mask}, dict(sleeves, in_bore, collar_mm3, steel_mm3 (all collars), by_material, assumed_sleeves, ...))."""
    P, T, V = mesh["points"], mesh["tet"], mesh["vol"]
    cent = P[T].mean(axis=1)
    reach = np.linalg.norm(P[T] - cent[:, None, :], axis=2).max(axis=1)
    sites = sleeve_bores(ctx, part, spec["of"], float(spec.get("r_max_mm", 5.0)))
    site_mats, decl = sleeve_site_materials(ctx, spec["of"], sites, default)
    sels = {}
    sel, target = np.zeros(len(T), bool), 0.0
    for c, (mat, _) in zip(sites, site_mats):
        before = sel.copy()
        R, L = c["r"] + float(spec["r_extra_mm"]), c["half_len"] + float(spec["end_extra_mm"])
        rel = cent - c["centre"]; al = rel @ c["axis"]
        cand = np.flatnonzero((np.linalg.norm(rel - np.outer(al, c["axis"]), axis=1) <= R + reach) & (np.abs(al) <= L + reach))
        s = np.einsum("sk,ekd->esd", _SAMPLES, P[T[cand]]) - c["centre"]
        sa = s @ c["axis"]
        share = ((np.linalg.norm(s - sa[..., None] * c["axis"], axis=2) <= R) & (np.abs(sa) <= L)).mean(axis=1)
        vol = float((share * V[cand]).sum())
        o = np.argsort(-share, kind="stable")           # largest share first
        cum = np.concatenate([[0.0], np.cumsum(V[cand][o])])
        sel[cand[o[:int(np.argmin(np.abs(cum - vol)))]]] = True
        target += vol
        sels.setdefault(mat, np.zeros(len(T), bool))[sel & ~before] = True
    by_mat = {}
    for (mat, src), c in zip(site_mats, sites):
        b = by_mat.setdefault(mat, dict(sleeves=0, sources=[]))
        b["sleeves"] += 1
        if src not in b["sources"]:
            b["sources"].append(src)
    for mat, m in sels.items():
        by_mat[mat]["collar_elements_mm3"] = round(float(V[m].sum()), 1)
    return sels, dict(sleeves=len(sites), in_bore=sum(1 for c in sites if c["source"] == "bore"), collar_mm3=round(target, 1), steel_mm3=round(float(V[sel].sum()), 1),
                      by_material=by_mat, assumed_sleeves=sum(1 for _, s in site_mats if s.startswith("assumed")), **decl)


def band_pred(ctx, spec):
    """Predicate over points: within dist_mm of the surface of any of the named mating parts (the clamped seat against a metal housing; follows the parts)."""
    trees = [_surface_points(ctx, n) for n in spec["parts"]]
    return lambda c: np.logical_or.reduce([t.query(np.asarray(c))[0] <= float(spec["dist_mm"]) for t in trees])


def apply_regions(ctx, mesh, setup):
    """mesh copy whose mat_ids carry the setup's material regions (elements by centroid: hand-typed sleeve collars, or bore collars of a sleeve part)."""
    regs = setup.get("regions")
    if not regs:
        return mesh
    lm = _label_map(setup)
    ids = mesh["mat_ids"].copy()
    cent = mesh["points"][mesh["tet"]].mean(axis=1)
    counts, collars = {}, {}
    for i, r in enumerate(regs):
        w = r["where"]
        if "bore_collars" in w:
            sels, collars[f"{r['material']} {i}"] = collar_elements(ctx, mesh, mesh["part"], w["bore_collars"], default=r["material"])
        else:
            sels = {r["material"]: (band_pred(ctx, w["contact_band"]) if "contact_band" in w else point_pred(w))(cent)}
        n_sel = int(sum(int(s.sum()) for s in sels.values()))
        poly = ids == 1
        for mat, sel in sels.items():
            if r.get("only_polymer"):
                sel = sel & poly
            ids[sel] = LABEL_CODE[lm[mat]]
        counts[f"{r['material']} {i}"] = n_sel
    return mesh | {"mat_ids": ids, "region_elements": counts, "bore_collars": collars}


MESH_RETRY_FACTOR = 0.85     # gmsh failed on the STEP as built and cleaned: one retry at 0.85 x the setup's mesh size


def get_mesh(ctx, part, setup):
    """gmsh tets of the part (+ bonded companions) in the manifest frame, cached by the STEP files' sha256 and h."""
    steps = [("polymer", frame_step(ctx, part))] + [(lab, frame_step(ctx, b["part"])) for (_, lab, _), b in zip(bonded_slots(setup), setup.get("bonded", []))]
    h = float(setup.get("mesh_h_mm", 2.2))
    key = "-".join(f"{lab}{sha(p)[:12]}" for lab, p in steps) + f"-h{h:g}"
    f = CACHE / "fem-mesh" / f"{key}.npz"
    d = cachefs.load_npz(f) if f.exists() else None
    if d is not None:
        say(f"mesh h {h:g} from the cache ({len(d['points'])} nodes, {len(d['tet'])} tets)")
        extra = {"mesh_retry": dict(requested_h_mm=h, used_h_mm=float(d["used_h_mm"]), note="gmsh failed at the requested h; this cached mesh is the retry at used_h_mm")} if "used_h_mm" in d else {}
        return apply_regions(ctx, {k: d[k] for k in ("points", "tet", "tri", "mat_ids", "vol")} | {"cached": True, "key": key, "part": part} | extra, setup)
    t0 = time.time()
    say(f"meshing h {h:g} (gmsh, {len(steps)} STEP files)")
    tries, extra = [], {}
    try:
        mesh = LL().mesh_compound_steps(steps, [], h_max=h)
    except Exception as e:   # noqa: BLE001  gmsh raises plain exceptions
        tries.append(dict(step="as built", h_mm=h, error=" ".join(f"{type(e).__name__}: {e}".split())[:600]))
        # near-coincident faces make the surface mesh intersect (arm roll-to-yaw part: 214 faces); the same part with merged same-domain faces meshes
        try:
            steps_c = [("polymer", cleaned_step(steps[0][1]))] + steps[1:]
            mesh = LL().mesh_compound_steps(steps_c, [], h_max=h) | {"healed_step": f"clean() after {type(e).__name__}"}
        except Exception as e2:  # noqa: BLE001
            # gmsh "1D mesh not forming a closed loop" on BRepCheck-valid STEPs: the boundary discretisation of one face; another mesh size
            # discretises its edges differently. One retry at MESH_RETRY_FACTOR x h, reported (the numbers are then at that h).
            tries.append(dict(step="clean()", h_mm=h, error=" ".join(f"{type(e2).__name__}: {e2}".split())[:600]))
            h2 = round(h * MESH_RETRY_FACTOR, 3)
            say(f"gmsh failed at h {h:g} ({str(e).splitlines()[0][:160]}); retrying once at h {h2:g}")
            try:
                mesh = LL().mesh_compound_steps(steps, [], h_max=h2)
            except Exception as e3:  # noqa: BLE001
                tries.append(dict(step="as built", h_mm=h2, error=" ".join(f"{type(e3).__name__}: {e3}".split())[:600]))
                nm = os.path.basename(str(steps[0][1]))      # run.py keeps 300 characters of the message: the attempts first, paths shortened
                raise RuntimeError(f"{part}: gmsh could not mesh {nm} (tried as built and clean() at h {h:g}, retry at h {h2:g}): " + " | ".join(
                    dict.fromkeys(t["error"].replace(str(steps[0][1]), nm) for t in reversed(tries)))) from e3
            extra = {"mesh_retry": dict(requested_h_mm=h, used_h_mm=h2, failures=tries)}
            say(f"meshed at the retry h {h2:g}")
    cachefs.savez(f, **{k: mesh[k] for k in ("points", "tet", "tri", "mat_ids", "vol")}, **({"used_h_mm": np.array(extra["mesh_retry"]["used_h_mm"])} if extra else {}))
    say(f"meshed: {len(mesh['points'])} nodes, {len(mesh['tet'])} tets in {time.time() - t0:.0f} s")
    return apply_regions(ctx, mesh | extra | {"cached": False, "key": key, "part": part, "mesh_seconds": round(time.time() - t0, 1)}, setup)


def _sample(mesh, n):
    """Surface samples, reproducible: the global generator is reseeded (this trimesh has no seed argument)."""
    np.random.seed(0)
    return mesh.sample(n)


def _surface_points(ctx, name):
    """Vertices + 20000 surface samples of a part's solid (manifest frame), as KD-tree (cached on the ctx)."""
    c = ctx.__dict__.setdefault("_fem_trees", {})
    if name not in c:
        s = ctx.solid(name)[0]
        c[name] = cKDTree(np.vstack([s.vertices, _sample(s, 20000)]))
    return c[name]


def hole_faces(ctx, name, r_max=5.0):
    """Cylindrical faces of the part's STEP up to r_max radius, manifest frame: [dict(r, centre, axis, half_len, arc_deg, concave, face)], cached on
    the ctx. arc_deg = the face's angular span about its axis; concave = the material lies outside the cylinder (a bore wall, or a concave round);
    face = the face's index in the STEP."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepGProp import BRepGProp_Face
    from OCP.BRepTools import BRepTools
    from OCP.GeomAbs import GeomAbs_Cylinder
    from OCP.gp import gp_Pnt, gp_Vec
    from build123d import import_step
    c = ctx.__dict__.setdefault("_fem_holes", {})
    key = (name, r_max)
    if key not in c:
        out = []
        for i, f in enumerate(import_step(str(frame_step(ctx, name))).faces()):
            a = BRepAdaptor_Surface(f.wrapped)
            if a.GetType() == GeomAbs_Cylinder and a.Cylinder().Radius() <= r_max:
                d = a.Cylinder().Axis().Direction(); ax = np.array([d.X(), d.Y(), d.Z()])
                v = np.array([p.to_tuple() for p in (x.center() for x in f.vertices())])
                loc = a.Cylinder().Location(); loc = np.array([loc.X(), loc.Y(), loc.Z()])
                com = np.array(f.center().to_tuple()); cen = loc + ((com - loc) @ ax) * ax      # centre of mass projected onto the axis
                al = (v - cen) @ ax if len(v) else np.zeros(1)
                u0, u1, v0, v1 = BRepTools.UVBounds_s(f.wrapped)
                p, nv = gp_Pnt(), gp_Vec()
                BRepGProp_Face(f.wrapped).Normal(0.5 * (u0 + u1), 0.5 * (v0 + v1), p, nv)     # outward normal of the solid at the face's middle
                pm = np.array([p.X(), p.Y(), p.Z()]) - loc
                concave = float(np.array([nv.X(), nv.Y(), nv.Z()]) @ (pm - (pm @ ax) * ax)) < 0.0
                out.append(dict(r=a.Cylinder().Radius(), centre=cen, axis=ax, half_len=float(max(abs(al.max()), abs(al.min()), 0.5)),
                                arc_deg=float(np.degrees(u1 - u0)), concave=concave, face=i))
        c[key] = out
    return c[key]


HOLE_SEARCH_R_MM = 5.0       # radius up to which a screw's coaxial bore is looked for (above the patch's r_max: listed as bore_above_r_max)
MIN_BORE_ARC_DEG = 270.0     # a bore without a joining screw: its coaxial faces of one radius span at least this angle (a fillet spans 90, a slot end 180)


def _box(ctx, n, pad=0.0):
    b = np.array(ctx.parts[n]["bounds_mm"], float)
    return b[0] - pad, b[1] + pad


def _boxes_meet(a, b):
    return bool(np.all(a[0] <= b[1]) and np.all(b[0] <= a[1]))


def _screws(ctx):
    """{name: None} of the parts that are screws (mount_check.kind: body definition, screw_plan, else the mesh shape), geometry filled lazily."""
    c = ctx.__dict__.get("_fem_screws")
    if c is None:
        from checks import screwlib as SL
        MC = SL.mount_check()
        c = ctx._fem_screws = {k: None for k, p in ctx.parts.items() if MC.kind(p) == "screw"}
    return c


def screw_geometry(ctx, name):
    """mount_check.screw_geometry of a screw part (head point _head, unit axis _d from the head, length_mm, r_shank_mm), None when unreadable."""
    c = _screws(ctx)
    if c.get(name) is None:
        from checks import screwlib as SL
        try:
            c[name] = SL.mount_check().screw_geometry(BL.load_mesh(ctx.parts[name]["mesh"]))
        except Exception:  # noqa: BLE001
            c[name] = False
    return c[name] or None


def joining_screws(ctx, part, mate):
    """[(screw, geometry)] of the screws that join `part` (with the sleeve sets pressed into it) and `mate`: at least 2 points of rings 0.8 and 3 mm
    outside the shank, along it and under the head (screwlib.ring_points), lie inside the part or its sleeves and at least 2 inside the mate (coarse
    screening solids, manifest frame). A mate that is itself a screw joins when it passes through the part. Cached on the ctx."""
    c = ctx.__dict__.setdefault("_fem_joins", {})
    if (part, mate) in c:
        return c[(part, mate)]
    from checks import screwlib as SL
    own = [part] + [s for s in ctx.rel.sleeve_sets_of(part) if s in ctx.parts]
    screws = _screws(ctx)
    pb, mb = _box(ctx, part, 1.0), _box(ctx, mate, 1.0)
    cand = [mate] if mate in screws else [k for k in screws if k != part and _boxes_meet(_box(ctx, k), pb) and _boxes_meet(_box(ctx, k), mb)]

    def inside(n, pts):
        try:
            return int(SL.screen(ctx, n)[0].contains(pts).sum())
        except Exception:  # noqa: BLE001
            return 0
    out = []
    for k in cand:
        g = screw_geometry(ctx, k)
        if g is None:
            continue
        pts = np.vstack([SL.ring_points(g, SL.RING_MM), SL.ring_points(g, SL.NEAR_MM)])
        if sum(inside(n, pts) for n in own) >= 2 and (k == mate or inside(mate, pts) >= 2):
            out.append((k, g))
    c[(part, mate)] = out
    return out


def _on_screw_axis(h, g, tol=0.5):
    """A cylinder face around the screw's axis: parallel within 2 deg, the screw's axis inside the face's circle (axis lines within max(tol, r) mm:
    a round hole, or the rounded ends of a slot the screw sits in) and along its length (head to tip, 1 mm margin)."""
    d, h0 = np.asarray(g["_d"], float), np.asarray(g["_head"], float)
    if abs(float(h["axis"] @ d)) < np.cos(np.radians(2.0)):
        return False
    rel = h["centre"] - h0
    t = float(rel @ d)
    return float(np.linalg.norm(rel - t * d)) <= max(tol, float(h["r"])) and t + h["half_len"] >= -1.0 and t - h["half_len"] <= float(g["length_mm"]) + 1.0


def _bore_arc(h, faces):
    """Angular span (deg) of the faces coaxial with h (axis lines within 0.1 mm) at h's radius: a hole split into two half-cylinder faces is one bore."""
    return sum(x["arc_deg"] for x in faces if abs(x["r"] - h["r"]) < 1e-3 and x["concave"] == h["concave"] and _coaxial(h, x))


def _face_row(h):
    return dict(face=h.get("face"), r_mm=round(float(h["r"]), 3), centre_mm=np.round(h["centre"], 2).tolist(), axis=np.round(h["axis"], 3).tolist(),
                arc_deg=round(h.get("arc_deg", 360.0), 1), concave=h.get("concave"))


def select_holes(ctx, part, mate, hs):
    """Hole faces of a holes PATCH for one mating part, and how they were chosen (dict for the report).
    Sites are declared screw holes: the faces of the source (the part, or the companion "of") on the axis of a screw that joins the part and the mate
    (joining_screws: the screw is the joint, so gap_mm is not applied), or of the screws the patch names ("screws": [...]); r_min_mm..r_max_mm,
    concave for the part's own faces; a screw whose bores are all wider than r_max_mm keeps its narrowest one (bore_above_r_max).
    A mate no screw joins (the screws are not in the manifest) falls back to the part's bores within gap_mm of the mate: concave faces of one radius
    on one axis spanning at least min_arc_deg (default 270) in all. A concave round or fillet is never a hole."""
    src = hs.get("of", part)
    own = src == part
    r_max, r_min, gap = float(hs.get("r_max_mm", 5.0)), float(hs.get("r_min_mm", 0.0)), float(hs["gap_mm"])
    axis = _axis_filter(hs)
    faces = [h for h in hole_faces(ctx, src, max(r_max, HOLE_SEARCH_R_MM))
             if not (axis and abs(float(h["axis"] @ axis["unit"])) < axis["cos"])]    # sleeve not pointing along the requested axis (sign ignored)
    tree = _surface_points(ctx, mate) if mate else None

    def near(h):           # axis centre or either end within gap of the mate's surface
        pts = np.array([h["centre"] + t * h["half_len"] * h["axis"] for t in (-1.0, 0.0, 1.0)])
        return tree is not None and float(tree.query(pts)[0].min()) <= gap
    in_range = [h for h in faces if r_min <= h["r"] <= r_max and near(h)]     # every cylindrical face in range near the mate
    named = hs.get("screws")
    if named:
        screws, mode = [(k, screw_geometry(ctx, k)) for k in named], "named screws (setup)"
        missing = [k for k, g in screws if g is None]
        screws = [(k, g) for k, g in screws if g is not None]
    else:
        screws, mode, missing = joining_screws(ctx, part, mate), "screws joining the part and the mate", []
    picked, per_screw, above = [], {}, []
    if screws:
        for k, g in screws:
            co = [h for h in faces if (h["concave"] or not own) and h["r"] >= r_min and _on_screw_axis(h, g)]
            sel = [h for h in co if h["r"] <= r_max]
            if not sel and co:
                rr = min(h["r"] for h in co)
                sel = [h for h in co if abs(h["r"] - rr) < 1e-3]
                above.append(dict(screw=k, bore_r_mm=round(float(rr), 3), r_max_mm=r_max))
            per_screw[k] = len(sel)
            picked += [h for h in sel if not any(h is x for x in picked)]
    else:
        mode = "bores: no screw in the manifest joins the part and the mate"
        picked = [h for h in in_range if not own or (h["concave"] and _bore_arc(h, faces) >= float(hs.get("min_arc_deg", MIN_BORE_ARC_DEG)))]
    dropped = [h for h in in_range if not any(h is x for x in picked)]
    fil = [h for h in dropped if own and not (h["concave"] and _bore_arc(h, faces) >= float(hs.get("min_arc_deg", MIN_BORE_ARC_DEG)))]
    det = dict(mode=mode, faces=len(picked), screws=list(per_screw), hole_faces_per_screw=per_screw, screws_without_bore=[k for k, v in per_screw.items() if not v],
               bore_above_r_max=above, named_screws_not_found=missing,
               dropped_faces=dict(round_or_partial_arc=len(fil), bore_without_joining_screw=len(dropped) - len(fil)),
               picked=[_face_row(h) for h in picked][:24], dropped=[dict(_face_row(h), why="round / partial arc" if any(h is x for x in fil) else "no joining screw on its axis")
                                                                  for h in dropped][:24])
    return picked, det


def holes_near(ctx, name, mates, gap, r_max=5.0, r_min=0.0, axis=None):
    """The part's cylinder faces (radius r_min..r_max) with ANY of axis centre / either end within gap of one of the mating parts' surfaces (every
    cylindrical face: kept for comparisons; the patches use select_holes)."""
    trees = [_surface_points(ctx, m) for m in mates]
    sel = []
    for h in hole_faces(ctx, name, r_max):
        if h["r"] < r_min:
            continue
        if axis and abs(float(h["axis"] @ axis["unit"])) < axis["cos"]:     # sleeve not pointing along the requested axis (sign ignored)
            continue
        pts = np.array([h["centre"] + t * h["half_len"] * h["axis"] for t in (-1.0, 0.0, 1.0)])
        if min(float(t.query(pts)[0].min()) for t in trees) <= gap:
            sel.append(h)
    return sel


def _axis_filter(hs):
    a = hs.get("axis_within_deg")
    if not a:
        return None
    u = np.asarray(a["axis"], float)
    return dict(unit=u / np.linalg.norm(u), cos=float(np.cos(np.radians(a["deg"]))))


def point_of(ctx, o):
    """A body point: [x, y, z], or {"joint": J | "joints": [A, B], "offset_mm": [..]} = the joint's origin (the midpoint of two) plus an offset."""
    if isinstance(o, dict):
        names = o["joints"] if "joints" in o else [o["joint"]]
        base = np.mean([ctx.body.joints[n]["origin_world_mm"] for n in names], axis=0)
        return base + np.asarray(o.get("offset_mm", [0, 0, 0]), float)
    return np.asarray(o, float)


def origin_of(ctx, setup, mesh=None, w_load=None, where=None):
    """Wrench origin: a point, {"joint": J, "offset_mm": [..]} = the joint's origin plus an offset (follows the joint; "joints": [A, B] = the midpoint of two joints),
    or "load_centroid" = the load patch's nodal-weight centroid (needs mesh and w_load). An empty load patch has no centroid: ValueError naming
    `where` (part [variant]) and the patch."""
    o = setup["origin_mm"]
    if o == "load_centroid":
        if w_load is None or not (w_load > 0).any():
            raise ValueError(f"{where or (mesh or {}).get('part')}: origin_mm load_centroid: the load patch selects no node of the part, so it has no centroid; "
                             f"load patch: {json.dumps(setup['load'])}")
        return np.average(mesh["points"][w_load > 0], axis=0, weights=w_load[w_load > 0]).tolist()
    return point_of(ctx, o).tolist()


def setup_notes(setup):
    return {k: setup[k] for k in ("_doc", "load_cases_doc", "open_difference") if setup.get(k)}


def _cent_area(mesh):
    v = mesh["points"][mesh["tri"]]
    return v.mean(axis=1), 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)


def _one_patch(ctx, mesh, spec):
    """(predicate(c, v) for LL().select_patch_weights, mask over surface triangles, per-mating-part triangle masks, hole selection per mate)."""
    cent, _ = _cent_area(mesh)
    if "contact" in spec:
        g = float(spec["gap_mm"])
        dist = {n: _surface_points(ctx, n).query(cent)[0] for n in spec["contact"]}
        near = {n: d <= g for n, d in dist.items()}
        mask = np.logical_or.reduce(list(near.values()))
        trees = [_surface_points(ctx, n) for n in spec["contact"]]
        pred = lambda c, v: np.logical_or.reduce([t.query(np.asarray(c))[0] <= g for t in trees])  # noqa: E731
        return pred, mask, near, {}
    if "holes" in spec:
        hs = spec["holes"]
        per, near, det = [], {}, {}
        groups = [(m, None) for m in hs.get("mate") or []] if not hs.get("screws") else [(", ".join(hs["screws"]), None)]
        for m, _ in groups:       # per mating part: its declared screw holes (a hole next to two parts counts for both)
            sel, d_ = select_holes(ctx, mesh["part"], None if hs.get("screws") else m, hs)
            mk = np.zeros(len(cent), bool)
            for h in sel:
                d = cent - h["centre"]
                along = d @ h["axis"]
                rad = np.linalg.norm(d - np.outer(along, h["axis"]), axis=1)
                mk |= (rad <= h["r"] + float(hs.get("rim_mm", 2.0))) & (np.abs(along) <= h["half_len"] + 0.5)
            near[f"{m} ({len(sel)} hole faces)"] = mk
            det[m] = d_
            per.append(mk)
        return None, np.logical_or.reduce(per), near, det
    if "axis_band" in spec:
        # follows the joint and the mating part: surface within r_mm of the joint axis, on one side of the mating part's centre along it
        ab = spec["axis_band"]
        j = ctx.body.joints[ab["joint"]]
        ax = np.asarray(j["axis_world_neutral"], float); ax /= np.linalg.norm(ax); o = np.asarray(j["origin_world_mm"], float)
        d = cent - o
        along = d @ ax
        rad = np.linalg.norm(d - np.outer(along, ax), axis=1)
        mb = ctx.solid(ab["mate"])[0].bounds
        mid = (((mb[0] + mb[1]) / 2 - o) @ ax)
        mask = (rad >= ab["r_mm"][0]) & (rad <= ab["r_mm"][1]) & ((along < mid) if ab["side"] == "low" else (along >= mid))
        return None, mask, {ab["mate"] + f" ({ab['side']} side)": mask}, {}
    if "box" in spec:
        pp = point_pred(spec)
        pred = lambda c, v: pp(c)  # noqa: E731
        return pred, pred(cent, None), {}, {}
    raise ValueError(f"unknown patch kind {sorted(spec)} (kinds: see the fem_core docstring)")


def _halves(mesh, cent, area, mask):
    """Patch area on each side of the part's bounding-box centre along the axis the patch is spread over most (one-cheek check)."""
    if not mask.any():
        return None
    ax = int(np.argmax(np.ptp(cent[mask], axis=0)))
    mid = (mesh["points"][:, ax].min() + mesh["points"][:, ax].max()) / 2
    lo = mask & (cent[:, ax] < mid)
    return dict(axis="xyz"[ax], mid_mm=round(float(mid), 1), low_mm2=round(float(area[lo].sum()), 1), high_mm2=round(float(area[mask & ~lo].sum()), 1))


def sites_of(mesh, cent, area, mask, keep=20):
    """The patch's sites: connected pieces of the selected surface triangles (sharing a node), largest first, each with its area-weighted centroid,
    area, nodes and triangles. A patch over two bolt sites reads as two sites, not as one centroid between them. Returns (rows[:keep], count)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    idx = np.flatnonzero(mask)
    if not len(idx):
        return [], 0
    T = mesh["tri"][idx]
    nn = len(mesh["points"])
    G = coo_matrix((np.ones(2 * len(T)), (np.concatenate([T[:, 0], T[:, 1]]), np.concatenate([T[:, 1], T[:, 2]]))), shape=(nn, nn))
    lab = connected_components(G, directed=False)[1][T[:, 0]]
    rows = []
    for k in np.unique(lab):
        s = lab == k
        a = area[idx[s]]
        rows.append(dict(area_mm2=round(float(a.sum()), 1), nodes=int(len(np.unique(T[s]))), triangles=int(s.sum()),
                         centroid_mm=np.round(np.average(cent[idx[s]], axis=0, weights=a if a.sum() > 0 else None), 2).tolist()))
    rows.sort(key=lambda r: -r["area_mm2"])
    return rows[:keep], len(rows)


def patch(ctx, mesh, spec):
    """Nodal area weights of a patch and what it selected. A PATCH dict with "within": {"centre_mm": POINT, "radius_mm": r} keeps only the surface
    within r of the point (POINT = [x, y, z] or {"joint": J, "offset_mm": [..]}): the load site held fixed to compare geometries."""
    specs = spec if isinstance(spec, list) else [spec]
    preds, masks, near, holes = [], [], {}, {}
    cent, area = _cent_area(mesh)
    for s in specs:
        p, m, nr, det = _one_patch(ctx, mesh, s)
        if isinstance(s, dict) and s.get("within"):
            c0, rw = point_of(ctx, s["within"]["centre_mm"]), float(s["within"]["radius_mm"])
            keep = np.linalg.norm(cent - c0, axis=1) <= rw
            m, nr = m & keep, {k: v & keep for k, v in nr.items()}
            if p is not None:
                p = (lambda p0: lambda c, v: p0(c, v) & (np.linalg.norm(np.asarray(c) - c0, axis=1) <= rw))(p)
        preds.append(p); masks.append(m); near.update(nr); holes.update(det)
    mask = np.logical_or.reduce(masks)
    if any(p is None for p in preds):      # mask-defined patches: lumped nodal area weights of the selected triangles
        w = np.bincount(mesh["tri"][mask].ravel(), weights=np.repeat(area[mask] / 3.0, 3), minlength=len(mesh["points"]))
    else:
        pred = lambda c, v: np.logical_or.reduce([p(c, v) for p in preds])  # noqa: E731
        w = LL().select_patch_weights(mesh["points"], mesh["tri"], pred)
    info = dict(triangles=int(mask.sum()), nodes=int(np.count_nonzero(w)), area_mm2=round(float(area[mask].sum()), 1),
                centroid_mm=np.round(np.average(cent[mask], axis=0, weights=area[mask]), 2).tolist() if mask.any() else None,
                by_half=_halves(mesh, cent, area, mask),
                follows_part=all(("contact" in s or "axis_band" in s or "holes" in s) for s in specs),
                by_mating_part={n: dict(area_mm2=round(float(area[m].sum()), 1), triangles=int(m.sum())) for n, m in near.items()})
    info["sites"], info["site_count"] = sites_of(mesh, cent, area, mask)
    if holes:
        info["hole_selection"] = holes
    return w, info, mask


def released_rotations(ctx, mesh, load, where):
    """Released rotations of a load union: one per joint named by a member's free_rotation_about_joint (the members naming one joint are one bearing),
    [dict(joint, axis, point, nodes, members)], or None. A released member whose patch selects no node raises, naming `where`, the member and its patch."""
    groups = {}
    for i, x in enumerate(load if isinstance(load, list) else [load]):
        if not (isinstance(x, dict) and x.get("free_rotation_about_joint")):
            continue
        j = x["free_rotation_about_joint"]
        spec = {k: v for k, v in x.items() if k != "free_rotation_about_joint"}
        w, _, _ = patch(ctx, mesh, spec)
        if not (w > 0).any():
            raise ValueError(f"{where}: load member {i} (free_rotation_about_joint {j}) selects no node of the part; its patch: {json.dumps(spec)}")
        jr = ctx.body.joints[j]
        g = groups.setdefault(j, dict(joint=j, axis=jr["axis_world_neutral"], point=jr["origin_world_mm"], nodes=[], members=[]))
        g["nodes"].append(np.flatnonzero(w > 0)); g["members"].append(i)
    return [dict(g, nodes=np.unique(np.concatenate(g["nodes"]))) for g in groups.values()] or None


# ------------------------------------------------------------------------------------------------ progress
_PROGRESS = dict(check="fem", part=None, t0=None, last=0.0)


def progress_start(part, check="fem"):
    _PROGRESS.update(check=check, part=part, t0=time.time(), last=0.0)


def say(msg, every_s=None):
    """A progress line on stderr, "[fem] PART +s: msg" (seconds since progress_start of the part); every_s: at most one such line per every_s seconds.
    Only stderr: the results do not change."""
    t = time.time() - (_PROGRESS["t0"] or time.time())
    if every_s and t - _PROGRESS["last"] < every_s:
        return
    _PROGRESS["last"] = t
    print(f"[{_PROGRESS['check']}] {_PROGRESS['part'] or ''} +{t:.0f} s: {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------ solver
def wrench_of(ctx, case, origin=None):
    """6-vector at the origin: given directly, a joint stall, or {"transfer": {F_N, M_Nm, scale, from_mm, lever}} = a force / moment given at from_mm,
    scaled, with the moment at the origin M*scale + lever * (from - origin) x F*scale (neck-head head-wrench stage scales)."""
    if "wrench_N_Nmm" in case:
        return np.asarray(case["wrench_N_Nmm"], float)
    if "transfer" in case:
        tr = case["transfer"]; s = float(tr.get("scale", 1.0))
        F = np.asarray(tr["F_N"], float) * s
        M = np.asarray(tr["M_Nm"], float) * 1000.0 * s + np.cross((np.asarray(tr["from_mm"], float) - np.asarray(origin, float)) * float(tr.get("lever", 1.0)), F)
        return np.concatenate([F, M])
    st = case["stall"]
    j = next(j for j in ctx.body.joints if j["name"] == st["joint"]) if not isinstance(ctx.body.joints, dict) else ctx.body.joints[st["joint"]]
    ax = np.asarray(j["axis_world_neutral"], float)
    w = np.zeros(6)
    w[3:] = float(st["torque_Nmm"]) * ax / np.linalg.norm(ax)
    return w


def rigid_modes(points):
    """The 6 rigid-body modes (3 translations, 3 rotations) of the node set, (3 N, 6): the near-nullspace for AMG on elasticity."""
    n = len(points); B = np.zeros((3 * n, 6))
    for i in range(3):
        B[i::3, i] = 1.0
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    B[1::3, 3], B[2::3, 3] = -z, y
    B[0::3, 4], B[2::3, 4] = z, -x
    B[0::3, 5], B[1::3, 5] = -y, x
    return B


CG_RTOL = 1e-8
SOLVE_LOG = []   # one entry per CG solve: iterations per load basis (6 unit wrenches), true relative residual ||Ax-b||/||b|| (max over them)


def spd_solve(A, rhs, method, B=None):
    """Solve A x = rhs for a symmetric positive-definite A (free-DOF stiffness). "spsolve" = SuperLU direct (the old path; CHECKS_FEM_SOLVER=spsolve); default "amg": R_thigh_roll_flip gives identical rounded numbers, peak 2.6 vs 3.8 GB;
    "amg" = conjugate gradients (relative residual CG_RTOL) preconditioned by pyamg smoothed aggregation, B = near-nullspace."""
    if method == "spsolve":
        SOLVE_LOG.append(dict(solver="spsolve"))
        say(f"direct solve (SuperLU, {A.shape[0]} unknowns, {rhs.shape[1]} right-hand sides)")
        return spsolve(A.tocsc(), rhs)
    import pyamg
    from scipy.sparse.linalg import cg
    A = A.tocsr()
    M = pyamg.smoothed_aggregation_solver(A, B=B, max_coarse=500).aspreconditioner(cycle="V")
    out, its, res = [], [], []
    for i in range(rhs.shape[1]):
        n = [0]
        def tick(_x, n=n):
            n[0] += 1
        t0 = time.time()
        x, info = cg(A, rhs[:, i], rtol=CG_RTOL, maxiter=2000, M=M, callback=tick)
        if info != 0:
            raise RuntimeError(f"CG did not converge (info {info})")
        say(f"CG solve {i + 1}/{rhs.shape[1]}: {n[0]} iterations, {time.time() - t0:.1f} s ({A.shape[0]} unknowns)")
        out.append(x); its.append(n[0]); res.append(float(np.linalg.norm(A @ x - rhs[:, i]) / max(np.linalg.norm(rhs[:, i]), 1e-300)))
    SOLVE_LOG.append(dict(solver="amg-cg", rhs=len(its), cg_iterations=its, true_relative_residual_max=max(res), rtol=CG_RTOL))
    return np.column_stack(out)


def solve(mesh, w_supports, w_load, origin, wrenches, material, ln_axis, bonded=None, field_path=None, field_case=None, solver=None, rigid_load=False, envelopes=None, release=None):
    """Static solve: moved from motor-refit/fem_refit.solve (same maths, same outputs) with the material table from
    data/materials.json: mesh material ids 0 / 2 = the setup's bonded materials (bonded = slot_materials(setup)), 1 = the polymer.
    release = released_rotations(...): load nodes that may also turn about a joint axis, one unloaded reference DOF per entry (rigid load only).
    envelopes = {case: (W (n, 6), labels, summary)}: a load case that stands for many wrenches (arm_loads.envelope); its row is the standard row of the
    wrench with the highest p99 plus "envelope" = distributions over all n wrenches (envelope_row)."""
    ll = LL()
    points, tet, mat_ids = mesh["points"], mesh["tet"], mesh["mat_ids"]
    origin = np.asarray(origin, float)
    fixed_nodes = np.unique(np.concatenate([np.flatnonzero(w > 0) for w in w_supports]))
    fixed = (3 * fixed_nodes[:, None] + np.arange(3)[None, :]).ravel()
    bonded = bonded or {}
    used = set(np.unique(mat_ids).tolist())
    code_label = {c: lab for lab, c in LABEL_CODE.items()}
    if any(c != 1 and code_label.get(c) not in bonded for c in used):
        raise ValueError("mesh has bonded companions whose material the setup does not name")
    Ds = [hooke(material, ln_axis) if c == 1 else hooke(bonded.get(code_label[c], material), ln_axis, material) for c in range(max(max(used), 2) + 1)]
    t0 = time.time()
    K, vol, B, _ = ll.assemble_chunked(points, tet, Ds, mat_ids)
    say(f"{material}, layer normal {'xyz'[ln_axis]}: stiffness assembled ({len(points)} nodes, {len(tet)} tets) in {time.time() - t0:.0f} s")
    nd = 3 * len(points)
    u_basis = np.zeros((nd, 6))
    method = solver or os.environ.get("CHECKS_FEM_SOLVER", "amg")
    if release and not rigid_load:
        raise ValueError("a released rotation (free_rotation_about_joint) needs a rigid load patch")
    if not rigid_load:
        unit = []
        for i in range(6):
            t = np.zeros(6); t[i] = 1.0
            f, _ = ll.ELAST.minimum_weight_wrench(points, w_load, t, origin); unit.append(f.ravel())
        unit = np.column_stack(unit)
        free = np.setdiff1d(np.arange(nd), fixed)
        u_basis[free] = spd_solve(K[free][:, free], unit[free], method, B=rigid_modes(points)[free])
    else:
        # master-slave: u_i = u0 + theta x (x_i - origin) on every free load node, 6 reference DOFs (u0, theta) carry the unit wrenches
        from scipy.sparse import coo_matrix
        ln = np.setdiff1d(np.flatnonzero(w_load > 0), fixed_nodes)
        ldofs = (3 * ln[:, None] + np.arange(3)[None, :]).ravel()
        other = np.setdiff1d(np.arange(nd), np.concatenate([fixed, ldofs]))
        no = len(other)
        r = points[ln] - origin
        # theta x r = -skew(r) theta with skew(r) = [[0,-rz,ry],[rz,0,-rx],[-ry,rx,0]]
        sk = np.zeros((len(ln), 3, 3)); sk[:, 0, 1], sk[:, 0, 2] = -r[:, 2], r[:, 1]; sk[:, 1, 0], sk[:, 1, 2] = r[:, 2], -r[:, 0]; sk[:, 2, 0], sk[:, 2, 1] = -r[:, 1], r[:, 0]
        R = -sk
        if release and method == "spsolve":
            raise ValueError("a load patch with a released rotation needs the amg path")
        if method == "spsolve":     # the old path: one direct solve of the 6-DOF-augmented system
            rows = [other]; cols = [np.arange(no)]; vals = [np.ones(no)]
            for a_ in range(3):
                rows.append(3 * ln + a_); cols.append(np.full(len(ln), no + a_)); vals.append(np.ones(len(ln)))
                for b_ in range(3):
                    rows.append(3 * ln + a_); cols.append(np.full(len(ln), no + 3 + b_)); vals.append(R[:, a_, b_])
            T = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nd, no + 6)).tocsr()
            Kr = (T.T @ K @ T).tocsr()
            rhs = np.zeros((no + 6, 6)); rhs[no:, :] = np.eye(6)
            u_basis = T @ spd_solve(Kr, rhs, "spsolve")
        else:
            # Schur complement on the 6 reference DOFs (the dense reference rows defeat AMG on the augmented system): Koo X = Kor (6 AMG-CG solves),
            # S = Krr - Kor^T X, reference motion u_r = S^-1 e_i, other nodes u_o = -X u_r; identical to the direct solve of the augmented system
            rel = release or []
            Tl = np.zeros((len(ldofs), 6 + len(rel)))
            for a_ in range(3):
                Tl[a_::3, a_] = 1.0
                Tl[a_::3, 3:6] = R[:, a_, :]
            for k, g in enumerate(rel):
                # g = {"nodes": node ids, "axis": a, "point": p}: those load nodes may also turn about the axis through p by an extra, unloaded
                # reference DOF phi_k (u_i += phi_k a x (x_i - p)): a bearing (servo idler) that carries forces and off-axis moments but no torque about its axis
                a_ax = np.asarray(g["axis"], float); a_ax = a_ax / np.linalg.norm(a_ax)
                sel = np.isin(ln, np.asarray(g["nodes"]))
                if not sel.any():
                    raise ValueError(f"released rotation about {g.get('joint')}: none of its {len(g['nodes'])} nodes is a free load node (all clamped by a support)")
                tang = np.cross(a_ax[None, :], points[ln] - np.asarray(g["point"], float)[None, :])
                for a_ in range(3):
                    Tl[a_::3, 6 + k] = np.where(sel, tang[:, a_], 0.0)
            Kc = K.tocsr()
            Kor = np.asarray(Kc[other][:, ldofs] @ Tl)
            Krr = Tl.T @ (Kc[ldofs][:, ldofs] @ Tl)
            X = spd_solve(Kc[other][:, other], Kor, method, B=rigid_modes(points)[other])
            Ur = np.linalg.inv(Krr - Kor.T @ X)[:, :6]    # the 6 unit wrenches load the first 6 reference DOFs; a released DOF carries no load
            u_basis[other] = -X @ Ur
            u_basis[ldofs] = Tl @ Ur
    # equilibrium: the clamped nodes' reaction (K u at the fixed DOFs, moment about the origin) must equal minus the applied wrench
    Rf = np.asarray(K.tocsr()[fixed] @ u_basis).reshape(len(fixed_nodes), 3, 6)
    react = np.vstack([Rf.sum(axis=0), np.cross((points[fixed_nodes] - origin)[:, :, None], Rf, axis=1).sum(axis=0)])    # (6, 6): reaction per unit-wrench basis
    D_el = np.array(Ds)[mat_ids]
    st_basis = np.empty((6, len(tet), 6))
    for i in range(6):
        ue = u_basis[:, i].reshape(-1, 3)[tet].reshape(-1, 12)
        st_basis[i] = np.einsum("eab,eb->ea", D_el, np.einsum("eij,ej->ei", B, ue))
    pm = mat_ids == 1
    info = materials()[material]["fem"]
    ref = info.get("reference_strength_MPa", info.get("tensile_XY_MPa")); ref_z = info.get("tensile_Z_layer_MPa", ref)   # None: no strength in the table, ratios null
    centers = points[tet].mean(axis=1)
    rows = {}
    for case, co in wrenches.items():
        dev = react @ co + co      # reaction + wrench, should be 0
        nf, nm, df, dm = np.linalg.norm(co[:3]), np.linalg.norm(co[3:]), np.linalg.norm(dev[:3]), np.linalg.norm(dev[3:])
        chk_ = dict(force_pct=round(100 * float(df / nf), 4) if nf > 1e-9 else None, moment_pct=round(100 * float(dm / nm), 4) if nm > 1e-9 else None,
                    force_abs_N=round(float(df), 6), moment_abs_Nmm=round(float(dm), 6))     # pct relative to the wrench part; None when that part is zero (stall: force 0)
        SOLVE_LOG[-1]["reaction_plus_wrench_pct_max"] = max([x for x in (chk_["force_pct"], chk_["moment_pct"], SOLVE_LOG[-1].get("reaction_plus_wrench_pct_max", 0.0)) if x is not None])
        u = (u_basis @ co).reshape(-1, 3); stress = np.einsum("i,iec->ec", co, st_basis); vm = ll.VON_MISES(stress)
        q = ll.volume_quantiles(vm[pm], vol[pm]); qz = ll.volume_quantiles(np.maximum(stress[pm, ln_axis], 0.0), vol[pm])
        if field_path is not None and case == field_case:
            np.savez_compressed(field_path, points_mm=points, tet=tet, mat_ids=mat_ids, von_mises_MPa=vm, displacement_mm=u)
        imax = np.flatnonzero(pm)[int(np.argmax(vm[pm]))]
        rows[case] = dict(p99_vm_MPa=round(q["p99"], 3), peak_vm_MPa=round(float(vm[pm].max()), 2), p99_layer_tension_MPa=round(qz["p99"], 3),
                          p99_strength_ratio=round(ref / max(q["p99"], 1e-6), 2) if ref else None, p99_layer_ratio=round(ref_z / max(qz["p99"], 1e-6), 2) if ref_z else None,
                          max_displacement_mm=round(float(np.linalg.norm(u, axis=1).max()), 4), reaction_plus_wrench_pct=chk_,
                          peak_vm_location_mm=np.round(centers[imax], 1).tolist())
    for case, (W, labels, summary) in (envelopes or {}).items():
        say(f"{material}, layer normal {'xyz'[ln_axis]}: envelope {case}, {len(W)} wrenches")
        env = envelope_row(W, labels, summary, st_basis, u_basis, pm, vol, ln_axis, ref, ref_z, centers, ll.VON_MISES)
        k = env.pop("_worst_index")
        sub = solve_rows_from_basis(dict(worst=np.asarray(W[k], float)), react, u_basis, st_basis, pm, vol, ln_axis, ref, ref_z, centers, ll, mesh, field_path, field_case if field_case != case else "worst")
        row = sub["worst"]
        if summary.get("symmetric"):    # layer tension of the opposite wrench (same von Mises and displacement)
            kz = int(np.argmax(env.pop("_layer_by_wrench")))
            neg = solve_rows_from_basis(dict(a=np.asarray(W[kz], float), b=-np.asarray(W[kz], float)), react, u_basis, st_basis, pm, vol, ln_axis, ref, ref_z, centers, ll, mesh)
            lz = max(neg.values(), key=lambda r: r["p99_layer_tension_MPa"])
            row = dict(row, p99_layer_tension_MPa=lz["p99_layer_tension_MPa"], p99_layer_ratio=lz["p99_layer_ratio"],
                       layer_tension_worst_wrench_N_Nmm=np.round(W[kz] * (1 if lz is neg["a"] else -1), 2).tolist())
        else:
            env.pop("_layer_by_wrench", None)
        rows[case] = dict(row, worst_wrench_N_Nmm=np.round(W[k], 2).tolist(), worst_label=labels[k], envelope=env)
    return rows


def solve_rows_from_basis(wrenches, react, u_basis, st_basis, pm, vol, ln_axis, ref, ref_z, centers, ll, mesh, field_path=None, field_case=None):
    """The standard per-case row (as in solve) for wrenches evaluated on an existing unit-wrench basis."""
    rows = {}
    for case, co in wrenches.items():
        dev = react @ co + co
        nf, nm, df, dm = np.linalg.norm(co[:3]), np.linalg.norm(co[3:]), np.linalg.norm(dev[:3]), np.linalg.norm(dev[3:])
        chk_ = dict(force_pct=round(100 * float(df / nf), 4) if nf > 1e-9 else None, moment_pct=round(100 * float(dm / nm), 4) if nm > 1e-9 else None,
                    force_abs_N=round(float(df), 6), moment_abs_Nmm=round(float(dm), 6))
        u = (u_basis @ co).reshape(-1, 3); stress = np.einsum("i,iec->ec", co, st_basis); vm = ll.VON_MISES(stress)
        q = ll.volume_quantiles(vm[pm], vol[pm]); qz = ll.volume_quantiles(np.maximum(stress[pm, ln_axis], 0.0), vol[pm])
        if field_path is not None and case == field_case:
            np.savez_compressed(field_path, points_mm=mesh["points"], tet=mesh["tet"], mat_ids=mesh["mat_ids"], von_mises_MPa=vm, displacement_mm=u)
        imax = np.flatnonzero(pm)[int(np.argmax(vm[pm]))]
        rows[case] = dict(p99_vm_MPa=round(q["p99"], 3), peak_vm_MPa=round(float(vm[pm].max()), 2), p99_layer_tension_MPa=round(qz["p99"], 3),
                          p99_strength_ratio=round(ref / max(q["p99"], 1e-6), 2) if ref else None, p99_layer_ratio=round(ref_z / max(qz["p99"], 1e-6), 2) if ref_z else None,
                          max_displacement_mm=round(float(np.linalg.norm(u, axis=1).max()), 4), reaction_plus_wrench_pct=chk_,
                          peak_vm_location_mm=np.round(centers[imax], 1).tolist())
    return rows


def _vm_batch(st):
    sx, sy, sz, txy, tyz, txz = (st[..., i] for i in range(6))
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3.0 * (txy * txy + tyz * tyz + txz * txz))


def _wq_batch(vals, v, q):
    """Volume-weighted quantile q of every row of vals (n, E) with element volumes v (E,), linear between sorted elements as volume_quantiles.
    For q >= 0.9 only the top slice of each row is sorted (argpartition, linear time): the slice holds the largest k elements with k chosen so that
    even the smallest elements fill 2 x (1 - q) of the volume; rows whose slice does not reach (1 - q) of the volume fall back to the full sort."""
    if q >= 0.9:
        E = vals.shape[1]
        tail = (1.0 - q) * float(v.sum())
        vs = np.sort(v)
        k = int(min(E - 1, max(2, np.searchsorted(np.cumsum(vs), 2.0 * tail) + 2)))
        idx = np.argpartition(vals, E - k, axis=1)[:, E - k:]                 # the k largest per row (unordered)
        top = np.take_along_axis(vals, idx, axis=1)
        o = np.argsort(-top, axis=1)
        top, vt = np.take_along_axis(top, o, axis=1), v[np.take_along_axis(idx, o, axis=1)]
        cum_top = np.cumsum(vt, axis=1)                                       # volume above-or-at each sorted value, from the top
        ok = cum_top[:, -1] >= tail
        out = np.empty(len(vals))
        if ok.any():
            # cum from below at the element with rank j from the top = total - cum_top[j] + vt[j]; quantile where cum_below/total crosses q
            tot = float(v.sum())
            r = np.flatnonzero(ok)
            cb = (tot - cum_top[r] + vt[r]) / tot                            # decreasing along the row
            j = np.clip((cb >= q).sum(axis=1), 1, k - 1)                     # first rank (from the top) whose cum_below < q
            rr = np.arange(len(r))
            hi_v, lo_v = top[r, j - 1], top[r, j]
            c_hi, c_lo = cb[rr, j - 1], cb[rr, j]
            t = np.clip((q - c_lo) / np.maximum(c_hi - c_lo, 1e-12), 0.0, 1.0)
            out[r] = lo_v + t * (hi_v - lo_v)
        if (~ok).any():
            out[~ok] = _wq_batch_full(vals[~ok], v, q)
        return out
    return _wq_batch_full(vals, v, q)


def _wq_batch_full(vals, v, q):
    order = np.argsort(vals, axis=1)
    sv = np.take_along_axis(vals, order, axis=1)
    cum = np.cumsum(v[order], axis=1); cum /= cum[:, -1:]
    k = np.clip((cum < q).sum(axis=1), 1, vals.shape[1] - 1)
    r = np.arange(len(vals))
    c0, c1 = cum[r, k - 1], cum[r, k]
    t = np.clip((q - c0) / np.maximum(c1 - c0, 1e-12), 0.0, 1.0)
    return sv[r, k - 1] + t * (sv[r, k] - sv[r, k - 1])


def _qs(x, ks=(("min", 0.0), ("p10", 0.1), ("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("max", 1.0))):
    return {k: round(float(np.quantile(x, v)), 3) for k, v in ks}


def envelope_row(W, labels, summary, st_basis, u_basis, pm, vol, ln_axis, ref, ref_z, centers, von_mises, chunk=128, keep=8):
    """Distributions over an envelope of wrenches W (n, 6) on the unit-wrench basis (linear, exact per wrench): p99 von Mises, p99 layer tension,
    peak, max displacement, strength ratios; the worst wrenches with their pose / direction labels; and the envelope field = per element the
    highest von Mises over all wrenches (its p99 and peak location show where the part needs material)."""
    symmetric = bool((summary or {}).get("symmetric"))
    v = vol[pm].astype(np.float64)
    S = np.ascontiguousarray(st_basis[:, pm, :], dtype=np.float32)
    U = u_basis.reshape(-1, 3, 6)
    n = len(W)
    p99, pz, peak, disp = (np.empty(n) for _ in range(4))
    env = np.zeros(S.shape[1], np.float32)
    for a in range(0, n, chunk):
        say(f"  {a}/{n} wrenches", every_s=15)
        w = np.asarray(W[a:a + chunk], np.float32)
        st = np.einsum("ni,iec->nec", w, S)
        vm = _vm_batch(st)
        env = np.maximum(env, vm.max(axis=0))
        peak[a:a + len(w)] = vm.max(axis=1)
        p99[a:a + len(w)] = _wq_batch(vm, v, 0.99)
        pz[a:a + len(w)] = _wq_batch(np.maximum(st[..., ln_axis], 0.0), v, 0.99)
        if symmetric:      # the wrench stands for itself and its opposite: layer tension of -w is max(-sigma_n, 0)
            pz[a:a + len(w)] = np.maximum(pz[a:a + len(w)], _wq_batch(np.maximum(-st[..., ln_axis], 0.0), v, 0.99))
        disp[a:a + len(w)] = np.linalg.norm(np.einsum("ni,dci->ndc", w.astype(np.float64), U), axis=2).max(axis=1)
    worst = np.argsort(-p99)[:keep]
    ie = int(np.argmax(env))
    env_q = volume_quantiles_np(env.astype(np.float64), v)
    ratio = (ref / np.maximum(p99, 1e-6)) if ref else None
    lratio = (ref_z / np.maximum(pz, 1e-6)) if ref_z else None
    return dict(_worst_index=int(worst[0]), _layer_by_wrench=pz, loads=summary, wrenches=n, layer_tension_both_signs=symmetric,
                p99_vm_MPa=_qs(p99), p99_layer_tension_MPa=_qs(pz), peak_vm_MPa=_qs(peak), max_displacement_mm=_qs(disp),
                p99_strength_ratio=_qs(ratio) if ratio is not None else None, p99_layer_ratio=_qs(lratio) if lratio is not None else None,
                wrenches_with_p99_strength_ratio_below_1=int((ratio < 1.0).sum()) if ratio is not None else None,
                wrenches_with_p99_layer_ratio_below_1=int((lratio < 1.0).sum()) if lratio is not None else None,
                worst=[dict(p99_vm_MPa=round(float(p99[i]), 2), p99_layer_tension_MPa=round(float(pz[i]), 2), max_displacement_mm=round(float(disp[i]), 3),
                            wrench_N_Nmm=np.round(W[i], 1).tolist(), **labels[i]) for i in worst],
                envelope_field=dict(what="per element the highest von Mises over all wrenches", p99_MPa=round(env_q["p99"], 2), p95_MPa=round(env_q["p95"], 2),
                                    peak_MPa=round(float(env[ie]), 2), peak_location_mm=np.round(centers[np.flatnonzero(pm)[ie]], 1).tolist()))


def volume_quantiles_np(vals, vols):
    return LL().volume_quantiles(vals, vols)
