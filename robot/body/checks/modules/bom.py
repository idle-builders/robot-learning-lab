"""bom: the robot BOM assembled from the body definitions (robot/body/bom/build.py over robot/**/<builder>.bom.yaml)
against the CAD. Reports what the definitions do not say yet, each finding naming the definition file to fix: manifest
parts no definition claims, entries / globs matching nothing, parts claimed twice, declared quantity vs manifest count,
declared files that resolve nowhere, hardware without a declared spec, printed parts without a print setup, generator
inputs no generator writes, parts without an assembly step, stale steps, declared sleeve contents vs the sleeves in each
sleeve STEP (one solid per sleeve, every solid counted whatever its size; each solid matched to the declared size it
measures, the sizes no declaration has listed). Plus supplier coverage and hardware variety by declared spec kind. Manifest-wide (the scope does not filter it). Measurements and counts only."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from checks import bodylib as BL
from checks.core import ROOT
from checks.registry import register

PRINTED = ("printed_structural_connector", "printed_structural_carrier")


def sleeve_sizes(p, q=0.05):
    """[(outer, inner, length) mm] per sleeve: one per solid of the native STEP (cylinder radii, extent along the axis);
    without a STEP, the tubes of the mesh (mount_check.tubes; touching tubes can merge there)."""
    import numpy as np
    r = lambda v: round(round(v / q) * q, 3)  # noqa: E731
    if p.get("native_step"):
        from build123d import import_step
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.GeomAbs import GeomAbs_Cylinder
        out = []
        for so in import_step(str(BL.src(p["native_step"]))).solids():
            rad, ax = [], None
            for f in so.faces():
                s = BRepAdaptor_Surface(f.wrapped, True)
                if s.GetType() == GeomAbs_Cylinder:
                    c = s.Cylinder(); rad.append(c.Radius()); d = c.Axis().Direction(); ax = np.array([d.X(), d.Y(), d.Z()])
            if not rad:
                out.append((None, None, None)); continue
            v = np.array([[x.X, x.Y, x.Z] for x in so.vertices()]) if so.vertices() else np.zeros((1, 3))
            t = v @ ax
            out.append((r(2 * max(rad)), r(2 * min(rad)), r(float(t.max() - t.min()))))
        return out
    sys.path.insert(0, str(ROOT / "robot/body/fastening"))
    from mount_check import tubes
    return [(r(od), r(idd), r(L)) for _, _, L, od, idd in tubes(BL.load_mesh(p["mesh"]))]


# How closely a STEP solid's measured size (rounded to sleeve_sizes' 0.05 mm) must agree with a declared spec (outer, inner,
# length) to count as that declared sleeve: a measurement tolerance, not a size limit. Which solids are sleeves comes from the
# declarations (contains: spec outer_dia_mm / inner_dia_mm / length_mm), never from a fixed diameter (the old 5 mm outer limit
# left the head carrier's 6.1-6.7 mm steel rings and the thigh's 1.25-1.6 mm sleeves uncounted, 2026-11-24).
MATCH_TOL_MM = 0.1


def match_sleeves(sizes, declared):
    """Solids of a sleeve STEP against its declared contents. sizes: [(outer, inner, length)] per solid; declared:
    [(line, quantity, spec)]. Each solid goes to the declared line whose spec it measures within MATCH_TOL_MM on every declared
    dimension (an insert without inner_dia_mm matches on outer and length), the closest line with quantity left first.
    Returns ({line: solids matched}, [(outer, inner, length) of solids matching no declaration])."""
    left = {ln: q for ln, q, _ in declared}
    got, other = {ln: 0 for ln, _, _ in declared}, []
    for x in sizes:
        cands = []
        for ln, _, sp in declared:
            dif = [abs(a - float(sp[k])) for a, k in zip(x, ("outer_dia_mm", "inner_dia_mm", "length_mm"))
                   if a is not None and (sp or {}).get(k) is not None]
            if dif and max(dif) <= MATCH_TOL_MM + 1e-9:
                cands.append((left[ln] <= 0, max(dif), ln))
        if cands:
            ln = min(cands)[2]
            got[ln] += 1; left[ln] -= 1
        else:
            other.append(x)
    return got, other


@register("bom", lambda ctx: bool(ctx.body.parts))
def run(ctx):
    """The BOM assembled from the body definitions against the CAD: every definition finding (part no definition claims,
    entries / globs matching nothing, parts claimed twice, quantity differences, declared files missing, hardware without a
    spec, printed parts without a print setup, generator inputs nobody writes, parts without an assembly step, stale steps),
    declared sleeve contents against the solids of each sleeve STEP, supplier coverage and hardware variety. Each finding
    names the definition file to fix."""
    sys.path.insert(0, str(ROOT / "robot/body/bom"))
    import build as BOM
    mp = Path(str(ctx.manifest)).resolve()
    bom = BOM.build(str(mp.relative_to(ROOT)) if mp.is_relative_to(ROOT) else str(mp))
    lines, c = bom["lines"], bom["counts"]
    findings = list(bom["findings"])
    # declared contents of sleeve-set parts vs the sleeves in their STEP (one solid per sleeve)
    for l in lines:
        for p in l["parts"]:
            decl = sum(x["quantity"] for x in l.get("contains_per_part", {}).get(p["part"], []))
            if not decl:
                continue
            spec = {x["id"]: x.get("spec") or {} for x in lines}
            declared = [(x["line"], x["quantity"], spec.get(x["line"], {})) for x in l.get("contains_per_part", {}).get(p["part"], [])]
            try:
                sizes = sleeve_sizes(ctx.body.parts[p["part"]])
            except Exception as e:  # noqa: BLE001
                findings.append(dict(kind="contains_not_counted", file=p["definition"], entry=l["id"], what=f"{p['part']}: STEP not read ({e})", part=p["part"])); continue
            cad = len(sizes)
            got, other = match_sleeves(sizes, declared)
            if cad != decl:
                findings.append(dict(kind="contains_differs_from_cad", file=p["definition"], entry=l["id"], what=f"{p['part']}: declared {decl} sleeves, STEP holds {cad} solids", part=p["part"], declared=decl, cad=cad))
            off = {ln: (q, got[ln]) for ln, q, _ in declared if got[ln] != q}
            if off or other:
                sz = Counter(other)
                findings.append(dict(kind="contains_size_differs_from_cad", file=p["definition"], entry=l["id"], part=p["part"],
                                     what=f"{p['part']}: " + "; ".join([f"{ln} declared {q}, STEP {g}" for ln, (q, g) in off.items()]
                                                                     + [f"{n} solid(s) of no declared size {o} x {i} x {L} mm (outer x inner x length)" for (o, i, L), n in sz.items()]),
                                     declared_vs_cad={ln: dict(declared=q, cad=g) for ln, (q, g) in off.items()},
                                     undeclared_sizes=[dict(outer_mm=o, inner_mm=i, length_mm=L, solids=n) for (o, i, L), n in sz.items()]))
    kinds = Counter(f["kind"] for f in findings)
    hw = [l for l in lines if l["kinds"] == ["hardware"] and l["make_buy"] == "buy"]
    spec_kinds = Counter((l.get("spec") or {}).get("kind") for l in hw)
    ev = dict(lines=len(lines), findings=len(findings), **{f"findings_{k}": v for k, v in sorted(kinds.items())},
              parts=c["parts"], parts_claimed=c["parts_claimed"], parts_with_assembly_step=c["parts_with_assembly_step"],
              lines_quantity_differs_from_declared=c["lines_quantity_differs_from_declared"],
              buy_lines=c["buy_lines"], buy_lines_supplier_known=c["buy_lines_supplier_known"], buy_lines_part_number_known=c["buy_lines_part_number_known"],
              buy_lines_supplier_empty=c["buy_lines_supplier_empty"],
              screw_kinds=spec_kinds.get("screw", 0), nut_kinds=spec_kinds.get("nut", 0), washer_kinds=spec_kinds.get("washer", 0),
              spacer_kinds=spec_kinds.get("spacer", 0), sleeve_sizes=spec_kinds.get("sleeve", 0),
              printed_without_native_step=c["printed_parts"] - c["printed_with_native_step"], printed_without_print_setup=c["printed_parts"] - c["printed_with_print_setup"],
              parts_mesh_only=c["parts_mesh_only"], parts_file_missing=c["parts_file_missing"], parts_file_in_bucket_not_indexed=c["parts_file_in_bucket_not_indexed"],
              parts_without_geometry=c["parts_without_geometry"], definitions=len(bom["definitions"]), generators=len(bom["generators"]))
    cases = [dict(part=f.get("part") or f.get("entry"), mating_part=None, pose=None, location_mm=None,
                  value=float(f["cad"] - f["declared"]) if "cad" in f and "declared" in f else 1.0, unit=f["kind"],
                  what=f"{f['file'] or '(no definition)'}: {f['what']}") for f in findings]
    out = dict(counts=c, findings=findings, definitions=bom["definitions"], generators=bom["generators"],
               lines=[{k: l.get(k) for k in ("id", "name", "make_buy", "quantity", "quantity_declared", "sections", "supplier", "part_number", "origin")} for l in lines])
    return dict(per_part={"manifest": out}, worst_cases=cases[:60], bad_events=ev, evidence=[],
                notes="BOM = robot/body/bom/build.py over the body definitions (robot/**/<builder>.bom.yaml); every finding names the definition file to fix")
