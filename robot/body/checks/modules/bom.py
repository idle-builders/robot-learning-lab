"""bom: the robot BOM (robot/body/bom/build.py: the manifest joined with the hand-maintained robot/body/bom/lines.json)
against the CAD, and the variety of hardware.

Per line: quantity (lines.json override) vs the count in the CAD, make / buy, supplier. Counts: lines, lines whose
quantity differs from the CAD, buy lines without a supplier or part number, lines with make / buy unknown, parts excluded
in lines.json, and per part what a builder cannot get from a script (mesh only, file missing, no geometry, no print
settings for a printed part, no assembly step). Variety: screw kinds (thread x length x head standard), nut / washer /
spacer kinds, steel sleeve sizes (outer x inner x length, 0.05 mm, from each <part>_steel_sleeves STEP; solids over 5 mm
outer or under 1.8 mm long are not sleeves). Manifest-wide (the scope does not filter it). Measurements and counts only."""
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


SLEEVE_MAX_OD_MM, SLEEVE_MIN_LEN_MM = 5.0, 1.8


def _is_sleeve(x):
    """A compression-limit sleeve: outer diameter at most SLEEVE_MAX_OD_MM and at least SLEEVE_MIN_LEN_MM long (the rings
    of 6-7 mm x 1.5 mm in some sleeve STEPs are not sleeves)."""
    od, idd, L = x
    return od is not None and od <= SLEEVE_MAX_OD_MM and L >= SLEEVE_MIN_LEN_MM and idd < od


@register("bom", lambda ctx: bool(ctx.body.parts))
def run(ctx):
    """The generated BOM against the CAD: quantity differences, open make / buy and supplier fields, hardware variety."""
    sys.path.insert(0, str(ROOT / "robot/body/bom"))
    import build as BOM
    mp = Path(str(ctx.manifest)).resolve()
    bom = BOM.build(str(mp.relative_to(ROOT)) if mp.is_relative_to(ROOT) else str(mp))
    lines = bom["lines"]
    cases = []
    for l in lines:
        if l.get("quantity") is not None and l.get("quantity_cad") is not None and l["quantity"] != l["quantity_cad"]:
            cases.append(dict(part=l["id"], mating_part=None, pose=None, location_mm=None, value=float(l["quantity"] - l["quantity_cad"]),
                              unit="BOM - CAD count", what=f"lines.json {l['quantity']}, CAD {l['quantity_cad']}"))
        if l.get("quantity_cad") is None and l["parts"]:
            cases.append(dict(part=l["id"], mating_part=None, pose=None, location_mm=None, value=float(len(l["parts"])),
                              unit="parts on a line the CAD could not count", what="sleeve STEP not readable"))
    hw = [l for l in lines if l["id"].startswith("hw:")]
    kinds = Counter((l.get("spec") or {}).get("kind") for l in hw)
    sleeves = [l for l in lines if l["id"].startswith("sleeve:") and not l["id"].startswith("sleeve:uncounted:")]
    c = bom["counts"]
    ev = dict(lines=len(lines), lines_quantity_differs_from_cad=sum(1 for x in cases if x["unit"] == "BOM - CAD count"),
              lines_make_buy_unknown=sum(1 for l in lines if l["make_buy"] not in ("make", "buy", "included", "context")),
              buy_lines_without_supplier=sum(1 for l in lines if l["make_buy"] == "buy" and not (l.get("supplier") or l.get("part_number"))),
              parts_excluded=len(bom["excluded"]), screw_kinds=kinds.get("screw", 0), nut_kinds=kinds.get("nut", 0),
              washer_kinds=kinds.get("washer", 0), spacer_kinds=kinds.get("spacer", 0), sleeve_sizes=len(sleeves),
              sleeves_in_cad=sum(l["quantity_cad"] or 0 for l in sleeves),
              printed_without_native_step=c["printed_parts"] - c["printed_with_native_step"],
              printed_without_print_settings=c["printed_parts"] - c["printed_with_print_settings_entry"],
              parts_without_assembly_step=c["parts"] - c["parts_with_assembly_step"], parts_mesh_only=c["parts_mesh_only"],
              parts_file_missing=c["parts_file_missing"], parts_without_geometry=c["parts_without_geometry"])
    out = dict(counts=c, lines=[{k: l.get(k) for k in ("id", "name", "make_buy", "quantity", "quantity_cad", "sections", "supplier", "part_number")}
                                for l in lines],
               hardware_by_spec={l["name"]: l["quantity"] for l in hw}, sleeve_sizes={l["name"]: l["quantity_cad"] for l in sleeves},
               hardware_specs=bom["hardware_specs"], excluded=bom["excluded"])
    cases.append(dict(part="manifest", mating_part=None, pose=None, location_mm=None, value=float(ev["buy_lines_without_supplier"]),
                      unit="buy lines without supplier or part number", what=f"{len(lines)} lines; {len(hw)} hardware kinds, {len(sleeves)} sleeve sizes"))
    cases.sort(key=lambda x: -abs(x["value"]))
    return dict(per_part={"manifest": out}, worst_cases=cases, bad_events=ev, evidence=[],
                notes="BOM = robot/body/bom/build.py (manifest + robot/body/bom/lines.json + robot/body/bom/hardware.py specs); "
                      "hardware specs come from .cache/work/bom-hardware/hardware.json when it matches the manifest, else they are measured now (adds about a minute)")
