"""bom: the body manifest's BOM against the parts the manifest (the CAD) actually contains, and the variety of hardware.

BOM lines (manifest `bom.items`) are mapped to links by data/bom_groups.json (id prefix -> links). Per line: the
quantity the BOM states, the count in the CAD (printed connectors / carriers by role; steel sleeves as the solids of each
part's sleeve STEP), and the difference. Then everything the CAD holds that no BOM line covers: screws by
thread and nominal length, nuts, washers, spacers, other hardware, printed parts outside the mapped groups and printed
shells, and every other part no line covers by role (motors, bearings, cells, modules ...). Solids of a sleeve STEP that
are not compression sleeves (outer diameter over 5 mm or shorter than 1.8 mm) are listed apart, not counted. Variety: distinct screw kinds (thread x length), nut / washer kinds, sleeve sizes (outer x inner x length, 0.05 mm).
Manifest-wide (the scope only filters which groups are listed). Measurements and counts only."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter

from checks import bodylib as BL
from checks.core import DATA, ROOT
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


def _thread(name):
    m = re.search(r"(?:^|_)(M\d(?:p\d)?)(?=_|x|$)", name, re.I)
    return m.group(1).upper().replace("P", ".") if m else "thread not in the name"


def _base(link):
    return re.sub(r"^[LR]_", "", link)


@register("bom", lambda ctx: bool(ctx.body.parts))
def run(ctx):
    """BOM lines vs the CAD (printed parts, sleeves), hardware the BOM does not list, and the number of hardware kinds."""
    sys.path.insert(0, str(ROOT / "robot/body/fastening"))
    from mount_check import kind, nominal, tubes
    man = json.loads(ctx.manifest.read_text()) if hasattr(ctx.manifest, "read_text") else json.loads(open(ctx.manifest).read())
    gd = json.loads((DATA / "bom_groups.json").read_text())
    groups = {k: set(v) for k, v in gd.items() if not k.startswith("_")}
    parts = {q["name"]: q for q in man.get("parts", [])}     # the raw manifest list: the body loader leaves some parts out (e.g. the hand bushings)
    grp_of = {n: gd.get("_parts", {}).get(n) or next((g for g, links in groups.items() if _base(p.get("owning_link") or "") in links), None) for n, p in parts.items()}
    sleeves = {n: sleeve_sizes(p) for n, p in parts.items() if p.get("role") == "compression_limit_sleeves" and (p.get("native_step") or p.get("mesh"))}
    lines, cases = [], []
    covered = set()
    for it in (man.get("bom") or {}).get("items", []):
        g = next((k for k in groups if it["id"].startswith(k)), None)
        cad = None
        if g and "sleeve" in it["id"]:
            cad = sum(sum(1 for x in v if _is_sleeve(x)) for n, v in sleeves.items() if grp_of[n] == g); covered |= {n for n in sleeves if grp_of[n] == g}
        elif g and ("fdm" in it["id"] or "Produced" in it.get("category", "")):
            names = [n for n, p in parts.items() if p.get("role") in PRINTED and grp_of[n] == g]
            cad = len(names); covered |= set(names)
        role = next((r for k, r in gd.get("_roles", {}).items() if it["id"].startswith(k)), None)
        if role:
            names = [n for n, p in parts.items() if p.get("role") == role]
            cad = len(names); covered |= set(names)
        row = dict(id=it["id"], group=g or role, bom_quantity=it.get("quantity"), cad_count=cad,
                   difference=None if cad is None or it.get("quantity") is None else cad - it["quantity"])
        lines.append(row)
        if row["difference"]:
            cases.append(dict(part=it["id"], mating_part=None, pose=None, location_mm=None, value=float(row["difference"]), unit="CAD - BOM count",
                              what=f"BOM {row['bom_quantity']}, CAD {cad} ({g})"))
    # hardware and parts no BOM line covers
    hw = Counter(); screws = Counter(); nuts = Counter(); washers = Counter()
    for n, p in parts.items():
        k = kind(p)
        if k == "screw":
            nm = nominal(n); screws[f"{nm.get('thread', '?')} x {nm.get('nominal_length_mm', '?')}"] += 1
        elif k == "nut" or (p.get("role") == "body_shell" and re.search(r"_nut(_\d+)?$", n)):
            nuts[nominal(n).get("thread") or _thread(n)] += 1
        elif k == "washer":
            washers[nominal(n).get("thread") or _thread(n)] += 1
        if k or (p.get("role") == "body_shell" and re.search(r"_nut(_\d+)?$", n)):
            hw[k or "nut"] += 1
    printed_unlisted = sorted(n for n, p in parts.items() if p.get("role") in PRINTED and n not in covered)
    other_by_role = Counter(p.get("role") for n, p in parts.items() if n not in covered and not kind(p) and p.get("role") not in PRINTED
                            and p.get("role") not in ("compression_limit_sleeves", "body_shell", "fastener", "fastener_proxy", "nominal_fastener", "nominal_hardware",
                                                      "standard_fastener_reference", "standard_nominal"))
    shells = sorted(n for n, p in parts.items() if p.get("role") == "body_shell" and not re.search(r"_nut(_\d+)?$", n))
    sizes = Counter(sz for v in sleeves.values() for sz in v if _is_sleeve(sz))
    not_sleeves = Counter(sz for v in sleeves.values() for sz in v if not _is_sleeve(sz))
    out = dict(bom_lines=lines, not_in_bom=dict(screws=dict(screws.most_common()), nuts=dict(nuts), washers=dict(washers),
                                               hardware_by_kind=dict(hw), printed_parts=printed_unlisted, shells=len(shells),
                                               other_parts_by_role=dict(other_by_role.most_common())),
               sleeve_sizes={f"{a:g} x {b:g} x {c:g}": k for (a, b, c), k in sizes.most_common()},
               other_solids_in_sleeve_steps={(f"{a:g} x {b:g} x {c:g}" if a else "no cylinder faces"): k for (a, b, c), k in not_sleeves.most_common()})
    ev = dict(bom_lines=len(lines), lines_with_count_difference=sum(1 for r in lines if r["difference"]), lines_not_mapped=sum(1 for r in lines if r["cad_count"] is None),
              screws_in_cad=sum(screws.values()), screw_kinds=len(screws), nut_kinds=len(nuts), washer_kinds=len(washers),
              sleeve_sizes=len(sizes), sleeves_in_cad=sum(sizes.values()), hardware_items_not_in_bom=sum(hw.values()),
              printed_parts_not_in_bom=len(printed_unlisted), shells_not_in_bom=len(shells), other_parts_not_in_bom=sum(other_by_role.values()),
              non_sleeve_solids_in_sleeve_steps=sum(not_sleeves.values()))
    cases.append(dict(part="manifest", mating_part=None, pose=None, location_mm=None, value=float(sum(hw.values())), unit="hardware items without a BOM line",
                      what=f"{sum(screws.values())} screws in {len(screws)} kinds, {sum(nuts.values())} nuts, {sum(washers.values())} washers; "
                           f"{len(printed_unlisted)} printed parts and {len(shells)} shells not in the BOM"))
    cases.sort(key=lambda c: -abs(c["value"]))
    return dict(per_part={"manifest": out}, worst_cases=cases, bad_events=ev, evidence=[],
                notes="BOM = manifest bom.items; groups from data/bom_groups.json; sleeves = solids of each <part>_steel_sleeves STEP (outer x inner x length from its cylinders, 0.05 mm); "
                      "hardware kinds from part names (fastening/mount_check.kind / nominal); manifest-wide, independent of the scope")
