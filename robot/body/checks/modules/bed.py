"""bed: first layer, footprint and warp screen per scoped printed part in its print direction (process.up).

First-layer contact = downward faces within one layer height of the bed, as a plan-view polygon (shapely union). Reported:
contact area, islands (separate contact patches, the smallest), contact as a share of the part's plan-view hull, the
contact's longest extent and aspect (minimum rotated rectangle), contact area in strips narrower than THIN_MM (thin bed
features curl first in CF nylon), sharp plan-view corners of the contact outline (corners lift), and stability: part height over the contact's hull size, and how far the centre of mass (of the
solid) projects inside the contact hull (negative = outside: it tips without a brim or support). Next to the part's own direction,
the first-layer contact it would have with each of the six axis directions up (contact_by_axis_mm2), so a seeded or chosen direction
can be read against the others without a run per direction.

Islands and corners are read on the contact as the first layer prints it: closed by half a line width (dilate, then
erode: a crevice narrower than a line width fuses, islands closer than a line width join), and islands smaller than one
line width squared are counted apart (`specks_under_line_width_sq`: a vertex or a sliver face on the bed, Z noise of a
few um), not as islands. A corner's angle is measured without the ring's orientation or its tessellation: 360 deg x the
share of a circle of CORNER_R_MM around an outline vertex that lies inside the contact (a reflex corner reads over 180,
a sharp 90 deg corner 90, a 90 deg corner rounded r 0.25 / 0.4 reads 106 / 117 at 0.5 mm); a sharp corner is a vertex
(the outline cut every CORNER_STEP_MM) under CORNER_DEG that is the sharpest within CORNER_SEP_MM (one per corner, however finely the outline is cut). Mesh-only
parts on their full closed mesh (checks/print_solid.py). Measurements and counts."""
from __future__ import annotations

import time

import numpy as np

from checks import print_solid as PS
from checks import process as P
from checks.core import tick
from checks.registry import register

THIN_MM = 4.0
CORNER_DEG = 100.0
CORNER_R_MM = 0.5          # circle radius the corner angle is read at
CORNER_SEP_MM = 1.0        # one sharp corner per this distance
CORNER_STEP_MM = 0.05      # outline points the corner angle is read at
LW_FALLBACK_MM = 0.42      # closing width when the part's line width does not resolve (said in the result)
RES = 0.2


def _targets(ctx):
    return [n for n in ctx.scope if P.is_printed(ctx, n) and P.effective(ctx, n)["up"] is not None]


AXES = ("+x", "-x", "+y", "-y", "+z", "-z")


def _bed_faces(m, layer):
    return (m.face_normals[:, 2] < -0.999) & (m.vertices[m.faces][:, :, 2].max(1) < layer)


def contact_area(m, layer):
    """First-layer contact area (mm2) of mesh m in its print frame, as measure() reads it (islands under 0.01 mm2 dropped)."""
    from shapely import polygons, union_all
    bed = _bed_faces(m, layer)
    if not bed.any():
        return 0.0
    poly = union_all(polygons(m.triangles[bed][:, :, :2])).buffer(0.01, join_style="mitre").buffer(-0.01, join_style="mitre")
    return round(float(sum(g.area for g in getattr(poly, "geoms", [poly]) if g.area > 0.01)), 1)


def by_axis(ctx, n, e):
    """{axis: first-layer contact mm2} with each of the six axis directions as the print direction (same plate yaw and layer)."""
    out = {}
    for ax in AXES:
        m = PS.print_mesh(ctx, n, dict(e, up=ax))
        out[ax] = contact_area(m, e["layer_height_mm"] or 0.2)
    return out


def _corners(poly, geoms):
    """Sharp outline corners of the contact: [(angle_deg, xy)], angle = 360 x the share of a circle of CORNER_R_MM around
    the vertex inside poly; vertices under CORNER_DEG, the sharpest within CORNER_SEP_MM kept."""
    from shapely import contains_xy, segmentize
    a = np.linspace(0.0, 2 * np.pi, 720, endpoint=False)
    # the outline cut every CORNER_STEP_MM: a rounded corner's sharpest point is read wherever the polygon's own vertices fall
    V = np.vstack([np.asarray(segmentize(r, CORNER_STEP_MM).coords)[:-1] for g in geoms for r in [g.exterior, *g.interiors]
                   if len(r.coords) > 3] or [np.zeros((0, 2))])
    if not len(V):
        return []
    ang = np.empty(len(V))
    for i in range(0, len(V), 2000):
        v = V[i:i + 2000]
        X = v[:, None, 0] + CORNER_R_MM * np.cos(a)[None]; Y = v[:, None, 1] + CORNER_R_MM * np.sin(a)[None]
        ang[i:i + 2000] = 360.0 * contains_xy(poly, X, Y).mean(1)
    cand = np.flatnonzero(ang < CORNER_DEG)
    cand = cand[np.lexsort((V[cand, 1], V[cand, 0], ang[cand]))]
    out = []
    for i in cand:
        if all(np.hypot(*(V[i] - p)) >= CORNER_SEP_MM for _, p in out):
            out.append((float(ang[i]), V[i]))
    return out


def measure(m, layer, lw=None):
    """Bed measurements of mesh m already in its print frame (bed at z 0); lw = line width (mm) the contact is closed by."""
    from scipy.ndimage import distance_transform_edt
    from shapely import contains_xy, polygons, union_all
    from shapely.geometry import MultiPoint, MultiPolygon, Point
    bed = _bed_faces(m, layer)
    out = dict(layer_height_mm=layer, height_mm=round(float(m.bounds[1][2]), 2))
    if not bed.any():
        return dict(out, contact_mm2=0.0, islands=0)
    raw = union_all(polygons(m.triangles[bed][:, :, :2])).buffer(0.01, join_style="mitre").buffer(-0.01, join_style="mitre").simplify(0.005)
    w = lw or LW_FALLBACK_MM
    out.update(closing_mm=round(w / 2, 3), closing_line_width_source="part's line width" if lw else f"no line width resolved: {LW_FALLBACK_MM} mm assumed")
    closed = raw.buffer(w / 2).buffer(-w / 2).simplify(0.005)
    allg = list(closed.geoms) if isinstance(closed, MultiPolygon) else [closed]
    geoms = [g for g in allg if g.area >= w * w]
    specks = [g for g in allg if 0 < g.area < w * w]
    out.update(specks_under_line_width_sq=len(specks), specks_mm2=round(float(sum(g.area for g in specks)), 3))
    rawg = [g for g in (list(raw.geoms) if isinstance(raw, MultiPolygon) else [raw]) if g.area > 0.01]
    out["raw_patches"] = len(rawg)
    if not geoms:                            # bed-facing faces whose patches are all under one line width squared
        return dict(out, contact_mm2=round(float(raw.area), 3), islands=0, bed_facing_area_under_0p01_mm2=round(float(raw.area), 4))
    poly = MultiPolygon(geoms) if len(geoms) > 1 else geoms[0]
    contact = float(raw.intersection(poly).area)     # the faces' own contact area inside the printed islands
    hull_all = MultiPoint(m.vertices[:, :2]).convex_hull      # the hull of the triangles' union is the hull of their vertices
    chull = poly.convex_hull
    mrr = poly.minimum_rotated_rectangle
    xy = np.asarray(mrr.exterior.coords)[:4]
    sides = sorted([np.linalg.norm(xy[1] - xy[0]), np.linalg.norm(xy[2] - xy[1])])
    # thin strips: contact pixels whose distance to the outline is under THIN_MM / 2
    x0, y0, x1, y1 = poly.bounds
    # raster on whole multiples of RES from the print-frame origin: a part and its x-mirror get the mirrored raster
    xs = np.arange(np.floor(x0 / RES) - 1, np.ceil(x1 / RES) + 2) * RES; ys = np.arange(np.floor(y0 / RES) - 1, np.ceil(y1 / RES) + 2) * RES
    X, Y = np.meshgrid(xs, ys)
    inside = contains_xy(poly, X, Y)
    d = distance_transform_edt(inside) * RES
    # a pixel is in a thin strip when the widest circle containing it is narrower than THIN_MM: dilate the "wide core"
    core = d >= THIN_MM / 2                  # morphological opening with a disc of diameter THIN_MM (Euclidean)
    wide = (distance_transform_edt(~core) * RES <= THIN_MM / 2 + 1e-9) & inside
    thin_area = float((inside & ~wide).sum() * RES * RES)
    corners = _corners(poly, geoms)
    com = m.center_mass if m.is_volume else m.vertices.mean(0)
    inside_com = chull.exterior.distance(Point(com[0], com[1])) * (1 if chull.contains(Point(com[0], com[1])) else -1)
    out.update(contact_mm2=round(contact, 1), islands=len(geoms), smallest_island_mm2=round(min(float(raw.intersection(g).area) for g in geoms), 2),
               contact_share_of_footprint=round(float(contact / max(hull_all.area, 1e-9)), 3),
               footprint_hull_mm2=round(float(hull_all.area), 1),
               contact_extent_mm=[round(float(sides[1]), 2), round(float(sides[0]), 2)],
               contact_aspect=round(float(sides[1] / max(sides[0], 1e-9)), 2),
               contact_in_strips_under_thin_mm2=round(thin_area, 1), thin_strip_width_mm=THIN_MM,
               sharp_corners=len(corners), sharpest_corners=[dict(angle_deg=round(a, 1), xy_mm=np.round(p, 2).tolist()) for a, p in corners[:6]],
               height_over_contact_hull=round(float(m.bounds[1][2] / max(np.sqrt(chull.area), 1e-9)), 2),
               com_height_mm=round(float(com[2]), 2), com_inside_contact_hull_mm=round(float(inside_com), 2))
    return out


@register("bed", lambda ctx: bool(_targets(ctx)))
def run(ctx):
    """First layer per printed part in its print direction: contact area and islands, thin strips, sharp corners, aspect, tipping."""
    per, cases = {}, []
    ev = dict(parts=0, parts_with_small_islands=0, sharp_contact_corners=0, parts_com_outside_contact=0, specks_under_line_width_sq=0)
    for n in _targets(ctx):
        t0 = time.time()
        e = P.effective(ctx, n)
        m, T = PS.print_mesh(ctx, n, e, with_transform=True)
        r = dict(material=e["material"], up=e["up"], up_source=e["source"]["up"], solid=PS.solid(ctx, n)[1],
                 **measure(m, e["layer_height_mm"] or 0.2, e["line_width_mm"]))
        r["contact_by_axis_mm2"] = by_axis(ctx, n, e)
        best = max(r["contact_by_axis_mm2"], key=r["contact_by_axis_mm2"].get)
        Ti = np.linalg.inv(T)
        for c in r.get("sharpest_corners", []):
            c["location_mm"] = np.round((Ti @ np.r_[c.pop("xy_mm"), 0.0, 1.0])[:3], 2).tolist()
        r["seconds"] = round(time.time() - t0, 1); tick("bed", t0)
        per[n] = r
        ev["parts"] += 1
        ev["specks_under_line_width_sq"] += r.get("specks_under_line_width_sq", 0)
        if not r.get("islands"):
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=None, value=0.0, unit="mm2 bed contact",
                              what=f"nothing on the bed in this direction ({r['up']}, {r['up_source']}); most contact with up {best}: {r['contact_by_axis_mm2'][best]} mm2"))
            continue
        ev["parts_with_small_islands"] += r["smallest_island_mm2"] < 10.0
        ev["sharp_contact_corners"] += r["sharp_corners"]
        ev["parts_com_outside_contact"] += r["com_inside_contact_hull_mm"] < 0
        cases.append(dict(part=n, mating_part=None, pose=None, location_mm=None, value=r["contact_mm2"], unit="mm2 bed contact",
                          what=f"{r['islands']} islands (smallest {r['smallest_island_mm2']} mm2), {round(100 * r['contact_share_of_footprint'])} % of the footprint, "
                               f"{r['contact_in_strips_under_thin_mm2']} mm2 in strips under {THIN_MM:g} mm, {r['sharp_corners']} corners under {CORNER_DEG:g} deg, "
                               f"centre of mass {r['com_inside_contact_hull_mm']} mm inside the contact hull at {r['com_height_mm']} mm height ({r['up']}, {r['up_source']}); "
                               f"most contact with up {best}: {r['contact_by_axis_mm2'][best]} mm2"))
    cases.sort(key=lambda c: c["value"])
    return dict(per_part=per, worst_cases=cases, bad_events=ev, evidence=[],
                notes=f"contact = faces facing the bed (normal within 2.6 deg) entirely within one layer of z 0; islands and corners on the contact closed by half a line width, "
                      f"patches under one line width squared counted as specks; thin strips = contact not covered by a {THIN_MM:g} mm disc inside it "
                      f"(raster {RES} mm; a square corner adds about 0.86 mm2); sharp corners = outline vertices whose angle (360 deg x share of a {CORNER_R_MM:g} mm circle inside the contact) is under {CORNER_DEG:g} deg, one per {CORNER_SEP_MM:g} mm; small island = under 10 mm2; "
                      f"locations in the manifest frame; contact_by_axis_mm2 = the same first-layer contact with each axis direction as up (plate yaw and layer of the part's setup)")
