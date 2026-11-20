"""bed: first layer, footprint and warp screen per scoped printed part in its print direction (process.up).

First-layer contact = downward faces within one layer height of the bed, as a plan-view polygon (shapely union). Reported:
contact area, islands (separate contact patches, the smallest), contact as a share of the part's plan-view hull, the
contact's longest extent and aspect (minimum rotated rectangle), contact area in strips narrower than THIN_MM (thin bed
features curl first in CF nylon), sharp convex plan-view corners of the contact outline (interior angle under
CORNER_DEG; corners lift), and stability: part height over the contact's hull size, and how far the centre of mass (of the
solid) projects inside the contact hull (negative = outside: it tips without a brim or support). Measurements and counts."""
from __future__ import annotations

import time

import numpy as np

from checks import process as P
from checks.core import tick
from checks.registry import register

THIN_MM = 4.0
CORNER_DEG = 100.0
RES = 0.2


def _targets(ctx):
    return [n for n in ctx.scope if P.is_printed(ctx, n) and P.effective(ctx, n)["up"] is not None]


def measure(m, layer):
    """Bed measurements of mesh m already in its print frame (bed at z 0)."""
    from scipy.ndimage import distance_transform_edt
    from shapely import contains_xy, polygons, union_all
    from shapely.geometry import MultiPolygon, Point
    nz = m.face_normals[:, 2]
    zmax_f = m.vertices[m.faces][:, :, 2].max(1)
    bed = (nz < -0.999) & (zmax_f < layer)
    out = dict(layer_height_mm=layer, height_mm=round(float(m.bounds[1][2]), 2))
    if not bed.any():
        return dict(out, contact_mm2=0.0, islands=0)
    poly = union_all(polygons(m.triangles[bed][:, :, :2])).buffer(0.01, join_style="mitre").buffer(-0.01, join_style="mitre").simplify(0.005)
    geoms = list(poly.geoms) if isinstance(poly, MultiPolygon) else [poly]
    geoms = [g for g in geoms if g.area > 0.01]
    hull_all = union_all(polygons(m.triangles[:, :, :2])).convex_hull
    chull = poly.convex_hull
    mrr = poly.minimum_rotated_rectangle
    xy = np.asarray(mrr.exterior.coords)[:4]
    sides = sorted([np.linalg.norm(xy[1] - xy[0]), np.linalg.norm(xy[2] - xy[1])])
    # thin strips: contact pixels whose distance to the outline is under THIN_MM / 2
    x0, y0, x1, y1 = poly.bounds
    xs = np.arange(x0 - RES, x1 + 2 * RES, RES); ys = np.arange(y0 - RES, y1 + 2 * RES, RES)
    X, Y = np.meshgrid(xs, ys)
    inside = contains_xy(poly, X, Y)
    d = distance_transform_edt(inside) * RES
    # a pixel is in a thin strip when the widest circle containing it is narrower than THIN_MM: dilate the "wide core"
    core = d >= THIN_MM / 2                  # morphological opening with a disc of diameter THIN_MM (Euclidean)
    wide = (distance_transform_edt(~core) * RES <= THIN_MM / 2 + 1e-9) & inside
    thin_area = float((inside & ~wide).sum() * RES * RES)
    # sharp convex corners of the outline
    corners = []
    for g in geoms:
        for ring in [g.exterior]:
            c = np.asarray(ring.coords)[:-1]
            if len(c) < 3:
                continue
            ccw = ring.is_ccw
            for i in range(len(c)):
                a, b, e = c[i - 1], c[i], c[(i + 1) % len(c)]
                u, v = a - b, e - b
                if np.linalg.norm(u) < 0.3 or np.linalg.norm(v) < 0.3:
                    continue
                ang = np.degrees(np.arccos(np.clip(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
                cross = (b - a)[0] * (e - b)[1] - (b - a)[1] * (e - b)[0]
                convex = cross > 0 if ccw else cross < 0
                if convex and ang < CORNER_DEG:
                    corners.append((float(ang), b))
    corners.sort(key=lambda t: t[0])
    com = m.center_mass if m.is_volume else m.vertices.mean(0)
    inside_com = chull.exterior.distance(Point(com[0], com[1])) * (1 if chull.contains(Point(com[0], com[1])) else -1)
    out.update(contact_mm2=round(float(poly.area), 1), islands=len(geoms), smallest_island_mm2=round(min(g.area for g in geoms), 2),
               contact_share_of_footprint=round(float(poly.area / max(hull_all.area, 1e-9)), 3),
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
    ev = dict(parts=0, parts_with_small_islands=0, sharp_contact_corners=0, parts_com_outside_contact=0)
    for n in _targets(ctx):
        t0 = time.time()
        e = P.effective(ctx, n)
        m, T = P.print_mesh(ctx, n, e, with_transform=True)
        r = dict(material=e["material"], up=e["up"], up_source=e["source"]["up"], **measure(m, e["layer_height_mm"] or 0.2))
        Ti = np.linalg.inv(T)
        for c in r.get("sharpest_corners", []):
            c["location_mm"] = np.round((Ti @ np.r_[c.pop("xy_mm"), 0.0, 1.0])[:3], 2).tolist()
        r["seconds"] = round(time.time() - t0, 1); tick("bed", t0)
        per[n] = r
        ev["parts"] += 1
        if not r["contact_mm2"]:
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=None, value=0.0, unit="mm2 bed contact", what="nothing on the bed in this direction"))
            continue
        ev["parts_with_small_islands"] += r["smallest_island_mm2"] < 10.0
        ev["sharp_contact_corners"] += r["sharp_corners"]
        ev["parts_com_outside_contact"] += r["com_inside_contact_hull_mm"] < 0
        cases.append(dict(part=n, mating_part=None, pose=None, location_mm=None, value=r["contact_mm2"], unit="mm2 bed contact",
                          what=f"{r['islands']} islands (smallest {r['smallest_island_mm2']} mm2), {round(100 * r['contact_share_of_footprint'])} % of the footprint, "
                               f"{r['contact_in_strips_under_thin_mm2']} mm2 in strips under {THIN_MM:g} mm, {r['sharp_corners']} corners under {CORNER_DEG:g} deg, "
                               f"centre of mass {r['com_inside_contact_hull_mm']} mm inside the contact hull at {r['com_height_mm']} mm height ({r['up']}, {r['up_source']})"))
    cases.sort(key=lambda c: c["value"])
    return dict(per_part=per, worst_cases=cases, bad_events=ev, evidence=[],
                notes=f"contact = faces facing the bed (normal within 2.6 deg) entirely within one layer of z 0; thin strips = contact not covered by a {THIN_MM:g} mm disc inside it "
                      f"(raster {RES} mm; a square corner adds about 0.86 mm2); sharp corners = convex outline corners under {CORNER_DEG:g} deg with both sides >= 0.3 mm; small island = under 10 mm2; "
                      f"locations in the manifest frame")
