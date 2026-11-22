"""Standard hardware of the body manifest as specs and as generated geometry: screws, nuts, washers and spacers.

Each hardware part gets a buyable spec (kind, thread, nominal length, head standard, nut / washer / spacer size) from its
name (fastening/mount_check.nominal) and from measurements of its current mesh (axis, head diameter and height, outer /
inner diameter, thickness), snapped to the standard sizes in robot/body/checks/data/hardware.json and the tables below.
Then the part is generated from that spec at the measured placement (axis, head end, centre) as a native STEP and an
STL, so no hardware depends on a mesh from an old study. Per part the report gives the fit of the generated geometry to
the old mesh (volume ratio, largest bounds difference in mm) as scores. Steel sleeves and bearings are not generated
here (sleeves come from their connector's script; bearings and bushes are bought parts).

  .venv/bin/python -B robot/body/bom/hardware.py [--manifest PATH] [--out DIR] [--only GLOB] [--no-geometry]

Output (recreated by this script): DIR/hardware.json (spec, placement and fit per part) and DIR/parts/<name>.step / .stl.
Default DIR .cache/work/bom-hardware. `specs(manifest)` is what robot/body/bom/build.py reads (no geometry).
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "robot/body")); sys.path.insert(0, str(ROOT / "robot/body/fastening"))
from mount_check import kind as hw_kind, nominal  # noqa: E402

DATA = ROOT / "robot/body/checks/data"
STD = json.loads((DATA / "hardware.json").read_text())["screw_standards"]
THREADS = {"M2": 2.0, "M2.5": 2.5, "M3": 3.0, "M4": 4.0}
# ISO 4032 hex nut: width across flats s, height m (mm)
HEX_NUT = {"M2": (4.0, 1.6), "M2.5": (5.0, 2.0), "M3": (5.5, 2.4), "M4": (7.0, 3.2)}
# ISO 7089 plain washer: inner d1, outer d2, thickness h (mm)
WASHER = {"M2": (2.2, 5.0, 0.3), "M2.5": (2.7, 6.0, 0.5), "M3": (3.2, 7.0, 0.5), "M4": (4.3, 9.0, 0.8)}
# ISO 10642 countersunk head (90 deg): head diameter dk, height k (mm)
CSK = {"M2": (4.4, 1.2), "M2.5": (5.5, 1.5), "M3": (6.3, 1.7), "M4": (8.96, 2.48)}
GENERATED = {"screw", "nut", "washer", "spacer"}


def thread_of(name):
    n = nominal(name).get("thread")
    if not n:
        return None
    n = n.upper().replace("B", "")
    return "M" + n[2:] if n.startswith("PA") else n


def family_of(name, plan=None):
    """Screw family: the screw plan's thread_label when the part carries one (PA2 -> PA), else the name's thread."""
    label = (plan or {}).get("thread_label")
    n = (label or nominal(name).get("thread") or "").upper()
    return "PA" if "PA" in n else "M"


def kind_of(p):
    k = hw_kind(p)
    if k in GENERATED:
        return k
    n = p["name"].lower()
    if re.search(r"(^|_)nut(_|\d|$)|hex_nut|m\dnut", n):
        return "nut"
    if "spacer" in n or "standoff" in n or "_ff" in n:
        return "spacer"
    if "washer" in n:
        return "washer"
    if k == "other_hardware" and (thread_of(p["name"]) or re.search(r"(^|_)screw(_|\d|$)", n)):
        return "screw"     # fan_M3_11_430 (thread and position); shoulder_u_roll_horn_screw_0 (thread and length measured)
    return k


def load_mesh(path):
    from checks import bodylib as BL
    return BL.load_mesh(path)


def measure(mesh, k):
    """Axis (unit), extents along it (t0 .. t1), radial profile and centre of the mesh. For a screw the axis is the
    longest principal direction and points from the head end to the tip; for nuts / washers / spacers it is the
    principal direction with the most rotational symmetry (smallest spread of radius around it)."""
    v = np.asarray(mesh.vertices, float)
    c = v.mean(axis=0)
    _, _, vt = np.linalg.svd(v - c, full_matrices=False)
    cands = [vt[0], vt[1], vt[2]]

    def radial(ax):
        t = (v - c) @ ax
        r = np.linalg.norm((v - c) - np.outer(t, ax), axis=1)
        return t, r

    if k == "screw":
        ax = cands[0]
    else:
        # the hole axis: a ray along it through the centre of the bounding box passes without hitting the part
        # (through hole); fallback: the axis the outer rim's radius varies least around
        bc = np.asarray(mesh.bounds, float).mean(axis=0)
        span = float(np.ptp(v, axis=0).max()) * 2

        def blocked(ax):
            try:
                return bool(mesh.ray.intersects_any([bc - ax * span], [ax])[0])
            except Exception:  # noqa: BLE001
                return True

        def spread(ax):
            t, r = radial(ax)
            rim = r > np.percentile(r, 70)
            return np.std(r[rim]) / (np.mean(r[rim]) + 1e-9)
        free = [a for a in cands if not blocked(a)]
        ax = min(free or cands, key=spread)
    t, r = radial(ax)
    # centre the axis on the bounding circle in the plane (vertex mean is biased by tessellation)
    u = np.cross(ax, [1.0, 0, 0] if abs(ax[0]) < 0.9 else [0, 1.0, 0]); u /= np.linalg.norm(u); w = np.cross(ax, u)
    q = np.c_[(v - c) @ u, (v - c) @ w]
    mid = (q.max(axis=0) + q.min(axis=0)) / 2
    c = c + mid[0] * u + mid[1] * w
    t, r = radial(ax)
    t0, t1 = float(t.min()), float(t.max())
    L = t1 - t0
    if k == "screw":
        near0 = r[t < t0 + 0.2 * L].max(initial=0); near1 = r[t > t1 - 0.2 * L].max(initial=0)
        if near1 > near0:     # head at t1: flip so the axis runs head -> tip
            ax = -ax; t = -t; t0, t1 = -t1, -t0
        head_r = r[t < t0 + 0.2 * L].max(initial=0)
        shank = r[(t > t0 + 0.45 * L) & (t < t1 - 0.1 * L)]
        shank_r = float(np.percentile(shank, 90)) if shank.size else head_r / 2
        # head height: where the radius drops below midway between head and shank
        order = np.argsort(t)
        thr = (head_r + shank_r) / 2
        big = t[order][r[order] > thr]
        k_h = float(big.max() - t0) if big.size else 0.2 * L
        return dict(axis=ax, head_point=c + ax * t0, centre=c, length_total=L, head_dia=2 * head_r,
                    head_height=k_h, shank_dia=2 * shank_r)
    inner = r[r < np.percentile(r, 50)]
    return dict(axis=ax, centre=c + ax * (t0 + t1) / 2, thickness=L, outer_dia=2 * float(r.max()),
                inner_dia=2 * float(np.percentile(inner, 10)) if inner.size else 0.0)


def nearest(value, options):
    return min(options, key=lambda o: abs(o - value))


def spec_of(p, mesh=None):
    """Buyable spec of one hardware part: from its name, refined by the mesh measurement when given."""
    k = kind_of(p)
    name = p["name"]
    thr = thread_of(name)
    nom = nominal(name)
    m = measure(mesh, k) if mesh is not None and k in GENERATED else None
    s = dict(kind=k, thread=thr, source_of_spec=[])
    if k == "screw":
        fam = family_of(name, p.get("screw_plan"))
        if not thr and m:
            thr = min(THREADS, key=lambda t: abs(THREADS[t] - m["shank_dia"])); s["source_of_spec"].append("thread from shank diameter")
        L = nom.get("nominal_length_mm")
        if (p.get("screw_plan") or {}).get("length_mm"):
            L = float(p["screw_plan"]["length_mm"]); s["source_of_spec"].append("length from screw_plan")   # names are stale (T-tt3v4q)
        csk = bool(re.search(r"(_CS\d|csk|countersunk)", name, re.I))
        plan = p.get("screw_plan") or {}
        head = plan.get("head_standard")
        if head:
            s["source_of_spec"].append("head from screw_plan")
        elif csk:
            head = "ISO 10642"
        elif m and thr in THREADS:
            heads = {h: STD["head_types"][h][thr] for h in STD["head_types"] if thr in STD["head_types"][h]
                     and not STD["head_types"][h].get("countersunk")}     # a countersunk head is named, never guessed
            head = min(heads, key=lambda h: abs(heads[h]["dk"] - m["head_dia"]) + abs(heads[h]["k"] - m["head_height"]))
            s["source_of_spec"].append("head standard nearest to the measured head")
        else:
            head = "ISO 7380-1"
        if head == "ISO 10642" and thr not in STD["head_types"].get(head, {}):
            dk, kk = CSK.get(thr, (2 * THREADS.get(thr, 3), 0.6 * THREADS.get(thr, 3)))
        else:
            hh = STD["head_types"].get(head, {}).get(thr) or {}
            dk, kk = hh.get("dk", 2 * THREADS.get(thr, 3)), hh.get("k", THREADS.get(thr, 3) * 0.55)
        if L is None:
            m_cs = re.search(r"_CS(\d+)", name)
            if m_cs:
                L = float(m_cs.group(1)); s["source_of_spec"].append("length from _CS<n> in the name")
            elif m:
                Lm = m["length_total"] - (0 if head == "ISO 10642" else kk)
                lengths = STD["head_types"].get(head, {}).get(thr, {}).get("lengths") or [4, 5, 6, 8, 10, 12, 16, 20, 25, 30]
                L = float(nearest(Lm, lengths)); s["source_of_spec"].append("length = nearest standard to the measured length")
        s.update(thread=thr, family=fam, nominal_length_mm=L, head_standard=head, head_dia_mm=dk, head_height_mm=kk,
                 label=f"{'PA' + thr[1:] if fam == 'PA' else thr} x {L:g} {head}" if thr and L else f"screw {name}")
    elif k == "nut":
        if not thr and m:
            thr = min(HEX_NUT, key=lambda t: abs(THREADS[t] - m["inner_dia"])); s["source_of_spec"].append("thread from the measured hole")
        thr = thr if thr in HEX_NUT else "M3"
        s.update(thread=thr, standard="ISO 4032", across_flats_mm=HEX_NUT[thr][0], height_mm=HEX_NUT[thr][1], label=f"{thr} hex nut ISO 4032")
    elif k == "washer":
        if not thr and m:
            thr = min(WASHER, key=lambda t: abs(WASHER[t][0] - m["inner_dia"])); s["source_of_spec"].append("thread from the measured hole")
        thr = thr if thr in WASHER else "M3"
        d1, d2, h = WASHER[thr]
        if m and (abs(m["outer_dia"] - d2) > 0.6 or abs(m["thickness"] - h) > 0.3):
            d2, h = round(m["outer_dia"], 1), round(m["thickness"], 2)
            s.update(standard="custom (measured)", note="outer diameter or thickness differs from ISO 7089; kept as measured")
        else:
            s.update(standard="ISO 7089")
        s.update(thread=thr, inner_dia_mm=d1, outer_dia_mm=d2, thickness_mm=h, label=f"{thr} washer {d1:g}/{d2:g} x {h:g}")
    elif k == "spacer":
        if m:
            od, idd, L = round(m["outer_dia"], 1), round(m["inner_dia"], 1), round(m["thickness"], 1)
        else:
            od = idd = L = None
        mm = re.search(r"(\d+(?:p\d)?)\s*mm", name) or re.search(r"FF(\d+)", name)
        if mm and m:
            L = float(mm.group(1).replace("p", "."))
        if not thr and idd:
            thr = min(THREADS, key=lambda t: abs(THREADS[t] + 0.2 - idd))
        hexs = bool(re.search(r"FF|standoff", name, re.I))
        s.update(thread=thr, outer_dia_mm=od, inner_dia_mm=idd, length_mm=L, standard="hex standoff F-F" if hexs else "round spacer",
                 label=f"{thr} {'F-F standoff' if hexs else 'spacer'} {od}/{idd} x {L}" if od else f"spacer {name}")
    if m:
        s["measured"] = {k2: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else round(float(v), 4)) for k2, v in m.items()}
    return s


def geometry(spec):
    """build123d solid of the spec at its measured placement (manifest frame, mm), or None."""
    from build123d import Cone, Cylinder, Location, Plane, RegularPolygon, Solid, extrude, Pos
    m = spec.get("measured")
    if not m:
        return None
    ax = np.asarray(m["axis"], float); ax /= np.linalg.norm(ax)
    k = spec["kind"]
    d = THREADS.get(spec.get("thread"), 3.0)
    if k == "screw":
        L, dk, kk = spec["nominal_length_mm"], spec["head_dia_mm"], spec["head_height_mm"]
        # local +z = head -> tip, origin = head top
        if spec["head_standard"] == "ISO 10642":
            head = Cone(dk / 2, d / 2, kk, align=None)
            shank = Pos(0, 0, kk) * Cylinder(d / 2, L - kk, align=None)
        else:
            head = Cylinder(dk / 2, kk, align=None)
            shank = Pos(0, 0, kk) * Cylinder(d / 2, L, align=None)
        part = head + shank
        origin = np.asarray(m["head_point"], float)
    elif k == "nut":
        s_af, h = spec["across_flats_mm"], spec["height_mm"]
        prism = extrude(RegularPolygon(s_af / 2, 6, major_radius=False), h)
        part = Pos(0, 0, -h / 2) * (prism - Cylinder(d / 2, h, align=None))
        origin = np.asarray(m["centre"], float)
    elif k in ("washer", "spacer"):
        od = spec.get("outer_dia_mm"); idd = spec.get("inner_dia_mm") or d; h = spec.get("thickness_mm") or spec.get("length_mm")
        if not od or not h:
            return None
        if k == "spacer" and "standoff" in spec.get("standard", ""):
            outer = extrude(RegularPolygon(od / 2, 6, major_radius=True), h)
        else:
            outer = Cylinder(od / 2, h, align=None)
        part = Pos(0, 0, -h / 2) * (outer - Cylinder(idd / 2, h, align=None))
        origin = np.asarray(m["centre"], float)
    else:
        return None
    z = ax
    x = np.cross(z, [1.0, 0, 0] if abs(z[0]) < 0.9 else [0, 1.0, 0]); x /= np.linalg.norm(x)
    pl = Plane(origin=tuple(origin), x_dir=tuple(x), z_dir=tuple(z))
    return pl.location * part


def fit(solid, mesh, stl_path):
    import trimesh
    g = trimesh.load(stl_path, force="mesh")
    vb = float(abs(mesh.volume)) if mesh.is_volume else float(mesh.convex_hull.volume)
    db = np.abs(np.asarray(g.bounds) - np.asarray(mesh.bounds)).max()
    return dict(volume_generated_mm3=round(float(g.volume), 3), volume_old_mm3=round(vb, 3),
                volume_ratio=round(float(g.volume) / vb, 3) if vb > 1e-6 else None, bounds_max_difference_mm=round(float(db), 3))


def specs(manifest, with_mesh=True, only=None):
    """{part name: spec} for every generated hardware kind in the manifest (screw / nut / washer / spacer)."""
    out = {}
    for p in manifest["parts"]:
        if only and not any(fnmatch.fnmatchcase(p["name"], g) for g in only):
            continue
        k = kind_of(p)
        if k not in GENERATED:
            continue
        mesh = None
        if with_mesh and p.get("mesh"):
            try:
                mesh = load_mesh(p["mesh"])
            except Exception as e:  # noqa: BLE001  (a missing old mesh: spec from the name only)
                mesh = None
                out.setdefault("_errors", {})[p["name"]] = f"mesh not loadable: {type(e).__name__}: {e}"
        out[p["name"]] = spec_of(p, mesh)
        if mesh is not None:
            out[p["name"]]["_mesh"] = mesh
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=json.loads((ROOT / "robot/design.json").read_text())["body_manifest"])
    ap.add_argument("--out", default=str(ROOT / ".cache/work/bom-hardware"))
    ap.add_argument("--only", help="comma-separated part names / globs")
    ap.add_argument("--no-geometry", action="store_true", help="specs only, no STEP / STL")
    a = ap.parse_args()
    from build123d import export_step, export_stl
    man = json.loads((ROOT / a.manifest).read_text())
    out = Path(a.out).resolve(); (out / "parts").mkdir(parents=True, exist_ok=True)
    sp = specs(man, only=a.only.split(",") if a.only else None)
    errors = sp.pop("_errors", {})
    rows = {}
    for name, s in sp.items():
        mesh = s.pop("_mesh", None)
        row = dict(s)
        if not a.no_geometry:
            try:
                solid = geometry(s)
                if solid is not None:
                    step, stl = out / "parts" / f"{name}.step", out / "parts" / f"{name}.stl"
                    export_step(solid, str(step)); export_stl(solid, str(stl), tolerance=0.01, angular_tolerance=0.2)
                    row.update(step=str(step.relative_to(ROOT)), stl=str(stl.relative_to(ROOT)))
                    if mesh is not None:
                        row["fit_to_old_mesh"] = fit(solid, mesh, stl)
            except Exception as e:  # noqa: BLE001
                row["geometry_error"] = f"{type(e).__name__}: {e}"
        rows[name] = row
    labels = {}
    for r in rows.values():
        labels[r.get("label")] = labels.get(r.get("label"), 0) + 1
    ratios = [r["fit_to_old_mesh"]["volume_ratio"] for r in rows.values() if r.get("fit_to_old_mesh", {}).get("volume_ratio")]
    bdiff = [r["fit_to_old_mesh"]["bounds_max_difference_mm"] for r in rows.values() if r.get("fit_to_old_mesh")]
    import hashlib
    summary = dict(manifest=a.manifest, manifest_sha256=hashlib.sha256((ROOT / a.manifest).read_bytes()).hexdigest(), parts=len(rows), by_kind={k: sum(1 for r in rows.values() if r["kind"] == k) for k in GENERATED},
                   generated=sum(1 for r in rows.values() if r.get("step")), geometry_errors=sum(1 for r in rows.values() if r.get("geometry_error")),
                   mesh_errors=len(errors), kinds_to_buy=len(labels),
                   volume_ratio_p10_p50_p90=[round(float(np.percentile(ratios, q)), 3) for q in (10, 50, 90)] if ratios else None,
                   bounds_difference_mm_p50_p90_max=[round(float(np.percentile(bdiff, q)), 3) for q in (50, 90)] + [max(bdiff)] if bdiff else None,
                   buy_list=dict(sorted(labels.items(), key=lambda kv: -kv[1])))
    (out / "hardware.json").write_text(json.dumps(dict(summary=summary, parts=rows, mesh_errors=errors), indent=1, default=str) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
