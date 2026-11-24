"""Drafting aid for the hardware specs of a body definition (robot/**/<builder>.bom.yaml): proposes kind, thread, nominal
length and head standard for screws, nuts, washers and spacers from the part name (fastening/mount_check.nominal), the
screw_plan a builder recorded, and a measurement of the mesh, snapped to the standard sizes in
robot/body/checks/data/hardware.json. The BOM never runs this: the owner of the definition decides the spec and writes it
there. Prints YAML `spec:` blocks to paste.

  .venv/bin/python -B robot/body/standard-hardware/propose_specs.py [--manifest PATH] --only GLOB[,GLOB]
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
sys.path.insert(0, str(ROOT / "robot/body/bom"))
import bom_config as BOM_CONFIG  # noqa: E402
CFG = BOM_CONFIG.section("standard-hardware")   # thread, nut, washer, countersink sizes: robot/body/standard-hardware/generate.bom.yaml config (shared with generate.py)

DATA = ROOT / "robot/body/checks/data"
STD = json.loads((DATA / "hardware.json").read_text())["screw_standards"]
THREADS = CFG["THREADS"]
# ISO 4032 hex nut: width across flats s, height m (mm)
HEX_NUT = CFG["HEX_NUT"]
# ISO 7089 plain washer: inner d1, outer d2, thickness h (mm)
WASHER = CFG["WASHER"]
# ISO 10642 countersunk head (90 deg): head diameter dk, height k (mm)
CSK = CFG["CSK"]
GENERATED = set(CFG["GENERATED"])


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
    import yaml
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=json.loads((ROOT / "robot/design.json").read_text())["body_manifest"])
    ap.add_argument("--only", required=True, help="comma-separated part names / globs")
    a = ap.parse_args()
    man = json.loads((ROOT / a.manifest).read_text())
    sp = specs(man, only=a.only.split(","))
    sp.pop("_errors", None)
    keep = {"kind": "kind", "thread": "thread", "family": "family", "head_standard": "standard", "standard": "standard", "nominal_length_mm": "length_mm",
            "length_mm": "length_mm", "outer_dia_mm": "outer_dia_mm", "inner_dia_mm": "inner_dia_mm", "thickness_mm": "thickness_mm",
            "head_dia_mm": "head_dia_mm", "head_height_mm": "head_height_mm", "across_flats_mm": "across_flats_mm", "label": "label"}
    out = {}
    for n, s in sp.items():
        s.pop("_mesh", None)
        out[n] = dict(spec={keep[k]: v for k, v in s.items() if k in keep and v is not None}, how=s.get("source_of_spec"))
    print(yaml.safe_dump(out, sort_keys=False, width=140))


if __name__ == "__main__":
    main()
