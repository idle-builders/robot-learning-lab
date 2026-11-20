"""Print setup of a part: the per-part `process` setting resolved against the material and printer tables, and the
part's solid turned into its print frame. Every printability check reads the process through `effective`.

Resolution, per key (the source of each value is reported):
  1. run override      `run.py --process KEY=VALUE` (cross-evaluation: another nozzle, material, layer height for every
                       printed part in scope; never stored)
  2. part setting      part_settings.json <part>.process
  3. material default  materials.json <material>.nozzle_mm_default (manufacturer recommendation)
  4. printer default   printers.json <printer>.by_nozzle[<nozzle>] (layer height, line width)

Print direction: `up` is the manifest-frame direction that points away from the bed (the layer normal, build +Z), as
"+x" / "-y" ... or a vector; `plate_yaw_deg` turns the part about it on the plate (default 0). The manifest frame is the
frame of the solids every check uses (ctx.solid), the same frame as the FEM setups.
"""
from __future__ import annotations

import json

import numpy as np

from checks.core import DATA

AX = {"x": 0, "y": 1, "z": 2}
KEYS_NUM = ("nozzle_mm", "layer_height_mm", "line_width_mm", "plate_yaw_deg")
KEYS_ALL = KEYS_NUM + ("printer", "material", "up")


def tables():
    return json.loads((DATA / "materials.json").read_text()), json.loads((DATA / "printers.json").read_text())


def parse_override(items):
    """['nozzle_mm=0.4', 'material=Bambu PETG-CF', 'up=+z'] -> {key: value}; numbers for the numeric keys."""
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        k = k.strip()
        if not k or not _:
            raise SystemExit(f"--process wants KEY=VALUE, got {it!r}")
        if k not in KEYS_ALL:
            raise SystemExit(f"--process: unknown key {k!r}; known {list(KEYS_ALL)}")
        out[k] = float(v) if k in KEYS_NUM else v.strip()
        if k == "up":
            up_vector(out[k])                       # raises SystemExit on a bad direction
    return out


def is_printed(ctx, n, settings=None):
    p = ctx.parts[n]
    s = (settings if settings is not None else ctx.settings(n)).get("process") or {}
    if "printed" in s:
        return bool(s["printed"])
    return p.get("role") in ctx._settings.get("_printed", {}).get("roles", [])


def up_vector(up):
    """'+x' / '-z' / 'y' / [x, y, z] -> unit vector, or None."""
    if up is None:
        return None
    if isinstance(up, str):
        u = up.strip().lower()
        if u.lstrip("+-") not in AX:
            raise SystemExit(f"print direction {up!r}: use +x, -x, +y, -y, +z, -z or a vector")
        sgn = -1.0 if u.startswith("-") else 1.0
        v = np.zeros(3); v[AX[u.lstrip("+-")]] = sgn
        return v
    v = np.asarray(up, float)
    return v / np.linalg.norm(v)


def rotation(up, yaw_deg=0.0):
    """3x3 R with R @ up = +Z (shortest rotation), then yaw_deg about +Z. Rows map manifest-frame vectors to the print frame."""
    u = up_vector(up)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(u, z))
    if c > 1 - 1e-12:
        Rm = np.eye(3)
    elif c < -1 + 1e-12:
        Rm = np.diag([1.0, -1.0, -1.0])           # turn 180 deg about x
    else:
        k = np.cross(u, z); s = np.linalg.norm(k); k = k / s
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        Rm = np.eye(3) + s * K + (1 - c) * K @ K
    a = np.radians(yaw_deg or 0.0)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return Rz @ Rm


def effective(ctx, n):
    """Resolved print setup of part n: values plus `source` per key. Missing values are None with source 'missing'."""
    mats, prns = tables()
    s = dict(ctx.settings(n).get("process") or {})
    ov = getattr(ctx, "process_override", {}) or {}
    out, src = {}, {}

    def put(k, v, where):
        out[k] = v; src[k] = where if v is not None else "missing"
    for k in ("printer", "material", "up", "plate_yaw_deg", "nozzle_mm", "layer_height_mm", "line_width_mm"):
        if k in ov:
            put(k, ov[k], "run override")
        elif s.get(k) is not None:
            put(k, s[k], "seeded" if s.get("seeded") and k in s.get("seeded_keys", ()) else "part setting")
        else:
            put(k, None, "missing")
    mat = mats.get(out["material"]) if out["material"] else None
    if out["nozzle_mm"] is None and mat and mat.get("nozzle_mm_default"):
        put("nozzle_mm", float(mat["nozzle_mm_default"]), "material default")
    prn = prns.get(out["printer"]) if out["printer"] else None
    by_noz = (prn or {}).get("by_nozzle", {}).get(f"{out['nozzle_mm']:g}" if out["nozzle_mm"] else "", {})
    for k in ("layer_height_mm", "line_width_mm"):
        if out[k] is None and by_noz.get(k) is not None:
            put(k, float(by_noz[k]), "printer default")
    if out["plate_yaw_deg"] is None:
        put("plate_yaw_deg", 0.0, "default")
    out.update(printed=is_printed(ctx, n), material_known=mat is not None, printer_known=prn is not None,
               material_alternatives=s.get("material_alternatives", []), seeded=s.get("seeded"), source=src,
               nozzle_offered=out["nozzle_mm"] is None or not prn or out["nozzle_mm"] in prn.get("nozzles_mm", [out["nozzle_mm"]]))
    return out


def print_mesh(ctx, n, eff=None, with_transform=False):
    """The part's solid (ctx.solid) in its print frame: up -> +Z, plate_yaw applied, lowest point at z = 0. None without `up`.
    with_transform: (mesh, T) where T (4x4) maps the part's manifest frame into the print frame (inverse maps locations back)."""
    eff = eff or effective(ctx, n)
    if eff["up"] is None:
        return (None, None) if with_transform else None
    m = ctx.solid(n)[0].copy()
    T = np.eye(4); T[:3, :3] = rotation(eff["up"], eff["plate_yaw_deg"])
    m.apply_transform(T)
    dz = -float(m.bounds[0][2])
    m.apply_translation([0.0, 0.0, dz])
    T[2, 3] = dz
    return (m, T) if with_transform else m
