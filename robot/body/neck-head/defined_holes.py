"""Neck / head hole positions from the standard-part definitions (owner 2026-11-18, M-jpppfb): the builders read the screw
sites of the XM335 horns and cases from robot/actuators/interfaces/standard-parts.json, registered to the body's instance of
each part (robot/body/fastening/interfaces.py), instead of typed numbers.

  registered(manifest)      registered standard-part instances and their defined holes (body frame, mm; default manifest: shell-pass)
  seat_sites(connector)     per screw of a connector in robot/body/neck-head/connector_screws.json: the hole axis and the point
                            of its seat plane (the standard part's outer face around the hole, measured on its mesh), the
                            thread, the hole depth and the reference engagement
Everything is a measurement in mm; nothing here accepts or rejects."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").exists())
HERE = Path(__file__).resolve().parent
for _p in (ROOT / "robot/body/fastening", ROOT / "robot/body"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RING_MM = 2.6           # radius of the rays that find the standard part's outer face around a hole (outside the tapped bore)
N_RAYS = 8
_CACHE: dict = {}


def connectors() -> dict:
    return json.loads((HERE / "connector_screws.json").read_text())["connectors"]


def manifest_path() -> Path:
    return ROOT / json.loads((ROOT / "robot/design.json").read_text())["body_manifest"]


# The default manifest of the neck / head hole readers is the declared upstream shell-pass manifest, not the selected body
# (robot/design.json): that is this chain's final output, so reading it made roll-flip-manifest (NP.apply -> head_screws) and the neck
# connector builds circular in a clean rebuild. The neck servos, the head carrier's supplier spacers / nuts / boards and the skin sit
# there as supplied: seat_sites of the three neck connectors and head_holes posts / threaded_posts / screw_plans / plate_faces are
# identical from both manifests (max difference 0.0, 2026-11-22). The neck_parts CLI still edits the selected body (manifest_path).
SHELL_PASS = ROOT / "artifacts/robot/body/ape-proportions/shell-pass/assembly-parts.json"


def registered(manifest: Path | None = None):
    """(body, manifest dict, iface) of the body manifest (default: the shell-pass manifest, SHELL_PASS)."""
    path = Path(manifest) if manifest else SHELL_PASS
    if path in _CACHE:
        return _CACHE[path]
    from checks import bodylib
    import interfaces as I
    man = json.loads(path.read_text())
    body = bodylib.Body(man)
    out = (body, man, I.instances(body, man))
    _CACHE[path] = out
    return out


def seat_offset(mesh, opening: np.ndarray, axis_out: np.ndarray) -> float:
    """Distance (mm) from a hole opening to the part's outer face around it along the outward hole axis (>= 0: the face lies
    outside the opening); median of rays on a ring that misses the bore."""
    a = np.asarray(axis_out, float)
    u = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0]); u /= np.linalg.norm(u)
    v = np.cross(a, u)
    o = np.asarray(opening, float)
    hits = []
    for k in range(N_RAYS):
        t = 2 * np.pi * k / N_RAYS
        start = o + RING_MM * (np.cos(t) * u + np.sin(t) * v) + 6.0 * a
        loc, _, _ = mesh.ray.intersects_location([start], [-a], multiple_hits=False)
        if len(loc):
            hits.append(float(np.dot(loc[0] - o, a)))
    return float(np.median(hits)) if hits else 0.0


def seat_sites(connector: str, manifest: Path | None = None) -> list[dict]:
    """The screw sites of one connector, in the order of connector_screws.json groups and holes."""
    body, man, iface = registered(manifest)
    import screw_plan as SP
    spec = connectors()[connector]
    sites = []
    for gname, g in spec["groups"].items():
        mesh = body.mesh(g["standard_part"])
        for i, h in enumerate(SP.hole_positions(iface, g["standard_part"], g["holes"])):
            op, ax = np.asarray(h["opening"], float), np.asarray(h["axis_out"], float)
            off = seat_offset(mesh, op, ax)
            sites.append(dict(group=gname, index=i, hole_id=h["hole_id"], centre=op + off * ax, axis_out=ax, opening=op,
                              seat_offset_mm=round(off, 3), thread=h["thread"], depth_mm=h["depth_mm"],
                              reference_engagement_mm=h["reference_engagement_mm"], terminal_material=h["terminal_material"]))
    return sites
