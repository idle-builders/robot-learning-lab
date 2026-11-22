"""Head-carrier screw posts from the supplier parts' own holes (Idle T-5bqc3m).

The compute-board, camera and microphone posts of combined__head_sensor_compute_carrier are never typed: the bore of each
spacer (and of its terminal nut) is measured on its mesh (section through the mid plane, circle fit of the inner loop), the
carrier plate faces come from where the spacers rest, and the screws are generated from them.
Everything is a measurement in mm; nothing here accepts or rejects.

  posts(manifest)         per post: spacer / terminal bore centre and axis, their coaxiality (mm), plate faces
  plate_faces(manifest)   (front, rear) coordinate of the carrier plate along Y
  screw_plans(manifest)   per screw of the carrier: the plan dict screw_plan.screw_mesh draws
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import importlib.util as _u
_spec = _u.spec_from_file_location("neck_head_defined_holes", HERE / "defined_holes.py")
DH = sys.modules.get("neck_head_defined_holes") or _u.module_from_spec(_spec)
if "neck_head_defined_holes" not in sys.modules:
    sys.modules["neck_head_defined_holes"] = DH
    _spec.loader.exec_module(DH)

# @lore Posts are measured on the spacer / nut / shim meshes, not on the (coarse, non-watertight) ROCK4D / camera board meshes: the
# supplier boards' mount holes carry triangle wedges that move a probe's hole centre by 1.6-2.1 mm although the bore itself is
# concentric with the spacer to 0.03 mm (T-5bqc3m).
POSTS = json.loads((HERE / "head_carrier_posts.json").read_text())["groups"]
_CACHE: dict = {}


def bore(mesh) -> dict:
    """Axis (unit, +Y or +X/+Z positive), bore centre on the mid plane, radius and extent along the axis of a part with a
    round through bore: the coordinate axis whose mid-plane section has a circular inner loop."""
    warnings.filterwarnings("ignore")
    lo, hi = mesh.bounds
    mid = (lo + hi) / 2
    best = None
    for k in range(3):
        n = np.zeros(3); n[k] = 1.0
        sec = mesh.section(plane_origin=mid, plane_normal=n)
        if sec is None:
            continue
        p2, T = sec.to_2D() if hasattr(sec, "to_2D") else sec.to_planar()
        for poly in p2.polygons_full:
            for ring in poly.interiors:
                c = np.array(ring.coords)
                cen = c[:-1].mean(0)
                r = np.linalg.norm(c - cen, axis=1)
                if 0.5 < r.mean() < 2.0 and np.ptp(r) / r.mean() < 0.08:
                    w = (T @ np.array([cen[0], cen[1], 0, 1]))[:3]
                    if best is None or np.ptp(r) < best[3]:
                        best = (k, w, float(r.mean()), float(np.ptp(r)))
    if best is None:
        raise ValueError("no round bore found")
    k, w, r, _ = best
    return dict(axis=int(k), centre=w, radius_mm=r, extent=(float(lo[k]), float(hi[k])))


def _body(manifest=None):
    body, man, _ = DH.registered(manifest)
    return body, man


def posts(manifest: Path | None = None) -> dict:
    key = ("posts", str(manifest))
    if key in _CACHE:
        return _CACHE[key]
    body, _ = _body(manifest)
    out = {}
    for g, spec in POSTS.items():
        rows = []
        for i, sp in enumerate(spec["spacers"]):
            b = bore(body.mesh(sp))
            row = dict(index=i, spacer=sp, screw=spec["screws"][i], spacer_bore=b)
            if "terminals" in spec:
                t = bore(body.mesh(spec["terminals"][i]))
                lat = [a for a in range(3) if a != b["axis"]]
                row["terminal"] = spec["terminals"][i]
                row["terminal_bore"] = t
                row["coaxiality_mm"] = round(float(np.linalg.norm((b["centre"] - t["centre"])[lat])), 3)
            rows.append(row)
        out[g] = rows
    _CACHE[key] = out
    return out


def plate_faces(manifest: Path | None = None) -> tuple[float, float]:
    """(front, rear) plate coordinates along the bore axis: the camera spacers rest on the front face, the compute
    spacers on the rear face."""
    P = posts(manifest)
    cam = {r["spacer_bore"]["extent"][1] for r in P["camera"]}
    comp = {r["spacer_bore"]["extent"][0] for r in P["compute"]}
    return float(np.mean(list(cam))), float(np.mean(list(comp)))


def sites(manifest: Path | None = None) -> dict:
    """group -> list of (centre on the plate front face, normal +axis), body frame."""
    P = posts(manifest)
    front, _ = plate_faces(manifest)
    out = {}
    for g, rows in P.items():
        pts = []
        for r in rows:
            c = np.array(r["spacer_bore"]["centre"], float)
            c[r["spacer_bore"]["axis"]] = r["spacer_bore"]["extent"][1] if g == "mic" else front
            pts.append(c)
        out[g] = pts
    return out


def screw_plans(manifest: Path | None = None) -> list[dict]:
    """The generated screw of every post: head on the sleeve end, standard length that reaches the terminal."""
    import screw_plan as SP
    P = posts(manifest)
    front, rear = plate_faces(manifest)
    st = SP.standards()
    plans = []
    for g, rows in P.items():
        if g == "mic":      # screws stay as they are: no terminal yet (needs a scope decision, see Idle T-5bqc3m)
            continue
        spec = POSTS[g]
        h = spec["head"]
        std = st["head_types"].get(h["standard"], {}).get(h["thread"], {})
        dk, k = h.get("dk", std.get("dk")), h.get("k", std.get("k"))
        for r in rows:
            b = r["spacer_bore"]
            ax = np.zeros(3); ax[b["axis"]] = 1.0
            sp_len = b["extent"][1] - b["extent"][0]
            if g == "compute":
                # head on the front face, thread in the nut: axis_out points to the head (-axis)
                t = r["terminal_bore"]
                opening = np.array(t["centre"], float); opening[t["axis"]] = t["extent"][0]
                axis_out = -ax
                stack = float(t["extent"][0] - front)
                depth = float(t["extent"][1] - t["extent"][0])
                ref, mat, hole = depth, "steel", r["terminal"]
            else:
                opening = np.array(b["centre"], float); opening[b["axis"]] = front
                axis_out = ax
                stack = float(rear - front)
                depth = sp_len / 2.0
                ref, mat, hole = 1.5 * 2.0, "aluminium", r["spacer"]
            pick = SP.choose_length(stack, depth, ref, [float(x) for x in h["lengths"]], st["point_mm"])
            thr = h["thread"]
            plans.append(dict(
                name=r["screw"], group=g, thread=thr, thread_label=thr, head_standard=h["standard"], head_dia_mm=dk,
                head_height_mm=k, countersunk=bool(h.get("countersunk")), terminal_material=mat, opening=opening,
                axis_out=axis_out, stack_mm=stack, hole_depth_mm=depth, reference_engagement_mm=ref,
                point_mm=st["point_mm"], hole_id=f"{hole}:thread", owning_link="head", spacer=r["spacer"], **pick))
    return plans
