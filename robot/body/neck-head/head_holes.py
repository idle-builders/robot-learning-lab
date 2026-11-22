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


def _man_parts(manifest=None):
    return _body(manifest)[1]["parts"]


def _body(manifest=None):
    body, man, _ = DH.registered(manifest)
    return body, man


MEASURED = HERE / "head_carrier_measured.json"


def _measure(body, part: str) -> dict:
    """Bore of a supplier part on its mesh; a part the manifest no longer has (the camera standoffs and mic shims are replaced by
    printed bosses, T-5bqc3m) is read from head_carrier_measured.json, which stores what its bore measured while it was there."""
    cache = json.loads(MEASURED.read_text()) if MEASURED.exists() else {}
    if part in body.parts:
        b = bore(body.mesh(part))
        cache[part] = dict(axis=b["axis"], centre=[round(float(x), 4) for x in b["centre"]], radius_mm=round(b["radius_mm"], 4),
                           extent=[round(x, 4) for x in b["extent"]])
        MEASURED.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n")
        return b
    c = cache[part]
    return dict(axis=c["axis"], centre=np.array(c["centre"], float), radius_mm=c["radius_mm"], extent=tuple(c["extent"]))


def posts(manifest: Path | None = None) -> dict:
    key = ("posts", str(manifest))
    if key in _CACHE:
        return _CACHE[key]
    body, _ = _body(manifest)
    out = {}
    for g, spec in POSTS.items():
        rows = []
        for i, sp in enumerate(spec["spacers"]):
            b = _measure(body, sp)
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
    """group -> list of (centre on the plate front face / the mic board's back face, normal +axis), body frame."""
    P = posts(manifest)
    front, _ = plate_faces(manifest)
    out = {}
    for g, rows in P.items():
        pts = []
        for r in rows:
            c = np.array(r["spacer_bore"]["centre"], float)
            c[r["spacer_bore"]["axis"]] = r["spacer_bore"]["extent"][0] if g == "mic" else front
            pts.append(c)
        out[g] = pts
    return out


def thread(name: str) -> dict:
    return json.loads((HERE / "head_carrier_posts.json").read_text())["threads"][name]


def sleeve(group: str) -> dict:
    """Outer radius, bore radius (the thread's minor diameter for the threaded groups), length and thread of a post's sleeve / insert."""
    s = dict(POSTS[group]["sleeve"])
    if "thread" in s:
        s["ri"] = thread(s["thread"])["minor_dia"] / 2.0
    return s


def _plan_dict(name, group, spec, std, row, opening, axis_out, stack, S, pick, mat, hole):
    st_pt = _standards()["point_mm"]
    h = spec["head"]
    thr = h["thread"]
    # the length goes into the name (mount_check reads a screw by its M<d>x<length> name): B0410_camera_M2_0 -> B0410_camera_M2x8_0
    name = name.replace("_M2_", f"_M2x{pick['length_mm']:g}_")
    return dict(name=name, group=group, thread=thr, thread_label=thr, head_standard=h["standard"], head_dia_mm=h.get("dk", std["dk"]),
                head_height_mm=h.get("k", std["k"]), countersunk=bool(h.get("countersunk")), terminal_material=mat,
                opening=opening, axis_out=axis_out, stack_mm=stack, hole_depth_mm=S, reference_engagement_mm=pick["ref_mm"],
                point_mm=st_pt, hole_id=f"{hole}:thread", owning_link="head", spacer=row["spacer"], head_gap_mm=0.0,
                **{k: v for k, v in pick.items() if k != "ref_mm"})


def _standards():
    import screw_plan as SP
    return SP.standards()


def threaded_posts(manifest: Path | None = None) -> dict:
    """The camera and mic posts: per post the insert (start, length), the screw plan (screw_plan.choose_length: smallest standard
    length that reaches the 1 d engagement with its 0.5 mm point inside the insert) and the numbers of the choice. Cached; the insert
    geometry feeds common.stage_interface_specs, the plans feed screw_plans."""
    import screw_plan as SP
    key = ("threaded", str(manifest))
    if key in _CACHE:
        return _CACHE[key]
    body, _ = _body(manifest)
    P = posts(manifest)
    front, rear = plate_faces(manifest)
    st = _standards()
    out = {}
    # camera: head on the camera board front (board, printed column), insert in the column's rear end, flush with the plate's rear face
    spec = POSTS["camera"]
    sl = sleeve("camera")
    th = thread(sl["thread"])
    ref = sl["engagement_x_d"] * th["major"]
    std = st["head_types"][spec["head"]["standard"]][spec["head"]["thread"]]
    rows = []
    for i, r in enumerate(P["camera"]):
        b = r["spacer_bore"]
        wb = _measure(body, spec["washers"][i])       # the 0.3 mm washer is gone (owner 2026-11-22): its board-side face is the board's front face
        seat = wb["extent"][1]                          # the board's mount pad: the ISO 4762 head bears on it directly
        start = rear - sl.get("recess_mm", 0.0) - sl["length"]     # the insert's front end (the screw enters here)
        stack = float(start - seat)
        ax = np.zeros(3); ax[b["axis"]] = 1.0
        c = np.array(b["centre"], float); c[b["axis"]] = start
        pick = SP.choose_length(stack, sl["length"], ref, std["lengths"], st["point_mm"])
        pick.update(ref_mm=ref, head_seat=round(float(seat), 3), insert_start=round(float(start), 3))
        rows.append(dict(plan=_plan_dict(spec["screws"][i], "camera", spec, std, r, c, -ax, stack, sl["length"], pick, "brass", f"camera_insert_{i}"),
                         sleeve_len=sl["length"], start=float(start)))
    out["camera"] = rows
    # mic: head on the board side, through the board into the insert in the skin boss; the board rests on the boss end (no shim)
    spec = POSTS["mic"]
    sl = sleeve("mic")
    th = thread(sl["thread"])
    ref = sl["engagement_x_d"] * th["major"]
    std = st["head_types"][spec["head"]["standard"]][spec["head"]["thread"]]
    rows = []
    shift = float(spec["boss"].get("board_shift_mm", 0.0))     # the board sits deeper on its bosses (neck_parts.skin_mic_bosses moves its mesh)
    applied = float(next(p for p in _man_parts(manifest) if p["name"] == spec["clamped"][0]).get("mic_board_shift_mm", 0.0))
    for i, r in enumerate(P["mic"]):
        b = r["spacer_bore"]
        ax = np.zeros(3); ax[b["axis"]] = 1.0
        c = np.array(b["centre"], float); c[b["axis"]] = b["extent"][0] + applied    # the shim's board-side face = the board's back face (where the mesh is now)
        stk = SP.stack_along_axis([body.mesh(n) for n in spec["clamped"]], c, -ax, radii=(1.5, 2.0))
        stack = float(stk["stack_mm"])
        c[b["axis"]] = b["extent"][0] + shift
        fixed = spec["head"].get("fixed_lengths")
        pick = SP.choose_length(stack, sl["length"], ref, [float(fixed[i])] if fixed else spec["head"]["lengths"], st["point_mm"])
        pick.update(ref_mm=ref, head_seat=round(float(c[b["axis"]] - stack), 3), insert_start=round(float(c[b["axis"]]), 3))
        rows.append(dict(plan=_plan_dict(spec["screws"][i], "mic", spec, std, r, c, -ax, stack, sl["length"], pick, "brass", f"mic_insert_{i}"),
                         sleeve_len=sl["length"], start=float(c[b["axis"]])))
    out["mic"] = rows
    _CACHE[key] = out
    return out


def sleeve_lens(group: str, manifest: Path | None = None) -> list[float]:
    return [float(r["sleeve_len"]) for r in threaded_posts(manifest)[group]]


def sleeve_starts(group: str, manifest: Path | None = None) -> list[float]:
    """Coordinate along the post axis of each insert's front end (where the screw enters)."""
    return [float(r["start"]) for r in threaded_posts(manifest)[group]]


def screw_plans(manifest: Path | None = None) -> list[dict]:
    """The generated screw of every post: head on the sleeve end, standard length that reaches the terminal."""
    import screw_plan as SP
    P = posts(manifest)
    front, rear = plate_faces(manifest)
    st = SP.standards()
    plans = []
    for g in ("camera", "mic"):
        plans += [r["plan"] for r in threaded_posts(manifest)[g]]
    spec = POSTS["compute"]
    h = spec["head"]
    std = st["head_types"].get(h["standard"], {}).get(h["thread"], {})
    dk, k = h.get("dk", std.get("dk")), h.get("k", std.get("k"))
    for r in P["compute"]:
        b = r["spacer_bore"]
        ax = np.zeros(3); ax[b["axis"]] = 1.0
        # head on the front face, thread in the nut: axis_out points to the head (-axis)
        t = r["terminal_bore"]
        opening = np.array(t["centre"], float); opening[t["axis"]] = t["extent"][0]
        stack = float(t["extent"][0] - front)
        depth = float(t["extent"][1] - t["extent"][0])
        pick = SP.choose_length(stack, depth, depth, [float(x) for x in h["lengths"]], st["point_mm"])
        plans.append(dict(
            name=r["screw"], group="compute", thread=h["thread"], thread_label=h["thread"], head_standard=h["standard"], head_dia_mm=dk,
            head_height_mm=k, countersunk=bool(h.get("countersunk")), terminal_material="steel",
            opening=opening, axis_out=-ax, stack_mm=stack, hole_depth_mm=depth, reference_engagement_mm=depth,
            point_mm=st["point_mm"], hole_id=f"{r['terminal']}:thread", owning_link="head", spacer=r["spacer"], **pick))
    return plans
