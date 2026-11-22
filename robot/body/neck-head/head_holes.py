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


# parts a screw tip must keep clear of: every head-link part that is not hardware, a carrier or one of the post's own parts
_HARDWARE_ROLES = {"fastener", "fastener_proxy", "standard_nominal", "spacer", "compression_limit_sleeves", "threaded_sleeve",
                   "nominal_hardware", "case_mount_spacer"}


def thread(name: str) -> dict:
    return json.loads((HERE / "head_carrier_posts.json").read_text())["threads"][name]


def margin_mm() -> float:
    return float(json.loads((HERE / "head_carrier_posts.json").read_text())["tip_margin_mm"])


def sleeve(group: str) -> dict:
    """Outer radius, bore radius (the thread's minor diameter for the threaded groups) and thread of a post's sleeve."""
    s = dict(POSTS[group]["sleeve"])
    if "thread" in s:
        s["ri"] = thread(s["thread"])["minor_dia"] / 2.0
    return s


def _neighbours(manifest, lo, hi, skip=()):
    """Meshes (cut to the faces inside the box lo..hi, mm) of the head parts a tip / head must clear."""
    import trimesh
    body, man = _body(manifest)
    out = []
    for p in man["parts"]:
        n = p["name"]
        if p.get("owning_link") != "head" or not p.get("mesh") or n.startswith("combined__") or n in skip \
                or p.get("role") in _HARDWARE_ROLES or "B0410" in n or "B0196" in n:
            continue
        m = body.mesh(n)
        if np.any(m.bounds[1] < lo) or np.any(m.bounds[0] > hi):
            continue
        c = m.triangles_center
        k = np.all((c >= lo) & (c <= hi), axis=1)
        if k.any():
            out.append((n, trimesh.Trimesh(m.vertices, m.faces[k], process=False)))
    return out


def _clear_run(neigh, centre, axis, a0, a1, r, step=0.1):
    """Distance along `axis` from a0 (up to a1) over which a disc of radius r around `centre` keeps margin_mm from `neigh`
    (list of (name, mesh)); returns (run, name of the first part met or None)."""
    import trimesh
    ang = np.linspace(0, 2 * np.pi, 9)[:-1]
    u = np.cross(axis, [1.0, 0, 0] if abs(axis[0]) < 0.9 else [0, 1.0, 0]); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    ring = [np.zeros(3)] + [r * (np.cos(t) * u + np.sin(t) * v) for t in ang]
    t = a0
    while t <= a1 + 1e-9:
        pts = np.array([centre + axis * t + q for q in ring])
        for n, m in neigh:
            if trimesh.proximity.closest_point(m, pts)[1].min() < margin_mm():
                return max(0.0, t - step - a0), n
        t += step
    return a1 - a0, None


def pick_length(stack: float, S: float, lengths, free: float) -> dict:
    """Shortest stocked length whose tip reaches the end of the threaded sleeve (length S, entered at `stack` from the head
    seat) with the tip no further than `free` beyond it; else the longest that stays clear (partial engagement)."""
    ok = [float(L) for L in lengths if L <= stack + S + free + 1e-9]
    full = [L for L in ok if L >= stack + S - 1e-9]
    L = full[0] if full else (ok[-1] if ok else float(lengths[0]))
    return dict(length_mm=L, engagement_mm=round(min(S, max(0.0, L - stack)), 3), sleeve_mm=round(S, 3),
                tip_beyond_sleeve_mm=round(L - stack - S, 3), fully_engaged=bool(full))


def _mic_sleeve_len(manifest, c, ro, wing):
    """Mic sleeve length from the wing's inboard face: the wing thickness, cut back where the head skin's inner surface
    (rays over the sleeve's outline) comes within margin_mm of the sleeve's end."""
    body, _ = _body(manifest)
    skin = body.mesh("original_profile_head_skin_v23")
    ang = np.linspace(0, 2 * np.pi, 13)[:-1]
    o = np.array([[c[0] - 1.0, c[1] + ro * np.cos(t), c[2] + ro * np.sin(t)] for t in ang] + [[c[0] - 1.0, c[1], c[2]]])
    loc, ri, _ = skin.ray.intersects_location(o, np.tile([1.0, 0, 0], (len(o), 1)))
    x = min(float(loc[ri == i][:, 0].min()) for i in range(len(o)) if np.any(ri == i))
    return float(min(wing, x - margin_mm() - c[0]))


def _plan_dict(name, group, spec, std, row, opening, axis_out, stack, S, pick, mat, hole):
    st_pt = _standards()["point_mm"]
    h = spec["head"]
    thr = h["thread"]
    # the length goes into the name (mount_check reads a screw by its M<d>x<length> name): B0410_camera_M2_0 -> B0410_camera_M2x10_0
    name = name.replace("_M2_", f"_M2x{pick['length_mm']:g}_")
    return dict(name=name, group=group, thread=thr, thread_label=thr, head_standard=h["standard"], head_dia_mm=h.get("dk", std["dk"]),
                head_height_mm=h.get("k", std["k"]), countersunk=bool(h.get("countersunk")), terminal_material=mat,
                opening=opening, axis_out=axis_out, stack_mm=stack, hole_depth_mm=S, reference_engagement_mm=S,
                point_mm=st_pt, tip_margin_mm=round(pick["tip_beyond_sleeve_mm"], 3), hole_id=f"{hole}:thread",
                owning_link="head", spacer=row["spacer"], head_gap_mm=0.0, shortfall_mm=round(max(0.0, S - pick["engagement_mm"]), 3),
                overrun_mm=0.0, stack_change_to_fit_mm=0.0, **pick)


def _standards():
    import screw_plan as SP
    return SP.standards()


def threaded_posts(manifest: Path | None = None) -> dict:
    """The camera and mic posts: per post the sleeve length, screw plan (shortest clear stocked length) and the numbers of the
    choice. Cached; the sleeve lengths feed common.stage_interface_specs, the plans feed screw_plans."""
    key = ("threaded", str(manifest))
    if key in _CACHE:
        return _CACHE[key]
    body, _ = _body(manifest)
    P = posts(manifest)
    front, rear = plate_faces(manifest)
    st = _standards()
    out = {}
    # camera: head on the front, through washer + board + plain spacer, into the sleeve that starts at the plate's front face
    spec = POSTS["camera"]
    sl = sleeve("camera")
    th = thread(sl["thread"])
    S = max(rear - front, sl["engagement_x_d"] * th["major"])
    std = st["head_types"][spec["head"]["standard"]][spec["head"]["thread"]]
    rows = []
    for i, r in enumerate(P["camera"]):
        b = r["spacer_bore"]
        wb = bore(body.mesh(spec["washers"][i]))
        seat = wb["extent"][0]                          # the washer's front face: the head's seat
        stack = float(front - seat)
        ax = np.zeros(3); ax[b["axis"]] = 1.0
        c = np.array(b["centre"], float); c[b["axis"]] = front
        lo = c + np.array([-4.0, S, -4.0]); hi = c + np.array([4.0, S + 8.0, 4.0])
        neigh = _neighbours(manifest, np.minimum(lo, hi), np.maximum(lo, hi))
        free, who = _clear_run(neigh, c, ax, S, S + 6.0, th["major"] / 2.0)
        pick = pick_length(stack, S, std["lengths"], free)
        pick.update(free_beyond_sleeve_mm=round(free, 2), first_neighbour=who, head_seat=round(float(seat), 3))
        opening = c
        rows.append(dict(plan=_plan_dict(spec["screws"][i], "camera", spec, std, r, opening, -ax, stack, S, pick, "steel", r["spacer"]),
                         sleeve_len=S))
    out["camera"] = rows
    # mic: head on the board side, through board + shim into the sleeve in the wing
    spec = POSTS["mic"]
    sl = sleeve("mic")
    th = thread(sl["thread"])
    std = st["head_types"][spec["head"]["standard"]][spec["head"]["thread"]]
    import screw_plan as SP
    rows = []
    for i, r in enumerate(P["mic"]):
        b = r["spacer_bore"]
        ax = np.zeros(3); ax[b["axis"]] = 1.0
        c = np.array(b["centre"], float); c[b["axis"]] = b["extent"][1]          # the wing's inboard face
        stk = SP.stack_along_axis([body.mesh(n) for n in spec["clamped"]], c, -ax, radii=(1.5, 2.0))
        stack = float(stk["stack_mm"])
        S = _mic_sleeve_len(manifest, c, sl["ro"], sl["wing_thickness"])
        lo = c + np.array([S - 0.5, -4.0, -4.0]); hi = c + np.array([S + 6.0, 4.0, 4.0])
        neigh = _neighbours(manifest, lo, hi)
        free, who = _clear_run(neigh, c, ax, S, S + 5.0, th["major"] / 2.0)
        pick = pick_length(stack, S, spec["head"]["lengths"], free)
        S2 = round(min(S, pick["engagement_mm"]), 3)     # the sleeve ends where the screw's thread does
        pick = dict(pick, sleeve_mm=S2, engagement_mm=S2, tip_beyond_sleeve_mm=round(pick["length_mm"] - stack - S2, 3))
        pick.update(free_beyond_sleeve_mm=round(free, 2), first_neighbour=who, head_seat=round(float(c[b["axis"]] - stack), 3),
                    sleeve_before_trim_mm=round(S, 3))
        rows.append(dict(plan=_plan_dict(spec["screws"][i], "mic", spec, std, r, c, -ax, stack, S2, pick, "steel", r["spacer"]),
                         sleeve_len=S2))
    out["mic"] = rows
    _CACHE[key] = out
    return out


def sleeve_lens(group: str, manifest: Path | None = None) -> list[float]:
    return [float(r["sleeve_len"]) for r in threaded_posts(manifest)[group]]


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
