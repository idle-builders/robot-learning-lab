"""Head-carrier screw posts from the supplier parts' own holes (Idle T-5bqc3m).

The compute-board, camera and microphone posts of combined__head_sensor_compute_carrier are never typed: the bore of each
spacer (and of its terminal nut) is measured on its mesh (section through the mid plane, circle fit of the inner loop), the
carrier plate faces come from where the spacers rest, and the screws are generated from them.
Everything is a measurement in mm; nothing here accepts or rejects.

  posts(manifest)         per post: spacer / terminal bore centre and axis, their coaxiality (mm), plate faces
  plate_faces(manifest)   (front, rear) coordinate of the carrier plate along Y
  threaded_posts(manifest) camera / mic inserts (start, length) and their screw plans
  column_profile(manifest) camera columns: radius bands from the board's components round each post (T-5bqc3m K1)
  compute_posts(manifest) compute posts: plain washer seat or printed column + counterbore, screw length (T-5bqc3m K2)
  root_plans(sites, ...)  neck-root M3 screws into the cradle tongue's heat-set inserts (T-9wjdqc)
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


def _measure(body, part: str) -> dict:
    """Bore of a supplier part on its mesh, measured live (the shell-pass manifest, defined_holes.SHELL_PASS, holds every supplier
    spacer / washer / shim the printed posts replace)."""
    if part not in body.parts:
        raise KeyError(f"{part}: not in the manifest head_holes reads; its bore places a head-carrier post (use the shell-pass manifest, "
                       f"defined_holes.SHELL_PASS)")
    return bore(body.mesh(part))


def posts(manifest: Path | None = None) -> dict:
    key = ("posts", str(manifest))
    if key in _CACHE:
        return _CACHE[key]
    body, _ = _body(manifest)
    out = {}
    for g, spec in POSTS.items():
        if "spacers" not in spec:        # neck_root: sites from the connector specs (common.stage_interface_specs)
            continue
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


def head_std(head: dict) -> dict:
    """dk, k, lengths of a head standard and thread: robot/body/checks/data/hardware.json, else the values the post gives with their
    source (a standard that table does not carry, e.g. ISO 14580)."""
    std = _standards()["head_types"].get(head["standard"], {}).get(head["thread"])
    if std:
        return dict(std, source="robot/body/checks/data/hardware.json")
    return dict(dk=head["dk"], k=head["k"], lengths=head["lengths"], source=head["source"])


def thread_token(thread: str) -> str:
    return thread.replace(".", "p")


def named(name: str, thread: str, length: float) -> str:
    """The length goes into the name (mount_check reads a screw by its M<d>x<length> name): B0410_camera_M2_0 -> B0410_camera_M2x8_0."""
    t = thread_token(thread)
    return name.replace(f"_{t}_", f"_{t}x{length:g}_")


def _plan_dict(name, group, spec, std, row, opening, axis_out, stack, S, pick, mat, hole):
    st_pt = _standards()["point_mm"]
    h = spec["head"]
    thr = h["thread"]
    name = named(name, thr, pick["length_mm"])
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
    # camera: head on the camera board front (board, printed column), insert in the column / plate, its rear end just behind the plate's rear face
    spec = POSTS["camera"]
    sl = sleeve("camera")
    th = thread(sl["thread"])
    ref = sl["engagement_x_d"] * th["major"]
    std = head_std(spec["head"])
    rows = []
    for i, r in enumerate(P["camera"]):
        b = r["spacer_bore"]
        wb = _measure(body, spec["washers"][i])       # the 0.3 mm washer is gone (owner 2026-11-22): its board-side face is the board's front face
        seat = wb["extent"][1]                          # the board's mount pad: the ISO 4762 head bears on it directly
        # the insert's rear end is flush with the plate's rear face (an iron tip ends on a face: a recessed end left 0.17-0.24 mm of plate
        # in the 8 mm tip's way), its front end on the column bore's ledge; the screw is the smallest standard length that reaches the
        # 1 d reference with its point inside the insert
        start = float(rear - sl["length"])
        L_pick = next(L for L in sorted(std["lengths"]) if L - st["point_mm"] - (start - seat) >= ref - 1e-6 and L - (start - seat) <= sl["length"] + 1e-6)
        stack = float(start - seat)
        ax = np.zeros(3); ax[b["axis"]] = 1.0
        c = np.array(b["centre"], float); c[b["axis"]] = start
        pick = SP.choose_length(stack, sl["length"], ref, [float(L_pick)], st["point_mm"])
        pick.update(ref_mm=ref, head_seat=round(float(seat), 3), insert_start=round(float(start), 3))
        rows.append(dict(plan=_plan_dict(spec["screws"][i], "camera", spec, std, r, c, -ax, stack, sl["length"], pick, "brass", f"camera_insert_{i}"),
                         sleeve_len=sl["length"], start=float(start)))
    out["camera"] = rows
    # mic: head on the board's front face (PCB pad), through the PCB and the wing's floor into the insert in the boss behind it; the board's
    # back face is where the supplier shim's board-side face was (it rests on the wing, face_gap_mm off)
    spec = POSTS["mic"]
    sl = sleeve("mic")
    th = thread(sl["thread"])
    ref = sl["engagement_x_d"] * th["major"]
    std = head_std(spec["head"])
    wing = spec["wing"]
    rows = []
    for i, r in enumerate(P["mic"]):
        b = r["spacer_bore"]
        ax = np.zeros(3); ax[b["axis"]] = 1.0
        back = float(b["extent"][0])                                   # the board's back face
        c = np.array(b["centre"], float); c[b["axis"]] = back
        stk = SP.stack_along_axis([body.mesh(n) for n in spec["clamped"]], c, -ax, radii=(1.5, 2.0))
        seat = back - float(stk["stack_mm"])                          # the PCB's front face under the head
        face = back + float(wing["face_gap_mm"])                      # the wing face the board rests on
        # the insert's front end where the screw reaches engagement_target_mm: start = seat + L - point - target, for the smallest standard
        # length that leaves the floor (wing face to insert) at least floor_min_mm thick
        L_pick, start = next((L, float(seat + L - st["point_mm"] - sl["engagement_target_mm"])) for L in sorted(std["lengths"])
                             if seat + L - st["point_mm"] - sl["engagement_target_mm"] - face >= float(wing["floor_min_mm"]) - 1e-6)
        stack = float(start - seat)
        c[b["axis"]] = start
        pick = SP.choose_length(stack, sl["length"], ref, [float(L_pick)], st["point_mm"])
        pick.update(ref_mm=ref, head_seat=round(float(seat), 3), insert_start=round(start, 3), board_back=round(back, 3), wing_face=round(face, 3),
                    floor_mm=round(start - face, 3))
        rows.append(dict(plan=_plan_dict(spec["screws"][i], "mic", spec, std, r, c, -ax, stack, sl["length"], pick, "brass", f"mic_insert_{i}"),
                         sleeve_len=sl["length"], start=start, back=back, face=face, seat=seat))
    out["mic"] = rows
    _CACHE[key] = out
    return out


def sleeve_lens(group: str, manifest: Path | None = None) -> list[float]:
    return [float(r["sleeve_len"]) for r in threaded_posts(manifest)[group]]


def pin_site(manifest: Path | None = None) -> np.ndarray:
    """Centre of the mic board's second mounting hole on its back face (the bore of the dropped upper shim): the locating pin's root."""
    body, _ = _body(manifest)
    b = _measure(body, POSTS["mic"]["drop"][1])
    c = np.array(b["centre"], float); c[b["axis"]] = b["extent"][0]
    return c


def sleeve_starts(group: str, manifest: Path | None = None) -> list[float]:
    """Coordinate along the post axis of each insert's front end (where the screw enters)."""
    return [float(r["start"]) for r in threaded_posts(manifest)[group]]


def _board_points(body, name: str) -> np.ndarray:
    """Dense surface samples + vertices of a supplier board mesh (its components are small boxes: the vertices alone miss their faces)."""
    import trimesh
    key = ("board_pts", name)
    if key not in _CACHE:
        m = body.mesh(name)
        _CACHE[key] = np.vstack([trimesh.sample.sample_surface_even(m, 1500000, seed=0)[0], np.asarray(m.vertices)])
    return _CACHE[key]


def column_profile(manifest: Path | None = None) -> list[dict]:
    """Per camera post the printed column as radius bands along its axis (+y, board end -> plate front), from the board's mesh: in each
    height band the column keeps column.component_clear_mm to the nearest board component (points behind the board's back face within
    reach_margin_mm above or below the band), never less than rim_mm round the clearance bore (insert_rim_mm round the insert, from 0.2 mm before the insert
    is), at most column.ro; the radius never shrinks toward the plate (printable with the plate on the bed, 45 deg cones between bands) and
    steps under min_step_mm are merged into the narrower band (into the wider one where that band reaches the insert). Rows: segments [(kind, y0, y1, r0, r1)], the measured clearance per band and the
    smallest one. Cached."""
    key = ("columns", str(manifest))
    if key in _CACHE:
        return _CACHE[key]
    body, _ = _body(manifest)
    P = posts(manifest)
    front, _rear = plate_faces(manifest)
    spec = POSTS["camera"]
    col = spec["column"]
    sl = sleeve("camera")
    starts = sleeve_starts("camera", manifest)
    ro, clear, mg, step = float(col["ro"]), float(col["component_clear_mm"]), float(col["reach_margin_mm"]), 0.05
    y_end = front - float(col["height"])
    out = []
    for i, r in enumerate(P["camera"]):
        b = r["spacer_bore"]
        c = np.asarray(b["centre"], float)
        back = float(b["extent"][0])                       # the board's back face (the standoff's board-side end)
        pts = _board_points(body, spec["boards"][i])
        rr = np.hypot(pts[:, 0] - c[0], pts[:, 2] - c[2])
        near = (pts[:, 1] > back + 0.03) & (rr < ro + clear + 1.0)
        pp, pr = pts[near, 1], rr[near]
        ys = np.arange(y_end, front - 1e-9, step)
        t, env = [], []
        for y0 in ys:
            s_ = (pp >= y0 - mg) & (pp <= y0 + step + mg)
            e = float(pr[s_].min()) if s_.any() else None
            lo = float(sl["ro"]) + float(col["insert_rim_mm"]) if y0 + step > starts[i] - 0.2 + 1e-6 else float(col["bore_dia"]) / 2 + float(col["rim_mm"])
            t.append(min(ro, max(lo, (e - clear) if e is not None else ro)))
            env.append(e)
        rad = [min(t[k:]) for k in range(len(t))]          # never narrower toward the plate
        rad = [float(np.floor(x * 100 + 1e-6) / 100) for x in rad]
        bands = []                                          # [y0, y1, r] from the board end
        for y0, x in zip(ys, rad):
            if bands and abs(bands[-1][2] - x) < 1e-9:
                bands[-1][1] = y0 + step
            else:
                bands.append([float(y0), float(y0 + step), x])
        bands[-1][1] = float(front)
        k = 0                                               # merge steps under min_step_mm: into the narrower band, unless the wider one
        while k < len(bands) - 1:                           # reaches the insert (its rim is the floor there): then into the wider one
            if bands[k + 1][2] - bands[k][2] < float(col["min_step_mm"]) - 1e-6:
                if bands[k + 1][1] <= starts[i] - 0.2 + 1e-6:
                    bands[k][1] = bands[k + 1][1]
                    del bands[k + 1]
                else:
                    bands[k + 1][0] = bands[k][0]
                    del bands[k]
                continue
            k += 1
        segs = []
        for k, (y0, y1, x) in enumerate(bands):
            if k and x > bands[k - 1][2]:                    # 45 deg cone from the narrower band below
                dl = min(x - bands[k - 1][2], y1 - y0)
                segs.append(("cone", y0, y0 + dl, bands[k - 1][2], bands[k - 1][2] + dl))
                y0 = y0 + dl
            if y1 - y0 > 1e-6:
                segs.append(("cyl", y0, y1, x, x))
        # clearance of the built profile to the components, per band (None: no component within reach)
        def r_at(y):
            for kind, a, bb, r0, r1 in segs:
                if a - 1e-9 <= y <= bb + 1e-9:
                    return r0 + (r1 - r0) * (y - a) / max(bb - a, 1e-9)
            return ro
        cl = [round(float(pr[(pp >= y0) & (pp <= y0 + step)].min() - r_at(y0 + step / 2)), 3) if ((pp >= y0) & (pp <= y0 + step)).any() else None
              for y0 in ys]
        cl_ok = [x for x in cl if x is not None]
        out.append(dict(index=i, centre=c, board=spec["boards"][i], segments=segs, bands=[[round(a, 3), round(bb, 3), x] for a, bb, x in bands],
                        clearance_min_mm=(min(cl_ok) if cl_ok else None), component_reach_mm=(round(float(pp.max() - back), 3) if len(pp) else None)))
    _CACHE[key] = out
    return out


def compute_posts(manifest: Path | None = None) -> list[dict]:
    """The 4 compute posts (T-5bqc3m K2): ISO 4762 M2.5 + ISO 7092 washer on the front, plain clearance hole, into the supplier nut. Per post the
    smallest standard length whose thread ends engagement_margin_mm past the nut's far face with the washer on the plate front ("plain"); where
    none reaches, the supplier spacer becomes a printed column and the washer sits in a counterbore whose floor lets the longest standard length
    reach ("column"). Rows: kind, washer floor (y of the washer's plate-side face), column top, the screw plan."""
    import screw_plan as SP
    key = ("compute", str(manifest))
    if key in _CACHE:
        return _CACHE[key]
    P = posts(manifest)
    front, rear = plate_faces(manifest)
    st = _standards()
    spec = POSTS["compute"]
    h = spec["head"]
    std = head_std(h)
    w, col = spec["washer"], spec["column"]
    pt, mgn = float(st["point_mm"]), float(spec["engagement_margin_mm"])
    out = []
    for r in P["compute"]:
        b, t = r["spacer_bore"], r["terminal_bore"]
        ax = np.zeros(3); ax[b["axis"]] = 1.0
        near, far = float(t["extent"][0]), float(t["extent"][1])
        floor = float(front)                                  # the washer's plate-side face
        fits = [L for L in sorted(std["lengths"]) if floor - float(w["h"]) + L - pt >= far + mgn - 1e-6]
        if fits:
            L, kind = float(fits[0]), "plain"
        else:
            L, kind = float(max(std["lengths"])), "column"
            floor = far + mgn + pt + float(w["h"]) - L
        top = float(b["extent"][1]) - float(col["gap_mm"])    # the spacer's board end, less the gap (column posts)
        seat = floor - float(w["h"])                          # head underside
        stack, depth = near - seat, far - near
        opening = np.array(t["centre"], float); opening[t["axis"]] = near
        pick = SP.choose_length(stack, depth, depth, [L], pt)
        pick.update(engagement_mm=round(min(L - pt - stack, depth), 3), thread_past_far_face_mm=round(L - pt - (far - seat), 3))
        plan = dict(name=named(r["screw"], h["thread"], L), group="compute", thread=h["thread"], thread_label=h["thread"], head_standard=h["standard"],
                    head_dia_mm=std["dk"], head_height_mm=std["k"], countersunk=False, terminal_material="steel", opening=opening, axis_out=-ax,
                    stack_mm=stack, hole_depth_mm=depth, reference_engagement_mm=depth, point_mm=pt, hole_id=f"{r['terminal']}:thread",
                    owning_link="head", spacer=r["spacer"], head_gap_mm=0.0, **pick)
        out.append(dict(index=r["index"], kind=kind, centre=np.array(b["centre"], float), floor=floor, column_top=top, spacer=r["spacer"],
                        washer=spec["washer"]["names"][r["index"]], plan=plan, floor_material_mm=round(top - floor, 3) if kind == "column" else None))
    _CACHE[key] = out
    return out


def root_insert() -> dict:
    """The cradle tongue's M3 heat-set insert (head_carrier_posts.json neck_root.insert): ro (hole), ri (thread minor), length, thread."""
    s = dict(POSTS["neck_root"]["insert"])
    s["ri"] = thread(s["thread"])["minor_dia"] / 2.0
    return s


def root_plans(sites: list, plate_rear: float, tongue_rear: float) -> list[dict]:
    """The 2 neck-root screws (T-9wjdqc): ISO 7380-1 M3 head on the carrier's rear face (plate_rear), through the carrier into the brass M3
    heat-set insert that fills the cradle tongue from its rear face (tongue_rear) to its front face; the smallest standard length that reaches
    the 1 d reference with its point inside the insert (screw_plan.choose_length)."""
    import screw_plan as SP
    st = _standards()
    spec = POSTS["neck_root"]
    h = spec["head"]
    std = head_std(h)
    ins = root_insert()
    pt, depth = float(st["point_mm"]), float(ins["length"])
    ref = float(ins["engagement_x_d"]) * float(thread(ins["thread"])["major"])
    out = []
    for i, c in enumerate(sites):
        ax = np.array([0.0, 1.0, 0.0])                     # head side (+y, the carrier's rear)
        opening = np.array(c, float); opening[1] = tongue_rear
        stack = plate_rear - tongue_rear
        pick = SP.choose_length(stack, depth, ref, [float(L) for L in sorted(std["lengths"])], pt)
        L = float(pick["length_mm"])
        pick.update(engagement_mm=round(min(L - pt - stack, depth), 3), tip_past_insert_front_mm=round(L - stack - depth, 3))
        out.append(dict(name=named(spec["screws"][i], h["thread"], L), group="neck_root", thread=h["thread"], thread_label=h["thread"],
                        head_standard=h["standard"], head_dia_mm=std["dk"], head_height_mm=std["k"], countersunk=False, terminal_material="brass",
                        opening=opening, axis_out=ax, stack_mm=stack, hole_depth_mm=depth, reference_engagement_mm=ref,
                        point_mm=pt, hole_id=f"root_insert_{i}:thread", owning_link="head", spacer=None, head_gap_mm=0.0,
                        **pick))
    return out


def screw_plans(manifest: Path | None = None) -> list[dict]:
    """The generated screw of every camera / mic / compute post (the neck-root screws: root_plans, sites from the connector specs)."""
    plans = []
    for g in ("camera", "mic"):
        plans += [r["plan"] for r in threaded_posts(manifest)[g]]
    plans += [r["plan"] for r in compute_posts(manifest)]
    return plans
