"""Shoulder shells: torso shells and upper-arm covers cleared for the shoulder U connectors, covers mounted on the green U
(Idle T-tph7xb, parent T-ce88t9; owner M-s9tnc9: shells and cover redesigned once the connectors passed their checks).

Same method as the shell pass (robot/body/ape-proportions/shells.py, "openings cut by the motion they have to allow"):
each moving group is voxelized in its link's neutral frame, grown by CLEARANCE, looked up from a grid over the target
shell at every pose of the joint range (1 deg steps), smoothed, turned into one surface (marching cubes) and subtracted
from the shell once. The openings are as small as the motion allows. Shells stay mesh-defined.

  torso shells (torso link)        minus the blue U (stage 1, its sleeves, its roll-horn / idler screws, the roll idler)
                                   swept over shoulder pitch -60..150 (roll and the arm joints do not move it vs the torso)
  upper-arm cover (roll link)      minus the same blue U group swept over shoulder roll -15..95 (in the roll-link frame),
                                   plus the green U (stage 2, its sleeves and roll-case screws) statically, grown by
                                   STATIC_CLEARANCE (same link: no motion), except on the two mount pads
  cover mount                      two pads on the green U at the cover's existing screw holes: a boss with a pressed
                                   304 sleeve (4.5 x 3.2, the M3 sleeve of the U) and a captive ISO 4032 M3 nut in a
                                   slot (front open to +z, rear to +x); ISO 7380-1 M3 button head from outside through
                                   the cover's well (front M3x8, rear M3x6).

Runs after `shoulder_clevis.py manifest`: reads the selected body it wrote (assembly-parts-shoulder-u.json, never
written here), takes the shell meshes from the roll-flip source manifest and the green U from the build output
(cad-manifest.json), and writes its own candidate shells/assembly-parts-shoulder-u-shells.json plus meshes / STEP under
artifacts/robot/body/arm/shoulder-clevis/shells/. Seeded voxel sampling: a rerun gives the same files (idempotent).

  python3 infra/tools/memguard.py --limit-gb 6 --timeout-s 1800 -- .venv/bin/python -B robot/body/arm/shoulder_shells.py apply
  python3 infra/tools/memguard.py --limit-gb 4 --timeout-s 900 -- .venv/bin/python -B robot/body/arm/shoulder_shells.py views
Check the candidate with robot/body/checks/run.py --manifest artifacts/robot/body/arm/shoulder-clevis/shells/assembly-parts-shoulder-u-shells.json
Every result is a measurement (mm, mm3); nothing accepts or rejects.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "robot/body"))
sys.path.insert(0, str(HERE))
from checks import bodylib as F  # noqa: E402
import shoulder_clevis as SC  # noqa: E402

OUT = SC.OUT / "shells"
PARTS = OUT / "parts"
REPORT = OUT / "report.json"
CANDIDATE = OUT / "assembly-parts-shoulder-u-shells.json"   # own candidate; the selected body (SC.CANDIDATE) is only read
VIEWS = SC.VIEWS
CLEARANCE = 2.0              # moving groups (as the shell pass)
STATIC_CLEARANCE = 0.5       # same-link parts (green U in the cover): print tolerance only, nothing moves
PITCH = 0.4                  # voxel (mm)
STEP = 1.0                   # joint step (deg)
TORSO_SHELLS = ("center_front_shell", "center_back_shell")


def _side(n, s):
    return n.replace("R_", s, 1) if n.startswith("R_") else n.replace("upper__R_", "upper__" + s, 1)


def blue_group(body, s):
    """The blue U and what moves with it on the pitch link near the roll servo."""
    names = [SC.S1, SC.S1 + "_steel_sleeves", SC.IDLER] + [n for n in body.parts if n.startswith("R_shoulder_u_roll_horn_")
                                                             or n.startswith("R_shoulder_u_roll_idler_")]
    return [_side(n, s) for n in names]


def green_group(body, s):
    names = [SC.S2, SC.S2 + "_steel_sleeves"] + [n for n in body.parts if n.startswith("R_shoulder_u_roll_case_")]
    return [_side(n, s) for n in names]


# ---------------------------------------------------------------------------------------------- voxel sweep
def _solid_voxels(m, lo, shape, pitch=PITCH):
    from scipy.ndimage import binary_fill_holes
    n = int(min(4e6, max(2e4, m.area / (0.4 * pitch) ** 2)))
    pts = np.vstack([trimesh.sample.sample_surface(m, n, seed=0)[0], m.vertices])   # seeded: reruns give the same cut
    idx = np.round((pts - lo) / pitch).astype(np.int32)
    a = np.maximum(idx.min(0) - 1, 0); b = np.minimum(idx.max(0) + 2, shape)
    sub = np.zeros(np.maximum(b - a, 1), bool)
    k = idx - a; ok = np.all((k >= 0) & (k < sub.shape), 1)
    sub[tuple(k[ok].T)] = True
    sub = binary_fill_holes(sub)
    if m.is_watertight:       # the hole fill leaks through sub-voxel walls (sleeve bores, slots): add the exact inside test
        ii = np.argwhere(~sub)
        if len(ii):
            sub[tuple(ii[m.contains(lo + (ii + a) * pitch)].T)] = True
    out = np.zeros(shape, bool); out[a[0]:b[0], a[1]:b[1], a[2]:b[2]] = sub
    return out


class Grid:
    """Parts as an occupancy grid in their neutral frame, grown by `clear` mm."""
    def __init__(self, meshes, clear):
        from scipy.ndimage import distance_transform_edt
        pad = clear + 3 * PITCH
        self.lo = np.min([m.bounds[0] for m in meshes], 0) - pad
        hi = np.max([m.bounds[1] for m in meshes], 0) + pad
        self.shape = np.ceil((hi - self.lo) / PITCH).astype(int) + 1
        occ = np.zeros(self.shape, bool)
        for m in meshes:
            occ |= _solid_voxels(m if m.is_watertight else m.convex_hull, self.lo, self.shape)
        self.hit = distance_transform_edt(~occ) * PITCH <= clear
        self.corners = np.array(list(itertools.product(*zip(self.lo, self.lo + (self.shape - 1) * PITCH))))


class Sweep:
    def __init__(self, lo, hi):
        self.lo = np.asarray(lo, float); self.shape = np.ceil((np.asarray(hi) - self.lo) / PITCH).astype(int) + 1
        self.mask = np.zeros(self.shape, bool)

    def points(self):
        return self.lo + np.argwhere(self.mask) * PITCH

    def add(self, g, T):
        """T maps target-neutral coordinates into the grid's neutral frame."""
        w = trimesh.transform_points(g.corners, np.linalg.inv(T))
        a = np.clip(np.floor((w.min(0) - self.lo) / PITCH).astype(int), 0, self.shape)
        b = np.clip(np.ceil((w.max(0) - self.lo) / PITCH).astype(int) + 1, 0, self.shape)
        if np.any(b <= a):
            return
        ii = np.stack(np.meshgrid(*[np.arange(a[k], b[k]) for k in range(3)], indexing="ij"), -1).reshape(-1, 3)
        q = (self.lo + ii * PITCH) @ T[:3, :3].T + T[:3, 3]
        j = np.round((q - g.lo) / PITCH).astype(np.int32)
        ok = np.all((j >= 0) & (j < g.shape), 1)
        hit = np.zeros(len(ii), bool); hit[ok] = g.hit[tuple(j[ok].T)]
        self.mask[tuple(ii[hit].T)] = True

    def surface(self):
        from scipy.ndimage import gaussian_filter
        from skimage.measure import marching_cubes
        f = gaussian_filter(np.pad(self.mask, 2).astype(np.float32), 0.6)
        v, faces, _, _ = marching_cubes(f, 0.5)
        return trimesh.Trimesh(self.lo + (v - 2) * PITCH, faces[:, ::-1], process=True)


def _mf(m):
    import manifold3d as mf
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32), tri_verts=np.asarray(m.faces, np.uint32)))


def _tm(x):
    r = x.to_mesh()
    return trimesh.Trimesh(np.asarray(r.vert_properties)[:, :3], np.asarray(r.tri_verts), process=True)


def joint_poses(body, jn):
    lo, hi = body.joints[jn]["range_deg"]
    return np.arange(lo, hi + 1e-6, STEP)


def torso_sweep(body, s, box):
    """Blue group swept over the shoulder pitch range, in the torso frame (= neutral world)."""
    g = Grid([body.mesh(n) for n in blue_group(body, s)], CLEARANCE)
    sw = Sweep(*box)
    for q in joint_poses(body, f"{s}shoulder_pitch"):
        sw.add(g, np.linalg.inv(body.link_T(f"{s}shoulder_pitch_link", {f"{s}shoulder_pitch": q})))
    return sw


def cover_sweep(body, s, box, pads=()):
    """Blue group swept over the shoulder roll range in the roll-link frame, plus the green group grown by
    STATIC_CLEARANCE (static, same link); the green clearance is left out inside the mount pad cylinders (contact)."""
    g = Grid([body.mesh(n) for n in blue_group(body, s)], CLEARANCE)
    sw_b = Sweep(*box)
    for q in joint_poses(body, f"{s}shoulder_roll"):
        sw_b.add(g, body.link_T(f"{s}shoulder_roll_link", {f"{s}shoulder_roll": q}))
    sw_g = Sweep(*box)
    sw_g.add(Grid([body.mesh(n) for n in green_group(body, s)], STATIC_CLEARANCE), np.eye(4))
    return sw_b, sw_g


def shell_volume_in(mask_sweep, shell):
    """Volume (mm3) of a sweep mask inside a closed shell mesh."""
    P = mask_sweep.points()
    if not len(P):
        return 0.0
    lo, hi = shell.bounds
    P = P[np.all((P >= lo) & (P <= hi), 1)]
    return float(shell.contains(P).sum() * PITCH ** 3) if len(P) else 0.0


def measure():
    """Quick look: removed volume per target and group, no files written."""
    man = json.loads(SC.CANDIDATE.read_text()); body = F.Body(man)     # quick look on the selected body
    t0 = time.time()
    for s in ("R_",):
        cov = body.mesh(f"{s}upper_arm_hard_cover")
        sw_b, sw_g = cover_sweep(body, s, (cov.bounds[0] - 1, cov.bounds[1] + 1))
        print(f"{s} cover: blue swept {shell_volume_in(sw_b, cov):.0f} mm3, green static {shell_volume_in(sw_g, cov):.0f} mm3 "
              f"({time.time() - t0:.0f} s)", flush=True)
        sh = {n: body.mesh(n) for n in TORSO_SHELLS}
        x0, x1 = (60.0, 125.0) if s == "R_" else (-125.0, -60.0)
        box = (np.array([x0, -40.0, 300.0]), np.array([x1, 60.0, 400.0]))
        sw = torso_sweep(body, s, box)
        for n, m in sh.items():
            print(f"{s} {n}: blue swept {shell_volume_in(sw, m):.0f} mm3 ({time.time() - t0:.0f} s)", flush=True)


# ---------------------------------------------------------------------------------------------- cover mount (B-rep)
# The cover keeps its two screw wells (inward tubes r 3.25..4.75 ending in a flat pad with an M3 clearance hole, made
# for the old stage-2 posts); a boss on the green U meets each pad. R frame (body mm); L = mirror x, as the build.
#   axis: (x, z) of the well (along y); head: side of the screw head (-1 face / -y, +1 back / +y)
#   sleeve: pressed 304 sleeve length from the pad face (the clamp: head - cover pad - sleeve - nut)
#   back: boss end away from the pad; chamfer: 45 deg underside where the boss overhangs the arm (print up ~ +z)
#   bore_slot: the screw bore behind the nut runs out of the arm's top edge (front: a 0.004 mm skin), so it is
#   opened to +z like the nut slot
#   slot: nut insertion direction in (x, z), default +z; rear tilted 15 deg toward +x: straight up the slot passes 0.25 mm
#   from the +y arm's M2 sleeve bore, straight along +x it cuts the yaw-screw counterbore (sleeve support 82 -> 60 %)
#   pocket_extra: pocket depth past the nut (default 0.2); rear 0.6 so the pocket opens cleanly into the yaw-screw
#   driver channel (r 2.25 at x ~133.4) instead of leaving a 0.08 mm membrane against it
NUT = dict(s=5.5, m=2.4, clear=0.15, depth_clear=0.2)    # ISO 4032 M3; pocket: flats along the slot walls
NUT_HW = NUT["s"] / 2 + NUT["clear"]
MOUNTS = {
    "front": dict(axis=(116.5, 336.5), head=-1, sleeve=2.5, screw=8, back=-16.9, chamfer=True, bore_slot=True),
    "rear": dict(axis=(130.5, 329.8), head=+1, sleeve=1.5, screw=6, back=25.3, chamfer=False, bore_slot=False, pocket_extra=0.6,
                 slot=(0.258819, 0.965926)),
}
BOSS_R = 4.5                  # boss radius = the well's flat pad (r 4.75) less 0.25
PAD_GAP = 0.05                # boss face to pad face
PAD_KEEP_R = 5.5              # no static clearance cut of the cover inside this radius of a mount axis (the pads)
SCREW_BORE_R = 1.7            # M3 clearance behind the nut
SLOT_TOP = 2.0                # boss top over the nut slot (above the axis): slot walls >= 1.15 mm wide (0.62 mm line)
HEAD = "ISO 7380-1"           # button head as the old cover screws, seated in the well


def pad_faces(cover, x, z, head):
    """Pad inner face (toward the green U) and screw seat (outer face of the pad, in the well) along the well axis,
    from rays parallel to the axis at r 2.6 (seat ring) and r 4.2 (pad ring)."""
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    rx = RayMeshIntersector(cover)
    out = {}
    for key, r in (("seat", 2.6), ("pad", 4.2)):
        ys = []
        for a in np.radians(np.arange(0, 360, 30)):
            o = np.array([[x + r * np.cos(a), float(cover.bounds[:, 1].mean()), z + r * np.sin(a)]])   # from inside the cover
            loc, _, _ = rx.intersects_location(o, np.array([[0.0, float(head), 0.0]]), multiple_hits=True)
            h = np.sort(loc[:, 1] * head)
            ys.append(h[:2] * head)
        ys = np.array(ys)
        out[key] = ys
    pad = float(np.median(out["pad"][:, 0]))          # first wall met going outward = the pad's inner face
    seat = float(np.median(out["seat"][:, 1]))        # second hit at the seat ring = the seat (pad outer face)
    return pad, seat, dict(pad_spread=float(np.ptp(out["pad"][:, 0])), seat_spread=float(np.ptp(out["seat"][:, 1])))


def _ccw(pts):
    """Counter-clockwise order (build123d extrudes along the face normal, which flips for a clockwise outline)."""
    a = np.asarray(pts, float)
    area = 0.5 * float(np.sum(a[:, 0] * np.roll(a[:, 1], -1) - np.roll(a[:, 0], -1) * a[:, 1]))
    return [tuple(map(float, q)) for q in (a if area > 0 else a[::-1])]


def _prism_y(pts_xz, y0, y1):
    """Prism of a polygon (x, z) between y0 < y1."""
    from build123d import Polygon, extrude
    s = extrude(Polygon(*_ccw(pts_xz), align=None), amount=y1 - y0)
    T = np.eye(4); T[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]; T[1, 3] = y1
    return SC.transformed(s, T)


def _prism_x(pts_yz, x0, x1):
    from build123d import Polygon, extrude
    s = extrude(Polygon(*_ccw(pts_yz), align=None), amount=x1 - x0)
    T = np.eye(4); T[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]; T[0, 3] = x0
    return SC.transformed(s, T)


def mount_design(cover):
    """Per mount: the pad / seat from the cover, the boss, sleeve, nut and screw positions (R frame)."""
    out = {}
    for k, m in MOUNTS.items():
        (x, z), h = m["axis"], m["head"]
        pad, seat, spread = pad_faces(cover, x, z, h)
        face = pad - h * PAD_GAP
        s_end = face - h * m["sleeve"]                       # sleeve end = nut seat
        n_far = s_end - h * (NUT["m"] + m.get("pocket_extra", NUT["depth_clear"]))  # pocket far face
        nut_far = s_end - h * NUT["m"]
        tip = seat - h * m["screw"]
        out[k] = dict(axis_xz=[x, z], head_side=h, pad_face_y=round(pad, 3), seat_y=round(seat, 3), cover_pad_mm=round(abs(seat - pad), 3),
                      boss_face_y=round(face, 3), boss_back_y=m["back"], boss_r=BOSS_R, sleeve_y=[round(face, 3), round(s_end, 3)],
                      sleeve_len=m["sleeve"], nut_y=[round(s_end, 3), round(nut_far, 3)], pocket_y=[round(s_end, 3), round(n_far, 3)],
                      screw=f"{HEAD} M3x{m['screw']}", screw_tip_y=round(tip, 3),
                      thread_past_nut_mm=round((nut_far - tip) * h, 3), nut_engaged_mm=round(min(NUT["m"], (s_end - tip) * h - 0.5), 3),
                      stack_head_to_nut_mm=round(abs(seat - nut_far), 3), pad_ray_spread_mm=spread)
    return out


def build_mounts(design):
    """Green U (R, from the build output) with the two bosses, sleeve bores, nut pockets and screw bores; the 2 new
    sleeves. Returns (part, [sleeve solids])."""
    from build123d import import_step
    cad = json.loads((SC.OUT / "cad-manifest.json").read_text())["parts"]
    green = SC.largest(import_step(str(ROOT / cad[SC.S2]["native_step"])))
    add, cut, sleeves = [], [], []
    for k, d in design.items():
        (x, z), h, m = d["axis_xz"], d["head_side"], MOUNTS[k]
        face, back = d["boss_face_y"], m["back"]
        y0, y1 = sorted((face, back))
        boss = SC._bool("fuse", SC.occ_cyl((x, y0, z), (x, y1, z), BOSS_R), [SC.occ_box(x - BOSS_R, x + BOSS_R, y0, y1, z - 14.0, z)])
        if m["chamfer"]:          # underside: z_bottom = z - R at the pad face, falling 45 deg toward the back
            zb = z - BOSS_R
            yf, yb = face + h * 1.0, face - h * 16.0
            tri = [(yf, zb + 1.0), (yb, zb - 16.0), (yb, z - 40.0), (yf, z - 40.0)]
            boss = SC._bool("cut", SC.largest(boss), [_prism_x(tri, x - BOSS_R - 1, x + BOSS_R + 1)])
        else:
            boss = SC._bool("cut", SC.largest(boss), [SC.occ_box(x - 9, x + 9, y0 - 1, y1 + 1, z - 40.0, z - BOSS_R)])
        # over the nut slot the round boss top would leave knife-edged slot walls: flat top at z + SLOT_TOP
        s_end, back_ = d["sleeve_y"][1], m["back"]
        ya, yb = sorted((s_end, back_ - h * 0.5))
        boss = SC._bool("cut", SC.largest(boss), [SC.occ_box(x - BOSS_R - 1, x + BOSS_R + 1, ya, yb, z + m.get("slot_top", SLOT_TOP), z + 20.0)])
        add.append(SC.largest(boss))
        # green material beyond the pad face (the plate at the rear reaches 0.16 mm into the cover's pad)
        cut.append(SC.occ_cyl((x, face, z), (x, face + h * 4.0, z), PAD_KEEP_R))
        # ... and run that trim out through the plate's bottom edge (a disc alone leaves a 0.06 mm lip under it)
        cut.append(SC.occ_box(x - PAD_KEEP_R, x + PAD_KEEP_R, face, face + h * 4.0, z - 20.0, z))
        s0, s1 = d["sleeve_y"]
        cut.append(SC.occ_cyl((x, s0 + h * 0.5, z), (x, s1, z), SC.M3_SLEEVE[0]))
        p0, p1 = d["pocket_y"]
        hw, e = NUT["s"] / 2 + NUT["clear"], (NUT["s"] / np.sqrt(3) + NUT["clear"] / np.cos(np.pi / 6))
        su = np.array(m.get("slot", (0.0, 1.0)), float)          # slot direction in (x, z); flats face across it
        sv = np.array([-su[1], su[0]])
        prof = [tuple(np.array([x, z]) + u * su + v * sv) for u, v in ((-e, 0.0), (-e / 2, -hw), (20.0, -hw), (20.0, hw), (-e / 2, hw))]
        a, b = sorted((p0, p1))
        cut.append(_prism_y(prof, a, b))
        cut.append(SC.occ_cyl((x, p1 + h * 0.01, z), (x, d["screw_tip_y"] - h * 0.5, z), SCREW_BORE_R))
        if m["bore_slot"]:
            a2, b2 = sorted((p1 + h * 0.01, d["screw_tip_y"] - h * 0.5))
            cut.append(SC.occ_box(x - SCREW_BORE_R, x + SCREW_BORE_R, a2, b2, z, z + 20.0))
        sleeves.append(SC.sleeve((x, s1, z), (x, s0, z), *SC.M3_SLEEVE))
    part = SC.largest(SC._bool("cut", SC.largest(SC._bool("fuse", green, add)), cut))
    return part, sleeves


def nut_mesh(x, z, y_near, y_far, slot=(0.0, 1.0)):
    """ISO 4032 M3 nut (s 5.5, m 2.4, bore dia 3) between y_near and y_far, a corner toward the slot direction (x, z)
    so its flats lie along the slot walls."""
    a = np.radians(np.arange(6) * 60.0 + np.degrees(np.arctan2(slot[1], slot[0])))
    hexa = trimesh.creation.extrude_polygon(__import__("shapely.geometry", fromlist=["Polygon"]).Polygon(
        np.c_[np.cos(a), np.sin(a)] * NUT["s"] / np.sqrt(3)).difference(
        __import__("shapely.geometry", fromlist=["Point"]).Point(0, 0).buffer(1.5, 32)), NUT["m"])
    lo = min(y_near, y_far)
    T = np.eye(4); T[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]     # (u, v, w) -> (u, w, v): w along y
    m = hexa.copy(); m.apply_transform(T)
    if m.volume < 0:
        m.invert()
    m.apply_translation([x, lo, z])
    return m


def screw_plan(k, d):
    import screw_plan as SP
    sp = SP.screw_spec("M3", "M", HEAD)
    (x, z), h = d["axis_xz"], d["head_side"]
    return dict(thread="M3", head_dia_mm=sp["dk"], head_height_mm=sp["k"], length_mm=float(MOUNTS[k]["screw"]),
                point_mm=sp["point_mm"], axis_out=np.array([0.0, float(h), 0.0]),
                opening=np.array([x, d["nut_y"][0], z]), stack_mm=abs(d["seat_y"] - d["nut_y"][0]))


def _save_npz(name, m):
    PARTS.mkdir(parents=True, exist_ok=True)
    f = PARTS / f"{name}.npz"
    np.savez_compressed(f, vertices=np.asarray(m.vertices, float), faces=np.asarray(m.faces, np.int64))
    return dict(mesh=SC._rel(f), mesh_sha256=SC._sha(f), mesh_units="mm",
                bounds_mm=[np.round(m.bounds[0], 3).tolist(), np.round(m.bounds[1], 3).tolist()])


def _subtract(target, env):
    import manifold3d as mf
    e = _mf(env)
    if e.status() != mf.Error.NoError:
        raise ValueError(f"swept surface is not a valid manifold: {e.status()}")
    m = _tm(_mf(target) - e)
    big = [q for q in m.split(only_watertight=False) if (q.volume if q.is_watertight else 0) > 20.0]
    return trimesh.util.concatenate(big) if big else m


def openings(before, after):
    """The removed material as connected pieces: volume (mm3), box extents (mm), max extent, centre."""
    rem = _tm(_mf(before) - _mf(after))
    rows = []
    for q in rem.split(only_watertight=False):
        v = float(abs(q.volume))            # signed volume; approximate when the piece is not closed (flagged)
        if not v > 1.0:
            continue
        ext = q.bounds[1] - q.bounds[0]
        rows.append(dict(volume_mm3=round(v, 1), closed=bool(q.is_watertight), extent_xyz_mm=np.round(ext, 1).tolist(),
                         max_extent_mm=round(float(ext.max()), 1),
                         centre_mm=np.round(q.bounds.mean(0), 1).tolist()))
    rows.sort(key=lambda r: -r["volume_mm3"])
    return dict(removed_mm3=round(float(before.volume - after.volume), 1), before_mm3=round(float(before.volume), 1),
                after_mm3=round(float(after.volume), 1), pieces=rows)


def _pad_keep(sw, design, s):
    """Clear the static mask inside the mount pad cylinders (the cover's pads touch the bosses)."""
    sx = 1.0 if s == "R_" else -1.0
    P = sw.lo + np.argwhere(sw.mask) * PITCH
    keep = np.zeros(len(P), bool)
    for d in design.values():
        (x, z), h = d["axis_xz"], d["head_side"]
        ylo, yhi = sorted((d["seat_y"] + h * 1.0, d["boss_back_y"]))
        keep |= (np.hypot(P[:, 0] - sx * x, P[:, 2] - z) < PAD_KEEP_R) & (P[:, 1] >= ylo) & (P[:, 1] <= yhi)
    idx = np.argwhere(sw.mask)[keep]
    sw.mask[tuple(idx.T)] = False
    return int(keep.sum())


def _export(name, shapes):
    from build123d import Compound, export_step, export_stl
    PARTS.mkdir(parents=True, exist_ok=True)
    comp = Compound(children=list(shapes))
    st, sl = PARTS / f"{name}.step", PARTS / f"{name}.stl"
    export_step(comp, str(st))
    for tol, ang in ((0.02, 0.08), (0.035, 0.09)):
        export_stl(comp, str(sl), tolerance=tol, angular_tolerance=ang)
        if trimesh.load(str(sl), force="mesh", process=True).is_watertight:
            break
    bb = comp.bounding_box()
    return dict(native_step=SC._rel(st), native_step_sha256=SC._sha(st), mesh=SC._rel(sl), mesh_sha256=SC._sha(sl), mesh_units="mm",
                bounds_mm=[[round(bb.min.X, 3), round(bb.min.Y, 3), round(bb.min.Z, 3)], [round(bb.max.X, 3), round(bb.max.Y, 3), round(bb.max.Z, 3)]],
                volume_mm3=round(float(sum(x.volume for x in shapes)), 2), native_solid_count=len(shapes))


def hardware_rows(design, templ):
    """Cover screws and nuts (R and L) as manifest rows; names as the old cover posts' hardware."""
    import screw_plan as SP
    rows = []
    for s, sx in (("R_", 1.0), ("L_", -1.0)):
        M = np.diag([sx, 1.0, 1.0, 1.0])
        for k, d in design.items():
            (x, z) = d["axis_xz"]
            pl = screw_plan(k, d)
            sm = SP.screw_mesh(pl); nm = nut_mesh(x, z, d["nut_y"][0], d["nut_y"][1], MOUNTS[k].get("slot", (0.0, 1.0)))
            for m in (sm, nm):
                m.apply_transform(M)
                if m.volume < 0:
                    m.invert()
            for kind, m, nmx in (("screw", sm, f"{s}upper_arm_shell_{k}_M3x{MOUNTS[k]['screw']}"), ("nut", nm, f"{s}upper_arm_shell_{k}_M3_nut")):
                t = copy.deepcopy(templ[kind]); 
                for key in ("native_step", "native_step_sha256"):
                    t.pop(key, None)
                t.update(name=nmx, physical_part_id=nmx, owning_link=f"{s}shoulder_roll_link", role="fastener", **_save_npz(nmx, m),
                         volume_mm3=round(float(m.volume), 3), mass_kg=round(float(m.volume) * 7.85e-6, 7),
                         mass_basis="mesh volume x 7.85 g/cm3",
                         material=(f"{HEAD} M3x{MOUNTS[k]['screw']} through the cover pad and the green U sleeve into the captive nut"
                                   if kind == "screw" else "ISO 4032 M3 nut, captive in the green U slot"),
                         shoulder_shells="cover mount on the green U, robot/body/arm/shoulder_shells.py")
                if kind == "screw":
                    t["screw_plan"] = dict(thread_label="M3", head_standard=HEAD, length_mm=float(MOUNTS[k]["screw"]),
                                           stack_mm=round(pl["stack_mm"], 3), terminal="ISO 4032 M3 nut",
                                           nut_engaged_mm=d["nut_engaged_mm"], thread_past_nut_mm=d["thread_past_nut_mm"])
                rows.append(t)
    return rows


def apply():
    """Cut the torso shells and the upper-arm covers, add the cover mounts, update the candidate manifest."""
    sys.path.insert(0, str(ROOT / "robot/body/fastening"))
    from build123d import Compound, import_step
    t0 = time.time()
    man = json.loads(SC.CANDIDATE.read_text())      # the selected body (shoulder_clevis.py manifest); never written here
    srcm = json.loads(SC.MANIFEST.read_text())
    src = F.Body(srcm)
    SP_ = {p["name"]: p for p in srcm["parts"]}
    P = {p["name"]: p for p in man["parts"]}
    rep = dict(idle_task="T-tph7xb (parent T-ce88t9)", owner="M-s9tnc9: shells and cover redesigned once the U connectors passed",
               candidate=SC._rel(CANDIDATE), input_manifest=SC._rel(SC.CANDIDATE),
               input_manifest_sha256=SC._sha(SC.CANDIDATE), source_manifest=SC._rel(SC.MANIFEST),
               source_manifest_sha256=SC._sha(SC.MANIFEST), cad_manifest=SC._rel(SC.OUT / "cad-manifest.json"),
               method=dict(clearance_moving_mm=CLEARANCE, clearance_static_mm=STATIC_CLEARANCE, voxel_mm=PITCH, step_deg=STEP,
                           smoothing="gaussian sigma 0.6 voxel, marching cubes at 0.5", pad_keep_radius_mm=PAD_KEEP_R),
               source=dict(path=SC._rel(__file__), sha256=SC._sha(__file__)))
    # 1. mounts (R designed against the source cover's wells; L = mirror x)
    design = mount_design(src.mesh("R_upper_arm_hard_cover"))
    rep["mounts"] = design
    print("mounts:", json.dumps(design), flush=True)
    part, new_sl = build_mounts(design)
    cad = json.loads((SC.OUT / "cad-manifest.json").read_text())["parts"]
    old_sl = import_step(str(ROOT / cad[SC.S2 + "_steel_sleeves"]["native_step"])).solids()
    print(f"green U with mounts: {part.volume:.0f} mm3 (was {cad[SC.S2]['volume_mm3']}), valid {part.is_valid} ({time.time() - t0:.0f} s)", flush=True)
    for s, f in (("R_", lambda x: x), ("L_", SC.mirror_x)):
        nm = _side(SC.S2, s); r = _export(nm, [f(part)]); mat, rho = SC._density(nm)
        q = P[nm]; q.update(r, mass_kg=round(r["volume_mm3"] * rho * 1e-6, 6),
                            mass_basis=f"native STEP volume x {rho} g/cm3 ({mat}, part setting material)",
                            shoulder_shells="green U of the build + 2 cover-mount bosses (sleeve bore, captive M3 nut slot), shoulder_shells.py")
        q.pop("native_validity", None)
        nm2 = nm + "_steel_sleeves"
        sls = [f(x) for x in (list(old_sl) if s == "R_" else [])] + [f(x) for x in new_sl]
        if s == "L_":
            sls = [x for x in import_step(str(ROOT / cad[nm2]["native_step"])).solids()] + [SC.mirror_x(x) for x in new_sl]
        r = _export(nm2, sls)
        P[nm2].update(r, mass_kg=round(r["volume_mm3"] * 7.93e-6, 7), mass_basis="304 SS sleeves, native STEP volume x 7.93 g/cm3",
                      shoulder_shells=f"{len(sls) - 2} U sleeves of the build + 2 cover-mount sleeves (4.5 x 3.2), shoulder_shells.py")
    # 2. hardware rows (old names of the cover-post hardware)
    templ = dict(screw=SP_["R_upper_arm_shell_front_M3x8"], nut=SP_["R_upper_arm_shell_front_M3_nut"])
    hw = hardware_rows(design, templ)
    mine = {r["name"] for r in hw} | {f"{s}upper_arm_shell_{k}_{h}" for s in ("R_", "L_") for k in MOUNTS for h in ("M3x6", "M3x8", "M3_nut")}
    man["parts"] = [p for p in man["parts"] if p["name"] not in mine] + hw
    P = {p["name"]: p for p in man["parts"]}
    body = F.Body(man)
    print(f"hardware: {[r['name'] for r in hw]} ({time.time() - t0:.0f} s)", flush=True)
    # 3. covers: envelope swept on the right arm; the left arm gets its mirror (the U's, covers and joints are mirrors,
    #    so both covers come out as exact mirror parts instead of two voxelizations of the same motion)
    covers, torso = {}, {}
    cov = src.mesh("R_upper_arm_hard_cover")
    sw_b, sw_g = cover_sweep(body, "R_", (cov.bounds[0] - 1, cov.bounds[1] + 1))
    kept = _pad_keep(sw_g, design, "R_")
    blue_v, green_v = shell_volume_in(sw_b, cov), shell_volume_in(sw_g, cov)
    sw_b.mask |= sw_g.mask
    env = sw_b.surface()
    for s in ("R_", "L_"):
        cn = f"{s}upper_arm_hard_cover"
        c = src.mesh(cn)
        out = _subtract(c, env if s == "R_" else _mirror(env))
        o = openings(c, out)
        o.update(blue_swept_in_cover_mm3=round(blue_v, 1), green_static_in_cover_mm3=round(green_v, 1), pad_voxels_kept=kept,
                 envelope="swept on R" if s == "R_" else "mirror of the R envelope")
        covers[cn] = o
        _update_shell(P[cn], SP_[cn], c, out, cn,
                      f"shoulder U clearances: blue U swept over shoulder roll (+{CLEARANCE} mm) and green U static (+{STATIC_CLEARANCE} mm, pads kept)")
        print(f"{cn}: {o['before_mm3']} -> {o['after_mm3']} mm3, pieces {[(r['volume_mm3'], r['max_extent_mm']) for r in o['pieces']]} ({time.time() - t0:.0f} s)", flush=True)
    # 4. torso shells: the right blue U swept over shoulder pitch, its mirror for the left
    sh = {n: src.mesh(n) for n in TORSO_SHELLS}
    lo = np.array([55.0, max(-60.0, min(m.bounds[0][1] for m in sh.values()) - 1), 290.0])
    hi = np.array([125.0, max(m.bounds[1][1] for m in sh.values()) + 1, 410.0])
    sw = torso_sweep(body, "R_", (lo, hi))
    env = sw.surface()
    print(f"torso sweep done ({time.time() - t0:.0f} s)", flush=True)
    cur = {}
    for n in TORSO_SHELLS:
        torso[n] = dict(R_blue_swept_in_shell_mm3=round(shell_volume_in(sw, sh[n]), 1), L="mirror of the R envelope")
        cur[n] = _subtract(_subtract(sh[n], env), _mirror(env))
        torso[n].update(openings(sh[n], cur[n]))
        _update_shell(P[n], SP_[n], sh[n], cur[n], n,
                      f"shoulder U clearance: blue U (both arms) swept over shoulder pitch (+{CLEARANCE} mm)")
        print(f"{n}: {torso[n]['before_mm3']} -> {torso[n]['after_mm3']} mm3, pieces {[(r['volume_mm3'], r['max_extent_mm'], r['centre_mm']) for r in torso[n]['pieces']]}", flush=True)
    rep.update(covers=covers, torso_shells=torso, seconds=round(time.time() - t0, 1))
    man["shoulder_shells"] = dict(idle_task=rep["idle_task"], report=SC._rel(REPORT), source=rep["source"], method=rep["method"],
                                  changes=sorted([n for n in P if P[n].get("shoulder_shells")]),
                                  input_manifest=SC._rel(SC.CANDIDATE), input_manifest_sha256=SC._sha(SC.CANDIDATE),
                                  note="Candidate: the selected body + shoulder shell cuts + cover mounts. Reads the selected body, "
                                       "writes only its own files; rerun after shoulder_clevis.py manifest; idempotent.")
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=1, default=float) + "\n")
    man["study_id"] = man["result_id"] = "cove50-shoulder-u-shells"
    CANDIDATE.write_text(json.dumps(man, indent=1) + "\n")
    print(f"wrote {SC._rel(CANDIDATE)} and {SC._rel(REPORT)} ({time.time() - t0:.0f} s)")


def _mirror(m):
    """Mirror x -> -x (face order flipped so the normals stay outward)."""
    v = np.asarray(m.vertices, float).copy(); v[:, 0] *= -1.0
    return trimesh.Trimesh(v, np.asarray(m.faces)[:, ::-1], process=True)


def _update_shell(q, src_row, before, after, name, note):
    """Point a shell row at its cut mesh (the source row's mesh stays the input)."""
    r = _save_npz(name, after)
    keep = {k: src_row[k] for k in ("mass_kg",) if k in src_row}
    q.update(r, volume_mm3=round(float(after.volume), 2), mesh_note=(src_row.get("mesh_note", "") + "; " + note).lstrip("; "),
             shoulder_shells=dict(input_mesh=src_row["mesh"], input_mesh_sha256=src_row.get("mesh_sha256"),
                                  removed_mm3=round(float(before.volume - after.volume), 1)))
    if "mass_kg" in keep:
        q["mass_kg"] = round(keep["mass_kg"] * float(after.volume / before.volume), 6)
        q["mass_basis"] = (src_row.get("mass_basis", "source mass") + f"; x mesh volume ratio {after.volume / before.volume:.4f} after the shoulder cut")



# ---------------------------------------------------------------------------------------------- owner images
LIGHT = dict(shell=(0.86, 0.86, 0.88), cover=(0.93, 0.90, 0.82), nut=(0.85, 0.72, 0.30))


def _scene_rows(man, s="R_"):
    """(name, colour, link) of the right shoulder scene: torso shells and cover light, U's coloured."""
    names = {p["name"]: p for p in man["parts"]}
    rows = [(n, LIGHT["shell"]) for n in TORSO_SHELLS] + [(f"{s}upper_arm_hard_cover", LIGHT["cover"])]
    rows += [(_side(SC.S1, s), SC.COL["blue"]), (_side(SC.S2, s), SC.COL["green"]),
             (_side(SC.S1, s) + "_steel_sleeves", SC.COL["sleeve"]), (_side(SC.S2, s) + "_steel_sleeves", SC.COL["sleeve"]),
             (f"upper__{s}roll_C047_case", SC.COL["case"]), (f"upper__{s}roll_C047_main_horn", SC.COL["horn"]),
             (_side(SC.IDLER, s), SC.COL["idler"]), (f"upper__{s}pitch_C047_main_horn", SC.COL["horn"]),
             (f"upper__{s}yaw_C047_case", SC.COL["pitchcase"])]
    rows += [(n, SC.COL["screw"]) for n in names if n.startswith(f"{s}shoulder_u_")]
    rows += [(n, LIGHT["nut"] if n.endswith("_nut") else SC.COL["screw"]) for n in names if n.startswith(f"{s}upper_arm_shell_")]
    return [(n, c) for n, c in rows if n in names and names[n].get("mesh")]


def _posed(body, rows, q):
    out = []
    for n, c in rows:
        m = body.mesh(n).copy(); m.apply_transform(body.link_T(body.parts[n]["owning_link"], q)); out.append((m, c))
    return out


def views():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    man = json.loads(CANDIDATE.read_text()); body = F.Body(man)
    old = json.loads(SC.CANDIDATE.read_text()); ob = F.Body(old)     # before = the selected body (input of apply)
    rows = _scene_rows(man)
    legend = [("torso shells", LIGHT["shell"]), ("upper-arm cover", LIGHT["cover"]), ("blue U (pitch link)", SC.COL["blue"]),
              ("green U (roll link)", SC.COL["green"]), ("roll servo case", SC.COL["case"]), ("horns", SC.COL["horn"]),
              ("roll idler", SC.COL["idler"]), ("yaw servo case", SC.COL["pitchcase"]), ("sleeves / screws", SC.COL["sleeve"]),
              ("cover-mount nuts", LIGHT["nut"])]
    lo, hi = np.array([62.0, -52.0, 296.0]), np.array([166.0, 60.0, 396.0])
    VIEWS.mkdir(parents=True, exist_ok=True)
    # (a) before / after, 4 views
    fig, axs = plt.subplots(2, 4, figsize=(28, 14.5), dpi=80)
    for r, (lab, b, rr) in enumerate((("BEFORE", ob, _scene_rows(old)), ("AFTER", body, rows))):
        for c, (t, d, up, xl, yl) in enumerate(SC.VIEWS4):
            img, ext = SC._render(_posed(b, rr, {}), d, up, lo, hi, px_mm=0.16)
            ax = axs[r, c]; ax.imshow(img, extent=ext, interpolation="bilinear"); ax.set_title(f"{lab}: {t}", fontsize=12)
            if xl:
                ax.set_xlabel(xl); ax.set_ylabel(yl); ax.tick_params(labelsize=8)
            else:
                ax.axis("off")
    fig.legend(handles=[Patch(facecolor=c, edgecolor="#444444", label=l) for l, c in legend], loc="lower center", ncol=5, fontsize=12, frameon=False)
    fig.suptitle("Right shoulder at standing: torso shells and upper-arm cover before / after the U clearance cuts", fontsize=14)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97)); f1 = VIEWS / "shoulder-shells-before-after.png"; fig.savefig(f1, facecolor="white"); plt.close(fig)
    print(f"wrote {SC._rel(f1)}")
    # (b) range poses
    poses = [("pitch -60", {"R_shoulder_pitch": -60.0}), ("pitch +150", {"R_shoulder_pitch": 150.0}),
             ("roll -15", {"R_shoulder_roll": -15.0}), ("roll +95", {"R_shoulder_roll": 95.0})]
    vv = [("3/4 (front, outside, above)", (-0.75, 0.9, -0.55), (0, 0, 1)), ("FRONT (-y)", (0, 1, 0), (0, 0, 1)),
          ("SIDE (+x)", (-1, 0, 0), (0, 0, 1))]
    lo2, hi2 = np.array([55.0, -75.0, 270.0]), np.array([200.0, 80.0, 430.0])
    fig, axs = plt.subplots(len(vv), len(poses), figsize=(28, 21), dpi=80)
    for c, (pl, q) in enumerate(poses):
        sc = _posed(body, rows, q)
        for r, (t, d, up) in enumerate(vv):
            img, ext = SC._render(sc, d, up, lo2, hi2, px_mm=0.2)
            ax = axs[r, c]; ax.imshow(img, extent=ext, interpolation="bilinear"); ax.set_title(f"{pl}: {t}", fontsize=12); ax.axis("off")
    fig.legend(handles=[Patch(facecolor=c, edgecolor="#444444", label=l) for l, c in legend], loc="lower center", ncol=5, fontsize=12, frameon=False)
    fig.suptitle("Right shoulder at the range ends (after): the openings the U's need", fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97)); f2 = VIEWS / "shoulder-shells-range-poses.png"; fig.savefig(f2, facecolor="white"); plt.close(fig)
    print(f"wrote {SC._rel(f2)}")
    # (c) the cover mount: green U with bosses, sleeves, nuts, screws (cover off), and sections through both mount axes
    rep = json.loads(REPORT.read_text()); dsg = rep["mounts"]
    mrows = [(n, c) for n, c in rows if n not in TORSO_SHELLS and not n.endswith("hard_cover")]
    fig, axs = plt.subplots(2, 3, figsize=(24, 16), dpi=80)
    lo3, hi3 = np.array([96.0, -30.0, 312.0]), np.array([144.0, 38.0, 352.0])
    for c, (t, d, up) in enumerate((("FRONT (-y), cover off", (0, 1, 0), (0, 0, 1)), ("BACK (+y), cover off", (0, -1, 0), (0, 0, 1)),
                                    ("3/4 from front-above, cover off", (-0.6, 0.8, -0.6), (0, 0, 1)))):
        img, ext = SC._render(_posed(body, mrows, {}), d, up, lo3, hi3, px_mm=0.06)
        axs[0, c].imshow(img, extent=ext, interpolation="bilinear"); axs[0, c].set_title(t, fontsize=12); axs[0, c].axis("off")
    secs = [n for n, _ in rows if n in body.parts]
    colr = {n: c for n, c in rows}
    for c, (k, plane) in enumerate((("front", "x"), ("rear", "x"), ("front", "y"))):
        d = dsg[k]; (x, z) = d["axis_xz"]
        ax = axs[1, c]
        o, nrm, ij = ((x, 0, 0), (1, 0, 0), (1, 2)) if plane == "x" else ((0, float(np.mean(d["nut_y"])), 0), (0, 1, 0), (0, 2))
        for n in secs:
            sct = body.mesh(n).section(plane_origin=o, plane_normal=nrm)
            if sct is None:
                continue
            from shapely.geometry import Polygon as SPoly
            geo = None
            for e in sct.discrete:
                if len(e) < 4:
                    continue
                pg = SPoly(e[:, list(ij)]).buffer(0)
                geo = pg if geo is None else geo.symmetric_difference(pg)      # even-odd fill
            for poly in ([] if geo is None else getattr(geo, "geoms", [geo])):
                if poly.geom_type != "Polygon":
                    continue
                g = np.array(poly.exterior.coords)
                ax.fill(g[:, 0], g[:, 1], color=colr[n], ec="#333333", lw=0.6)
                for h in poly.interiors:
                    hh = np.array(h.coords); ax.fill(hh[:, 0], hh[:, 1], color="white", ec="#333333", lw=0.6)
        cx = (d["axis_xz"][1] if plane == "x" else d["axis_xz"][0])
        cy0 = (np.mean(d["sleeve_y"]) if plane == "x" else z)
        if plane == "x":
            ax.set_xlim(min(d["seat_y"], d["boss_back_y"]) - 8, max(d["seat_y"], d["boss_back_y"]) + 8); ax.set_ylim(z - 12, z + 12)
            ax.set_xlabel("y mm"); ax.set_ylabel("z mm")
        else:
            ax.set_xlim(x - 14, x + 14); ax.set_ylim(z - 12, z + 12); ax.set_xlabel("x mm"); ax.set_ylabel("z mm")
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_title(f"{k} mount, section {'x' if plane == 'x' else 'y'} = {o[0] if plane == 'x' else round(o[1], 2)}: "
                     f"cover pad {d['cover_pad_mm']} mm, sleeve {d['sleeve_len']} mm, nut slot open to {'+z' if not MOUNTS[k].get('slot', (0, 1))[0] else '+z tilted 15 deg to +x'}, {d['screw']}", fontsize=11)
    fig.legend(handles=[Patch(facecolor=c, edgecolor="#444444", label=l) for l, c in legend], loc="lower center", ncol=5, fontsize=12, frameon=False)
    fig.suptitle("Cover mount on the green U: boss + pressed 304 sleeve (4.5 x 3.2) + captive M3 nut; screw from outside through the cover's well", fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97)); f3 = VIEWS / "shoulder-shells-cover-mount.png"; fig.savefig(f3, facecolor="white"); plt.close(fig)
    print(f"wrote {SC._rel(f3)}")


if __name__ == "__main__":
    {"measure": measure, "apply": apply, "views": views}[sys.argv[1] if len(sys.argv) > 1 else "apply"]()
