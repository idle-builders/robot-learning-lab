"""Shared kit for the hip-shell studio (Idle T-czv2d8): one base body, one render, one interference measure.

Every concept is a list of Concept parts: a closed mesh in the body's neutral world frame (mm; z up, -y is forward,
+x is the robot's right side), owned by one link, so it moves with that link. A concept may also hide or replace
existing shells (e.g. the lower belly of center_front_shell).

  import kit
  body = kit.base()
  parts = [kit.Concept("R_hip_cap", mesh, "R_pitch_link", (0.86, 0.83, 0.76))]
  kit.render(body, parts, "out.png", hide=("R_thigh_front_hard_cover",), poses=kit.LOOK_POSES)
  card = kit.interference(body, parts)        # contact-free hip ranges and worst overlaps, native units

Geometry facts (right side; the left mirrors x):
  hip yaw axis   vertical through (59, 2), range in assembly -30..30 (target -40..40)
  hip pitch axis along x through (y 2, z 219); negative = flexion (thigh forward/up); target -90..+40
  hip roll axis  along y through (x 56, z 190), 29 mm below the pitch axis; negative = abduction; target -45..+25
  yaw and pitch axes meet at C = (59, 2, 219). Links: torso -> R_yaw_link -> R_pitch_link -> R_roll_link (thigh)
  -> R_shin. Standing pose: hip roll -2.68, ankle roll +2.68 (splay).
Run everything under memguard with at most 4 GB: the kit decimates and only loads the hip region.
"""
from __future__ import annotations

import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "robot/body/motor-refit"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "robot/body"))
from checks import bodylib as F  # noqa: E402  (body model: Body, load_mesh, src, ...)

BASE = ROOT / "artifacts/robot/body/ape-proportions/assembly-parts.json"  # uncut shells, reviewed (T-gqwdrn, T-xdcjn9)
SKETCH = ROOT / "robot/shell/sketch/reference/sketch.png"
SKIP = {"cut_material", "adhesive_allocation", "authored_harness"}
HIP_C = np.array([59.0, 2.0, 219.0])
PITCH_AXIS = (np.array([0.0, 2.0, 219.0]), np.array([1.0, 0.0, 0.0]))
ROLL_AXIS = (np.array([56.0, 0.0, 190.0]), np.array([0.0, 1.0, 0.0]))
YAW_AXIS = (np.array([59.0, 2.0, 0.0]), np.array([0.0, 0.0, 1.0]))
LOOK_POSES = [("standing", {}),
              ("hip -90 flexion", {"R_hip_pitch": -90.0, "R_knee": 90.0}),
              ("hip -45, abducted -30, yaw 30", {"R_hip_pitch": -45.0, "R_hip_roll": -30.0, "R_hip_yaw": 30.0, "R_knee": 40.0}),
              ("hip +40 extension", {"R_hip_pitch": 40.0})]
GRID = dict(pitch=np.arange(-90.0, 40.1, 5.0), roll=(-45.0, -30.0, -15.0, 0.0, 15.0, 25.0), yaw=(-40.0, -20.0, 0.0, 20.0, 40.0))
HIP_REGION_Z = (110.0, 330.0)


@dataclass
class Concept:
    name: str
    mesh: trimesh.Trimesh          # closed, neutral world frame
    link: str                      # owning link (moves with it)
    color: tuple = (0.88, 0.85, 0.78)
    note: str = ""


def base(path=BASE):
    man = json.load(open(path))
    b = F.Body(man)
    b.standing = man.get("standing_pose_deg", {})
    return b


def mirror_x(m):
    v = m.vertices.copy(); v[:, 0] *= -1
    return trimesh.Trimesh(v, m.faces[:, ::-1], process=True)


def mf(m):
    import manifold3d as M
    return M.Manifold(M.Mesh(vert_properties=np.asarray(m.vertices, np.float32), tri_verts=np.asarray(m.faces, np.uint32)))


def tm(x):
    r = x.to_mesh()
    return trimesh.Trimesh(np.asarray(r.vert_properties)[:, :3], np.asarray(r.tri_verts), process=True)


def revolve(profile_rz, axis_point, axis_dir, sections=96, angle=(0.0, 360.0)):
    """Closed surface of revolution of a closed 2D profile [(r, z_along_axis), ...] about an axis (degrees range).
    The result is invariant under rotation about that axis when angle is the full turn."""
    import manifold3d as M
    prof = [list(map(float, p)) for p in profile_rz]
    a0, a1 = angle
    solid = M.Manifold.revolve(M.CrossSection([prof]), sections, a1 - a0)  # about +z, profile in (x=r, y=z)
    if a0:
        solid = solid.rotate((0.0, 0.0, a0))
    m = tm(solid)
    # map +z to axis_dir, origin to axis_point
    z = np.array([0.0, 0.0, 1.0]); d = np.asarray(axis_dir, float) / np.linalg.norm(axis_dir)
    R = trimesh.geometry.align_vectors(z, d)
    m.apply_transform(R); m.apply_translation(axis_point)
    return m


def shell_of(outer, thickness):
    """A closed shell 'thickness' mm thick inside a closed outer solid (inward scaling about the centroid is crude;
    use for block models only)."""
    c = outer.centroid; e = np.ptp(outer.vertices, axis=0)
    inner = outer.copy(); inner.vertices = c + (outer.vertices - c) * (1 - 2 * thickness / e.min())
    return tm(mf(outer) - mf(inner))


def _hip_parts(body, hide=()):
    out = {}
    for n, p in body.parts.items():
        if n in hide or p.get("role") in SKIP or F.is_fastener(n, p):
            continue
        b = np.array(p["bounds_mm"])
        if b[1][2] < HIP_REGION_Z[0] - 60 or b[0][2] > HIP_REGION_Z[1] + 80:
            continue
        out[n] = p
    return out


_MESH = {}


CACHE = ROOT / ".cache/work/hip-shell/meshes"


_TAGS: dict = {}


def mesh_tag(p):
    """First 12 hex of the sha256 of the part's mesh file as it is on disk (by path, size, mtime in this process). The
    manifest's recorded mesh_sha256 goes stale when a mesh is rebuilt in place, and a cache keyed by it then serves the old
    geometry (found 2026-11-22: the owner six-side drawing drew a moved part where it had been)."""
    import hashlib
    try:
        f = F.src(p["mesh"])
        st = f.stat()
        k = (str(f), st.st_size, st.st_mtime_ns)
        if k not in _TAGS:
            _TAGS[k] = hashlib.sha256(f.read_bytes()).hexdigest()[:12]
        return _TAGS[k]
    except (OSError, KeyError, TypeError):
        return (p.get("mesh_sha256") or p.get("native_step_sha256") or "nohash")[:12]


def _mesh(body, n, faces=2500):
    """Decimated part mesh, cached on disk by the sha256 of the part's mesh file (decimating the big shells costs GBs)."""
    key = (n, faces)
    if key not in _MESH:
        p = body.parts[n]
        tag = mesh_tag(p)
        f = CACHE / f"{n}-{faces}-{tag}.npz"
        if f.exists():
            d = np.load(f); m = trimesh.Trimesh(d["v"], d["f"], process=False)
        else:
            m = body.mesh(n)
            if len(m.faces) > faces:
                m = m.simplify_quadric_decimation(face_count=faces)
            CACHE.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(f, v=np.asarray(m.vertices), f=np.asarray(m.faces))
        _MESH[key] = m
    return _MESH[key]


def _rot(yaw, pitch):
    a, b = np.radians(yaw), np.radians(pitch)
    return np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]]) @ \
        np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def _draw(ax, items, yaw, title, xlim, zlim):
    from matplotlib.collections import PolyCollection
    R = _rot(yaw, 8); L = np.array([0.4, -0.7, 0.6]); L /= np.linalg.norm(L); P, C, D = [], [], []
    for m, T, c in items:
        v = (m.vertices @ T[:3, :3].T + T[:3, 3]) @ R.T; nn = m.face_normals @ T[:3, :3].T
        tri = v[m.faces]; vis = (nn @ R.T)[:, 1] < 0.05; sh = 0.35 + 0.65 * np.clip(np.abs(nn @ L), 0, 1)
        P.extend(tri[vis][:, :, [0, 2]]); D.extend(tri[vis][:, :, 1].mean(1)); C.extend(np.clip(np.outer(sh[vis], c), 0, 1))
    o = np.argsort(-np.array(D))
    ax.add_collection(PolyCollection([P[i] for i in o], facecolors=[C[i] for i in o], edgecolors="none"))
    ax.set_xlim(*xlim); ax.set_ylim(*zlim); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=11)


def render(body, concepts, out, hide=(), poses=None, views=((0, "front"), (-35, "3/4"), (90, "side"), (180, "back")),
           title="", sketch=True, full=False, mirror_left=True):
    """Rows = poses, columns = views. Concept parts named R_* are mirrored to the left side unless given."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    poses = poses or LOOK_POSES
    cs = list(concepts)
    if mirror_left:
        have = {c.name for c in cs}
        for c in list(cs):
            if c.name.startswith("R_") and "L_" + c.name[2:] not in have:
                cs.append(Concept("L_" + c.name[2:], mirror_x(c.mesh), "L_" + c.link[2:], c.color, c.note))
    parts = body.parts if full else _hip_parts(body, hide)
    xlim, zlim = ((-200, 200), (-15, 510)) if full else ((-140, 140), (90, 360))
    ncol = len(views) + (1 if sketch else 0)
    fig, axs = plt.subplots(len(poses), ncol, figsize=(5.2 * ncol, 5.6 * len(poses)), squeeze=False)
    img = plt.imread(SKETCH) if sketch else None
    for r, (lab, dq) in enumerate(poses):
        q = {**body.standing, **dq}
        items = []
        for n, p in parts.items():
            if n in hide:
                continue
            col = np.array(p.get("color") or [180, 180, 180, 255], float)[:3] / 255
            items.append((_mesh(body, n), body.link_T(p["owning_link"], q), col))
        for c in cs:
            items.append((c.mesh, body.link_T(c.link, q), np.array(c.color)))
        k = 0
        if sketch:
            crop = img[300:620, 120:560] if r == 0 else img[300:620, 1150:1500] if r == 1 else img[300:620, 650:1120]
            axs[r, 0].imshow(crop); axs[r, 0].axis("off"); axs[r, 0].set_title("sketch", fontsize=11); k = 1
        for c, (yaw, v) in enumerate(views):
            _draw(axs[r, c + k], items, yaw, f"{lab} - {v}", xlim, zlim)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.97)); plt.savefig(out, dpi=60); plt.close(fig)


def interference(body, concepts, hide=(), grid=None, side="R", pairs_out=8):
    """Overlap volume (mm3) of every concept part against every part on another link (and the other concepts),
    over the hip pose grid; returns contact-free hip pitch/roll/yaw ranges walking out from 0 (others held at 0)
    and the worst combined poses. Continuous values, no verdicts."""
    import manifold3d as M
    grid = grid or GRID
    others = {n: p for n, p in _hip_parts(body, hide).items() if n not in hide}
    cm = {c.name: (mf(c.mesh), c.link) for c in concepts if c.name.startswith(side + "_") or not c.name[:2] in ("R_", "L_")}
    om = {}
    for n, p in others.items():
        m = _mesh(body, n, 1500)
        if not m.is_watertight:
            m = m.convex_hull
        om[n] = (mf(m), p["owning_link"])

    def overlap(q):
        res = {}
        posed = {}
        for n, (g, lk) in list(cm.items()) + list(om.items()):
            posed[n] = (g.transform(body.link_T(lk, q)[:3, :4]), lk)
        for a, (ga, la) in posed.items():
            if a not in cm:
                continue
            for b, (gb, lb) in posed.items():
                if b == a or lb == la or (b in cm and b < a):
                    continue
                v = (ga ^ gb).volume()
                if v > 0.5:
                    res[(a, b)] = v
        return res

    base0 = overlap(body.standing)

    def growth(q):
        o = overlap({**body.standing, **q})
        return {k: v for k, v in o.items() if v > base0.get(k, 0.0) + 0.5}

    ranges = {}
    for j, vals in (("pitch", grid["pitch"]), ("roll", grid["roll"]), ("yaw", grid["yaw"])):
        name = f"{side}_hip_{j}"; free = [0.0, 0.0]; first = {}
        for sgn, k in ((-1, 0), (1, 1)):
            for v in sorted([x for x in vals if sgn * x > 0], key=abs):
                g = growth({name: float(v)})
                if g:
                    first[k] = (float(v), sorted(g.items(), key=lambda t: -t[1])[:3]); break
                free[k] = float(v)
        ranges[j] = dict(contact_free_deg=free, first_contact={str(k): [v[0], [(f"{a} x {b}", round(o, 1)) for (a, b), o in v[1]]]
                                                               for k, v in first.items()})
    worst = []
    for p_, r_, y_ in itertools.product(grid["pitch"][::2], grid["roll"], grid["yaw"]):
        g = growth({f"{side}_hip_pitch": float(p_), f"{side}_hip_roll": float(r_), f"{side}_hip_yaw": float(y_)})
        if g:
            (a, b), v = max(g.items(), key=lambda t: t[1])
            worst.append(((float(p_), float(r_), float(y_)), f"{a} x {b}", round(v, 1), round(sum(g.values()), 1)))
    worst.sort(key=lambda t: -t[3])
    combined = len(grid["pitch"][::2]) * len(grid["roll"]) * len(grid["yaw"])
    return dict(neutral_overlap_mm3={f"{a} x {b}": round(v, 1) for (a, b), v in base0.items()}, ranges=ranges,
                combined_poses=combined, combined_poses_with_contact=len(worst),
                worst_combined=[dict(pose_pitch_roll_yaw=w[0], pair=w[1], max_mm3=w[2], total_mm3=w[3]) for w in worst[:pairs_out]])
