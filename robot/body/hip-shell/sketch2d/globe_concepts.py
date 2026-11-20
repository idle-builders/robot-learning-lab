"""S3 sketch round on the concurrent hip (P-04, D-08): hip-globe concepts as nested shell layers about C, shaped
from the measured free space, scored over the whole-leg poses and drawn on the whole robot (Idle T-czv2d8).

Internals (A-20, "tight" P-04 brackets): the roll-pitch servo block (J2); the yaw U-bracket as a low arch over the
block within r 59 of C (J1); the thigh fork arms (x 31-37 and 86-91) and a crossbar at z 150-155 (J3). The yaw
servo, battery keel and cells, cradle plate and the left leg (standing) are torso keep-outs. The hanging arm is
scored at standing only (A-19).

Method (A-21): with all three axes through C, every layer is a surface about C owned by one link: a sphere, or a
slim "egg" (surface of revolution about the pitch axis, half-width W, superellipse exponent P). A layer is its
link's free space at that radius (no other link's material reaches the 2.5 mm shell + 1.5 mm clearance in any of
the 151 realistic poses), with a smoothed edge kept inside the free space. Layers sit 4 mm apart and slide over one
another. The static pelvis hood follows the envelope the moving layers sweep (+4 mm); the torso shell gets a raised
hem wherever the moving layers sweep its walls.
Concepts: K1 ball (spheres: roll-link tread inside, thigh globe outside), K2 slim egg (same order, slim), K3 saddle
(slim; roll-link layer outside the thigh egg). K2 is also shown with the hood cut to the torso width and with no hood.

Scores (no pass/fail): "seen into the hip" = share of directions from C not blocked by any layer, the torso wall
above its hem, the thigh shell or the standing left leg, as a distribution over the poses; hip width (all pieces,
and the moving ones at standing), depth, hem height, arm gap at standing, and the share of each piece that the A-12
range combinations would hit.

Run: python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B robot/body/hip-shell/sketch2d/globe_concepts.py
     --final: redo only the final variants from globe-concepts-search.json; --render: redraw the sheet
"""
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import hip_layouts as H  # noqa: E402

C = H.C
STEP = 2.0
DB = 2.0                      # direction bin (deg)
DR = 1.0                      # radius bin (mm)
RMAX = 120.0
SHELL, CLEAR = 2.5, 1.5
NAZ, NEL, NR = int(360 / DB), int(180 / DB), int(RMAX / DR)
WORK = H.kit.ROOT / ".cache/work/hip-shell"
ARM = [("right arm, hanging (upper arm lower cover)", (103, -27, 223), (159, 31, 293)),
       ("right forearm, hanging", (107, -29, 141), (163, 29, 216))]
THIGH_SHELL = ((11, -25, 124), (88, 38, 207))     # today's thigh covers: the thigh body continues below the globe


# ------------------------------------------------------------------ internals (A-20)
def _ring_mesh(r0, r1, xw, psi0, psi1):
    parts = []
    a = psi0
    while a < psi1:
        b = min(a + 30.0, psi1)
        th = np.radians(np.linspace(a, b, 6))
        pts = [C + np.array([x, r * np.sin(t), -r * np.cos(t)]) for r in (r0, r1) for x in (-xw, xw) for t in th]
        parts.append(trimesh.convex.convex_hull(np.array(pts)))
        a = b
    return trimesh.util.concatenate(parts)


def _spoke(x0, x1, r=10.0):
    m = trimesh.creation.cylinder(radius=r, height=x1 - x0, sections=24)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    m.apply_translation([(x0 + x1) / 2, C[1], C[2]])
    return m


def internals():
    top = 229.0
    lay = H.op3_style()
    keep = {n: (m, l) for n, m, l in lay.parts}
    # yaw U-bracket as a low arch (A-20): front arm to the idler on the pitch servo's front face, a bar over the
    # block at z 238-243 (clears the block's top corners under roll -45..25 by >= 3 mm), a chamfered rear corner
    # and the rear arm behind the roll horn; every point within r 59 of C
    rear = trimesh.convex.convex_hull(np.array([[x, y, z] for x in (46, 72) for y, z in
                                                ((46, 238), (46, 243), (55.5, 234), (51.5, 234), (51.5, 238))]))
    parts = [("yaw servo", keep["yaw servo"][0], "torso"),
             ("yaw neck", H.bx((52, -5, 243), (66, 9, 271)), "J1"),
             ("yaw U front arm", H.bx((46, -17, 210), (72, -12, 243)), "J1"),
             ("yaw U bar", H.bx((46, -17, 238), (72, 46, 243)), "J1"),
             ("yaw U rear corner", rear, "J1"),
             ("yaw U rear arm", H.bx((46, 51.5, 210), (72, 55.5, 234)), "J1"),
             ("roll horn", keep["roll horn"][0], "J1"),
             ("roll servo", keep["roll servo"][0], "J2"),
             ("pitch servo", keep["pitch servo"][0], "J2"),
             ("pitch horn", keep["pitch horn"][0], "J3"),
             # A-22: the thigh shell is the fork. Two hub spokes on the pitch axis (horn outboard, idler inboard)
             # join the thigh layer at its sides; the layer carries the load round the roll-link layer to the thigh
             ("thigh hub spoke outboard", _spoke(81.0, 100.0), "J3"),
             ("thigh hub spoke inboard", _spoke(18.0, 37.0), "J3"),
             ("knee servo", keep["knee servo"][0], "J3"),
             ("shin", keep["shin"][0], "shin")]
    parts += [(n, H.bx(lo, hi), "torso") for n, lo, hi in H.TORSO_KEEP]
    for n, m, l in list(parts):
        if l not in ("torso",):
            mm = m.copy(); mm.vertices[:, 0] *= -1; mm.faces = mm.faces[:, ::-1]
            parts.append(("left " + n, mm, "torso"))
    return lay, parts


def vol_points(m):
    lo, hi = m.bounds
    g = np.mgrid[lo[0]:hi[0] + 1e-6:STEP, lo[1]:hi[1] + 1e-6:STEP, lo[2]:hi[2] + 1e-6:STEP].reshape(3, -1).T
    inside = m.contains(g) if len(g) else np.zeros(0, bool)
    return np.vstack([g[inside], m.vertices])


def sph(Q):
    v = Q - C
    r = np.linalg.norm(v, axis=1)
    az = np.degrees(np.arctan2(v[:, 0], -v[:, 1]))
    el = np.degrees(np.arcsin(np.clip(v[:, 2] / np.maximum(r, 1e-9), -1, 1)))
    return r, az, el


def bins(az, el):
    return ((az + 180) / DB).astype(int) % NAZ, np.clip(((el + 90) / DB).astype(int), 0, NEL - 1)


FRAMES = ("torso", "J1", "J2", "J3")


def lk(l):
    return "J3" if l in ("thigh",) else l


def occupancy(lay, parts, poses):
    """occ[f][az, el, r]: other links' material in frame f, over the given poses."""
    pts = [(n, vol_points(m), l) for n, m, l in parts]
    occ = {f: np.zeros((NAZ, NEL, NR), bool) for f in FRAMES}
    for fam, p in poses:
        q = dict(zip(("pitch", "roll", "yaw", "knee"), p))
        Ts = H.link_T(lay, q)
        for n, P, l in pts:
            W = P @ Ts[l][:3, :3].T + Ts[l][:3, 3]
            for f in FRAMES:          # own-link parts count too: a layer may not pass through its own servo or fork
                if lk(l) == f and "spoke" in n:      # the hub spokes join their own shell (A-22)
                    continue
                Ti = np.linalg.inv(Ts[f])
                r, az, el = sph(W @ Ti[:3, :3].T + Ti[:3, 3])
                k = r < RMAX - 1
                ia, ie = bins(az[k], el[k])
                occ[f][ia, ie, (r[k] / DR).astype(int)] = True
    # dilate one bin in direction (sampling)
    for f in FRAMES:
        o = occ[f]
        occ[f] = o | np.roll(o, 1, 0) | np.roll(o, -1, 0) | np.pad(o, ((0, 0), (1, 0), (0, 0)))[:, :-1] | np.pad(o, ((0, 0), (0, 1), (0, 0)))[:, 1:]
    return occ


# ------------------------------------------------------------------ pieces (clean seams)
@dataclass
class Piece:
    name: str
    owner: str               # torso, J1, J2, J3
    R: float                 # rho of the outer surface at the middle (sphere radius when W = 0)
    W: float = 0.0           # half-width along the pitch axis x (0 = sphere)
    P: float = 2.0           # superellipse exponent of the profile
    holes: tuple = ()        # circular openings: (azimuth, elevation, half-angle) in the owner frame (deg)
    band: float = 0.0        # if > 0: keep only |x - Cx| <= band (a tread around the pitch axis)
    psi: tuple = ()          # if set: keep only angles about x (0 = down, +90 = back) in these (lo, hi) ranges
    slot: tuple = ()         # (b, lo, hi): remove |x - Cx| < b where the angle about x is outside lo..hi
    color: tuple = (0.9, 0.87, 0.8)
    grid: np.ndarray = None
    mask: np.ndarray = None

    @property
    def pts(self):
        return self.grid[self.mask]


def unit(az, el):
    a, e = np.radians(az), np.radians(el)
    return np.array([np.cos(e) * np.sin(a), -np.cos(e) * np.cos(a), np.sin(e)])


def surface(pc, n_u=141, n_t=241):
    W = pc.W if pc.W else pc.R
    u = W * np.sin(np.linspace(-np.pi / 2, np.pi / 2, n_u))          # denser near the poles of x
    rho = pc.R * np.clip(1 - np.abs(u / W) ** pc.P, 0, 1) ** (1 / pc.P)
    t = np.radians(np.linspace(-180, 180, n_t))
    X = np.broadcast_to(u[:, None], (n_u, n_t))
    return np.stack([X, rho[:, None] * np.sin(t)[None, :], -rho[:, None] * np.cos(t)[None, :]], -1) + C, u, t


def build(pc):
    G, u, t = surface(pc)
    d = G - C
    d = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
    keep = np.ones(G.shape[:2], bool)
    for az, el, h in pc.holes:
        keep &= d @ unit(az, el) < np.cos(np.radians(h))
    if pc.band:
        keep &= np.abs(u)[:, None] <= pc.band
    if pc.psi:
        ps = np.degrees(t)[None, :] * np.ones((len(u), 1))
        k = np.zeros_like(keep)
        for lo, hi in pc.psi:
            k |= (ps >= lo) & (ps <= hi)
        keep &= k
    if pc.slot:
        b, lo, hi = pc.slot
        ps = np.degrees(t)[None, :] * np.ones((len(u), 1))
        keep &= ~((np.abs(u)[:, None] < b) & ((ps < lo) | (ps > hi)))
    pc.grid, pc.mask = G, keep
    return pc


def hits(pc, occ):
    """Share of the piece's area that another link's material reaches (shell R-2.5-1.5 .. R+1.5) in some pose."""
    P = pc.pts
    r, az, el = sph(P)
    ia, ie = bins(az, el)
    lo = np.clip(((r - SHELL - CLEAR) / DR).astype(int), 0, NR - 1)
    hi = np.clip(((r + CLEAR) / DR).astype(int) + 1, 0, NR)
    cum = np.concatenate([np.zeros((NAZ, NEL, 1), np.int32), np.cumsum(occ[pc.owner], axis=2, dtype=np.int32)], axis=2)
    h = (cum[ia, ie, hi] - cum[ia, ie, lo]) > 0
    return float(h.mean()), P[h]


def mesh_of(pc):
    G, k = pc.grid, pc.mask
    nu, nt = k.shape
    idx = np.arange(nu * nt).reshape(nu, nt)
    q = k[:-1, :-1] & k[1:, :-1] & k[1:, 1:] & k[:-1, 1:]
    a, b = np.nonzero(q)
    f = np.concatenate([np.stack([idx[a, b], idx[a + 1, b], idx[a + 1, b + 1]], 1),
                        np.stack([idx[a, b], idx[a + 1, b + 1], idx[a, b + 1]], 1)])
    return trimesh.Trimesh(G.reshape(-1, 3), f, process=False)


IVORY, WARM, GREY = (0.93, 0.90, 0.83), (0.80, 0.74, 0.64), (0.80, 0.78, 0.74)


def blocked_grid(pc, occ, r_off=None):
    """Per surface grid point: does another link's material reach the shell there in some pose?"""
    G = pc.grid
    flat = G.reshape(-1, 3)
    r, az, el = sph(flat)
    ia, ie = bins(az, el)
    lo = np.clip(((r - SHELL - CLEAR) / DR).astype(int), 0, NR - 1)
    hi = np.clip(((r + CLEAR) / DR).astype(int) + 1, 0, NR)
    cum = np.concatenate([np.zeros((NAZ, NEL, 1), np.int32), np.cumsum(occ[pc.owner], axis=2, dtype=np.int32)], axis=2)
    return ((cum[ia, ie, hi] - cum[ia, ie, lo]) > 0).reshape(G.shape[:2])


def free_piece(pc, occ, sigma=4.0, erode=2, min_cells=400):
    """Shape a piece from its free space: free = intent minus blocked; the boundary is smoothed (Gaussian on the
    surface grid, wrapped around the pitch axis) and kept inside the eroded free set, so seams are continuous curves
    and the piece clears every realistic pose."""
    from scipy.ndimage import gaussian_filter, binary_erosion, label
    build(pc)
    intent = pc.mask.copy()
    free = intent & ~blocked_grid(pc, occ)
    sm = gaussian_filter(free.astype(float), sigma=(sigma, sigma * 1.5), mode=("nearest", "wrap")) > 0.5
    core = binary_erosion(free, iterations=erode, border_value=1)
    keep = sm & core
    lab, n = label(keep)
    if n:
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        keep = np.isin(lab, np.nonzero(sizes >= min_cells)[0])
    pc.mask = keep
    pc.kept = round(float(keep.sum()) / max(1.0, float(intent.sum())), 3)
    return pc


def envelope_hood(name, inner, lay, poses, gap=4.0, sigma=3.0):
    """Static pelvis hood: in every torso-frame direction, the largest radius the moving pieces reach over the poses
    plus `gap`, smoothed; drawn as a sphere-like surface about C with that radius map."""
    from scipy.ndimage import gaussian_filter, maximum_filter
    env = np.zeros((NAZ, NEL))
    for fam, p in poses:
        q = dict(zip(("pitch", "roll", "yaw", "knee"), p))
        Ts = H.link_T(lay, q)
        for pc in inner:
            P = pc.pts[::2]
            W = P @ Ts[pc.owner][:3, :3].T + Ts[pc.owner][:3, 3]
            r, az, el = sph(W)
            ia, ie = bins(az, el)
            np.maximum.at(env, (ia, ie), r)
    env = maximum_filter(env, size=5, mode=("wrap", "nearest"))
    env = gaussian_filter(env, sigma=sigma, mode=("wrap", "nearest"))
    env = np.maximum(env, 1.0)
    pc = Piece(name, "torso", float(env.max()) + gap, color=GREY)
    a = np.radians(np.arange(-180, 180, DB) + DB / 2); e = np.radians(np.arange(-90, 90, DB) + DB / 2)
    A, E = np.meshgrid(a, e, indexing="ij")
    D = np.stack([np.cos(E) * np.sin(A), -np.cos(E) * np.cos(A), np.sin(E)], -1)
    pc.grid = C + (env + gap)[..., None] * D
    pc.mask = np.ones(env.shape, bool)
    pc.env = env
    return pc


def concept_pieces(kind, occ, lay, poses, R=62.0, W=42.0, P=2.4, sigma=4.0, hood="full"):
    """Three layers 4 mm apart, each shaped from its own free space; a static pelvis hood outside them follows the
    envelope the moving layers sweep. kind: 'ball' (spheres), 'egg' (slim: surfaces of revolution about the pitch
    axis, half-width W), 'saddle' (slim, with the roll-link layer outside the thigh layer instead of inside)."""
    Wt = W if kind != "ball" else 0.0
    if kind == "saddle":
        layers = [Piece("thigh egg", "J3", R, W=Wt, P=P, color=IVORY),
                  Piece("roll-link saddle", "J2", R + 4, W=Wt + 4 if Wt else 0, P=P, color=WARM)]
    else:
        layers = [Piece("inner tread (roll link)", "J2", R, W=Wt, P=P, color=WARM),
                  Piece("thigh " + ("globe" if kind == "ball" else "egg"), "J3", R + 4, W=Wt + 4 if Wt else 0, P=P, color=IVORY)]
    layers = [free_piece(pc, occ, sigma=sigma) for pc in layers]
    if hood == "none":
        return layers
    hood_kind = hood
    hood = envelope_hood("pelvis hood", layers, lay, poses[::2])
    # the hood exists only where no moving material would cross it: carve with the torso occupancy at its radius
    flat = hood.grid.reshape(-1, 3)
    r, az, el = sph(flat)
    ia, ie = bins(az, el)
    lo = np.clip(((r - SHELL - CLEAR) / DR).astype(int), 0, NR - 1); hi = np.clip(((r + CLEAR) / DR).astype(int) + 1, 0, NR)
    cum = np.concatenate([np.zeros((NAZ, NEL, 1), np.int32), np.cumsum(occ["torso"], axis=2, dtype=np.int32)], axis=2)
    blk = ((cum[ia, ie, hi] - cum[ia, ie, lo]) > 0).reshape(hood.mask.shape)
    from scipy.ndimage import gaussian_filter, binary_erosion
    free = ~blk
    sm = gaussian_filter(free.astype(float), sigma=(sigma, sigma), mode=("wrap", "nearest")) > 0.5
    hood.mask = sm & binary_erosion(free, iterations=2, border_value=1)
    if hood_kind == "torso width":       # a skirt under the torso: no wider than the torso shell (x 90) + 4
        hood.mask &= np.abs(hood.grid[..., 0]) <= 94.0
    # the hood is only drawn where it is outside the torso box or below the torso's hem region (the rest is the torso)
    hood.kept = round(float(hood.mask.mean()), 3)
    return layers + [hood]


# ------------------------------------------------------------------ torso hem and exposure
TORSO = dict(x=90.0, y0=-49.0, y1=60.0, z0=186.0)


def sweep_voxels(lay, pieces, poses, vox=3.0):
    """Torso-frame voxels swept by the moving pieces over the poses (both legs; the left mirrored)."""
    S = set()
    for fam, p in poses:
        q = dict(zip(("pitch", "roll", "yaw", "knee"), p))
        Ts = H.link_T(lay, q)
        for pc in pieces:
            if pc.owner == "torso":
                continue
            P = pc.pts[::2]
            W = P @ Ts[pc.owner][:3, :3].T + Ts[pc.owner][:3, 3]
            for M in (W, W * np.array([-1, 1, 1])):
                S.update(map(tuple, np.floor(M / vox).astype(int)))
    return S, vox


def hem(S, vox, clear=3.0):
    """Hem of the torso shell: on each wall (outboard x=+-90, front y=-49, back y=60), the height below which the
    shell is cut so that it clears the swept pieces by `clear`; a column keeps its original bottom (z 186) where
    nothing sweeps. Returned as functions of position along each wall, smoothed."""
    from scipy.ndimage import maximum_filter1d, gaussian_filter1d
    arr = np.array(list(S)) * vox + vox / 2
    out = {}
    walls = {"outboard": (0, TORSO["x"], 1, np.arange(TORSO["y0"], TORSO["y1"] + 0.1, 2.0)),
             "front": (1, TORSO["y0"], 0, np.arange(-TORSO["x"], TORSO["x"] + 0.1, 2.0)),
             "back": (1, TORSO["y1"], 0, np.arange(-TORSO["x"], TORSO["x"] + 0.1, 2.0))}
    for w, (ax, val, along, s) in walls.items():
        sel = np.abs(np.abs(arr[:, ax]) - abs(val)) <= clear + vox if ax == 0 else np.abs(arr[:, ax] - val) <= clear + vox
        near = arr[sel]
        if ax == 0:
            near = near.copy(); near[:, 0] = np.abs(near[:, 0])
        h = np.full(len(s), TORSO["z0"])
        for i, v in enumerate(s):
            c = near[np.abs(near[:, along] - v) <= clear + vox / 2]
            if len(c):
                h[i] = max(h[i], c[:, 2].max() + clear)
        h = maximum_filter1d(h, 7)
        h = np.maximum(gaussian_filter1d(h, 3.0), h - 0.0)
        out[w] = (s, h)
    return out


def hem_z(hm, p):
    """Hem height at a point on the torso walls."""
    x, y = p[..., 0], p[..., 1]
    z = np.full(x.shape, TORSO["z0"])
    on_out = np.abs(np.abs(x) - TORSO["x"]) < 1e-6
    on_f = np.abs(y - TORSO["y0"]) < 1e-6
    on_b = np.abs(y - TORSO["y1"]) < 1e-6
    s, h = hm["outboard"]; z = np.where(on_out, np.interp(y, s, h), z)
    s, h = hm["front"]; z = np.where(on_f, np.interp(x, s, h), z)
    s, h = hm["back"]; z = np.where(on_b, np.interp(x, s, h), z)
    return z


def ray_exit(d):
    """Exit point of rays from C through the torso box walls (x=+-90, y=-49/60) or its open bottom (z=186)."""
    t = np.full(d.shape[:-1], np.inf)
    for ax, val in ((0, TORSO["x"]), (0, -TORSO["x"]), (1, TORSO["y0"]), (1, TORSO["y1"]), (2, TORSO["z0"])):
        dd = d[..., ax]
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = (val - C[ax]) / dd
        tt = np.where((tt > 0) & np.isfinite(tt), tt, np.inf)
        t = np.minimum(t, tt)
    p = C + t[..., None] * d
    for ax, val in ((0, TORSO["x"]), (0, -TORSO["x"]), (1, TORSO["y0"]), (1, TORSO["y1"]), (2, TORSO["z0"])):
        p[..., ax] = np.where(np.abs(p[..., ax] - val) < 1e-6, val, p[..., ax])
    return p, t


def dir_map(pc):
    m = np.zeros((NAZ, NEL), bool)
    r, az, el = sph(pc.pts)
    ia, ie = bins(az, el)
    m[ia, ie] = True
    from scipy.ndimage import binary_closing
    return binary_closing(m, iterations=1)


def exposure(lay, pieces, hm, poses, egg_inboard_x):
    a = np.radians(np.arange(-180, 180, DB) + DB / 2); e = np.radians(np.arange(-90, 90, DB) + DB / 2)
    A, E = np.meshgrid(a, e, indexing="ij")
    D = np.stack([np.cos(E) * np.sin(A), -np.cos(E) * np.cos(A), np.sin(E)], -1)
    w = np.cos(E); w /= w.sum()
    p, t = ray_exit(D)
    bottom = np.abs(p[..., 2] - TORSO["z0"]) < 1e-6
    torso = ~bottom & (p[..., 2] > hem_z(hm, p))
    crotch = bottom & (np.abs(p[..., 0]) < egg_inboard_x)        # keel cover between the eggs (static)
    # the left leg's pieces, standing (mirrored): a ray from C that runs into them is hidden
    vox = 3.0
    L = set()
    for pc in pieces:
        if pc.owner == "torso":
            continue
        M = pc.pts * np.array([-1, 1, 1])
        L.update(map(tuple, np.floor(M / vox).astype(int)))
    left = np.zeros(D.shape[:2], bool)
    for rr in np.arange(30.0, 230.0, 2.0):
        k = np.floor((C + rr * D) / vox).astype(int).reshape(-1, 3)
        left |= np.array([tuple(v) in L for v in k]).reshape(D.shape[:2]) & (rr < t)
    torso = torso | left
    maps = [(pc.owner, dir_map(pc)) for pc in pieces]
    for o, m in maps:
        if o == "torso":
            torso = torso | m
    flat = D.reshape(-1, 3)
    ex = []
    for fam, pq in poses:
        q = dict(zip(("pitch", "roll", "yaw", "knee"), pq))
        Ts = H.link_T(lay, q)
        cov = torso | crotch
        for f in ("J2", "J3"):
            Ri = Ts[f][:3, :3].T
            dl = flat @ Ri.T
            r, az, el = sph(C + 70 * dl)
            ia, ie = bins(az, el)
            for o, m in maps:
                if o == f:
                    cov = cov | m[ia, ie].reshape(NAZ, NEL)
            if f == "J3":
                for rr in (75, 90, 105):
                    pp = C + rr * dl
                    lo, hi = THIGH_SHELL
                    cov = cov | (np.all((pp >= lo) & (pp <= hi), axis=1) & (pp[:, 2] < C[2] - 25)).reshape(NAZ, NEL)
        ex.append(float((w * ~cov).sum()))
    return np.array(ex)


# ------------------------------------------------------------------ search, report, sheet
def evaluate(kind, params, lay, occ, occ_c, poses, with_hem=True):
    pcs = concept_pieces(kind, occ, lay, poses, **params)
    hs = {pc.name: hits(pc, occ)[0] for pc in pcs}
    hc = {pc.name: hits(pc, occ_c)[0] for pc in pcs}
    S, vox = sweep_voxels(lay, [p for p in pcs if p.owner != "torso"] + [p for p in pcs if p.owner == "torso"], poses[::3])
    hm = hem(S, vox)
    egg_in = min(float(pc.pts[:, 0].min()) for pc in pcs) - 2.0
    ex = exposure(lay, pcs, hm, poses, egg_in)
    out_x = max(float(pc.pts[:, 0].max()) for pc in pcs)
    out_x_moving = max([float(pc.pts[:, 0].max()) for pc in pcs if pc.owner != "torso"])
    sel = [pc.pts[(pc.pts[:, 2] > 141) & (pc.pts[:, 2] < 293)] for pc in pcs]
    arm_gap = min(float(103 - s[:, 0].max()) for s in sel if len(s))
    hem_max = {w: round(float(h.max()), 1) for w, (s, h) in hm.items()}
    return dict(kind=kind, params=params,
                hit_share_realistic={k: round(v, 4) for k, v in hs.items()},
                hit_share_combinations={k: round(v, 4) for k, v in hc.items()},
                exposed_pct=dict(median=round(100 * float(np.median(ex)), 2), p90=round(100 * float(np.percentile(ex, 90)), 2),
                                 max=round(100 * float(ex.max()), 2), standing=round(100 * float(ex[0]), 2)),
                worst_poses=[(poses[i][0], [round(float(v), 1) for v in poses[i][1]], round(100 * float(ex[i]), 2))
                             for i in np.argsort(-ex)[:4]],
                outboard_x_mm=round(out_x, 1), outboard_x_moving_standing_mm=round(out_x_moving, 1), inboard_x_mm=round(egg_in + 2, 1),
                depth_y_mm=[round(min(float(pc.pts[:, 1].min()) for pc in pcs), 1), round(max(float(pc.pts[:, 1].max()) for pc in pcs), 1)],
                arm_gap_standing_mm=round(arm_gap, 1), hem_top_z_mm=hem_max), pcs, hm, ex


SEARCH = {   # round 4: A-22 (shell fork with hub spokes) replaces the straight fork arms that cut the thigh layer apart
    "ball": [dict(R=R, sigma=3.0) for R in (60.0, 62.0, 64.0)],
    "egg": [dict(R=R, W=W, P=2.4, sigma=3.0) for R in (60.0, 62.0, 64.0) for W in (34.0, 38.0, 42.0)],
    "saddle": [dict(R=R, W=W, P=2.4, sigma=3.0) for R in (60.0, 62.0) for W in (34.0, 38.0, 42.0)],
}
NAMES = {"ball": "K1 ball: thigh globe over a roll-link tread, under a pelvis hood (spheres)",
         "egg": "K2 slim egg: the same layers, round from the side and slim from the front",
         "saddle": "K3 saddle: slim thigh egg with the roll-link layer riding outside it as a saddle"}


def main():
    resume = "--final" in sys.argv
    lay, parts = internals()
    poses = H.poses()
    occ = occupancy(lay, parts, poses)
    occ_c = occupancy(lay, parts, [("all combinations", g) for g in H.GRID])
    allres, best = {}, {}
    if resume:
        allres = json.loads((HERE / "globe-concepts-search.json").read_text())
        best = {k: v[0] for k, v in allres.items()}
    for kind, plist in ([] if resume else SEARCH.items()):
        rows = []
        for prm in plist:
            r, pcs, hm, ex = evaluate(kind, prm, lay, occ, occ_c, poses)
            rows.append(r)
            print(kind, prm, "hits", r["hit_share_realistic"], "exposed", r["exposed_pct"]["median"], r["exposed_pct"]["max"],
                  "out", r["outboard_x_mm"], "hem", r["hem_top_z_mm"], flush=True)
        # rank: no realistic hits first (largest hit share), then exposure (median + p90), then hem height
        def key(r):
            return (round(max(r["hit_share_realistic"].values()), 3), r["exposed_pct"]["median"] + r["exposed_pct"]["p90"],
                    max(r["hem_top_z_mm"].values()))
        rows.sort(key=key)
        allres[kind] = rows
        best[kind] = rows[0]
    if not resume:
        (HERE / "globe-concepts-search.json").write_text(json.dumps(allres, indent=1))
    store = {}
    final = dict(best)
    final["egg, hood cut to torso width"] = dict(best["egg"], params={**best["egg"]["params"], "hood": "torso width"})
    final["egg, no hood"] = dict(best["egg"], params={**best["egg"]["params"], "hood": "none"})
    for key, r in final.items():
        kind = key.split(",")[0]
        rr, pcs, hm, ex = evaluate(kind, r["params"], lay, occ, occ_c, poses)
        rr["name"] = NAMES[kind] + ("" if key == kind else f" ({key.split(', ')[1]})")
        rr["hit_points_realistic"] = {pc.name: len(hits(pc, occ)[1]) for pc in pcs}
        best[key] = rr
        store[key] = dict(pieces=[(pc.name, pc.owner, pc.color, mesh_of(pc)) for pc in pcs], hem=hm,
                          exposed=ex.tolist())
    (HERE / "globe-concepts.json").write_text(json.dumps(best, indent=1))
    import pickle
    WORK.mkdir(parents=True, exist_ok=True)
    pickle.dump(store, open(WORK / "globe-concepts.pkl", "wb"))
    render(best)


VIEWS = {"front": np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], float),
         "side": np.array([[0, -1, 0], [0, 0, 1], [1, 0, 0]], float),
         "3/4": None, "back 3/4": None}


def view_R(v):
    if VIEWS[v] is not None:
        return VIEWS[v]
    a = np.radians(35 if v == "3/4" else 145)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    b = np.radians(-10)
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return VIEWS["front"] @ Rx.T @ Rz


BODY = ["center_front_shell", "center_back_shell", "original_profile_head_skin_v23", "head_neck_collar",
        "R_upper_arm_hard_cover", "L_upper_arm_hard_cover", "R_upper_arm_lower_cover", "L_upper_arm_lower_cover",
        "R_fore_front_hard_cover", "R_fore_back_hard_cover", "L_fore_front_hard_cover", "L_fore_back_hard_cover"]
TORSO_SHELLS = ("center_front_shell", "center_back_shell")
LEG_R = [("R_thigh_front_hard_cover", "J3"), ("R_thigh_back_hard_cover", "J3"), ("R_shin_front_hard_cover", "shin"),
         ("R_shin_back_hard_cover", "shin"), ("R_foot_boot_shell", "shin"), ("R_foot_sole_shell", "shin")]
LEG_L = ["L_thigh_front_hard_cover", "L_thigh_back_hard_cover", "L_shin_front_hard_cover", "L_shin_back_hard_cover",
         "L_foot_boot_shell", "L_foot_sole_shell"]


def trim_torso(m, hm):
    """Cut a torso shell below the hem (faces whose centroid is under the hem of the nearest wall)."""
    c = m.triangles_center
    x, y, z = c[:, 0], c[:, 1], c[:, 2]
    p = np.stack([np.where(np.abs(x) > 70, np.sign(x) * TORSO["x"], x),
                  np.where(np.abs(x) > 70, y, np.where(y < 5, TORSO["y0"], TORSO["y1"])), z], 1)
    # outboard wall for |x| > 70, else the front/back wall
    hz = hem_z(hm, p)
    keep = z >= hz
    mm = m.copy(); mm.update_faces(keep); mm.remove_unreferenced_vertices()
    return mm


def crotch_cover(xg):
    """Static keel cover between the eggs: a rounded bar under the battery keel."""
    m = trimesh.creation.capsule(height=max(2 * xg - 30, 4), radius=15, count=[16, 16])
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    m.apply_scale([1, 2.6, 1]); m.apply_translation([0, 6, 200])
    return m


def draw(ax, items, view, lim):
    from matplotlib.collections import PolyCollection
    R = view_R(view)
    L = np.array([0.35, 0.75, 0.55]); L /= np.linalg.norm(L)
    P, Cc, D = [], [], []
    for m, T, col in items:
        v = (m.vertices @ T[:3, :3].T + T[:3, 3]) @ R.T
        nn = (m.face_normals @ T[:3, :3].T) @ R.T
        tri = v[m.faces]
        sh = 0.55 + 0.45 * np.abs(nn @ L)
        P.extend(tri[:, :, :2]); D.extend(tri[:, :, 2].mean(1))
        Cc.extend(np.clip(np.outer(sh, col), 0, 1))
    o = np.argsort(np.array(D))
    ax.add_collection(PolyCollection([P[i] for i in o], facecolors=[Cc[i] for i in o], edgecolors=[Cc[i] for i in o],
                                     linewidths=0.15))
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_aspect("equal"); ax.axis("off")


POSES = [("standing", dict(pitch=0, roll=0, yaw=0, knee=0), "front"),
         ("standing", dict(pitch=0, roll=0, yaw=0, knee=0), "side"),
         ("standing", dict(pitch=0, roll=0, yaw=0, knee=0), "3/4"),
         ("standing", dict(pitch=0, roll=0, yaw=0, knee=0), "back 3/4"),
         ("squat -85, knee 95", dict(pitch=-85, roll=0, yaw=0, knee=95), "side"),
         ("leg out: roll -40, pitch -20", dict(pitch=-20, roll=-40, yaw=0, knee=20), "front"),
         ("climb: -85 / -30 / toes out 30 / 95", dict(pitch=-85, roll=-30, yaw=30, knee=95), "3/4"),
         ("rear leg of a lunge +40, knee 40", dict(pitch=40, roll=0, yaw=0, knee=40), "back 3/4")]


def render(res):
    import pickle
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    body = H.kit.base()
    store = pickle.load(open(WORK / "globe-concepts.pkl", "rb"))
    lay, parts = internals()
    shells = {n: H.kit._mesh(body, n) for n in BODY + [a for a, _ in LEG_R] + LEG_L}
    col = lambda n: np.array(body.parts[n].get("color") or [200, 200, 200, 255], float)[:3] / 255
    lims = {"front": ((-175, 175), (95, 385)), "side": ((-150, 150), (95, 385)), "3/4": ((-175, 175), (95, 385)),
            "back 3/4": ((-175, 175), (95, 385))}
    cols = ["today"] + [k for k in store if k != "egg, no hood"]
    fig, axs = plt.subplots(len(POSES), len(cols), figsize=(6.4 * len(cols), 5.6 * len(POSES)))
    for j, cname in enumerate(cols):
        trimmed = {n: trim_torso(shells[n], store[cname]["hem"]) for n in TORSO_SHELLS} if cname != "today" else {}
        for i, (lab, q, view) in enumerate(POSES):
            ax = axs[i, j]
            items = [(trimmed.get(n, shells[n]), np.eye(4), col(n)) for n in BODY]
            items += [(shells[n], np.eye(4), col(n)) for n in LEG_L]
            if cname == "today":
                qq = {**body.standing, "R_hip_pitch": q["pitch"], "R_hip_roll": q["roll"], "R_hip_yaw": q["yaw"],
                      "R_knee": q["knee"]}
                Tb = {l: body.link_T(l, qq) for l in ("R_roll_link", "R_shin")}
                for n, l in LEG_R:
                    items.append((shells[n], Tb["R_roll_link" if l == "J3" else "R_shin"], col(n)))
            else:
                Ts = H.link_T(lay, q)
                Ts0 = H.link_T(lay, dict(pitch=0, roll=0, yaw=0, knee=0))
                for n, l in LEG_R:
                    items.append((shells[n], Ts[l], col(n)))
                xin = res[cname]["inboard_x_mm"]
                items.append((crotch_cover(xin), np.eye(4), np.array(GREY)))
                for pn, owner, c, m in store[cname]["pieces"]:
                    items.append((m, Ts[owner], np.array(c)))
                    mm = m.copy(); mm.vertices[:, 0] *= -1; mm.faces = mm.faces[:, ::-1]
                    items.append((mm, Ts0[owner], np.array(c)))
            draw(ax, items, view, lims[view])
            ax.set_title(f"{lab} - {view}", fontsize=12)
        if cname == "today":
            head = "today's hip\n(yaw -> pitch -> roll, roll axis 29 mm low)\nthigh covers x 11..88"
        else:
            r = res[cname]
            head = (f"{r['name'].split(':')[0]}{' - ' + cname.split(', ')[1] if ', ' in cname else ''}\n"
                    f"seen into the hip: median {r['exposed_pct']['median']:.1f}%, worst {r['exposed_pct']['max']:.1f}%\n"
                    f"outboard x {r['outboard_x_mm']:.0f} (torso 90, arm 103)\ndepth y {r['depth_y_mm'][0]:.0f}..{r['depth_y_mm'][1]:.0f} (torso -49..60)")
        axs[0, j].text(0.5, 1.10, head, transform=axs[0, j].transAxes, ha="center", va="bottom", fontsize=14)
    fig.suptitle("Hip globe concepts on the concurrent hip (P-04, S3 sketch round). Ivory = thigh, tan = roll-link "
                 "tread, grey = static pelvis pieces; the torso shell is cut back where the globes sweep.", fontsize=18)
    plt.tight_layout(rect=(0, 0, 1, 0.965))
    out = HERE / "globe-concepts.png"
    plt.savefig(out, dpi=48); plt.close(fig)
    print(out)


if __name__ == "__main__":
    if "--render" in sys.argv:
        render(json.loads((HERE / "globe-concepts.json").read_text()))
    else:
        main()
