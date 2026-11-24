"""Design kit, shells: the parametric master surface of docs/design/design-kit.md section 2 (the arm-shell recipe).

A shell is one smooth master surface per limb or region, cut into parts afterwards at analytic seams. The surface is a
sweep of planar superellipse sections along a spine:

  station        (name, centre (3,), tilt w, a_out, a_in, b_front, b_back, n)
                 centre and tilt are set by rule (joint centres, axes); the half-widths and n come from `Sweep.fit`
  along the spine every column is ONE natural cubic spline over the chord length of the centres (C2: no kinks)
  section plane  normal N = (1 - w) up + w tilt_axis (normalised); in-plane axes U = u_ref projected on the plane,
                 V = U x N; half-widths a_out along +U, a_in along -U, b_front along +V, b_back along -V
  section        superellipse |u / a|^n + |v / b|^n = 1
  cap (optional) above the first station the section shrinks as a dome toward `cap["z_top"]` (height along `up`):
                 U half-widths by g(p), V half-widths by g(p_y), g(q) = (1 - h^q)^(1/q)
  wall           constant, inward along the vertex normal; the far end is left open

Requirements are point sets with a radial need (mm): `margin` gives the room between each point + need and the skin,
`fit` finds the half-widths / exponents (least section area + smoothness + rounder exponents) that hold every point.
Seam cutters (planes, sectors and cylinders about a joint axis, tubes) are in `seam_*` below. Nothing here knows a part,
link or joint; every fact comes in as an argument. Measurements only; nothing accepts or rejects.
"""
from __future__ import annotations

import numpy as np
import trimesh

COLS = ("cx", "cy", "cz", "w", "a_out", "a_in", "b_front", "b_back", "n")


def dome(h, p):
    """Dome profile g(h) = (1 - h^p)^(1/p) for the height share h in 0..1."""
    return np.clip(1.0 - np.clip(h, 0.0, 1.0) ** p, 0.0, 1.0) ** (1.0 / p)


def ring_r(th, prm, gx=1.0, gy=1.0):
    """Superellipse radius at angle th (from U toward V) for ring values prm (..., 5: a_out, a_in, b_front, b_back, n);
    gx / gy scale the U / V half-widths."""
    a = np.where(np.cos(th) > 0, prm[..., 0], prm[..., 1]) * gx
    b = np.where(np.sin(th) > 0, prm[..., 2], prm[..., 3]) * gy
    e = prm[..., 4]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (np.abs(np.cos(th) / a) ** e + np.abs(np.sin(th) / b) ** e) ** (-1.0 / e)
    return np.nan_to_num(r, nan=0.0, posinf=0.0)


def closed_solid(G, pole, bottom):
    """Closed mesh from a ring grid G (rows, nth, 3; rows run from the pole end): a fan to `pole`, the grid, a fan to
    `bottom`."""
    nz, nth = G.shape[:2]
    i, j = np.meshgrid(np.arange(nz - 1), np.arange(nth), indexing="ij")
    a, b = i * nth + j, i * nth + (j + 1) % nth
    Fc = np.r_[np.c_[a.ravel(), b.ravel(), (b + nth).ravel()], np.c_[a.ravel(), (b + nth).ravel(), (a + nth).ravel()]]
    P, B = nz * nth, nz * nth + 1
    j = np.arange(nth); last = (nz - 1) * nth
    Fc = np.r_[Fc, np.c_[(j + 1) % nth, j, np.full(nth, P)], np.c_[last + j, last + (j + 1) % nth, np.full(nth, B)]]
    m = trimesh.Trimesh(np.vstack([G.reshape(-1, 3), pole, bottom]), Fc, process=True)
    if m.volume < 0:
        m.invert()
    return m


class Sweep:
    """Master surface of superellipse sections on natural cubic splines (see the module docstring).

    stations   sequence of (name, centre, tilt, a_out, a_in, b_front, b_back, n)
    tilt_axis  the section normal at tilt 1 (e.g. a joint axis along the limb); at tilt 0 the normal is `up`
    up         the section normal at tilt 0 and the dome's height direction
    u_ref      the direction the in-plane U axis is taken from (projected on each section plane)
    cap        None (open top) or dict(z_top, p, p_y): the dome above the first station
    wall       shell wall along the surface normal (mm)
    """

    def __init__(self, stations, *, tilt_axis, up=(0.0, 0.0, 1.0), u_ref=(1.0, 0.0, 0.0), cap=None, wall=0.9):
        self.st = tuple(stations)
        self.tilt_axis = np.asarray(tilt_axis, float)
        self.up = np.asarray(up, float)
        self.u_ref = np.asarray(u_ref, float)
        self.cap = cap
        self.wall = float(wall)
        self._dense = {}

    def with_stations(self, stations):
        """The same sweep over other stations (a fit candidate)."""
        return Sweep(stations, tilt_axis=self.tilt_axis, up=self.up, u_ref=self.u_ref, cap=self.cap, wall=self.wall)

    @property
    def names(self):
        return [s[0] for s in self.st]

    # ------------------------------------------------------------------------------------------ spine and frames
    def knots(self):
        """Chord parameters of the centres (n,) and the station values (n, 9) in COLS order."""
        C = np.array([s[1] for s in self.st], float)
        t = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
        V = np.array([list(s[1]) + list(s[2:]) for s in self.st], float)
        return t, V

    @staticmethod
    def spline(tk, Vk):
        """Natural cubic spline per column (C2)."""
        from scipy.interpolate import CubicSpline
        return CubicSpline(tk, Vk, bc_type="natural", axis=0)

    def frames(self, t):
        """Centre (n, 3), frame U, V, N (n, 3) each and the ring values (n, 5) at chord parameters t."""
        tk, Vk = self.knots()
        t = np.clip(np.asarray(t, float), tk[0], tk[-1])
        val = self.spline(tk, Vk)(t)
        C = val[:, :3]
        w = np.clip(val[:, 3], 0.0, 1.0)
        N = (1.0 - w)[:, None] * self.up + w[:, None] * self.tilt_axis
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        U = self.u_ref - (N @ self.u_ref)[:, None] * N
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        V = np.cross(U, N)
        return C, U, V, N, val[:, 4:]

    # ------------------------------------------------------------------------------------------ surface
    def _cap_height(self, c0):
        return self.cap["z_top"] - float(c0 @ self.up)

    def _pole(self, c0):
        return c0 - float(c0 @ self.up) * self.up + self.cap["z_top"] * self.up

    def grid(self, dt=0.6, nth=256, ncap=48):
        """Outer skin as rings (rows, nth, 3) from the pole end down to the last station, the pole (the first centre
        when there is no cap) and vertex normals (outward)."""
        tk, _ = self.knots()
        th = np.linspace(-np.pi, np.pi, nth, endpoint=False)
        t = np.arange(0.0, tk[-1] + 1e-9, dt)
        C, U, V, N, prm = self.frames(t)
        R = ring_r(th[None, :], prm[:, None, :])
        body = C[:, None, :] + R[..., None] * (np.cos(th)[None, :, None] * U[:, None, :] + np.sin(th)[None, :, None] * V[:, None, :])
        c0, p0 = C[0], prm[0]
        if self.cap is None:
            G, pole, cen = body, c0 + 2.0 * N[0], C
        else:
            p = self.cap["p"]
            phi = np.linspace(0.0, np.pi / 2, ncap)[1:-1][::-1]           # dome rows from near the pole down to the first ring
            hz = np.sin(phi) ** (2.0 / p)                                 # height share h of each row
            gx = np.cos(phi) ** (2.0 / p)
            gy = dome(hz, self.cap["p_y"])
            Rc = ring_r(th[None, :], p0[None, None, :], gx[:, None], gy[:, None])
            cap = (c0 + self.up * (self._cap_height(c0) * hz)[:, None])[:, None, :] + \
                Rc[..., None] * (np.cos(th)[None, :, None] * U[0] + np.sin(th)[None, :, None] * V[0])
            G = np.concatenate([cap, body], 0)
            pole = self._pole(c0)
            cen = np.concatenate([np.broadcast_to(c0, (len(cap), 3)), C], 0)
        dth = np.roll(G, -1, 1) - np.roll(G, 1, 1)
        up = np.concatenate([np.broadcast_to(pole, (1,) + G.shape[1:]), G[:-1]], 0)
        dn = np.concatenate([G[1:], G[-1:] + (G[-1:] - G[-2:-1])], 0)
        Nv = np.cross(up - dn, dth)
        Nv /= np.linalg.norm(Nv, axis=-1, keepdims=True)
        if np.mean(np.einsum("ijk,ijk->ij", Nv, G - cen[:, None, :])) < 0:
            Nv = -Nv
        return G, pole, Nv

    def master(self, end_out=4.0):
        """(outer solid, inner solid); the shell is outer - inner (wall along the normal). The inner solid runs out past
        the open far end, so the shell is open there."""
        G, pole, Nv = self.grid()
        tk, _ = self.knots()
        Cend, _, _, Nend, _ = self.frames([tk[-1]])
        bot = Cend[0] - end_out * Nend[0]
        outer = closed_solid(G, pole, bot)
        Gi = G - self.wall * Nv
        Gi = np.concatenate([Gi, Gi[-1:] - 3.0 * Nend[0]], 0)
        inner = closed_solid(Gi, pole - self.wall * self.up, bot - (end_out + 2.0) * Nend[0])
        return outer, inner

    # ------------------------------------------------------------------------------------------ local coordinates
    def local(self, P, dt=0.25):
        """Station parameter t, angle th and radius rho of points P in their section plane, the skin radius r_skin at
        that angle, and the dome share h (> 0 above the first station when there is a cap). Points beyond the ends get
        the end station."""
        from scipy.spatial import cKDTree
        if dt not in self._dense:
            tk, _ = self.knots()
            t = np.arange(0.0, tk[-1] + 1e-9, dt)
            self._dense[dt] = (t,) + self.frames(t)
        t, C, U, V, N, prm = self._dense[dt]
        P = np.asarray(P, float)
        _, i0 = cKDTree(C).query(P)
        win = np.arange(-60, 61)
        idx = np.clip(i0[:, None] + win[None, :], 0, len(t) - 1)
        f = np.einsum("ijk,ijk->ij", P[:, None, :] - C[idx], N[idx])        # signed distance from each nearby plane
        sc = np.sign(f[:, :-1]) != np.sign(f[:, 1:])
        k = np.where(sc.any(1), np.argmin(np.where(sc, np.abs(win[None, :-1]), 10 ** 6), axis=1), np.argmin(np.abs(f), axis=1))
        best = idx[np.arange(len(P)), k]
        d = P - C[best]
        d = d - np.einsum("ij,ij->i", d, N[best])[:, None] * N[best]
        u, v = np.einsum("ij,ij->i", d, U[best]), np.einsum("ij,ij->i", d, V[best])
        th = np.arctan2(v, u); rho = np.hypot(u, v)
        rs = ring_r(th, prm[best])
        h = np.zeros(len(P))
        if self.cap is not None:
            top = (best == 0) & ((P - C[0]) @ self.up > 0)
            if top.any():                                                  # the dome above the first station
                d0 = P[top] - C[0]
                h[top] = (d0 @ self.up) / self._cap_height(C[0])
                u0, v0 = d0 @ U[0], d0 @ V[0]
                th[top] = np.arctan2(v0, u0); rho[top] = np.hypot(u0, v0)
                rs[top] = ring_r(th[top], prm[0][None, :], dome(h[top], self.cap["p"]), dome(h[top], self.cap["p_y"]))
        return t[best], th, rho, rs, h

    def margin(self, P, need, exclude=False):
        """Radial room (mm) between points P + need and the outer skin (< 0: the point with its need pokes out); for
        `exclude` sets (parts that must stay outside) the room outside the skin."""
        _, _, rho, rs, h = self.local(P)
        if exclude:
            return np.where(h >= 1.0, 99.0, rho - rs - need)
        return np.where(h >= 1.0, -need - 1.0, rs - rho - need)

    def report_margins(self, requirements, exclude_prefix="EXCLUDE", verbose=True):
        """Per requirement {name: (P, need)}: points, worst margin and where, count of points short."""
        rows = {}
        for k, (P, need) in requirements.items():
            m = self.margin(P, need, exclude=k.startswith(exclude_prefix))
            i = int(np.argmin(m))
            rows[k] = dict(points=int(len(P)), min_mm=round(float(m[i]), 2), at=np.round(P[i], 1).tolist(),
                           short_points=int((m < 0).sum()))
            if verbose:
                print(f"{k:50s} min {m[i]:6.2f} mm at {np.round(P[i], 1)}  short {int((m < 0).sum())} / {len(P)}", flush=True)
        return rows

    # ------------------------------------------------------------------------------------------ measures
    def smoothness(self, dt=0.5):
        """Spine curvature (1/mm) and its change per mm along the centre curve; ring-value slopes; exponent range."""
        tk, _ = self.knots()
        t = np.arange(0.0, tk[-1] + 1e-9, dt)
        C, _, _, _, prm = self.frames(t)
        d1 = np.gradient(C, dt, axis=0); d2 = np.gradient(d1, dt, axis=0)
        k = np.linalg.norm(np.cross(d1, d2), axis=1) / np.linalg.norm(d1, axis=1) ** 3
        dk = np.gradient(k, dt)
        W = prm[:, :4]; dW = np.gradient(W, dt, axis=0); ddW = np.gradient(dW, dt, axis=0)
        return dict(length_mm=round(float(tk[-1]), 1), max_curvature_per_mm=round(float(k.max()), 5),
                    min_bend_radius_mm=round(float(1.0 / k.max()), 1), max_curvature_change_per_mm2=round(float(np.abs(dk).max()), 6),
                    max_half_width_slope=round(float(np.abs(dW).max()), 3), max_half_width_curvature_per_mm=round(float(np.abs(ddW).max()), 4),
                    exponent_range=[round(float(prm[:, 4].min()), 2), round(float(prm[:, 4].max()), 2)])

    # ------------------------------------------------------------------------------------------ fit
    def fit(self, requirements, bounds, *, fixed=(), tied=(), smooth=400.0, n_pref=2.4, n_weight=0.5,
            exclude_prefix="EXCLUDE", band=8.0, maxiter=400):
        """Half-widths and exponents of the free stations (centres and tilts stay): the least section area (mean over a
        dense sampling of (a_out + a_in) (b_front + b_back)) + smooth x the mean squared second derivative of every
        half-width along the spine + n_weight x the squared pull of each exponent toward n_pref, with every requirement
        point inside with its need (exclude sets: outside).

        requirements  {name: (P (k, 3), need mm)}; names starting with exclude_prefix must stay outside
        bounds        bounds(station_index, column 0..4) -> (lo, hi) for a_out, a_in, b_front, b_back, n
        fixed         station names left as they are
        tied          (source, target) station names: target copies source (a constant band, e.g. about a hinge)
        band          only points within this many mm of their need on the start surface constrain the fit
        Returns (values (stations, 5), scipy result, margins at the result)."""
        from scipy.interpolate import CubicSpline
        from scipy.optimize import minimize
        names = self.names
        P = np.vstack([r[0] for r in requirements.values()])
        need = np.concatenate([np.full(len(r[0]), r[1]) for r in requirements.values()])
        sg = np.concatenate([np.full(len(r[0]), -1.0 if k.startswith(exclude_prefix) else 1.0) for k, r in requirements.items()])
        t, th, rho, rs0, h = self.local(P)
        keep = (sg * (rs0 - rho) - need < band) & (h < 1.0)
        need, t, th, rho, sg = need[keep], t[keep], th[keep], rho[keep], sg[keep]
        tk, Vk = self.knots()
        basis = lambda x, d=0: np.stack([CubicSpline(tk, np.eye(len(tk))[i], bc_type="natural")(x, d) for i in range(len(tk))], -1)  # noqa: E731
        B = basis(np.clip(t, tk[0], tk[-1]))
        td = np.linspace(0, tk[-1], 400)
        Bd, Bdd = basis(td), basis(td, 2)
        targets = {names.index(b) for _, b in tied}
        free = [i for i, n in enumerate(names) if n not in fixed and i not in targets]
        V0 = Vk[:, 4:].copy()                                       # (stations, 5)

        def unpack(x):
            V = V0.copy(); k = 0
            for i in free:
                for c in range(5):
                    V[i, c] = x[k]; k += 1
            for a, b in tied:
                V[names.index(b)] = V[names.index(a)]
            return V

        x0 = np.array([V0[i, c] for i in free for c in range(5)])
        bnd = [bounds(i, c) for i in free for c in range(5)]

        def obj(x):
            V = unpack(x); W = Bd @ V
            area = np.mean((W[:, 0] + W[:, 1]) * (W[:, 2] + W[:, 3]))
            dd = Bdd @ V[:, :4]
            return float(area / 100.0 + smooth * np.mean(dd ** 2) * 100.0 + n_weight * np.sum((V[:, 4] - n_pref) ** 2))

        def cons(x):
            prm = B @ unpack(x)
            return sg * (ring_r(th, prm) - rho) - need
        r = minimize(obj, x0, method="SLSQP", bounds=bnd, constraints=[dict(type="ineq", fun=cons)],
                     options=dict(maxiter=maxiter, ftol=1e-8))
        return unpack(r.x), r, cons(r.x)

    def stations_text(self, V, fmt="{:.2f}"):
        """The STATIONS literal for fitted values V (stations, 5), to paste into a builder."""
        lines = ["STATIONS = ("]
        for i, s in enumerate(self.st):
            c = s[1]
            lines.append(f'    ("{s[0]}", ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}), {s[2]}, ' + ", ".join(fmt.format(v) for v in V[i]) + "),")
        return "\n".join(lines + [")"])


# ---------------------------------------------------------------------------------------------- growth against a skin
def growth(outer, covers, n=30000):
    """Outward growth of a new outer skin `outer` against today's covers {name: mesh}: from each point of a cover's
    outer skin along its outward normal to the new skin (+ = the new skin lies outside), mm."""
    out = {}
    rx = trimesh.ray.ray_pyembree.RayMeshIntersector(outer)
    for name, c in covers.items():
        P, fi = trimesh.sample.sample_surface(c, n, seed=1)
        Nn = c.face_normals[fi]
        own = trimesh.ray.ray_pyembree.RayMeshIntersector(c)
        far = ~own.intersects_any(P + Nn * 0.05, Nn)                    # outer skin: nothing of the cover outside it
        P, Nn = P[far], Nn[far]
        ins = outer.contains(P)
        d = np.full(len(P), np.nan)
        for sgn, sel in ((1.0, ins), (-1.0, ~ins)):
            if not sel.any():
                continue
            loc, ir, _ = rx.intersects_location(P[sel], Nn[sel] * sgn, multiple_hits=False)
            idx = np.flatnonzero(sel)[ir]
            d[idx] = sgn * np.linalg.norm(loc - P[idx], axis=1)
        ok = np.isfinite(d) & (np.abs(d) < 40.0)
        d, Pk = d[ok], P[ok]
        out[name] = dict(points=int(len(d)), p05_p50_p95_mm=np.round(np.percentile(d, [5, 50, 95]), 2).tolist(),
                         max_outward_mm=round(float(d.max()), 2), at=np.round(Pk[int(np.argmax(d))], 1).tolist(),
                         max_inward_mm=round(float(d.min()), 2))
    return out


# ---------------------------------------------------------------------------------------------- seam cutters (manifold3d)
def to_manifold(m):
    """trimesh -> manifold3d.Manifold."""
    import manifold3d
    return manifold3d.Manifold(manifold3d.Mesh(vert_properties=np.asarray(m.vertices, np.float32),
                                               tri_verts=np.asarray(m.faces, np.uint32)))


def to_trimesh(x):
    """manifold3d.Manifold -> trimesh, unprocessed (2-manifold as manifold gives it; merging coincident vertices would
    join two sheets that touch along an edge)."""
    r = x.to_mesh()
    return trimesh.Trimesh(np.asarray(r.vert_properties)[:, :3], np.asarray(r.tri_verts), process=False)


def seam_sector(origin, e1, e2, e_zero, e_ninety, phi0, phi1, s0, s1, r=400.0, n=90):
    """Prism along the axis e1 x e2 over the sector phi0..phi1 (deg) about the axis through `origin`, the angle measured
    from e_zero toward e_ninety (both in the e1-e2 plane); s0..s1 are absolute positions along the axis (p . axis).
    Hinge seams: each part keeps a sector and the parts turn inside each other. e1, e2 orthonormal. trimesh."""
    from shapely.geometry import Polygon
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    axis = np.cross(e1, e2)
    o = np.asarray(origin, float)
    c = np.array([o @ e1, o @ e2])
    z2 = np.array([np.asarray(e_zero, float) @ e1, np.asarray(e_zero, float) @ e2])
    n2 = np.array([np.asarray(e_ninety, float) @ e1, np.asarray(e_ninety, float) @ e2])
    ph = np.radians(np.linspace(phi0, phi1, n))
    ring = c + r * (np.cos(ph)[:, None] * z2 + np.sin(ph)[:, None] * n2)
    pts = np.r_[[c], ring]
    poly = Polygon(pts)
    if not poly.exterior.is_ccw:
        poly = Polygon(pts[::-1])
    m = trimesh.creation.extrude_polygon(poly, s1 - s0)
    T = np.eye(4); T[:3, 0], T[:3, 1], T[:3, 2] = e1, e2, axis
    T[:3, 3] = s0 * axis
    m.apply_transform(T)
    if m.volume < 0:
        m.invert()
    return m


def seam_cylinder(p0, p1, r, sections=180):
    """Cylinder of radius r from p0 to p1 (the round part of a hinge seam). trimesh."""
    return trimesh.creation.cylinder(radius=r, segment=[np.asarray(p0, float), np.asarray(p1, float)], sections=sections)


def tube(p0, p1, r_in, r_out, axis=None, sections=240):
    """Tube from p0 to p1 (a lip under a rotating seam); the bore runs 1 mm past both ends along `axis` (unit; default
    p0 -> p1). Manifold."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    a = (p1 - p0) / np.linalg.norm(p1 - p0) if axis is None else np.asarray(axis, float)
    outer = trimesh.creation.cylinder(radius=r_out, segment=[p0, p1], sections=sections)
    inner = trimesh.creation.cylinder(radius=r_in, segment=[p0 - a, p1 + a], sections=sections)
    return to_manifold(outer) - to_manifold(inner)


def boss(axis_pt, axis_dir, r, t0, t1, sections=64):
    """Cylinder r on the axis axis_pt + t axis_dir, t0..t1 (bosses, wells, bores). Manifold."""
    a = np.asarray(axis_pt, float); d = np.asarray(axis_dir, float) / np.linalg.norm(axis_dir)
    return to_manifold(trimesh.creation.cylinder(radius=r, segment=[a + t0 * d, a + t1 * d], sections=sections))


def sleeve_tube(a, d, t0, t1, od, bore):
    """Steel sleeve solid on the axis a + t d, t0..t1, outer diameter od, bore diameter `bore`. Manifold."""
    return boss(a, d, od / 2.0, t0, t1) - boss(a, d, bore / 2.0, t0 - 1.0, t1 + 1.0)
