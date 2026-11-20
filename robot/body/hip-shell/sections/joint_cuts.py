"""Stage 1 of the hip-shell process: cuts through the bare skeleton (no shells, no covers), one sheet per hip motor
(Idle T-czv2d8).

For each hip joint, the cuts lie in planes the joint's rotation keeps (pitch: x = const, roll: y = const, yaw: z = const),
so every curve in a cut stays in that cut through the whole motion, and the 2D drawing is exact.

Columns per cut:
  1  standing: static internals (filled), moving internals (outlined), the turning point and rings r 20/40/60 mm
  2  motion: moving internals at the range ends and between, the swept region (red), and where the moving internals
     run into static ones (dark red; contact already present standing is left out), with the first contact angle
  3  space for a STATIC shell piece (green): what is left around the turning point once the static internals and
     everything the moving internals sweep are taken out
  4  space for a MOVING shell piece (blue, drawn standing): what is left once the moving internals and everything the
     static internals sweep relative to the moving side are taken out
Columns 3 and 4 are the 2D design space for the shells (stage 3).

Run: python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B robot/body/hip-shell/sections/joint_cuts.py [pitch|roll|yaw]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import kit  # noqa: E402

FACES = 6000
CLIP_R = 100.0
STEP = 2.5


def side(x0):   # u = forward (-y), v = z
    return np.array([[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, x0], [0, 0, 0, 1]], float)


def front(y0):  # seen from the front: u = -x (robot right on the left), v = z
    return np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, -y0], [0, 0, 0, 1]], float)


def plan(z0):   # seen from above: u = x (robot right to the right), v = forward (-y)
    return np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, -z0], [0, 0, 0, 1]], float)


# Joint ranges are assumption A-02 in DECISIONS.md (pitch relaxed to -85 per the owner, 2026-11-15).
JOINTS = {
    "pitch": dict(joint="R_hip_pitch", range=(-85.0, 40.0), axis_pt=np.array([59.0, 2.0, 219.0]),
                  note="pitch axis along x through (y 2, z 219); negative = flexion (thigh forward and up)",
                  cuts=[("x = 20, inboard edge of the hip", side(20)), ("x = 36, through the hip-pitch servo", side(36)),
                        ("x = 59, through the hip centre C", side(59)), ("x = 80, outboard side of the thigh", side(80))],
                  xlim=(-115, 115), ylim=(95, 335), ulab="forward (-y) mm", vlab="z mm"),
    "roll": dict(joint="R_hip_roll", range=(-45.0, 25.0), axis_pt=np.array([56.0, 2.0, 190.0]),
                 note="roll axis along y through (x 56, z 190), 29 mm below C; negative = abduction (leg out)",
                 cuts=[("y = -15, front of the hip", front(-15)), ("y = 2, through the pitch axis", front(2)),
                       ("y = 20, back of the hip", front(20))],
                 xlim=(-160, 40), ylim=(95, 335), ulab="robot right (+x) <-  mm", vlab="z mm"),
    "yaw": dict(joint="R_hip_yaw", range=(-40.0, 40.0), axis_pt=np.array([59.0, 2.0, 219.0]),
                note="yaw axis vertical through (x 59, y 2); sign as in the URDF",
                cuts=[("z = 245, top of the hip-pitch servo", plan(245)), ("z = 219, through C", plan(219)),
                      ("z = 195, through the roll servo", plan(195)), ("z = 160, upper thigh", plan(160))],
                xlim=(-60, 160), ylim=(-90, 110), ulab="robot right (+x) ->  mm", vlab="forward (-y) mm"),
}


ARM_LINKS = ("shoulder", "elbow", "arm", "wrist", "finger", "palm", "hand", "neck", "head")


def skeleton_parts(body):
    """Assumption A-01: the bare skeleton = every hip-region part except body shells and covers, fasteners and
    hardware (kit._hip_parts), and the arms and head (they move on their own; their clearance is a separate check)."""
    return {n: p for n, p in kit._hip_parts(body).items()
            if p.get("role") != "body_shell" and not any(k in p["owning_link"].lower() for k in ARM_LINKS)}


def cut_polys(mesh, M):
    """Shapely polygons of mesh posed by M (world->cut frame incl. pose), cut by the cut plane (local z = 0)."""
    from shapely.geometry import Polygon
    n = M[2, :3]; d = M[2, 3]
    lo, hi = mesh.bounds
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    s = corners @ n + d
    if s.min() > 0 or s.max() < 0:
        return []
    sec = mesh.section(plane_origin=-d * n / (n @ n), plane_normal=n)
    if sec is None:
        return []
    out = []
    for c in sec.discrete:
        if len(c) < 4:
            continue
        uv = (c @ M[:3, :3].T + M[:3, 3])[:, :2]
        g = Polygon(uv).buffer(0)
        if g.area > 0.5:
            out.append(g)
    return out


def union(gs):
    from shapely.ops import unary_union
    return unary_union(gs) if gs else None


def draw(ax, geom, **kw):
    from matplotlib.patches import Polygon as MP
    if geom is None or geom.is_empty:
        return
    for g in getattr(geom, "geoms", [geom]):
        if g.geom_type != "Polygon":
            continue
        ax.add_patch(MP(np.array(g.exterior.coords), closed=True, **kw))
        for h in g.interiors:
            ax.add_patch(MP(np.array(h.coords), closed=True, fc="white", ec=kw.get("ec", "k"), lw=0.4))


def analyse(body, spec):
    from shapely.geometry import Point
    j = spec["joint"]
    q0 = dict(body.standing)
    parts = skeleton_parts(body)
    probe = {**q0, j: 20.0}
    moving = {n: p for n, p in parts.items()
              if not np.allclose(body.link_T(p["owning_link"], q0), body.link_T(p["owning_link"], probe))}
    static = {n: p for n, p in parts.items() if n not in moving}
    # the link the joint turns: the moving part whose link is closest to the joint in the chain
    links = {p["owning_link"] for p in moving.values()}
    jl = next(l for l in links if body.child_joint.get(l, {}).get("name") == j)
    angles = np.arange(spec["range"][0], spec["range"][1] + 1e-6, STEP)
    meshes = {n: kit._mesh(body, n, FACES) for n in parts}
    results = []
    for title, Tc in spec["cuts"]:
        ap = (Tc @ np.r_[spec["axis_pt"], 1.0])[:2]
        clip = Point(*ap).buffer(CLIP_R, 128)
        stat = {n: union(cut_polys(meshes[n], Tc @ body.link_T(p["owning_link"], q0))) for n, p in static.items()}
        stat = {n: g for n, g in stat.items() if g is not None}
        S = union(list(stat.values()))

        def moving_at(a):
            q = {**q0, j: float(a)}
            return {n: union(cut_polys(meshes[n], Tc @ body.link_T(p["owning_link"], q))) for n, p in moving.items()}

        mov0 = {n: g for n, g in moving_at(0.0 if spec["range"][0] <= 0 <= spec["range"][1] else angles[0]).items() if g is not None}
        base = S.intersection(union(list(mov0.values()))) if S is not None and mov0 else None
        env, coll, first = [], [], {}
        ghosts = {}
        for a in angles:
            m = {n: g for n, g in moving_at(a).items() if g is not None}
            Ma = union(list(m.values()))
            if Ma is None:
                continue
            env.append(Ma)
            if S is not None:
                x = S.intersection(Ma)
                if base is not None:
                    x = x.difference(base.buffer(0.3))
                if x.area > 2.0:
                    coll.append(x)
                    sgn = "-" if a < 0 else "+"
                    first[sgn] = a if sgn not in first else (max(first[sgn], a) if a < 0 else min(first[sgn], a))
            if np.isclose(a, angles[0]) or np.isclose(a, angles[-1]) or np.isclose(a % 25, 0):
                ghosts[float(a)] = Ma
        E = union(env)
        E = E.buffer(2.5).buffer(-2.5) if E is not None else None   # close the gaps between 2.5-deg steps far out
        C = union(coll)
        if E is not None:
            E = E.intersection(clip)
        if C is not None:
            C = C.intersection(clip)
        ghosts = {a: g.intersection(clip) for a, g in ghosts.items()}
        # static internals as the moving side sees them over the range (moving link's neutral frame)
        rel = []
        for a in angles:
            q = {**q0, j: float(a)}
            Tm = np.linalg.inv(body.link_T(jl, q))
            for n, p in static.items():
                g = union(cut_polys(meshes[n], Tc @ Tm @ body.link_T(p["owning_link"], q)))
                if g is not None:
                    rel.append(g)
        R = union(rel)
        R = R.buffer(2.5).buffer(-2.5).intersection(clip) if R is not None else None
        M0 = union(list(mov0.values()))
        free_static = clip.difference(union([g for g in (S, E) if g is not None]))
        free_moving = clip.difference(union([g for g in (M0, R) if g is not None]))
        results.append(dict(title=title, ap=ap, stat=stat, mov0=mov0, S=S, E=E, C=C, ghosts=ghosts, R=R, M0=M0,
                            free_static=free_static, free_moving=free_moving, first=first))
    return moving, static, results


def contact_angles(results):
    """First contact per direction from the cuts (exact in each plane): the angle closest to 0 with new overlap."""
    out = {}
    for r in results:
        out[r["title"]] = {k: float(v) for k, v in r["first"].items()}
    return out


def sheet(body, name, spec, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    moving, static, res = analyse(body, spec)
    n = len(res)
    fig, axs = plt.subplots(n, 4, figsize=(30, 7.6 * n), squeeze=False)
    cmap = plt.get_cmap("viridis")
    lo, hi = spec["range"]
    for i, r in enumerate(res):
        ap = r["ap"]
        for k in range(4):
            ax = axs[i, k]
            ax.set_xlim(*spec["xlim"]); ax.set_ylim(*spec["ylim"]); ax.set_aspect("equal"); ax.grid(alpha=0.2)
            ax.set_xlabel(spec["ulab"]); ax.set_ylabel(spec["vlab"])
            for rad in (20, 40, 60):
                ax.add_patch(Circle(ap, rad, fill=False, ls=":", ec="#d33", lw=0.8))
            ax.plot(*ap, "+", color="#d00", ms=18, mew=2.5)
        a0, a1, a2, a3 = axs[i]
        # 1 standing
        for nm, g in r["stat"].items():
            col = np.array(static[nm].get("color") or [170, 170, 170, 255], float)[:3] / 255
            draw(a0, g, fc=col, ec="k", lw=0.5, alpha=0.9)
        for nm, g in r["mov0"].items():
            col = np.array(moving[nm].get("color") or [120, 120, 220, 255], float)[:3] / 255
            draw(a0, g, fc=(*col, 0.35), ec=col * 0.6, lw=1.6)
        lab = set()
        for nm, g in list(r["stat"].items()) + list(r["mov0"].items()):
            if g.area < 60:
                continue
            c = g.representative_point()
            key = nm.replace("R_", "").replace("combined__", "")[:22]
            if key in lab:
                continue
            lab.add(key)
            a0.annotate(key, (c.x, c.y), fontsize=7, ha="center", bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.3))
        a0.text(ap[0] + 3, ap[1] + 3, "turning point", color="#d00", fontsize=10)
        a0.set_title(f"{r['title']}\nstanding: filled = static, outlined = moves with {name}", fontsize=12)
        # 2 motion
        draw(a1, r["S"], fc="#9a9a9a", ec="k", lw=0.4)
        draw(a1, r["E"], fc="#f4b0b0", ec="#c0392b", lw=0.8, alpha=0.55)
        for a, g in sorted(r["ghosts"].items()):
            draw(a1, g, fc="none", ec=cmap((a - lo) / (hi - lo)), lw=1.2)
        draw(a1, r["C"], fc="#8b0000", ec="#8b0000", lw=0.5)
        f = r["first"]
        txt = ", ".join(f"first contact {f[s]:+.1f} deg" for s in ("-", "+") if s in f) or "no internal contact over the range"
        a1.set_title(f"{name} {lo:+.0f}..{hi:+.0f}: swept region (red), poses every 25 deg (yellow = {hi:+.0f})\n"
                     f"internal contact (dark red): {txt}", fontsize=12)
        # 3 static-side space
        draw(a2, r["free_static"], fc="#9ad19a", ec="#2e7d32", lw=0.8, alpha=0.7)
        draw(a2, r["S"], fc="#9a9a9a", ec="k", lw=0.4)
        draw(a2, r["E"], fc="#f4b0b0", ec="none", alpha=0.35)
        a2.set_title("space for a shell piece that does NOT move\nwith this joint (green)", fontsize=12)
        # 4 moving-side space
        draw(a3, r["free_moving"], fc="#9cc3ea", ec="#1f5fbf", lw=0.8, alpha=0.7)
        draw(a3, r["M0"], fc="#5a6fb0", ec="k", lw=0.4)
        draw(a3, r["R"], fc="#c9c9c9", ec="none", alpha=0.6)
        a3.set_title("space for a shell piece that moves WITH this joint (blue)\n"
                     "grey = static internals as that piece sees them over the range", fontsize=12)
    fig.suptitle(f"Hip {name} - bare skeleton and internals, no shells. {spec['note']}. Range {lo:+.0f}..{hi:+.0f} deg "
                 f"(other joints standing). Red cross = turning point, dotted rings r 20/40/60 mm.", fontsize=15, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    plt.savefig(out, dpi=60)
    plt.close(fig)
    return contact_angles(res)


def main():
    body = kit.base()
    which = [a for a in sys.argv[1:] if a in JOINTS] or list(JOINTS)
    summary = {}
    for name in which:
        out = HERE / f"cuts-hip-{name}.png"
        summary[name] = sheet(body, name, JOINTS[name], out)
        print(out, json.dumps(summary[name]))
    f = HERE / "cuts-contacts.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    old.update(summary)
    f.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
