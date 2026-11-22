"""S1 (redone): whole-leg motion analysis of the bare skeleton (no shells), Idle T-czv2d8.

The hip actuators move together, so this analysis poses the whole right leg (hip yaw, pitch, roll and knee) through a
library of realistic poses (A-11) and through every combination of the target ranges (A-12), including the path from
standing into each pose. It looks at the hip from the point of view of every link a shell piece could ride on (torso,
yaw link, pitch link, thigh): in that link's frame, everything on the other links moves, and a shell piece on the link
must stay out of where they go.

Outputs (in this folder):
  leg-motion-poses.png             every library pose, side and front view of all motors and connectors, contacts in red
  leg-motion-<link>-side.png       cuts x = 20/36/59/80 in the link's frame
  leg-motion-<link>-front.png      cuts y = -15/2/20 in the link's frame
  leg-motion-contacts.json         3D overlap (mm3) between parts on different links, per pose

Run: python3 infra/tools/memguard.py --limit-gb 8 -- .venv/bin/python -B robot/body/hip-shell/sections/leg_motion.py
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import kit  # noqa: E402
from joint_cuts import cut_polys, draw, front, side, skeleton_parts, union  # noqa: E402

SOLIDS = kit.ROOT / ".cache/work/hip-shell/solids"
VOX = 0.75
CLIP_R = 110.0
C = np.array([59.0, 2.0, 219.0])
J = ("R_hip_pitch", "R_hip_roll", "R_hip_yaw", "R_knee")

# A-11: realistic whole-leg poses, (pitch, roll, yaw, knee) in deg; pitch < 0 = flexion, roll < 0 = leg out,
# yaw > 0 = right toes out (measured: yaw +30 moves the toe to +x), knee > 0 = bent. Both yaw signs where the motion
# can go either way. Left leg stays standing (A-13).
LIBRARY = {
    "walk: swing": [(-35, 0, 0, 40), (-15, 0, 0, 60)],
    "walk: heel strike / push-off": [(-25, 0, 0, 5), (20, 0, 0, 30)],
    "walk: weight shift / turn": [(0, -8, 0, 10), (0, 8, 0, 10), (-25, 0, 15, 20), (-25, 0, -15, 20)],
    "high knee / jump landing": [(-70, 0, 0, 95), (-60, -10, 0, 95)],
    "squat / sit": [(-85, 0, 0, 95)],
    "wide squat, toes out / in": [(-85, -20, 25, 95), (-85, -20, -25, 95)],
    "front kick": [(-85, 0, 0, 0), (-85, 0, 0, 30)],
    "lunge, rear leg": [(40, 0, 0, 0), (40, 0, 0, 40)],
    "side step / leg out": [(0, -45, 0, 0), (-30, -45, 0, 30), (-30, -30, 20, 30)],
    "cross step": [(-20, 25, 30, 20), (-20, 25, -30, 20)],
    "turn in place": [(-20, 0, 40, 20), (-20, 0, -40, 20), (0, -10, 40, 0), (0, -10, -40, 0)],
    "ape crouch": [(-70, -20, 20, 95), (-70, -20, -20, 95)],
    "climb step": [(-85, -30, 30, 95), (-85, -30, -30, 95)],
}
KEY = {"squat": (-85, 0, 0, 95), "front kick": (-85, 0, 0, 0), "lunge rear": (40, 0, 0, 40),
       "leg out": (0, -45, 0, 0), "cross step": (-20, 25, 30, 20), "turn, toes out": (-20, 0, 40, 20),
       "climb, toes in": (-85, -30, -30, 95)}
KEY_COL = dict(zip(KEY, ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b", "#e377c2"]))
STEPS = 5          # path from standing into each pose, joint-space interpolation
# A-12: every combination of the target ranges (conservative bound)
GRID = list(itertools.product((-85, -60, -30, 0, 20, 40), (-45, -25, 0, 25), (-40, 0, 40), (0,)))
FRAMES = {"torso": "torso", "yaw link": "R_yaw_link", "pitch link": "R_pitch_link", "thigh (roll link)": "R_roll_link"}
SIDE_CUTS = [("x = 20", side(20)), ("x = 36", side(36)), ("x = 59, through C", side(59)), ("x = 80", side(80))]
FRONT_CUTS = [("y = -15", front(-15)), ("y = 2, through C", front(2)), ("y = 20", front(20))]
ARTEFACT_BASE = 1000.0   # A-14: a pair overlapping more than this standing is assembled inside one another; its change is
                         # not counted as contact (voxel filling of open meshes)
MECH_ROLES = {"motor", "purchased_servo_case_family_reference", "printed_structural_connector", "output",
              "purchased_servo_output_family_reference", "output_horn", "horn", "actuator", "bearing"}


def q_of(body, p):
    return {**body.standing, **dict(zip(J, map(float, p)))}


def library_poses():
    out = [("standing", (0, 0, 0, 0), True)]
    for fam, ps in LIBRARY.items():
        for p in ps:
            for k in range(1, STEPS + 1):
                out.append((fam, tuple(np.array(p, float) * k / STEPS), k == STEPS))
    return out


def solid(body, n):
    """Closed mesh of a part (voxel-filled when the CAD mesh is open), cached on disk."""
    p = body.parts[n]
    tag = kit.mesh_tag(p)                  # the file's own sha256, not the manifest's (stale after an in-place rebuild)
    f = SOLIDS / f"{n}-{tag}.npz"
    if f.exists():
        d = np.load(f); return trimesh.Trimesh(d["v"], d["f"], process=False)
    m = kit._mesh(body, n, 6000)
    if not m.is_watertight:
        from scipy.ndimage import binary_closing, binary_fill_holes
        from skimage.measure import marching_cubes
        full = body.mesh(n)
        lo = full.bounds[0] - 2 * VOX; shape = np.ceil((full.bounds[1] - lo) / VOX).astype(int) + 3
        pts = np.vstack([full.sample(int(min(3e6, max(5e4, full.area / (0.35 * VOX) ** 2)))), full.vertices])
        idx = np.clip(np.round((pts - lo) / VOX).astype(int), 0, shape - 1)
        g = np.zeros(shape, bool); g[tuple(idx.T)] = True
        g = binary_fill_holes(binary_closing(g, iterations=1))
        v, fc, _, _ = marching_cubes(np.pad(g, 1).astype(np.float32), 0.5)
        m = trimesh.Trimesh((v - 1) * VOX + lo, fc, process=True)
        if len(m.faces) > 8000:
            m = m.simplify_quadric_decimation(face_count=8000)
        if not m.is_watertight:
            m = m.convex_hull
    SOLIDS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, v=np.asarray(m.vertices), f=np.asarray(m.faces))
    return m


def short(n):
    s = n.replace("R_", "").replace("combined__", "").replace("_reference", "").replace("_combined", "")
    return s.replace("stage2_lower_pelvis_", "").replace("Honeycomb_", "")[:20]


# ---------------------------------------------------------------- 3D contacts
def contacts(body, parts, sol, poses):
    import manifold3d as M
    mf = {n: kit.mf(sol[n]) for n in parts}
    link = {n: p["owning_link"] for n, p in parts.items()}
    names = list(parts)

    def overlaps(q):
        T = {l: body.link_T(l, q) for l in set(link.values())}
        posed = {n: mf[n].transform(T[link[n]][:3, :4]) for n in names}
        bb = {n: np.array(posed[n].bounding_box()).reshape(2, 3) for n in names}
        res = {}
        for a, b in itertools.combinations(names, 2):
            if link[a] == link[b]:
                continue
            if np.any(bb[a][1] < bb[b][0]) or np.any(bb[b][1] < bb[a][0]):
                continue
            v = (posed[a] ^ posed[b]).volume()
            if v > 0.5:
                res[(a, b)] = v
        return res

    base = overlaps(q_of(body, (0, 0, 0, 0)))
    out = []
    for fam, p, end in poses:
        o = overlaps(q_of(body, p))
        new = {k: v - base.get(k, 0.0) for k, v in o.items() if v > base.get(k, 0.0) + 2.0}
        real = {k: v for k, v in new.items() if base.get(k, 0.0) < ARTEFACT_BASE}
        out.append(dict(family=fam, pose_pitch_roll_yaw_knee=[round(x, 1) for x in p], end_pose=end,
                        new_overlap_mm3=round(sum(new.values()), 1),
                        contact_mm3=round(sum(real.values()), 1),
                        contact_parts=[[a, b, round(v, 1)] for (a, b), v in sorted(real.items(), key=lambda t: -t[1]) if v > 20],
                        pairs={f"{short(a)} x {short(b)}": round(v, 1) for (a, b), v in sorted(new.items(), key=lambda t: -t[1])[:4]}))
    return dict(baseline_standing={f"{short(a)} x {short(b)}": round(v, 1) for (a, b), v in base.items()}, poses=out)


# ---------------------------------------------------------------- views
def ortho(ax, items, view, lim):
    """Flat-shaded orthographic view (painter's algorithm). view 'side' looks from +x (forward to the right),
    'front' looks from the front (-y)."""
    from matplotlib.collections import PolyCollection
    R = np.array([[0, -1, 0], [0, 0, 1], [1, 0, 0]], float) if view == "side" else np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], float)
    L = np.array([0.3, -0.6, 0.7]); L /= np.linalg.norm(L)
    P, Cc, D = [], [], []
    for m, T, c, a in items:
        v = (m.vertices @ T[:3, :3].T + T[:3, 3]) @ R.T
        nn = m.face_normals @ T[:3, :3].T
        tri = v[m.faces]; vis = (nn @ R.T)[:, 2] > -0.05
        sh = 0.45 + 0.55 * np.clip(np.abs(nn @ L), 0, 1)
        P.extend(tri[vis][:, :, :2]); D.extend(tri[vis][:, :, 2].mean(1))
        Cc.extend([(*np.clip(c * s, 0, 1), a) for s in sh[vis]])
    o = np.argsort(np.array(D))
    ax.add_collection(PolyCollection([P[i] for i in o], facecolors=[Cc[i] for i in o], edgecolors="none"))
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_aspect("equal"); ax.axis("off")


LINK_COL = {"torso": np.array([0.62, 0.62, 0.62]), "R_yaw_link": np.array([0.2, 0.65, 0.45]),
            "R_pitch_link": np.array([0.95, 0.55, 0.15]), "R_roll_link": np.array([0.25, 0.45, 0.85]),
            "R_shin": np.array([0.55, 0.35, 0.75]), "R_ankle_pitch": np.array([0.55, 0.35, 0.75]),
            "R_foot": np.array([0.55, 0.35, 0.75])}


def pose_sheet(body, parts, sol, cres, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ends = [("standing", (0, 0, 0, 0))] + [(f, p) for f, ps in LIBRARY.items() for p in ps]
    worst = {}
    for r in cres["poses"]:
        if r["end_pose"] or r["family"] == "standing":
            key = tuple(r["pose_pitch_roll_yaw_knee"])
        else:
            continue
        worst[key] = r
    path_worst = {}
    for r in cres["poses"]:
        fp = r["family"]; path_worst[fp] = max(path_worst.get(fp, 0), r["new_overlap_mm3"])
    n = len(ends); cols = 6; rows = int(np.ceil(n * 2 / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 5.6))
    axs = axs.ravel()
    k = 0
    for fam, p in ends:
        q = q_of(body, p)
        items = []
        r = worst.get(tuple(float(x) for x in np.round(p, 1)))
        hit = set()
        if r:
            for a, b_, v in r.get("contact_parts", []):
                hit.update((a, b_))
        for nme, pp in parts.items():
            lk = pp["owning_link"]
            col = LINK_COL.get(lk, np.array([0.7, 0.7, 0.7]))
            if lk.startswith("L_"):
                col = np.array([0.82, 0.82, 0.82])
            if nme in hit:
                col = np.array([0.9, 0.05, 0.05])
            items.append((sol[nme], body.link_T(lk, q), col, 1.0))
        for view, lim in (("side", ((-130, 130), (40, 330))), ("front", ((-150, 60), (40, 330)))):
            ax = axs[k]; k += 1
            ortho(ax, items, view, lim)
            ov = r.get("contact_mm3", 0.0) if r else 0.0
            ax.set_title(f"{fam}\n{view}: pitch {p[0]:+.0f} roll {p[1]:+.0f} yaw {p[2]:+.0f} knee {p[3]:.0f}\n"
                         f"contact {ov:.0f} mm3", fontsize=10, color="#b00" if ov > 20 else "k")
    for ax in axs[k:]:
        ax.axis("off")
    fig.suptitle("Whole-leg poses (A-11), bare skeleton: all motors, connectors and carriers. Grey = torso, green = yaw "
                 "link, orange = pitch link, blue = thigh, purple = shin, light = left leg (standing); red = parts in contact "
                 "at this pose (> 20 mm3, assembled pairs excluded, A-14)", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.975)); plt.savefig(out, dpi=50); plt.close(fig)


# ---------------------------------------------------------------- cuts in a link's frame
def frame_cuts(body, parts, sol, frame, cuts, lib, grid, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from shapely.geometry import Point
    own = {n: p for n, p in parts.items() if p["owning_link"] == frame}
    other = {n: p for n, p in parts.items() if p["owning_link"] != frame}
    q0 = q_of(body, (0, 0, 0, 0))
    T0 = body.link_T(frame, q0)
    fig, axs = plt.subplots(len(cuts), 3, figsize=(27, 8.6 * len(cuts)), squeeze=False)
    summary = {}
    side_view = cuts is SIDE_CUTS
    lim = ((-125, 125), (85, 345)) if side_view else ((-175, 45), (85, 345))
    for i, (ct, Tc) in enumerate(cuts):
        ap = (Tc @ np.r_[C, 1.0])[:2]
        clip = Point(*ap).buffer(CLIP_R, 128)

        def secs(pose, which):
            q = q_of(body, pose)
            Tf = np.linalg.inv(body.link_T(frame, q))
            return {n: union(cut_polys(sol[n], Tc @ Tf @ body.link_T(p["owning_link"], q))) for n, p in which.items()}

        own0 = {n: g for n, g in secs((0, 0, 0, 0), own).items() if g is not None}
        O = union(list(own0.values()))
        lib_env, grid_env, coll = [], [], []
        base_o = union([g for g in secs((0, 0, 0, 0), other).values() if g is not None])
        base_c = O.intersection(base_o) if O is not None and base_o is not None else None
        for _, p, _ in lib:
            g = union([x for x in secs(p, other).values() if x is not None])
            if g is None:
                continue
            lib_env.append(g)
            if O is not None:
                x = O.intersection(g)
                if base_c is not None:
                    x = x.difference(base_c.buffer(0.3))
                if x.area > 2:
                    coll.append(x)
        for p in grid:
            g = union([x for x in secs(p, other).values() if x is not None])
            if g is not None:
                grid_env.append(g)
        E = union(lib_env); E = E.intersection(clip) if E is not None else None
        G = union(grid_env); G = G.intersection(clip) if G is not None else None
        X = union(coll)
        free = clip.difference(union([g for g in (O, E) if g is not None]))
        free_grid = clip.difference(union([g for g in (O, G) if g is not None]))
        a0, a1, a2 = axs[i]
        for ax in (a0, a1, a2):
            ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_aspect("equal"); ax.grid(alpha=0.2)
            ax.set_xlabel("forward (-y) mm" if side_view else "robot right (+x) <-  mm"); ax.set_ylabel("z mm")
            for rad in (30, 50, 70):
                ax.add_patch(Circle(ap, rad, fill=False, ls=":", ec="#d33", lw=0.7))
            ax.plot(*ap, "+", color="#d00", ms=16, mew=2.5)
        # 1: own parts + other links' motors and connectors at key poses
        for n, g in own0.items():
            col = np.array(parts[n].get("color") or [160, 160, 160, 255], float)[:3] / 255
            draw(a0, g, fc=col, ec="k", lw=0.6)
        labelled = set()
        for n, g in own0.items():
            if g.area > 40 and short(n) not in labelled:
                c = g.representative_point(); labelled.add(short(n))
                a0.annotate(short(n), (c.x, c.y), fontsize=7, ha="center", bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.3))
        st = secs((0, 0, 0, 0), {n: p for n, p in other.items() if p.get("role") in MECH_ROLES or "battery" in (p.get("role") or "")})
        for n, g in st.items():
            if g is None:
                continue
            draw(a0, g, fc=(0.5, 0.5, 0.5, 0.15), ec="#333", lw=1.2)
            if g.area > 40 and short(n) not in labelled:
                c = g.representative_point(); labelled.add(short(n))
                a0.annotate(short(n), (c.x, c.y), fontsize=7, ha="center", color="#333",
                            bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.3))
        mech = {n: p for n, p in other.items() if p.get("role") in MECH_ROLES}
        for kname, p in KEY.items():
            for n, g in secs(p, mech).items():
                if g is not None:
                    draw(a0, g.intersection(clip), fc="none", ec=KEY_COL[kname], lw=1.3)
        for kname in KEY:
            a0.plot([], [], color=KEY_COL[kname], lw=2, label=kname)
        a0.legend(loc="lower left", fontsize=8, ncol=2)
        a0.set_title(f"{ct}: filled = parts on the {title}; grey outline = other motors/connectors standing;\n"
                     "coloured = other motors/connectors at the key poses", fontsize=11)
        # 2: envelopes and contacts
        draw(a1, G, fc="#fde0e0", ec="none")
        draw(a1, E, fc="#f19a9a", ec="#c0392b", lw=0.8, alpha=0.75)
        draw(a1, O, fc="#5a5a5a", ec="k", lw=0.5)
        draw(a1, X, fc="#8b0000", ec="#8b0000", lw=0.5)
        a1.set_title(f"where the other links go: realistic poses incl. the path into them (red), all target combinations "
                     f"(pink)\ncontact with parts on the {title} (dark red): {X.area if X is not None else 0:.0f} mm2 in this cut",
                     fontsize=11)
        # 3: free space for a shell piece on this frame
        draw(a2, free_grid, fc="#2e7d32", ec="none", alpha=0.35)
        draw(a2, free, fc="#9ad19a", ec="#2e7d32", lw=0.8, alpha=0.6)
        draw(a2, O, fc="#5a5a5a", ec="k", lw=0.5)
        a2.set_title(f"space for a shell piece riding on the {title}: light green = free over the realistic poses,\n"
                     "dark green = free even over all target combinations", fontsize=11)
        summary[ct] = dict(contact_mm2=round(X.area, 1) if X is not None else 0.0,
                           free_realistic_mm2=round(free.area, 0), free_all_combinations_mm2=round(free_grid.area, 0))
    fig.suptitle(f"Hip seen from the {title} (cuts in its own frame), bare skeleton, right leg through {len(lib)} "
                 f"library poses incl. paths (A-11) and {len(grid)} combinations of the target ranges (A-12). "
                 "Red cross = hip centre C, rings r 30/50/70.", fontsize=15, y=0.997)
    plt.tight_layout(rect=(0, 0, 1, 0.99)); plt.savefig(out, dpi=50); plt.close(fig)
    return summary


def main():
    body = kit.base()
    parts = {n: p for n, p in skeleton_parts(body).items()}
    sol = {n: solid(body, n) for n in parts}
    lib = library_poses()
    which = sys.argv[1:] or ["contacts", "poses", "cuts"]
    cf = HERE / "leg-motion-contacts.json"
    if "contacts" in which:
        cres = contacts(body, parts, sol, lib + [("all combinations", g, True) for g in GRID])
        cf.write_text(json.dumps(cres, indent=1))
    cres = json.loads(cf.read_text())
    if "poses" in which:
        pose_sheet(body, parts, sol, cres, HERE / "leg-motion-poses.png")
    if "cuts" in which:
        summ = {}
        for title, frame in FRAMES.items():
            tag = frame.replace("R_", "").replace("_link", "")
            for cuts, view in ((SIDE_CUTS, "side"), (FRONT_CUTS, "front")):
                summ[f"{title} {view}"] = frame_cuts(body, parts, sol, frame, cuts, lib, GRID,
                                                     HERE / f"leg-motion-{tag}-{view}.png", title)
                print(title, view, json.dumps(summ[f"{title} {view}"]), flush=True)
        (HERE / "leg-motion-space.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
