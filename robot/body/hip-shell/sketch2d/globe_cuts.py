"""S3 side cuts of the hip-globe concepts (Idle T-czv2d8, D-01, D-07): every concept's shell layers cut through the
hip centre C, with the servos, the yaw U-bracket and the thigh fork in the same cut, posed with the whole leg.

Columns: side cut x = 59 (standing, squat -85/95, lunge rear +40/40), front cut y = 2 (standing, leg out -40,
climb -85/-30/+30/95). Rows: the three concepts from globe-concepts.json (cached pieces in .cache/work/hip-shell).
Colours: grey = torso (keel, cradle plate, yaw servo, hood), green = yaw link (U-bracket), orange = roll link (servo
block, tread), blue = thigh (fork, knee servo, thigh layer). Dashed = the torso shell's raised hem.

Run after globe_concepts.py:
  python3 infra/tools/memguard.py --limit-gb 4 -- .venv/bin/python -B robot/body/hip-shell/sketch2d/globe_cuts.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import globe_concepts as G  # noqa: E402
import hip_layouts as H  # noqa: E402

COL = {"torso": (0.55, 0.55, 0.55), "J1": (0.2, 0.65, 0.45), "J2": (0.95, 0.55, 0.15), "J3": (0.25, 0.45, 0.85),
       "shin": (0.55, 0.35, 0.75)}
CUTS = [("side cut x 59, standing", "side", dict(pitch=0, roll=0, yaw=0, knee=0)),
        ("side cut x 59, squat -85 / knee 95", "side", dict(pitch=-85, roll=0, yaw=0, knee=95)),
        ("side cut x 59, lunge rear +40 / knee 40", "side", dict(pitch=40, roll=0, yaw=0, knee=40)),
        ("front cut y 2, standing", "front", dict(pitch=0, roll=0, yaw=0, knee=0)),
        ("front cut y 2, leg out -40 / -20", "front", dict(pitch=-20, roll=-40, yaw=0, knee=20)),
        ("front cut y 2, turn toes out 40", "front", dict(pitch=0, roll=0, yaw=40, knee=0))]


def section(m, T, plane):
    mm = m.copy(); mm.apply_transform(T)
    o, n = (np.array([59.0, 0, 0]), np.array([1.0, 0, 0])) if plane == "side" else (np.array([0, 2.0, 0]), np.array([0, 1.0, 0]))
    try:
        segs = trimesh.intersections.mesh_plane(mm, n, o)
    except Exception:
        return np.zeros((0, 2, 3))
    return segs


def uv(p, plane):
    return np.stack([-p[..., 1], p[..., 2]], -1) if plane == "side" else np.stack([-p[..., 0], p[..., 2]], -1)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    store = pickle.load(open(G.WORK / "globe-concepts.pkl", "rb"))
    res = json.loads((HERE / "globe-concepts.json").read_text())
    lay, parts = G.internals()
    parts = [(n, m, l) for n, m, l in parts if not n.startswith("left")]
    fig, axs = plt.subplots(len(store), len(CUTS), figsize=(5.2 * len(CUTS), 5.6 * len(store)), squeeze=False)
    for i, (cname, st) in enumerate(store.items()):
        for j, (lab, plane, q) in enumerate(CUTS):
            ax = axs[i, j]
            Ts = H.link_T(lay, q)
            for n, m, l in parts:
                segs = section(m, Ts[l], plane)
                if len(segs):
                    ax.add_collection(LineCollection(uv(segs, plane), colors=[COL[l]], linewidths=1.6))
            for pn, owner, c, m in st["pieces"]:
                segs = section(m, Ts[owner], plane)
                if len(segs):
                    lw = 3.2 if owner != "torso" else 2.4
                    ax.add_collection(LineCollection(uv(segs, plane), colors=[COL[owner]], linewidths=lw, alpha=0.9))
            # torso walls above the hem (side cut: front and back walls; front cut: the outboard wall)
            hm = st["hem"]
            if plane == "side":
                for w, y in (("front", G.TORSO["y0"]), ("back", G.TORSO["y1"])):
                    s, h = hm[w]
                    z = float(np.interp(59.0, s, h))
                    ax.plot([-y, -y], [z, 330], color="k", lw=1.2)
                    ax.plot([-y - 6, -y + 6], [z, z], color="r", lw=2)
            else:
                s, h = hm["outboard"]
                z = float(np.interp(2.0, s, h))
                ax.plot([-90, -90], [z, 330], color="k", lw=1.2); ax.plot([-96, -84], [z, z], color="r", lw=2)
                ax.plot([90, 90], [z, 330], color="k", lw=1.2)
            c = G.C
            cu = (-c[1], c[2]) if plane == "side" else (-c[0], c[2])
            ax.plot(*cu, "k+", ms=14, mew=2)
            ax.set_xlim(cu[0] - 115, cu[0] + 115); ax.set_ylim(110, 330); ax.set_aspect("equal"); ax.grid(alpha=0.2)
            ax.set_title(lab, fontsize=11)
            if j == 0:
                ax.set_ylabel(res[cname]["name"].split(":")[0] + ("\n" + cname.split(", ")[1] if ", " in cname else ""), fontsize=13)
            if plane == "side":
                ax.text(cu[0] - 110, 115, "front", fontsize=9); ax.text(cu[0] + 90, 115, "back", fontsize=9)
            else:
                ax.text(cu[0] - 110, 115, "outboard", fontsize=9); ax.text(cu[0] + 80, 115, "inboard", fontsize=9)
    fig.suptitle("Hip-globe concepts cut through the hip centre: servos (orange block), yaw U-bracket (green), thigh fork "
                 "and knee servo (blue), battery keel / cradle / yaw servo (grey); thick = shell layers; black = torso wall "
                 "above its raised hem (red tick)", fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out = HERE / "globe-cuts.png"
    plt.savefig(out, dpi=55); plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
