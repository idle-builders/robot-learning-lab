"""Synthetic shapes with known answers for modules/fits.py. Run: PYTHONPATH=robot/body .venv/bin/python -B robot/body/checks/test_fits.py"""
import numpy as np
import trimesh

from checks.modules import fits


def block(w=20.0, h=10.0, chamfer=0.0):
    if chamfer:
        a, b = w / 2 - chamfer, w / 2
        pts = [(sx * x, sy * x, z) for x, z in ((a, 0.0), (b, chamfer), (b, h)) for sx in (-1, 1) for sy in (-1, 1)]
        pts += [(sx * b, sy * b, h) for sx in (-1, 1) for sy in (-1, 1)]
        m = trimesh.convex.convex_hull(np.array(pts))
    else:
        m = trimesh.creation.box([w, w, h]); m.apply_translation([0, 0, h / 2])
        return m
    m.apply_translation([0, 0, 0])
    return m


def touching(q):
    return np.zeros(len(q)), np.array(["N"] * len(q), dtype=object)


def far(q):
    return np.full(len(q), 5.0), np.array([None] * len(q), dtype=object)


def main():
    sq = fits.foot(block(), touching)
    ch = fits.foot(block(chamfer=0.4), touching)
    nb = fits.foot(block(), far)
    print("square foot   mating %.1f mm, no chamfer %.1f mm, area %.1f mm2, inset %s" % (sq["mating_outline_mm"], sq["no_chamfer_outline_mm"], sq["no_chamfer_area_mm2"], sq["smallest_inset_mm"]))
    print("0.4 chamfer   mating %.1f mm, no chamfer %.1f mm, area %.1f mm2, inset %s" % (ch["mating_outline_mm"], ch["no_chamfer_outline_mm"], ch["no_chamfer_area_mm2"], ch["smallest_inset_mm"]))
    print("no neighbour  mating %.1f mm" % nb["mating_outline_mm"])
    assert abs(sq["no_chamfer_outline_mm"] - 80) < 1 and abs(sq["no_chamfer_area_mm2"] - 40) < 1
    assert ch["no_chamfer_outline_mm"] == 0 and abs(ch["smallest_inset_mm"] - 0.4) < 0.05
    assert nb["mating_outline_mm"] == 0
    # two blocks at a 0.15 mm gap and at 0.3 mm overlap
    from checks.core import kit
    a = trimesh.creation.box([10, 10, 10]); b = a.copy(); b.apply_translation([10.15, 0, 0])
    d = kit.mf(a).min_gap(kit.mf(b), 6.0)
    print("gap 0.15 -> min_gap %.3f" % d)
    assert abs(d - 0.15) < 1e-3
    b.apply_translation([-0.45, 0, 0])               # 9.7 apart: 0.3 overlap
    p = fits.penetration(a, b)
    print("overlap 0.3 -> depth %.3f" % p)
    assert abs(p - 0.3) < 0.02
    cl = fits.table()["classes"]
    for c, k in ((0.15, "slide"), (0.5, "slide"), (0.05, "slide"), (-0.1, "press"), (0.2, "press")):
        print("clearance %5.2f vs %-5s -> %s" % (c, k, fits.judge(c, k, cl)))
    assert fits.judge(0.15, "slide", cl)["margin_mm"] == 0.0 and fits.judge(0.5, "slide", cl)["margin_mm"] == -0.15
    assert fits.judge(0.2, "press", cl)["side"] == "too loose" and fits.judge(0.05, "slide", cl)["side"] == "too tight"
    print("ok")


main()
