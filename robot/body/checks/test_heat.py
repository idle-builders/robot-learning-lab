"""Synthetic two blocks with known contact area / distance for modules/heat.py. Run: PYTHONPATH=robot/body .venv/bin/python -B robot/body/checks/test_heat.py"""
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from types import SimpleNamespace

from checks.modules import heat, overhang


def main():
    tab = heat.table()
    # source: 20 x 20 x 10 block, z 0..10. printed part: 10 x 10 x 10 block resting on it at gap g, offset so 10 x 10 of its bottom is over the source.
    src = trimesh.creation.box([20, 20, 10]); src.apply_translation([0, 0, 5])
    pts_src = np.vstack([trimesh.sample.sample_surface(src, 200000, seed=1)[0], src.vertices])
    overhang._TREES["S"] = cKDTree(pts_src)
    T = np.eye(4)
    body = SimpleNamespace(standing=None, link_T=lambda lk, q: T)
    ctx = SimpleNamespace(body=body, parts={"S": dict(owning_link="l", bounds_mm=src.bounds.tolist()), "P": dict(owning_link="l")}, link=lambda n: "l")
    srcs = {"S": ("servo_case", tab["kinds"]["servo_case"])}
    for gap, want_area, want_close in ((0.0, 100.0, 0.0), (2.0, 0.0, 2.0)):
        p = trimesh.creation.box([10, 10, 10]); p.apply_translation([0, 0, 10 + 5 + gap])
        pts, _ = trimesh.sample.sample_surface(p, int(p.area / 0.5 ** 2), seed=1)
        r = heat.measure(ctx, "P", pts, np.full(len(pts), p.area / len(pts)), tab, srcs)["S"]
        print("gap %.1f -> contact %.1f mm2 (want %.1f), closest %.2f mm (want %.1f), within 3 mm %.1f mm2" % (gap, r["contact_area_mm2"], want_area, r["closest_mm"], want_close, r["area_within_near_mm2"]))
        assert abs(r["contact_area_mm2"] - want_area) < 0.1 * max(want_area, 1) and abs(r["closest_mm"] - want_close) < 0.3
    assert heat._margins(dict(hdt_045_C=74.0, hdt_18_C=68.0, glass_transition_C=68.0), 70.0) == dict(margin_hdt_045_C=4.0, margin_hdt_18_C=-2.0, margin_tg_C=-2.0)
    print("ok")


main()
