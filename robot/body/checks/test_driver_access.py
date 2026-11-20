"""Synthetic cases for screwlib.driver_path: a screw head under a plate, without and with a hole above it, and a plate beside the path.
Run: PYTHONPATH=robot/body .venv/bin/python -B robot/body/checks/test_driver_access.py"""
import numpy as np
import trimesh

from checks import screwlib as SL
from checks.core import kit


def plate(hole_r=0.0, z=10.0, t=2.0, w=30.0, dx=0.0):
    m = trimesh.creation.box([w, w, t]); m.apply_translation([dx, 0, z + t / 2])
    if hole_r:
        h = trimesh.creation.cylinder(radius=hole_r, height=t * 3, sections=64); h.apply_translation([0, 0, z + t / 2])
        m = kit.tm(kit.mf(m) - kit.mf(h))
    return kit.mf(m)


def main():
    head, out, r = np.zeros(3), np.array([0.0, 0.0, 1.0]), 1.5       # tool blade dia 3 mm, 40 mm path
    full = SL.driver_path(head, out, r, 40.0, {"plate": plate()})
    tight = SL.driver_path(head, out, r, 40.0, {"plate": plate(hole_r=1.2)})
    open_ = SL.driver_path(head, out, r, 40.0, {"plate": plate(hole_r=3.0)})
    aside = SL.driver_path(head, out, r, 40.0, {"plate": plate(dx=40.0)})
    for nm, x in (("no hole", full), ("hole 2.4 mm dia", tight), ("hole 6 mm dia", open_), ("plate beside", aside)):
        print("%-16s obstruction %8.3f mm3, share %5.1f %%, first %s at %s mm" % (nm, x["obstruction_mm3"], x["blocked_share_pct"], x["first_part"], x["first_distance_mm"]))
    # full plate: the whole 2 mm thick path through it, first contact 10 mm above the head
    want = np.pi * r ** 2 * 2.0
    assert abs(full["obstruction_mm3"] - want) < 0.03 * want and abs(full["first_distance_mm"] - 10.0) < 0.1
    # hole smaller than the blade: the ring between hole and blade is still in the way
    want_t = (np.pi * r ** 2 - np.pi * 1.2 ** 2) * 2.0
    assert abs(tight["obstruction_mm3"] - want_t) < 0.06 * want_t and abs(tight["first_distance_mm"] - 10.0) < 0.1
    assert open_["obstruction_mm3"] == 0.0 and open_["first_part"] is None
    assert aside["obstruction_mm3"] == 0.0
    print("ok")


main()
