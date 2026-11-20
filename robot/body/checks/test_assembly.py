"""Synthetic case for assembly.sweep_insertion: a 10 mm block inserted into an open box (free path), then with a 2 mm lid fitted first.
Run: PYTHONPATH=robot/body .venv/bin/python -B robot/body/checks/test_assembly.py"""
import trimesh

from checks.core import kit
from checks.modules import assembly as A


def box(size, lo):
    m = trimesh.creation.box(size); m.apply_translation([lo[i] + size[i] / 2 for i in range(3)])
    return kit.mf(m)


def main():
    shell = box([40, 40, 30], [-20, -20, 0]) - box([36, 36, 30], [-18, -18, 2])      # open at the top (z = 30)
    lid = box([40, 40, 2], [-20, -20, 30])                                              # lid z 30..32
    block = box([10, 10, 10], [-5, -5, 2])                                              # final position on the floor
    d = [0, 0, -1]                                                                      # travels downward, comes from above
    free = A.sweep_insertion(block, d, {"box": shell})
    shut = A.sweep_insertion(block, d, {"box": shell, "lid": lid})
    print("open box:", {k: free[k] for k in ("seat_overlap_mm3", "max_overlap_mm3", "first_contact_mm", "blocked_samples")})
    print("lid on :", {k: shut[k] for k in ("seat_overlap_mm3", "max_overlap_mm3", "max_overlap_at_mm", "first_contact_mm", "first_part", "blocked_samples")})
    assert free["first_contact_mm"] is None and free["max_overlap_mm3"] == 0 and free["seat_overlap_mm3"] == 0
    # block top at 12 + t reaches the lid underside z = 30 at t = 18; coming from above the first touch is its top face: block bottom 2 + t = 32 -> t = 30
    assert abs(shut["first_contact_mm"] - 30.0) < 0.02 and shut["first_part"] == "lid"
    assert abs(shut["max_overlap_mm3"] - 200.0) < 0.5      # block fully inside the 2 mm lid: 10 x 10 x 2
    assert shut["blocked_samples"] == 11                   # overlap for 18 < t < 30: samples 19..29
    print("ok")


main()
