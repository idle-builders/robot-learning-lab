"""Concept B2 'sketch crease' (lens B, converging geometry): look first.

The briefs' leg-opening line on the skin follows the sketch's V (front: z 186 at the crotch rising to z 229 at the
lateral hip, x 88; back: z 196 rising to z 226, both read on the actual briefs skin), but the opening's walls are still ruled surfaces through the hip pitch
axis, so the seam slopes inward toward the turning point and its width is r * (angle between wall and thigh), smallest
deep inside. The walls carry dark lips from the skin to 36 mm from the axis: the sketch's dark gap band. Same egg
thigh as B1. Flexion is whatever the V leaves (measured, not designed).

Run: python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B robot/body/hip-shell/converging/b2_sketch_crease.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import concepts as C  # noqa: E402
import cvg  # noqa: E402
import kit  # noqa: E402


def z_front(x):  # sketch V on the front skin
    return 186.0 + 43.0 * min(abs(x), 94.0) / 88.0


def z_back(x):  # back leg line on the back skin
    return 196.0 + 30.0 * min(abs(x), 94.0) / 88.0


def sketch_walls(env):
    """Wall angles about the pitch axis that put the seam on the sketch line where it meets the actual skin."""
    def wall(zf, side):
        def psi(x):
            z = zf(x)
            y = cvg.skin_y(env, max(abs(x), 0.0), z, side)
            if np.isnan(y):  # past the bowl: keep the last angle
                y = cvg.skin_y(env, 80.0, zf(80.0), side)
                z = zf(80.0)
            return float(np.degrees(np.arctan2(-(y - 2.0), 219.0 - z)))
        return psi
    return wall(z_front, -1.0), wall(z_back, 1.0)


body = kit.base()
parts, info = C.build(body, "b2_sketch_crease", x_out=100.0, psi_override=sketch_walls, r_lip=36.0,
                      lip_x_max=90.0)
info["sketch_edge"] = dict(front="z = 186 + 43 * x / 88 at the front skin", back="z = 196 + 30 * x / 88 at the back skin")
C.run(body, "b2_sketch_crease", parts, info,
      "B2 sketch crease: briefs edge on the sketch's V, walls through the hip pitch axis, dark lips as the gap band; egg thigh")
