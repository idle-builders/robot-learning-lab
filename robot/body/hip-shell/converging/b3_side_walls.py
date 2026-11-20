"""Concept B3 'side skirt' (lens B, converging geometry): seams in pitch planes, balanced range.

Hip pitch moves every leg point inside its own plane x = const, so a fixed panel lying lateral of the leg's whole x
range (x > 91, the thigh ends at x 89) is never swept by pitch. B3 adds such a skirt to the briefs over the hip joint
(x 91.5..94.5, z 207..240, y -12..42: behind the flexed thigh, above the egg top so yaw cannot reach it) and opens the
front/back with the axis wedge (walls through the pitch axis) set for -75..+40, between B1 (-90) and B2 (sketch V).
Same egg thigh and dark lips as B1. What it tests: whether covering the side costs roll/yaw.

Run: python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B robot/body/hip-shell/converging/b3_side_walls.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import concepts as C  # noqa: E402
import cvg  # noqa: E402
import kit  # noqa: E402
import manifold3d as M  # noqa: E402

body = kit.base()
skirt = (M.Manifold.cube((3.0, 54.0, 33.0)).translate((91.5, -12.0, 207.0))
         ^ cvg.ell((92.0, 15.0, 226.0), (40.0, 30.0, 21.0), 48))
parts, info = C.build(body, "b3_side_walls", flex=75.0, ext=40.0, rolls=(-15.0, 0.0, 15.0), r_lip=40.0,
                      x_out=90.0, lip_x_max=86.0, skirt=skirt)
info["skirt"] = "x 91.5..94.5, y -12..42, z 207..240 (rounded), both sides, part of the briefs"
C.run(body, "b3_side_walls", parts, info,
      "B3 side skirt: fixed hip panel lateral of the thigh (pitch-free), axis wedge set for -75..+40, lips; egg thigh")
