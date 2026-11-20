"""Concept B1 'axis wedge' (lens B, converging geometry): full-range first.

Fixed briefs whose leg opening is a wedge about the hip pitch axis: its front and back walls are ruled surfaces
through the axis (the owner's 'slope inward toward the turning point'), placed from the leg's own angular envelope so
the leg reaches pitch -90..+40 (roll -15..15 allowed for in the envelope). Dark funnel lips lie in the walls from the
skin to 40 mm from the axis and read as the gap band. Thigh: egg-topped shell on the roll link converging toward C,
its open top cut by the roll sweep of the pitch-link mechanism. A narrow crotch strip (|x| < 6) stays.

Run: python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B robot/body/hip-shell/converging/b1_axis_wedge.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import concepts as C  # noqa: E402
import kit  # noqa: E402

body = kit.base()
parts, info = C.build(body, "b1_axis_wedge", flex=90.0, ext=40.0, rolls=(-15.0, 0.0, 15.0), r_lip=40.0)
C.run(body, "b1_axis_wedge", parts, info,
      "B1 axis wedge: fixed briefs with walls through the hip pitch axis, set for -90..+40; dark funnel lips; egg thigh")
