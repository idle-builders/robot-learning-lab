"""Trade-off curve for lens B: where a fixed wall through the hip pitch axis must sit for a given flexion, and what
that leaves open when standing. Light (no render, no interference sweep).

For flexion targets 30..90 deg it builds the B1-style briefs (axis wedge from the leg envelope, egg thigh, lips) and
reports: front edge height on the skin (min / median / max over x 14..80), the wall angle, and the ray-cast visible
gap when standing (largest non-shell run between shell hits, and visible mechanism area, front view).
Writes edge_vs_flexion.json next to this file.

Run: python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B robot/body/hip-shell/converging/edge_vs_flexion.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import concepts as C  # noqa: E402
import cvg  # noqa: E402
import kit  # noqa: E402

body = kit.base()
rows = []
for flex in (30.0, 45.0, 60.0, 75.0, 90.0):
    for lips in (True, False):
        parts, info = C.build(body, "curve", flex=flex, lips_on=lips)
        z = np.array([v[1] for k, v in info["front_edge_yz_on_skin_mm"].items() if 14 <= float(k) <= 80 and v[1] == v[1]])
        g = cvg.gap_measure(body, parts, cvg.HIDE_BASE, {}, views=("front", "side"))
        rows.append(dict(flex_design_deg=flex, lips=lips, front_edge_z_min_med_max=[round(float(z.min()), 1),
                         round(float(np.median(z)), 1), round(float(z.max()), 1)],
                         psi_front_deg_mid=info["psi_front_deg"].get("46"), standing_gap=g))
        print(rows[-1], flush=True)
(HERE / "edge_vs_flexion.json").write_text(json.dumps(dict(
    note="B1-style axis wedge at several flexion targets (rolls -15..15 in the envelope, 3 deg margin, lips 40 mm "
         "from the axis). Edge height where the front wall leaves the briefs skin (x 14..80); standing gap from "
         "front/side ray casts at the right hip (visible_gaps.py method).",
    rows=rows), indent=1))
