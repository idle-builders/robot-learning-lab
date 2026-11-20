"""Visible gap of the lens-B concepts (light: builds the concept meshes and casts rays, no render or sweep).

Parallel rays at the right hip (front: x 0..104, back: x 0..104, side: y -56..64; z 150..272; arms excluded) find
the first part hit. The largest gap is the longest vertical run (mm) of non-shell hits (mechanism or see-through)
between two shell hits in one column; the area is the visible mechanism (mm2) inside such runs. Shell = concept
parts and parts with role body_shell. Silhouette columns can inflate the largest run; the area is the steadier number.
Writes visible_gaps.json.

Run: python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B robot/body/hip-shell/converging/visible_gaps.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import concepts as C  # noqa: E402
import cvg  # noqa: E402
import kit  # noqa: E402

POSES = (("standing", {}), ("hip -90 flexion", {"R_hip_pitch": -90.0, "R_knee": 90.0}),
         ("hip -60 flexion", {"R_hip_pitch": -60.0, "R_knee": 60.0}),
         ("hip -45, abducted -30, yaw 30", {"R_hip_pitch": -45.0, "R_hip_roll": -30.0, "R_hip_yaw": 30.0, "R_knee": 40.0}),
         ("hip +40 extension", {"R_hip_pitch": 40.0}))


def concepts(body):
    out = {}
    for f in ("b1_axis_wedge", "b2_sketch_crease", "b3_side_walls"):
        got = {}
        C_run = C.run
        C.run = lambda body_, name, parts, info, title, **k: got.update(parts=parts)
        try:
            exec(compile((HERE / f"{f}.py").read_text(), f, "exec"), {"__name__": "visible_gaps", "__file__": str(HERE / f"{f}.py")})
        finally:
            C.run = C_run
        out[f] = got["parts"]
    return out


body = kit.base()
res = {"baseline (current shells, no concept)": {lab: cvg.gap_measure(body, [], (), dq) for lab, dq in POSES}}
for name, parts in concepts(body).items():
    res[name] = {lab: cvg.gap_measure(body, parts, cvg.HIDE_BASE, dq) for lab, dq in POSES}
    print(name, json.dumps(res[name]["standing"]), json.dumps(res[name]["hip -90 flexion"]), flush=True)
(HERE / "visible_gaps.json").write_text(json.dumps(dict(method=__doc__.strip().splitlines()[2:6], results=res), indent=1))
