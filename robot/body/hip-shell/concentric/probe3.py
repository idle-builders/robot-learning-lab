"""Torso internals around the right pitch axis: min radius per x bin and angle sector (lens A sizing)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kit
body = kit.base()
for shells in (False, True):
    vs = []
    for n, p in kit._hip_parts(body).items():
        if p["owning_link"] != "torso" or (("shell" in n) != shells):
            continue
        m = kit._mesh(body, n)
        v = m.sample(max(2000, int(m.area / 4)))
        vs.append(v)
    v = np.vstack(vs)
    ang = np.degrees(np.arctan2(v[:, 2] - 219, v[:, 1] - 2)); r = np.hypot(v[:, 1] - 2, v[:, 2] - 219)
    print("TORSO", "shells" if shells else "internals", ": min r_pitch per x bin, sector (0 back, 90 up, 180 front, -90 down)")
    print("   x bin  " + " ".join(f"{a:>5d}" for a in range(-180, 180, 30)))
    for x0 in range(0, 100, 10):
        s = (v[:, 0] >= x0) & (v[:, 0] < x0 + 10) & (r < 120)
        row = []
        for a in range(-180, 180, 30):
            t = s & (ang >= a) & (ang < a + 30)
            row.append(f"{r[t].min():5.0f}" if t.any() else "    -")
        print(f"   {x0:3d}-{x0+10:3d} " + " ".join(row))
