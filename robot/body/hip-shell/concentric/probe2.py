"""Polar envelopes of the hip parts (lens A sizing)."""
import sys
from pathlib import Path
import numpy as np, trimesh
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kit
body = kit.base()
def pts(link, zmin=-1e9):
    out = []
    for n, p in kit._hip_parts(body).items():
        if p["owning_link"] == link and not any(k in n for k in ("cover",)):
            v = kit._mesh(body, n).sample(4000) if True else None
            out.append(v[v[:, 2] > zmin])
    return np.vstack(out)
P = pts("R_pitch_link"); Y = pts("R_yaw_link"); Rl = pts("R_roll_link", 150)
for lab, v in (("pitch link", P), ("yaw link", Y), ("roll link z>150", Rl)):
    ang = np.degrees(np.arctan2(v[:, 2] - 219, v[:, 1] - 2))  # 0=back(+y), 90=up, -90=down, 180=front
    r = np.hypot(v[:, 1] - 2, v[:, 2] - 219)
    print(lab, "about PITCH axis: angle(0 back, 90 up, +-180 front, -90 down): max r")
    print("   ", " ".join(f"{a:+d}:{r[(ang>=a)&(ang<a+30)].max():.0f}" if ((ang>=a)&(ang<a+30)).any() else f"{a:+d}:-" for a in range(-180, 180, 30)))
    ang = np.degrees(np.arctan2(v[:, 2] - 190, v[:, 0] - 56))  # 0=outboard, 90 up, 180 inboard
    r = np.hypot(v[:, 0] - 56, v[:, 2] - 190)
    print(lab, "about ROLL axis: angle(0 outboard, 90 up, 180 inboard, -90 down): max r")
    print("   ", " ".join(f"{a:+d}:{r[(ang>=a)&(ang<a+30)].max():.0f}" if ((ang>=a)&(ang<a+30)).any() else f"{a:+d}:-" for a in range(-180, 180, 30)))
    rc = np.linalg.norm(v - kit.HIP_C, axis=1)
    print(lab, "r_C max", rc.max().round(1), " r_yaw max", np.hypot(v[:,0]-59, v[:,1]-2).max().round(1))
m = kit._mesh(body, "center_back_shell")
print("back shell open edges:", len(trimesh.grouping.group_rows(m.edges_sorted, require_count=1)))
mm = m.copy(); trimesh.repair.fill_holes(mm); print("after fill_holes watertight", mm.is_watertight)
