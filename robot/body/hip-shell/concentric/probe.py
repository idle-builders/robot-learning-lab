"""Geometry probe for lens A: radii of the hip parts about the pitch axis, the roll axis and C; shell outlines."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kit

body = kit.base()
names = ["R_hip_pitch_ST3025_case", "R_hip_yaw_to_pitch_combined", "R_hip_pitch_ST3025_output", "R_roll_servo_case_reference",
         "R_hip_pitch_to_roll_combined", "R_roll_upper_horn_reference", "R_roll_passive_idler_reference", "R_thigh_combined",
         "R_thigh_front_hard_cover", "R_thigh_back_hard_cover", "R_knee_ST3025_case", "R_yaw_upper_horn_reference",
         "lower__R_yaw_servo_case_reference", "combined__stage2_lower_pelvis_battery_cradle"]
C = kit.HIP_C
for n in names:
    m = kit._mesh(body, n); v = m.vertices
    rp = np.hypot(v[:, 1] - 2, v[:, 2] - 219)
    rr = np.hypot(v[:, 0] - 56, v[:, 2] - 190)
    rc = np.linalg.norm(v - C, axis=1)
    print(f"{n:45s} {body.parts[n]['owning_link']:12s} r_pitch {rp.min():5.0f}..{rp.max():5.0f}  r_roll {rr.min():5.0f}..{rr.max():5.0f}  r_C {rc.min():5.0f}..{rc.max():5.0f}"
          f"  x {v[:,0].min():.0f}..{v[:,0].max():.0f}")
    if n.startswith("R_") and body.parts[n]['owning_link'] in ("R_pitch_link", "R_roll_link", "R_yaw_link"):
        for x0 in range(0, 100, 10):
            s = (v[:, 0] >= x0) & (v[:, 0] < x0 + 10)
            if s.any():
                print(f"      x{x0:3d}..{x0+10:3d}: r_pitch max {rp[s].max():5.1f}  z {v[s,2].min():.0f}..{v[s,2].max():.0f} y {v[s,1].min():.0f}..{v[s,1].max():.0f}")

# torso shells: lowest edge and outline
for n in ("center_front_shell", "center_back_shell"):
    m = kit._mesh(body, n); v = m.vertices
    print(n, "faces", len(m.faces), "watertight", m.is_watertight)
    for x0 in range(0, 95, 10):
        s = (np.abs(v[:, 0] - x0) < 5)
        if s.any():
            low = v[s][v[s, 2] < 240]
            print(f"   x~{x0:3d}: z min {v[s,2].min():.0f}; y range at z<240: {low[:,1].min() if len(low) else 'na'}..{low[:,1].max() if len(low) else 'na'}")
    for z0 in (190, 200, 210, 220, 230, 240, 260):
        s = np.abs(v[:, 2] - z0) < 4
        if s.any():
            print(f"   z~{z0}: x {v[s,0].min():.0f}..{v[s,0].max():.0f}  y {v[s,1].min():.0f}..{v[s,1].max():.0f}")
print("joints", {k: (j["origin_world_mm"], j["axis_world_neutral"]) for k, j in body.joints.items() if k.startswith("R_hip") or k == "R_knee"})
