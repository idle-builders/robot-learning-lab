"""Lens A diagnostic: which parts block a concentric cap. For a shell of revolution about the pitch axis (sphere about
C, R 52) and about the roll axis (thigh dome R 44), prints per axial slab the angles that stay free over the target
range, separately for each group of obstacles (real geometry, 2.5 mm clearance)."""
import numpy as np
import geom as G

hide = G.HIDE_BASE
covers = G.clipped_covers(172.0)
groups = {
    "thigh rolling (roll link + covers)": lambda n, lk: lk in ("R_roll_link", "R_shin"),
    "hip-pitch servo + yaw-to-pitch connector (yaw link)": lambda n, lk: lk == "R_yaw_link",
    "torso internals (cradle, yaw servo, chassis)": lambda n, lk: lk == "torso",
    "other (arms, left leg)": lambda n, lk: lk not in ("R_roll_link", "R_shin", "R_yaw_link", "torso", "R_pitch_link"),
}
all_clouds = G.obstacle_clouds(hide, extra=covers)


def show(label, ax, edges, ri, ro, link, sym):
    print(f"\n=== {label}: free angles per slab ('#' free, '.' blocked), 10-deg columns starting at 0 = {sym}")
    total = None
    for gname, f in groups.items():
        cl = [c for c in all_clouds if f(c[0], c[1]) and c[1] != link]
        _, cov = G.max_sector_shell(ax, edges, ri, ro, link, cl, G.grid_poses(), min_run_deg=1e9)
        a = cov["allowed"]
        total = a if total is None else total & a
        print(f"  {gname}: free fraction {a.mean():.2f}")
    print("  all groups combined: free fraction %.2f" % total.mean())
    for i in range(0, len(edges) - 1, 2):
        row = "".join("#" if total[i, j * 5:(j + 1) * 5].all() else "." for j in range(36))
        print(f"   w {edges[i]:6.1f}..{edges[i + 1]:6.1f}  {row}")


R, T = 52.0, 2.5
b_out = lambda w: float(np.sqrt(max(0.0, R * R - (w - 59.0) ** 2)))
b_in = lambda w: float(np.sqrt(max(0.0, (R - T) ** 2 - (w - 59.0) ** 2)))
show("sphere about C, R 52 (pitch link); w = x", G.PITCH, np.linspace(14, 102, 23), b_in, b_out, "R_pitch_link", "back, 90 up, 180 front, 270 down")
edges, ri, ro, keep = G.roll_dome()
show("thigh dome about roll axis, R 44 (roll link); w = -y", G.ROLL, edges, ri, ro, "R_roll_link", "outboard, 90 up, 180 inboard, 270 down")
