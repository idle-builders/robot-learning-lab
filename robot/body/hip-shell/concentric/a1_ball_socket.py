"""Lens A, concept A1 "ball socket": a hip ball that is a sphere about C (pitch and yaw invariant), nested over a thigh
dome of revolution about the roll axis, under fixed briefs whose leg openings are cut to clear the full target range.

  hip ball (R_pitch_link): sphere about C = (59, 2, 219), R 52 (wall 2.5), truncated to x 14..102 (flat end discs).
      Kept only at the angles no torso or yaw-link part (or the rolling thigh) ever reaches over pitch -90..+40,
      yaw -40..+40, roll -45..+25 (2.5 mm clearance), and outside the thigh dome.
  thigh dome (R_roll_link): spheroid of revolution about the roll axis (centre x 56, y 6.5, z 190; R 44 in the
      frontal plane, 34.5 front-back), upper part (roll angles -35..215), kept where nothing on the pitch link,
      yaw link or torso reaches it; existing thigh covers kept below z 172.
  briefs (torso): ellipsoid bowl below the waist z 255 (half-axes 80 x 45 x 78), leg openings cut by the voxel
      sweep (3 mm gap) of every right-leg part and concept part over the target range (left mirrored).
      Range-first: contact-free by construction; the render shows what is left exposed.

  python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B robot/body/hip-shell/concentric/a1_ball_socket.py
"""
import numpy as np

import geom as G
import kit

Z_W = 255.0
R, T = 52.0, 2.5
X0, X1 = 14.0, 102.0
hide = G.HIDE_BASE

# thigh: dome about the roll axis + existing covers below z 172
covers = G.clipped_covers(172.0)
edges, ri, ro, keep = G.roll_dome()
clouds_dome = G.obstacle_clouds(hide, skip_links=("R_roll_link",))
dome, dome_cov = G.max_sector_shell(G.ROLL, edges, ri, ro, "R_roll_link", clouds_dome, G.grid_poses(), keep=keep)
dome_c = kit.Concept("R_thigh_dome", G.tm(dome), "R_roll_link", G.IVORY)
print("dome", dome.volume(), "coverage of region", round(float(dome_cov["allowed"].mean()), 3))

# hip ball about C, truncated, maximal sector
def b_out(w):
    return float(np.sqrt(max(0.0, R * R - (w - 59.0) ** 2)))


def b_in(w):
    if w < X0 + T or w > X1 - T:
        return 0.0
    return float(np.sqrt(max(0.0, (R - T) ** 2 - (w - 59.0) ** 2)))


w_edges = np.r_[X0, X0 + T, np.linspace(X0 + T, X1 - T, 30)[1:-1], X1 - T, X1]
clouds_ball = G.obstacle_clouds(hide, extra=covers + [dome_c], skip_links=("R_pitch_link",))
ball, ball_cov = G.max_sector_shell(G.PITCH, w_edges, b_in, b_out, "R_pitch_link", clouds_ball, G.grid_poses())
dome_fill = G.spheroid_solid(G.ROLL, -6.5, 34.5 + 3.0, 44.0 + 3.0)
ball = ball - dome_fill
ball_c = kit.Concept("R_hip_ball", G.tm(ball), "R_pitch_link", G.SAND)
print("ball", ball.volume())

# briefs: bowl cut by the sweep of everything that moves on the right leg (mirrored to the left)
moving = G.moving_clouds(hide, [ball_c, dome_c] + covers)
cut = G.swept_cut(moving, G.target_poses(), gap=3.0)
c, s = (0.0, 3.0, Z_W), (80.0, 45.0, 78.0)
E_out, E_in = G.ellipsoid(c, s), G.ellipsoid(c, np.asarray(s) - T)
bowl = (E_out - E_in) ^ G.box((-200, -200, 100), (200, 200, Z_W + 2))
briefs = bowl - cut - cut.mirror((1.0, 0.0, 0.0))
briefs_c = kit.Concept("briefs", G.tm(briefs), "torso", G.IVORY)

parts = [briefs_c, ball_c, dome_c] + covers
G.run("a1-ball-socket", parts, hide,
      "A1 ball socket: sphere about C (pitch link) over a roll-axis thigh dome, briefs cut for the full target range",
      Z_W,
      notes=dict(ball_R=R, ball_x=[X0, X1], dome="R44 x 34.5 about roll axis", briefs="range-first sweep cut, gap 3 mm",
                 ball_coverage=float(ball_cov["allowed"].mean()), dome_coverage=float(dome_cov["allowed"].mean()),
                 upper_torso="original shells above z 255 drawn in the render only (not in interference)"))
