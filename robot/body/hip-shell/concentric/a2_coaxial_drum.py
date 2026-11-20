"""Lens A, concept A2 "coaxial drum" (look-first doll hip): a drum of revolution about the hip-pitch axis on the pitch
link, covering the front of the hip at standing up to 20 deg above horizontal, nested over the same roll-axis thigh
dome as A1, under fixed briefs whose leg-opening edge is drawn for the sketch look (a coaxial socket 3 mm outside the
drum, a V edge from the crotch up to the waist side). Nothing is cut for range: the interference card measures what
the look costs.

  drum (R_pitch_link): superellipse profile about the pitch axis, R 50 at x 57, half-length 44 (x 13..101), p 4,
      wall 2.5, inboard end an annulus (hole r 33 for the pitch servo), outboard end closed; kept at pitch-axis angles 160..380 (front 20 deg up, over the bottom, to back 20 deg up), trimmed
      only where the rolling thigh (dome, covers, thigh structure) passes, and outside the thigh dome.
  thigh dome + covers: as A1 (dome trimmed against everything, so it stays hidden under the drum).
  briefs (torso): ellipsoid bowl below the waist z 255 merged with a drum socket (R 53..55.5 about the pitch axis);
      leg opening = half-space through crotch (12, 2, 182), front (59, -60, 226) and back (59, 64, 226).

  python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B robot/body/hip-shell/concentric/a2_coaxial_drum.py
"""
import numpy as np

import geom as G
import kit

Z_W = 255.0
R, T, W0, A, P = 50.0, 2.5, 57.0, 44.0, 4.0
GAP = 3.0
SECTOR = (160.0, 380.0)
hide = G.HIDE_BASE


def barrel(Req, a):
    return lambda w: float(Req * max(0.0, 1 - abs((w - W0) / a) ** P) ** (1 / P))


# thigh: dome + covers (as A1)
covers = G.clipped_covers(172.0)
edges, ri, ro, keep = G.roll_dome()
dome, _ = G.max_sector_shell(G.ROLL, edges, ri, ro, "R_roll_link", G.obstacle_clouds(hide, skip_links=("R_roll_link",)),
                             G.grid_poses(), keep=keep)
dome_c = kit.Concept("R_thigh_dome", G.tm(dome), "R_roll_link", G.IVORY)

# drum: look-first sector, trimmed only by the rolling thigh
b_out = barrel(R, A)
b_in_raw = barrel(R - T, A - T)
HOLE = 33.0  # inboard end is an annulus around the hip-pitch servo (r <= 31 about the pitch axis)
b_in = lambda w: (HOLE if w < W0 else 0.0) if abs(w - W0) > A - T else b_in_raw(w)
w_edges = np.r_[W0 - A, W0 - A + T, np.linspace(W0 - A + T, W0 + A - T, 30)[1:-1], W0 + A - T, W0 + A]
roll_clouds = [c for c in G.obstacle_clouds(hide, extra=covers + [dome_c]) if c[1] in ("R_roll_link", "R_shin")]
in_sector = lambda w, a: SECTOR[0] <= a <= SECTOR[1] or SECTOR[0] <= a + 360 <= SECTOR[1]
drum, drum_cov = G.max_sector_shell(G.PITCH, w_edges, b_in, b_out, "R_pitch_link", roll_clouds, G.grid_poses(), keep=in_sector)
drum = drum - G.spheroid_solid(G.ROLL, -6.5, 34.5 + 3.0, 44.0 + 3.0)
drum_c = kit.Concept("R_hip_drum", G.tm(drum), "R_pitch_link", G.SAND)


# briefs: bowl + coaxial drum socket, V leg opening
def socket(r_add):
    f = barrel(R + r_add, A + r_add)
    w = np.linspace(W0 - A - r_add, W0 + A + r_add, 60)
    prof = [(0.0, w[0])] + [(f(x), x) for x in w[1:-1]] + [(0.0, w[-1])]
    return G.rev(prof, G.PITCH, 0, 360, 144)


p_c, p_f, p_b = np.array([12.0, 2.0, 182.0]), np.array([59.0, -60.0, 226.0]), np.array([59.0, 64.0, 226.0])
n = np.cross(p_f - p_c, p_b - p_c)
n = n if n[2] > 0 else -n                      # normal pointing up/inboard: the leg side is (x - p_c).n < 0
leg_cut = G.half_space(n, p_c) ^ G.box((0.0, -200, 0), (200, 200, 400))
c, s = (0.0, 3.0, Z_W), (80.0, 45.0, 78.0)
E_out, E_in = G.ellipsoid(c, s), G.ellipsoid(c, np.asarray(s) - T)
mir = lambda m: m.mirror((1.0, 0.0, 0.0))
sock_out, sock_in = socket(GAP + T), socket(GAP)
outer = G.union([E_out, sock_out, mir(sock_out)]) ^ G.box((-200, -200, 100), (200, 200, Z_W + 2))
inner = G.union([E_in, sock_in, mir(sock_in)])
briefs = outer - inner - leg_cut - mir(leg_cut)
briefs_c = kit.Concept("briefs", G.tm(briefs), "torso", G.IVORY)

parts = [briefs_c, drum_c, dome_c] + covers
G.run("a2-coaxial-drum", parts, hide,
      "A2 coaxial drum (look-first): pitch-axis drum covering the front, roll-axis thigh dome, sketch V briefs edge",
      Z_W,
      notes=dict(drum=f"R {R}, x {W0 - A}..{W0 + A}, p {P}, sector {SECTOR}", briefs="coaxial socket +3 mm, V edge plane",
                 drum_coverage=float(drum_cov["allowed"].mean())))
