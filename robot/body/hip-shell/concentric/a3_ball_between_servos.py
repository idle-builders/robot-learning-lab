"""Lens A, concept A3 "ball between the servos" (doll ball hip, with a requirement challenge): a closed hip ball,
sphere about C on the pitch link, sized to pass between the hip-pitch servo (r_C <= 43 for x >= 30) and the hip-yaw
horn (r_C >= 56) and servo bottom face (r_C 57), with 2.5 mm clearance. It covers the front of the hip at standing and rotates up under the briefs in flexion;
pitch and yaw never sweep it against the sphere-socket briefs.

  hip ball (R_pitch_link): sphere about C, R 53 (wall 2.5: inner 50.5), x 30..96; inboard end an annulus (hole
      r 35 about the pitch axis, the pitch servo passes through it), outboard end closed. Trimmed only where the rolling thigh, the
      arms, motors, horns, electronics or battery cells would reach it (2.5 mm clearance over the target range), plus
      a slot at x 47..74 for a yaw-to-pitch connector rerouted to cross the ball at 45 deg (back-up diagonal), so that
      the slot is under the briefs at standing.
  NOT trimmed against torso parts other than the hip-yaw servo (lower-pelvis cradle, spine chassis, front power
      carrier, upper battery honeycomb cell R2 at r_C ~50) or the current yaw-to-pitch connector: those would need a
      spherical pocket around C / a reroute. Their contacts in the
      interference card are the size of that requirement challenge.
  thigh (R_roll_link): existing covers below z 180 (the thigh top sits inside the ball's bottom opening).
  briefs (torso): convex hull of the waist ellipsoid (z 255) and a sphere socket about C (R 56..58.5), wall 2.5
      (hull of the inner surfaces), open at the waist; leg opening
      = half-space through crotch (12, 2, 176), front 30 deg above horizontal (59, -47.8, 247.8) and back 5 deg above
      horizontal (59, 59.3, 224), plus the voxel sweep of the thigh in pure pitch -90..+40 (3 mm gap).

  python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B robot/body/hip-shell/concentric/a3_ball_between_servos.py
"""
import numpy as np

import geom as G
import kit

Z_W = 255.0
R, T, GAP = 53.0, 2.5, 3.0
X0, X1, HOLE = 30.0, 96.0, 35.0
# not trimmed against: every torso part except the hip-yaw servo (printed pelvis structure, electronics carriers and
# the upper battery honeycomb would need a pocket around C), and the current yaw-to-pitch connector (reroute)
TORSO_KEEP = ("lower__R_yaw_servo_case_reference",)
hide = G.HIDE_BASE

covers = G.clipped_covers(180.0)


def b_out(w):
    return float(np.sqrt(max(0.0, R * R - (w - 59.0) ** 2)))


def b_in(w):
    if w > X1 - T:
        return 0.0
    if w < X0 + T:
        return HOLE
    return float(np.sqrt(max(0.0, (R - T) ** 2 - (w - 59.0) ** 2)))


w_edges = np.r_[X0, X0 + T, np.linspace(X0 + T, X1 - T, 30)[1:-1], X1 - T, X1]
clouds = [c for c in G.obstacle_clouds(hide, extra=covers, skip_links=("R_pitch_link",))
          if not ((c[1] == "torso" and c[0] not in TORSO_KEEP) or c[0] == "R_hip_yaw_to_pitch_combined")]
CHALLENGE = sorted({c[0] for c in G.obstacle_clouds(hide, skip_links=("R_pitch_link",))
                    if (c[1] == "torso" and c[0] not in TORSO_KEEP) or c[0] == "R_hip_yaw_to_pitch_combined"})
# slot for a rerouted connector crossing at 45 deg (pitch-axis angle), 15 deg wide, swept by pitch -90..+40
nb = 180; wc = 0.5 * (w_edges[:-1] + w_edges[1:])
slot = np.zeros((len(wc), nb), bool)
for i, x in enumerate(wc):
    if 47.0 <= x <= 74.0:
        for b in range(nb):
            a = (b + 0.5) * 2.0
            slot[i, b] = (45 - 40 - 7.5) <= a <= (45 + 90 + 7.5) or a >= 360 + (45 - 40 - 7.5)
ball, cov = G.max_sector_shell(G.PITCH, w_edges, b_in, b_out, "R_pitch_link", clouds, G.grid_poses(), extra_forbid=slot)
ball_c = kit.Concept("R_hip_ball", G.tm(ball), "R_pitch_link", G.SAND)
print("ball", ball.volume(), "free fraction", round(float(cov["allowed"].mean()), 3))

# briefs with a sphere socket and a leg-opening plane
mir = lambda m: m.mirror((1.0, 0.0, 0.0))
sock_out = G.M.Manifold.sphere(R + GAP + T, 144).translate(tuple(G.C))
sock_in = G.M.Manifold.sphere(R + GAP, 144).translate(tuple(G.C))
p_c, p_f, p_b = np.array([12.0, 2.0, 176.0]), np.array([59.0, -47.8, 247.8]), np.array([59.0, 59.3, 224.0])
n = np.cross(p_f - p_c, p_b - p_c); n = n if n[2] > 0 else -n
leg_cut = G.half_space(n, p_c) ^ G.box((0.0, -200, 0), (200, 200, 400))
# plus the sweep of the thigh (covers and roll-link structure) in pure pitch -90..+40 (roll, yaw at standing), so the
# edge clears straight flexion/extension; combined yaw/roll poses are measured, not cut for
thigh = G.moving_clouds(hide, covers, links=("R_roll_link",))
pitch_only = [{**G.BODY.standing, "R_hip_pitch": float(p)} for p in np.arange(-90.0, 40.1, 2.5)]
leg_cut = G.union([leg_cut, G.swept_cut(thigh, pitch_only, gap=3.0)])
c, s = (0.0, 3.0, Z_W), (80.0, 45.0, 78.0)
# the pelvis is the convex hull of the waist ellipsoid and the two sockets (no ledge where a socket is wider than the
# torso); its wall is the hull of the inner ellipsoid and inner sockets, open at the waist
clip = lambda m, z: m ^ G.box((-250, -250, 100), (250, 250, z))
outer = G.M.Manifold.batch_hull([clip(G.ellipsoid(c, s), Z_W + 2), clip(sock_out, Z_W + 2), clip(mir(sock_out), Z_W + 2)])
inner = G.M.Manifold.batch_hull([clip(G.ellipsoid(c, np.asarray(s) - T), Z_W + 12), clip(sock_in, Z_W + 12), clip(mir(sock_in), Z_W + 12)])
briefs = outer - inner - leg_cut - mir(leg_cut)
briefs_c = kit.Concept("briefs", G.tm(briefs), "torso", G.IVORY)

parts = [briefs_c, ball_c] + covers

# the same measures with the challenged parts removed (as if the pelvis had a pocket around C and the connector were
# rerouted through the slot): what the requirement challenge buys
POCKETED = ("combined__stage2_lower_pelvis_battery_cradle", "combined__stage1_center_torso_spine_chassis",
            "combined__stage3_front_power_electronics_carrier", "Honeycomb_Cell_05_Upper_Dorsal_R2",
            "Honeycomb_Cell_05_Upper_Dorsal_R2_top_insulator", "Honeycomb_Fin_05_Upper_Dorsal_R2", "R_hip_yaw_to_pitch_combined")
import json  # noqa: E402
pk = kit.interference(G.BODY, parts, hide=tuple(hide) + POCKETED)
pk["real_geometry_check"] = G.real_check(parts, tuple(hide) + POCKETED)
pk["note"] = "A3 measured with these parts removed (requirement challenge: pocket / reroute): " + ", ".join(POCKETED)
json.dump(pk, open(G.HERE / "a3-ball-between-servos-pocketed-interference.json", "w"), indent=1, default=float)
print("pocketed kit", json.dumps(pk["ranges"], default=float)[:1500], pk["combined_poses_with_contact"])
print("pocketed real", json.dumps(pk["real_geometry_check"]["ranges"], default=float)[:1500], pk["real_geometry_check"]["combined_poses_with_contact"])
G.run("a3-ball-between-servos", parts, hide,
      "A3 ball between the servos: closed sphere about C (R 53) under a sphere-socket briefs; needs pelvis pockets + connector reroute",
      Z_W, notes=dict(ball=f"R {R}, wall {T}, x {X0}..{X1}, inboard hole r {HOLE}", ball_free_fraction=float(cov["allowed"].mean()),
                      not_trimmed_against=list(CHALLENGE), connector_slot="x 47..74, pitch-axis angles -2.5..142.5 deg",
                      briefs="hull of waist ellipsoid + sphere socket R 56..58.5 about C; leg opening = plane (front 30 / back 5 deg above horizontal) + thigh sweep in pure pitch -90..+40, gap 3"))
