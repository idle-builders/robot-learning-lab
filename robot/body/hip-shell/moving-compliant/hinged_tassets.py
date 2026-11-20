"""Lens C, concept 2: hinged tassets (spring-returned flaps the thigh pushes open).

Same waistband cut as concept 1 but 6 mm higher (rim z 256 between x 8 and 66, centre gusset strip, side strip kept), but the briefs
below the rim are flaps hinged on the OUTSIDE of the shell at the rim, like armour tassets or a cat flap:
  - front tasset per hip (x 8 .. ~32) plus a front-side tasset (x ~32 .. ~64): curved 2 mm plates 0.8 mm outside the belly outline, from the rim (z 258; the front flap hinges along x
    at its top edge, the side flap along the rim tangent at its middle, so both open outward) down to the leg opening (z 192 at the crotch rising to ~208 at the side). A torsion
    spring holds it closed; in flexion the thigh top pushes its lower edge forward and up, and it rides on the thigh.
  - back and back-side tassets per hip, hinged at the back rim, pushed back and up by the thigh in extension.
  - side hip panel on the yaw link as in concept 1.
Nothing slides inside the torso; the flaps swing out in front of / behind the body instead.
Flap angle per pose = smallest opening angle with <= 0.3 mm3 overlap against the thigh covers, thigh structure and the
pitch-link parts that come forward above the thigh top (pitch-to-roll connector, roll servo).

Run: python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B robot/body/hip-shell/moving-compliant/hinged_tassets.py [render|interference|quick|all]
"""
from __future__ import annotations

import sys
import time

import numpy as np

import mc_common as mc
from mc_common import kit
from sliding_briefs import HIDE, SHELL_C, leg_opening

OUT = mc.HERE
NAME = "hinged-tassets"
Z_RIM = 256.0                 # rim/hinge a little higher than concept 1: the opened flap must clear the thigh top (z ~246 at -90)
TH1 = 68.0
TH_SPLIT = 40.0               # front/back flap | side flap split (deg from straight front/back)
OPEN_MAX = 130.0
PUSHERS = ("R_thigh_combined", "R_hip_pitch_to_roll_combined", "R_roll_servo_case_reference", "R_roll_upper_horn_reference",
           "R_hip_pitch_ST3025_output")


def rho_outer(mesh, z, side):
    s = mesh.section([0, 0, 1], [0, 0, z]); P = np.asarray(s.vertices)[:, :2] - SHELL_C
    ths = np.arange(0, 91, 5.0); rs = []
    for th in np.radians(ths):
        d = np.array([np.sin(th), -np.cos(th) if side == "front" else np.cos(th)])
        proj = P @ d; perp = np.abs(P @ np.array([-d[1], d[0]]))
        sel = (perp < 2.5) & (proj > 20)
        rs.append(proj[sel].max() if sel.any() else np.nan)
    rs = np.array(rs); ok = ~np.isnan(rs)
    rs = np.interp(ths, ths[ok], rs[ok])
    return lambda th: np.interp(th, ths, rs)


def flap(rho, side, th0, th1, z_bot, z_top, gap=0.8, thick=2.0):
    sg = -1.0 if side == "front" else 1.0

    def fn(u, v, w):
        th = th0 + (th1 - th0) * u
        zb = z_bot(th)
        z = zb + (z_top - zb) * v
        r = rho(th) + gap + thick * w
        t = np.radians(th)
        return SHELL_C[0] + r * np.sin(t), SHELL_C[1] + sg * r * np.cos(t), z
    return mc.mapped_cube(fn, (18, 3, 1))


def build(body, log=None):
    vb = mc.VBody(body)
    front = kit._mesh(body, "center_front_shell"); back = kit._mesh(body, "center_back_shell")
    rf = rho_outer(front, Z_RIM + 1, "front"); rb = rho_outer(back, Z_RIM + 1, "back")
    th0 = np.degrees(np.arcsin(8.5 / 46.0))
    z_top = Z_RIM + 3.0
    leg_b = lambda th: leg_opening(th) + 2.0 + 8.0 * np.clip((th - 35.0) / 25.0, 0, 1)
    covers = mc.chamfered_thigh_covers(body)
    obstacles = [(c.mesh, "roll_link") for c in covers]
    for n in PUSHERS:                     # everything below the flap that moves with the leg can push it
        m = kit._mesh(body, n, 1500)
        obstacles.append((m if m.is_watertight else m.convex_hull, body.parts[n]["owning_link"][2:]))

    def hinge(h, axis, sign):
        h = np.asarray(h, float); axis = np.asarray(axis, float)

        def motion(p, s):
            if s == "L":   # mirror in x: axis -> (-ax, ay, az), angle -> -angle
                return mc.rot(h * (-1, 1, 1), axis * (-1, 1, 1), -sign * p)
            return mc.rot(h, axis, sign * p)
        return motion

    flaps = []
    # (name, side, th range, profile, leg line, colour); front/back flaps hinge along x, side flaps along the rim
    # tangent at their middle, all on their top edge, all opening outward
    for nm, side, (a, b), rho, leg, col in (("front_tasset", "front", (th0, TH_SPLIT), rf, leg_opening, (0.93, 0.90, 0.83)),
                                            ("front_side_tasset", "front", (TH_SPLIT + 0.5, TH1), rf, leg_opening, (0.90, 0.87, 0.80)),
                                            ("back_tasset", "back", (th0, TH_SPLIT), rb, leg_b, (0.91, 0.88, 0.81)),
                                            ("back_side_tasset", "back", (TH_SPLIT + 0.5, TH1), rb, leg_b, (0.88, 0.85, 0.78))):
        m = flap(rho, side, a, b, leg, z_top)
        thc = 0.0 if "side" not in nm else 0.5 * (a + b)
        t = np.radians(thc); r = rho(thc) + 2.8
        sg = -1.0 if side == "front" else 1.0
        h = np.array([SHELL_C[0] + r * np.sin(t), SHELL_C[1] + sg * r * np.cos(t), z_top - 1.0])
        axis = np.array([np.cos(t), -sg * np.sin(t), 0.0])
        vb.virtual[nm] = mc.follower_link(m, obstacles, hinge(h, axis, sg), OPEN_MAX,
                                          log=None if log is None else log.setdefault(nm, {}))
        flaps.append(kit.Concept("R_" + nm, m, "R_" + nm, col))
    parts = mc.trimmed_torso(body, Z_RIM, Z_RIM)
    parts += covers
    parts += flaps + [mc.side_hip_panel()]
    return vb, parts


def main(what="all"):
    body = kit.base()
    log = {}
    vb, parts = build(body, log)
    for p in parts:
        assert p.mesh.is_watertight, p.name
    t = time.time()
    if what in ("render", "all"):
        mc.render(vb, parts, OUT / f"{NAME}.png", hide=HIDE,
                  title="C2 hinged tassets: spring-closed flaps hinged outside the waistband rim, pushed open by the thigh")
        print("render", round(time.time() - t), "s", flush=True)
    if what in ("interference", "all", "quick"):
        grid = None
        if what == "quick":
            grid = dict(pitch=np.array([-90., -80, -60, -45, -30, 0, 20, 40]), roll=(-45.0, -30, 0.0, 25.0), yaw=(-40.0, 0.0, 40.0))
        res = kit.interference(vb, parts, hide=HIDE, grid=grid)
        res["concept"] = NAME
        res["moving_parts"] = "R_front_tasset / R_back_tasset on follower hinge links (angle = smallest contact-free " \
                              "opening against the thigh, max %.0f deg); R_side_hip_panel on R_yaw_link" % OPEN_MAX
        sweep = mc.sweep(vb, parts, HIDE, "R_hip_pitch", [-90, -80, -70, -60, -50, -40, -30, 20, 30, 40])
        res["pitch_sweep_growth_mm3"] = {str(k): v for k, v in sweep.items()}
        res["tasset_open_deg_pitch_only"] = {k: {str(kk[2]): round(vv, 1) for kk, vv in sorted(v.items(), key=lambda t: t[0][2])
                                                if kk[1] == 0.0 and kk[3] == -2.68} for k, v in log.items()}
        res["tasset_open_deg_max"] = {k: round(max(v.values()), 1) for k, v in log.items() if v}
        mc.save_json(res, OUT / (f"{NAME}-interference.json" if what != "quick" else f"{NAME}-quick.json"))
        print("interference", round(time.time() - t), "s", flush=True)
        print("neutral", res["neutral_overlap_mm3"])
        for j, v in res["ranges"].items():
            print(j, v["contact_free_deg"], v["first_contact"])
        print(res["combined_poses_with_contact"], "/", res["combined_poses"], res["tasset_open_deg_max"])
        print(res["tasset_open_deg_pitch_only"])
        for k, v in sweep.items():
            print(k, v)
        for w in res["worst_combined"][:5]:
            print(w)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
