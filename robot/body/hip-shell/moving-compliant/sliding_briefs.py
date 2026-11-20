"""Lens C, concept 1: sliding briefs panels (telescoping, pushed up by the thigh, spring/gravity return).

The torso shells are cut back to a waistband rim at z 250 between x 8 and x 66 (and in front of y -12 / behind y 14
further out); the 16 mm centre strip stays down to the crotch as the briefs gusset and a side strip around the
hip-yaw connector keeps the original rim (z ~231). Below the rim each hip gets two rigid curved panels that run
0.5-1.4 mm inside the shell's inner wall (a 1.6 mm panel in a telescoping channel):
  - front panel (x 8 .. x ~66): its lower edge is the briefs leg opening (z 192 at the crotch rising to ~208 at the panel end, overlapping the thigh front).
    The thigh top pushes its lower lip up (a low-friction lip/roller), so it slides straight up behind the belly
    wall as far as the thigh needs; a light spring and gravity bring it back. No drive parts cross the hip.
  - back panel, the same behind, pushed up by the thigh's back in extension.
  - a side hip panel (oval plate on the yaw link, outside the thigh's outer wall) closes the side gap.
Thigh covers keep their shape with the front-top corner chamfered (y 4 / z 207 to y -25 / z 190) to pass the
yaw-to-pitch connector at deep flexion.
Panel lift per pose = smallest lift with <= 0.3 mm3 overlap against the thigh covers and thigh structure.

Run: python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B robot/body/hip-shell/moving-compliant/sliding_briefs.py [render|interference|quick|all]
"""
from __future__ import annotations

import sys
import time

import numpy as np

import mc_common as mc
from mc_common import kit

OUT = mc.HERE
NAME = "sliding-briefs"
Z_RIM = 250.0
SHELL_C = np.array([0.0, 4.0])
TH1 = 68.0                     # panel ends (deg from straight front/back toward the side), x ~ 64
LIFT_MAX = 70.0
HIDE = ("center_front_shell", "center_back_shell", "R_thigh_front_hard_cover", "R_thigh_back_hard_cover",
        "L_thigh_front_hard_cover", "L_thigh_back_hard_cover")


def rho_inner(mesh, zs, side):
    """Smallest inner-wall radius of the shell about SHELL_C over heights zs, per angle th (deg; 0 = straight
    front/back, 90 = side): where a panel can slide up without touching the shell."""
    ths = np.arange(0, 91, 5.0); best = np.full(len(ths), np.inf)
    for z in zs:
        s = mesh.section([0, 0, 1], [0, 0, z])
        if s is None:
            continue
        P = np.asarray(s.vertices)[:, :2] - SHELL_C
        for i, th in enumerate(np.radians(ths)):
            d = np.array([np.sin(th), -np.cos(th) if side == "front" else np.cos(th)])
            proj = P @ d; perp = np.abs(P @ np.array([-d[1], d[0]]))
            sel = (perp < 2.5) & (proj > 20)
            if sel.any():
                best[i] = min(best[i], proj[sel].min())
    ok = np.isfinite(best)
    rs = np.interp(ths, ths[ok], best[ok])
    return lambda th: np.interp(th, ths, rs)


def panel(rho, side, th0, th1, z_bot, z_top, gap=0.8, thick=2.0):
    sg = -1.0 if side == "front" else 1.0

    def fn(u, v, w):
        th = th0 + (th1 - th0) * u
        zb = z_bot(th)
        z = zb + (z_top - zb) * v
        r = rho(th) - gap - thick * w
        t = np.radians(th)
        return SHELL_C[0] + r * np.sin(t), SHELL_C[1] + sg * r * np.cos(t), z
    return mc.mapped_cube(fn, (18, 3, 1))


def leg_opening(th):           # briefs leg line: 192 at the crotch rising to ~208 at the panel end (overlaps the
    s = np.clip((np.asarray(th) - 10.0) / 80.0, 0, 1)   # thigh front by ~5-10 mm: thigh top tucked under the edge)
    return 192.0 + 24.0 * s ** 1.6


def build(body, log=None):
    vb = mc.VBody(body)
    front = kit._mesh(body, "center_front_shell"); back = kit._mesh(body, "center_back_shell")
    zs = np.arange(Z_RIM - 4, Z_RIM + LIFT_MAX + 8, 6.0)
    rf = rho_inner(front, zs, "front"); rb = rho_inner(back, zs, "back")
    th0 = np.degrees(np.arcsin(8.5 / 46.0))
    fp = panel(rf, "front", th0, TH1, leg_opening, Z_RIM + 5, gap=0.5, thick=1.6)
    bp = panel(rb, "back", th0, TH1, lambda th: leg_opening(th) + 2.0 + 8.0 * np.clip((th - 35.0) / 25.0, 0, 1), Z_RIM + 5, gap=1.4, thick=1.6)
    covers = mc.chamfered_thigh_covers(body)
    thigh = kit._mesh(body, "R_thigh_combined", 1500)
    thigh = thigh if thigh.is_watertight else thigh.convex_hull
    obstacles = [(c.mesh, "roll_link") for c in covers] + [(thigh, "roll_link")]
    lift = lambda p, s: mc.tr((0, 0, p))
    vb.virtual["front_slider"] = mc.follower_link(fp, obstacles, lift, LIFT_MAX, log=None if log is None else log.setdefault("front", {}))
    vb.virtual["back_slider"] = mc.follower_link(bp, obstacles, lift, LIFT_MAX, log=None if log is None else log.setdefault("back", {}))
    parts = mc.trimmed_torso(body, Z_RIM, Z_RIM)
    parts += covers
    parts += [kit.Concept("R_briefs_front_panel", fp, "R_front_slider", (0.93, 0.90, 0.83)),
              kit.Concept("R_briefs_back_panel", bp, "R_back_slider", (0.91, 0.88, 0.81)),
              mc.side_hip_panel()]
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
                  title="C1 sliding briefs: front/back panels pushed up behind the waistband rim by the thigh; side panel on yaw link")
        print("render", round(time.time() - t), "s", flush=True)
    if what in ("interference", "all", "quick"):
        grid = None
        if what == "quick":
            grid = dict(pitch=np.array([-90., -80, -60, -45, -30, 0, 20, 40]), roll=(-45.0, -30, 0.0, 25.0), yaw=(-40.0, 0.0, 40.0))
        res = kit.interference(vb, parts, hide=HIDE, grid=grid)
        res["concept"] = NAME
        res["moving_parts"] = "R_briefs_front_panel / R_briefs_back_panel on follower slider links (lift = smallest " \
                              "contact-free lift against the thigh, max %.0f mm); R_side_hip_panel on R_yaw_link" % LIFT_MAX
        sweep = mc.sweep(vb, parts, HIDE, "R_hip_pitch", [-90, -80, -70, -60, -50, -40, -30, 20, 30, 40])
        res["pitch_sweep_growth_mm3"] = {str(k): v for k, v in sweep.items()}
        res["panel_lift_mm_pitch_only"] = {k: {str(kk[2]): round(vv, 1) for kk, vv in sorted(v.items(), key=lambda t: t[0][2])
                                              if kk[1] == 0.0 and kk[3] == -2.68} for k, v in log.items()}
        lifts = {k: round(max(v.values()), 1) for k, v in log.items() if v}
        res["panel_lift_mm_max"] = lifts
        mc.save_json(res, OUT / (f"{NAME}-interference.json" if what != "quick" else f"{NAME}-quick.json"))
        print("interference", round(time.time() - t), "s", flush=True)
        print("neutral", res["neutral_overlap_mm3"])
        for j, v in res["ranges"].items():
            print(j, v["contact_free_deg"], v["first_contact"])
        print(res["combined_poses_with_contact"], "/", res["combined_poses"], lifts)
        print(res["panel_lift_mm_pitch_only"])
        for k, v in sweep.items():
            print(k, v)
        for w in res["worst_combined"][:5]:
            print(w)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
