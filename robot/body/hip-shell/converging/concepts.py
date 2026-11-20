"""Lens B concept builders and the shared run harness (render + kit.interference + visible gap + numbers).

Each concept is fixed briefs (one torso part: the trimmed torso shell plus the briefs, cut by a leg-opening wedge whose
walls are ruled surfaces through the hip pitch axis), dark funnel lips lying in those walls, and an egg-topped
thigh shell on the roll link. Concepts differ in where the seam runs and what they trade.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cvg  # noqa: E402
import kit  # noqa: E402
import manifold3d as M  # noqa: E402

WAIST_Z = 238.0


def _interp(xs, vals):
    ok = ~np.isnan(vals)
    xs, vals = np.asarray(xs)[ok], np.asarray(vals)[ok]
    return lambda x: float(np.interp(x, xs, vals))


def build(body, name, *, flex=90.0, ext=40.0, rolls=(-15.0, 0.0, 15.0), r_lip=40.0, margin=3.0, x_in=6.0,
          x_out=None, egg_extra=(), hips=None, bowl=cvg.BOWL, egg_kw=None, lip_x_max=86.0, lips_on=True,
          psi_override=None, skirt=None):
    """Return (concepts, info). The wedge walls are placed from the leg's own angular envelope (pitch-link parts +
    this egg thigh over `rolls`, points at r >= r_lip) so the leg can reach -flex..+ext."""
    egg = cvg.egg_thigh(body, extra=egg_extra, **(egg_kw or {}))
    egg_m = cvg.as_mesh(egg)
    pts = cvg.leg_points(body, egg_m, rolls=rolls)
    xs = np.arange(0.0, 112.0, 4.0)
    pf, pb = cvg.wedge_from_leg(pts, xs, r0=r_lip, flex=flex, ext=ext, margin=margin)
    x_hi = float(np.nanmax(np.where(~np.isnan(pf), xs, np.nan))) if x_out is None else float(x_out)
    psi_f, psi_b = _interp(xs, pf), _interp(xs, pb)
    env = cvg.body_envelope(body, WAIST_Z, bowl, hips)
    if psi_override is not None:  # f(env) -> (psi_front(x), psi_back(x)) drawn from the sketch, not the leg envelope
        psi_f, psi_b = psi_override(env)
    wx = np.arange(x_in, x_hi + 0.1, 4.0)
    wedge = cvg.twisted_wedge(wx, psi_b, psi_f)
    blank = cvg.briefs_blank(body, WAIST_Z, bowl, hips)
    torso = cvg.upper_torso(body, WAIST_Z) + blank
    # new fixed parts (briefs below the waist, lips) stay out of the hip pitch servo / connector yaw sweep; the kept
    # torso shell above the waist is left as it is
    keep_out = cvg.yaw_sweep(body)
    torso = torso - wedge - cvg.twisted_wedge(-wx[::-1], lambda x: psi_b(-x), lambda x: psi_f(-x)) - \
        keep_out.trim_by_plane((0.0, 0.0, -1.0), -WAIST_Z)
    if skirt is not None:  # fixed lateral panel lateral of the leg's x range: hip pitch never reaches it
        torso = torso + skirt + skirt.mirror((1.0, 0.0, 0.0))
    lips = []
    lx = wx[wx <= (lip_x_max if lip_x_max is not None else x_hi)]
    for sgn in (1.0, -1.0):
        xx = lx if sgn > 0 else -lx[::-1]
        f = (lambda x: psi_f(abs(x)))
        b = (lambda x: psi_b(abs(x)))
        lips.append(cvg.lip(xx, f, r_lip, 54.0, t=2.5))
        lips.append(cvg.lip(xx, b, r_lip, 54.0, t=2.5))
    lip = M.Manifold.batch_boolean(lips, M.OpType.Add) if lips_on else None
    # keep lips inside the body outline (trim at the briefs/torso outer hull) and off the shell itself
    lip = None if lip is None else (lip - keep_out) ^ cvg.mf(env)
    parts = [kit.Concept(f"{name}_torso_briefs", cvg.as_mesh(torso), "torso", cvg.SHELL),
             kit.Concept(f"R_{name}_egg_thigh", egg_m, "R_roll_link", cvg.THIGH)]
    if lip is not None:
        parts.insert(1, kit.Concept(f"{name}_funnel_lips", cvg.as_mesh(lip), "torso", cvg.DARK))
    info = dict(flex_design_deg=flex, ext_design_deg=ext, design_rolls_deg=list(rolls), lip_inner_radius_mm=r_lip,
                margin_deg=margin, wedge_x_mm=[float(x_in), x_hi],
                psi_front_deg={f"{x:.0f}": round(psi_f(x), 1) for x in wx[::2]},
                psi_back_deg={f"{x:.0f}": round(psi_b(x), 1) for x in wx[::2]},
                front_edge_yz_on_skin_mm={f"{x:.0f}": cvg.edge_on_skin(env, x, psi_f(x)) for x in wx[::2]},
                back_edge_yz_on_skin_mm={f"{x:.0f}": cvg.edge_on_skin(env, x, psi_b(x)) for x in wx[::2]},
                egg_bounds_mm=np.round(egg_m.bounds, 1).tolist(),
                watertight={p.name: bool(p.mesh.is_watertight) for p in parts},
                faces={p.name: int(len(p.mesh.faces)) for p in parts})
    return parts, info


def run(body, name, parts, info, title, *, hide=cvg.HIDE_BASE, grid=None, do_render=True, do_interf=True):
    out_dir = HERE
    t = time.time()
    res = dict(concept=name, title=title, hidden=list(hide), build=info)
    res["visible_gap"] = "see visible_gaps.json (visible_gaps.py, arms excluded from the rays)"
    if do_render:
        png = out_dir / f"{name}-look.png"
        kit.render(body, parts, str(png), hide=hide, title=title)
        res["render"] = str(png.relative_to(kit.ROOT))
        print("render", png, round(time.time() - t), "s", flush=True)
    if do_interf:
        res["interference"] = kit.interference(body, parts, hide=hide, grid=grid)
        print(json.dumps(res["interference"]["ranges"], indent=1), flush=True)
    res["seconds"] = round(time.time() - t)
    js = out_dir / f"{name}-interference.json"
    js.write_text(json.dumps(res, indent=1))
    print("wrote", js, flush=True)
    return res
