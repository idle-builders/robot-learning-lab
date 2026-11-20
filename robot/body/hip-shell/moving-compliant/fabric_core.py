"""Lens C, concept 3 (owner's direction 2026-11-15): segmented core shell + stretch-fabric zones.

Rigid islands (block model):
  - upper torso front/back shells (torso), cut back to a waistband rim at z 250 between x 8 and 66; a side strip
    around the hip-yaw connector keeps the original rim (z ~231);
  - centre briefs gusset (torso): the 16 mm strip |x| < 8 of the old shells, running down under the crotch;
  - side hip panel per hip (R_yaw_link): oval plate outside the thigh's outer wall, overlapping the thigh top; it
    yaws with the leg and never pitches, so the outer side needs no fabric;
  - thigh covers (R_roll_link) with the front-top corner chamfered as in concepts 1 and 2.
Fabric zones per hip (4-way stretch knit, rendered dark grey), sewn/clamped along rigid edges:
  - front (groin crease): waistband rim (torso) -> thigh front cover at z ~186 (lower toward the crotch);
  - back (seat): back rim (torso) -> thigh back cover at z ~168 (a long span on purpose: the length change at the
    back is set by the hip motion, so a longer panel sees less strain);
  - inner: gusset edge (torso) -> thigh inner wall at z ~165.
Per pose the fabric is a ruled surface; each ruling's taut length wraps over body parts that poke across it.

Run: python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B robot/body/hip-shell/moving-compliant/fabric_core.py [render|interference|strain|all]
"""
from __future__ import annotations

import sys
import time

import numpy as np

import fabric as fb
import mc_common as mc
from mc_common import kit

OUT = mc.HERE
NAME = "fabric-core"
Z_RIM = 250.0
HIDE = ("center_front_shell", "center_back_shell", "R_thigh_front_hard_cover", "R_thigh_back_hard_cover",
        "L_thigh_front_hard_cover", "L_thigh_back_hard_cover")
STRAIN_POSES = [("standing", {}),
                ("hip -90 flexion", {"R_hip_pitch": -90.0}),
                ("hip -45 flexion", {"R_hip_pitch": -45.0}),
                ("hip +40 extension", {"R_hip_pitch": 40.0}),
                ("roll -45 abduction", {"R_hip_roll": -45.0}),
                ("roll +25 adduction", {"R_hip_roll": 25.0}),
                ("yaw -40", {"R_hip_yaw": -40.0}),
                ("yaw +40", {"R_hip_yaw": 40.0}),
                ("hip -45, abducted -30, yaw 30", {"R_hip_pitch": -45.0, "R_hip_roll": -30.0, "R_hip_yaw": 30.0}),
                ("hip -90, abducted -30", {"R_hip_pitch": -90.0, "R_hip_roll": -30.0}),
                ("hip -90, yaw +40", {"R_hip_pitch": -90.0, "R_hip_yaw": 40.0}),
                ("hip +40, yaw -40", {"R_hip_pitch": 40.0, "R_hip_yaw": -40.0})]


def edge_from_section(mesh, plane_origin, plane_normal, key_axis, keys, pick_axis, pick, tol=2.0, zmax=None):
    """For each key value along key_axis, the extreme (min/max along pick_axis) section point near it."""
    s = mesh.section(plane_normal, plane_origin)
    P = np.concatenate([np.concatenate([np.linspace(a, b, max(2, int(np.linalg.norm(b - a) / 0.5)))
                                        for a, b in zip(d[:-1], d[1:])]) for d in s.discrete])   # densified
    if zmax is not None:
        P = P[P[:, 2] < zmax]
    out = []
    for k in keys:
        sel = np.abs(P[:, key_axis] - k) < tol
        Q = P[sel]
        if not len(Q):
            continue
        i = np.argmin(Q[:, pick_axis]) if pick == "min" else np.argmax(Q[:, pick_axis])
        out.append(Q[i])
    return np.array(out)


def build(body):
    vb = mc.VBody(body)
    parts = mc.trimmed_torso(body, Z_RIM, Z_RIM)
    covers = mc.chamfered_thigh_covers(body)
    parts += covers + [mc.side_hip_panel()]
    front = kit._mesh(body, "center_front_shell"); back = kit._mesh(body, "center_back_shell")
    cf, cb = covers[0].mesh, covers[1].mesh
    xa = np.linspace(10.0, 64.0, 9)
    # front zone
    A = edge_from_section(front, (0, 0, Z_RIM + 1), (0, 0, 1), 0, xa, 1, "min")
    xb = np.linspace(18.0, 86.0, 9)
    B = []
    for x in xb:
        zb = 186.0 if x >= 48 else 186.0 - (48 - x) * 0.6
        B.append(edge_from_section(cf, (0, 0, zb), (0, 0, 1), 0, [x], 1, "min")[0])
    front_zone = fb.Zone("front", A, ["torso"] * len(A), np.array(B) + (0, -0.3, 0), np.array([1.0, 0, 0]),
                         np.array([35.0, 2.0, 320.0]))
    # back zone
    A = edge_from_section(back, (0, 0, Z_RIM + 1), (0, 0, 1), 0, xa, 1, "max")
    xb = np.linspace(24.0, 86.0, 9)
    B = edge_from_section(cb, (0, 0, 168.0), (0, 0, 1), 0, xb, 1, "max") + (0, 0.3, 0)
    back_zone = fb.Zone("back", A, ["torso"] * len(A), B, np.array([1.0, 0, 0]), np.array([35.0, 2.0, 320.0]))
    # inner zone: gusset edge (x 7.5 section of the cut shells, lowest outer point per y) -> thigh inner wall z 165
    tf, tb = parts[0].mesh, parts[1].mesh
    ya = np.linspace(-36.0, 44.0, 11)
    ys = np.arange(-40.0, 50.0, 1.0)
    E = np.vstack([edge_from_section(m, (7.5, 0, 0), (1, 0, 0), 1, ys, 2, "min", tol=0.6, zmax=Z_RIM) for m in (tf, tb)])
    E = np.array([E[np.abs(E[:, 1] - y) < 0.6][np.argmin(E[np.abs(E[:, 1] - y) < 0.6][:, 2])]
                  for y in ys if np.any(np.abs(E[:, 1] - y) < 0.6)])
    E = E[(E[:, 1] >= -34) & (E[:, 1] <= 44)]
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(E, axis=0), axis=1))]
    A = np.array([[np.interp(t, cum, E[:, k]) for k in range(3)] for t in np.linspace(0, cum[-1], 11)]); A[:, 0] = 8.0
    print("gusset edge", A[:, 1:].round(0).tolist())
    yb = np.linspace(-22.0, 36.0, 11)
    Bf = [edge_from_section(cf if y < 8 else cb, (0, 0, 165.0), (0, 0, 1), 1, [y], 0, "min")[0] for y in yb]
    inner_zone = fb.Zone("inner", A, ["torso"] * len(A), np.array(Bf) + (-0.3, 0, 0), np.array([0, 1.0, 0]),
                         np.array([59.0, 2.0, 200.0]))
    return vb, parts, [front_zone, back_zone, inner_zone]


def back_long(body, zones, covers):
    """Variant of the seat zone with a longer span: clamped on the back shell 15 mm above its rim (z 265) and on the
    thigh back at z 158."""
    z0 = zones[1]
    back = kit._mesh(body, "center_back_shell")
    A = edge_from_section(back, (0, 0, 265.0), (0, 0, 1), 0, np.linspace(10.0, 64.0, len(z0.A)), 1, "max")
    B = edge_from_section(covers[1].mesh, (0, 0, 158.0), (0, 0, 1), 0, np.linspace(24.0, 86.0, len(z0.A)), 1, "max") + (0, 0.3, 0)
    return fb.Zone("back_long", A, z0.A_links, B, z0.lateral, z0.inside, z0.color)


def inner_high(body, zones, covers):
    """Variant of the inner zone attached higher on the thigh (inner wall at z 182, nearer the pitch axis)."""
    z0 = zones[2]
    yb = np.linspace(-22.0, 36.0, len(z0.A))
    B = [edge_from_section(covers[0].mesh if y < 8 else covers[1].mesh, (0, 0, 182.0), (0, 0, 1), 1, [y], 0, "min")[0]
         for y in yb]
    return fb.Zone("inner_high", z0.A, z0.A_links, np.array(B) + (-0.3, 0, 0), z0.lateral, z0.inside, z0.color)


def main(what="all"):
    body = kit.base()
    vb, parts, zones = build(body)
    for p in parts:
        assert p.mesh.is_watertight, p.name
    t = time.time()
    field = fb.Field(body, parts, HIDE)
    if what in ("strain", "all"):
        variants = [inner_high(body, zones, parts[2:4]), back_long(body, zones, parts[2:4])]
        tab = fb.measure(body, zones + variants, STRAIN_POSES, field)
        res = dict(concept=NAME, prestretch=fb.PRESTRETCH,
                   reference_working_strain=dict(knit_4way_nylon_elastane=fb.KNIT_WORKING_STRAIN,
                                                 neoprene_2mm=fb.NEOPRENE_WORKING_STRAIN,
                                                 printed_tpu_95A_skin=fb.TPU_SKIN_WORKING_STRAIN),
                   note="strain = taut ruling length / natural length - 1 (natural = standing length / 1.10); "
                        "negative = slack that must fold (excess_fabric_mm); shear = change of ruling-to-top-edge angle; "
                        "crease_gap = min distance from the top edge to leg-side parts (pitch/roll links)",
                   zones={z.name: dict(rulings=len(z.A), standing_length_mm=None) for z in zones}, poses=tab)
        P0, _ = field.posed(body, dict(body.standing))
        res["zones"]["inner_high"] = dict(rulings=len(zones[2].A), standing_length_mm=None,
                                          note="variant: inner zone attached at z 182 on the thigh instead of z 165")
        res["zones"]["back_long"] = dict(rulings=len(zones[1].A), standing_length_mm=None,
                                         note="variant: seat zone clamped on the back shell at z 265 and the thigh at z 158")
        for z in zones + variants:
            _, _, r = fb.zone_state(body, z, dict(body.standing), P0)
            res["zones"][z.name]["standing_length_mm"] = [round(x[0], 1) for x in r]
        mc.save_json(res, OUT / f"{NAME}-strain.json")
        for lab, row in tab.items():
            print(f"{lab:32s}", "  ".join(f"{k}: {v['strain_min']:+.2f}..{v['strain_max']:+.2f} sh{v['shear_deg_max']:.0f} "
                                       f"gap{v['crease_gap_mm_min']:.0f} xs{v['excess_fabric_mm_max']:.0f} w{v['rulings_wrapped_over_parts']}"
                                       for k, v in row.items()))
        print("standing lengths", {k: v["standing_length_mm"] for k, v in res["zones"].items()}, flush=True)
    if what in ("render", "all"):
        def extra(q):
            items = []
            for z in zones:
                m = fb.render_mesh(body, z, q, field)
                items.append((m, np.eye(4), np.array(z.color)))
                ml = fb.render_mesh(body, z, dict(body.standing), field)    # left leg stays standing in LOOK_POSES
                items.append((kit.mirror_x(ml), np.eye(4), np.array(z.color)))
            return items
        mc.render(vb, parts, OUT / f"{NAME}.png", hide=HIDE, extra=extra,
                  title="C3 segmented core + fabric: rigid upper torso, gusset, side panels (yaw link), thigh covers; "
                        "stretch knit (dark) at groin, seat and inner hip")
        print("render", round(time.time() - t), "s", flush=True)
    if what in ("interference", "all"):
        res = kit.interference(vb, parts, hide=HIDE)
        res["concept"] = NAME
        res["note"] = "rigid islands only; the fabric is soft and measured in fabric-core-strain.json"
        sweep = mc.sweep(vb, parts, HIDE, "R_hip_pitch", [-90, -80, -70, -60, -50, -40, -30, 20, 30, 40])
        res["pitch_sweep_growth_mm3"] = {str(k): v for k, v in sweep.items()}
        mc.save_json(res, OUT / f"{NAME}-interference.json")
        for j, v in res["ranges"].items():
            print(j, v["contact_free_deg"], v["first_contact"])
        print(res["combined_poses_with_contact"], "/", res["combined_poses"])
        for k, v in sweep.items():
            print(k, v)
        for w in res["worst_combined"][:4]:
            print(w)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
