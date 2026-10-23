"""One bounded 50 cm cheek-socket knee coupon against the offset drive package.

Run from the repository root with .venv/bin/python. All dimensions are mm.
The result is a sampled geometry/appearance trial, not a rated knee assembly.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import build123d as b
import trimesh

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_design.swarm_knee_flex_closure.closure_coupon import (
    AXIS, hard_parts,
)
from experiments.compact_morphology.ape_design.swarm_rated_leg_skin_v2.skin_v2_case import turn
from experiments.compact_morphology.ape_design.swarm_rated_leg_drive.rated_leg_coupon import mesh
from experiments.sketch_ape_shell.build import Study, ellipsoid

ANGLES = (0, 60, 115)
MARGIN = 0.5
WALL = 2.0


def volume(shape):
    return sum(s.volume for s in shape.solids()) if shape else 0.0


def box_gap(a, c):
    aa, cc = a.bounding_box(optimal=False), c.bounding_box(optimal=False)
    return math.sqrt(sum(max(getattr(aa.min, q)-getattr(cc.max, q),
                             getattr(cc.min, q)-getattr(aa.max, q), 0)**2
                         for q in "XYZ"))


def exact_pair(a, c):
    """Report both OCCT operand orders, including any asymmetric failure."""
    if box_gap(a, c) > 0:
        return (0.0, 0.0)
    return (volume(a & c), volume(c & a))


def candidate():
    # Start from the exact source-sized offset drive, but discard its annular
    # cuff. Source thigh/shin wall remains 2 mm. The cheek shell occupies a
    # deep side sector; a narrow front apron remains part of the thigh.
    source_fixed, source_moving, hardware = hard_parts()
    study = Study(500, wall=WALL, gap=.65)
    socket = study.hollow_ellipsoid((60, 6, 145), (35, 31, 36))
    left = socket & (b.Pos(20, 8, 145) * b.Box(46, 100, 75))
    right = socket & (b.Pos(100, 8, 145) * b.Box(36, 100, 75))
    # Tabs are integral with each cheek; they enter the existing opposed
    # supports above the bearing seats, creating a named load-path contact.
    tab_a = b.Pos(40.5, 23, 175) * b.Box(5, 10, 14)
    tab_b = b.Pos(88.5, 23, 175) * b.Box(5, 10, 14)
    cheeks = {"socket_cheek_A": left + tab_a,
              "socket_cheek_B": right + tab_b}
    apron = socket & (b.Pos(60, -25, 145) * b.Box(47, 19, 72))
    fixed = {n: s for n, s in source_fixed.items() if n != "R_knee_cuff"}
    fixed.update(cheeks)
    fixed["front_apron"] = apron

    # A 2 mm ellipsoidal crown is attached to the unmodified 2 mm shin wall;
    # its lower half overlaps both source shin halves. The crown rotates with
    # the yoke and cross-pad about the source-sized offset axis.
    crown = study.hollow_ellipsoid((60, 5, 121), (24, 25, 23))
    moving = {n: source_moving[n] for n in ("R_shin_front", "R_shin_back")}
    moving["shin_crown"] = crown

    # Recessed posterior shutter: a short rigid curved plate on its own hinge
    # at the knee axis. A half-angle rocker is a kinematic placeholder; the
    # hinge/link body is explicit so it cannot masquerade as a free membrane.
    shutter_shell = study.hollow_ellipsoid((60, 12, 142), (23, 25, 25))
    # Single bounded correction after the first build: keep the panel medial
    # to the x=69 mm belt edge, instead of the initial x=38..82 mm sector.
    shutter = shutter_shell & (b.Pos(55, 33, 139) * b.Box(26, 18, 43))
    hinge = b.Pos(55, 31, 156) * b.Box(26, 4, 5)
    shutter = shutter + hinge
    moving_hw = {n: source_moving[n] for n in ("yoke", "bridge", "shin_cross_pad")}
    return source_fixed, source_moving, fixed, moving, shutter, hardware, moving_hw


def pose(moving, shutter, moving_hw, angle):
    axis = tuple(AXIS)
    return ({n: turn(s, axis, angle) for n, s in moving.items()},
            turn(shutter, axis, angle/2),
            {n: turn(s, axis, angle) for n, s in moving_hw.items()})


def cut_socket(fixed, moving, shutter, moving_hw):
    """Recess mating shells at the three stated poses with a 0.5 mm cutter."""
    targets = {}
    for angle in ANGLES:
        m, sh, hw = pose(moving, shutter, moving_hw, angle)
        for n, s in {**m, **hw, "rear_shutter": sh}.items():
            if n == "shin_cross_pad":
                continue  # the pad deliberately bonds into the shin wall
            if n == "shin_crown":
                # OCCT cannot offset the closed 2 mm ellipsoid shell. Its
                # exterior solid with enlarged axes is a conservative cutter.
                cutter = turn(ellipsoid((60, 5, 121), (24.5, 25.5, 23.5)), tuple(AXIS), angle)
            elif n == "rear_shutter":
                # Offset the shutter's stock before clipping into the short
                # posterior sector; the tiny rocker tab is an enlarged box.
                stock = ellipsoid((60, 12, 142), (23.5, 25.5, 25.5))
                panel = stock & (b.Pos(55, 33, 139) * b.Box(27, 19, 44))
                tab = b.Pos(55, 31, 156) * b.Box(27, 5, 6)
                cutter = turn(panel + tab, tuple(AXIS), angle/2)
            else:
                cutter = b.offset(s, amount=MARGIN)
            targets[(angle, n)] = cutter
    result = {}
    removed = {}
    for name, raw in fixed.items():
        shape = raw
        for cutter in targets.values():
            if box_gap(shape, cutter) == 0 and max(exact_pair(shape, cutter)) > .05:
                shape = shape - cutter
        result[name] = shape
        removed[name] = round(volume(raw)-volume(shape), 3)
    return result, removed


def evaluate(fixed, moving, shutter, hardware, moving_hw):
    rows = []
    first_failure = None
    fixed_hits, fixed_short_gaps, fixed_order_disagreements = [], [], []
    for fn, fs in fixed.items():
        for hn, hs in hardware.items():
            if (fn, hn) in (("socket_cheek_A", "support_A"),
                            ("socket_cheek_B", "support_B")):
                continue  # designed support mounts
            ab, ba = exact_pair(fs, hs)
            if max(ab, ba) > .05:
                fixed_hits.append([fn, hn, round(ab, 3), round(ba, 3)])
            elif box_gap(fs, hs) < MARGIN:
                gap = fs.distance_to(hs)
                if gap < MARGIN-.01:
                    fixed_short_gaps.append([fn, hn, round(gap, 3)])
            if abs(ab-ba) > .1:
                fixed_order_disagreements.append([fn, hn, round(ab-ba, 3)])
    if fixed_hits:
        first_failure = f"fixed {fixed_hits[0][0]} × {fixed_hits[0][1]} penetration"
    elif fixed_short_gaps:
        first_failure = f"fixed {fixed_short_gaps[0][0]} × {fixed_short_gaps[0][1]} gap below 0.5 mm"
    for angle in ANGLES:
        print(f"exact BRep checks {angle}°", flush=True)
        posed, panel, posed_hw = pose(moving, shutter, moving_hw, angle)
        hits = []
        short_gaps = []
        order_disagreements = []
        # Intentional interfaces: support/cheek, crown/shin, yoke/bridge/pad,
        # and bearing/shaft/pulley. Everything listed below is nonmating.
        for mn, ms in {**posed, "rear_shutter": panel, **posed_hw}.items():
            for fn, fs in {**fixed, **hardware}.items():
                if mn == "shin_cross_pad" and fn.startswith("R_shin_"):
                    continue
                if mn in ("yoke", "bridge") and fn in ("shaft", "output_pulley"):
                    continue
                if mn == "rear_shutter" and fn in ("support_A", "support_B"):
                    continue  # shutter hinge is deliberately support-carried
                ab, ba = exact_pair(ms, fs)
                if max(ab, ba) > .05:
                    hits.append([mn, fn, round(ab, 3), round(ba, 3)])
                elif box_gap(ms, fs) < MARGIN:
                    gap = ms.distance_to(fs)
                    if gap < MARGIN-.01:
                        short_gaps.append([mn, fn, round(gap, 3)])
                if abs(ab-ba) > .1:
                    order_disagreements.append([mn, fn, round(ab-ba, 3)])
        for mn, ms in posed.items():
            ab, ba = exact_pair(ms, panel)
            if max(ab, ba) > .05:
                hits.append([mn, "rear_shutter", round(ab, 3), round(ba, 3)])
            elif box_gap(ms, panel) < MARGIN:
                gap = ms.distance_to(panel)
                if gap < MARGIN-.01:
                    short_gaps.append([mn, "rear_shutter", round(gap, 3)])
            if abs(ab-ba) > .1:
                order_disagreements.append([mn, "rear_shutter", round(ab-ba, 3)])
        rows.append(dict(angle_deg=angle, nonmating_hits_both_orders_mm3=hits,
                         nonmating_short_gaps_mm=short_gaps,
                         operand_order_disagreements_mm3=order_disagreements))
        if first_failure is None and hits:
            first_failure = f"{angle}° {hits[0][0]} × {hits[0][1]} penetration"
        if first_failure is None and short_gaps:
            first_failure = f"{angle}° {short_gaps[0][0]} × {short_gaps[0][1]} gap below 0.5 mm"
    return dict(hits_both_orders_mm3=fixed_hits, short_gaps_mm=fixed_short_gaps,
                operand_order_disagreements_mm3=fixed_order_disagreements), rows, first_failure


def make_scene(fixed, moving, shutter, angle, hardware=None, moving_hw=None):
    scene = trimesh.Scene()
    posed, panel, posed_hw = pose(moving, shutter, moving_hw or {}, angle)
    opaque = (225, 217, 199, 255)
    for n, s in fixed.items():
        mesh(scene, n, s, opaque if hardware is None else (225, 217, 199, 95))
    for n, s in posed.items():
        mesh(scene, n, s, opaque if hardware is None else (225, 217, 199, 95))
    mesh(scene, "rear_shutter", panel, (69, 72, 73, 255))
    if hardware is not None:
        for n, s in hardware.items():
            mesh(scene, n, s, (55, 100, 148, 255))
        for n, s in posed_hw.items():
            mesh(scene, n, s, (190, 122, 70, 255))
    return scene


def main():
    HERE.mkdir(exist_ok=True)
    src_f, src_m, raw, moving, shutter, hardware, moving_hw = candidate()
    fixed, removed = cut_socket(raw, moving, shutter, moving_hw)
    fixed_check, rows, failure = evaluate(fixed, moving, shutter, hardware, moving_hw)
    connections = {}
    for cheek, support in (("socket_cheek_A", "support_A"),
                           ("socket_cheek_B", "support_B")):
        connections[f"{cheek}_to_{support}_mm3"] = [round(x, 3) for x in exact_pair(fixed[cheek], hardware[support])]
    connections["crown_to_shin_front_mm3"] = [round(x, 3) for x in exact_pair(moving["shin_crown"], moving["R_shin_front"])]
    connections["crown_to_shin_back_mm3"] = [round(x, 3) for x in exact_pair(moving["shin_crown"], moving["R_shin_back"])]
    for cheek in ("socket_cheek_A", "socket_cheek_B"):
        connections[f"{cheek}_to_thigh_back_mm3"] = [round(x, 3) for x in exact_pair(fixed[cheek], fixed["R_thigh_back"])]
    connections["apron_to_thigh_front_mm3"] = [round(x, 3) for x in exact_pair(fixed["front_apron"], fixed["R_thigh_front"])]
    connections["shutter_to_support_A_neutral_mm3"] = [round(x, 3) for x in exact_pair(shutter, hardware["support_A"])]
    solids = {n: len(s.solids()) for n, s in fixed.items()}
    if failure is None and any(x != 1 for x in solids.values()):
        failure = "cut fixed skin has a disconnected named solid"
    if failure is None and any(min(v) < 1 for v in connections.values()):
        failure = "socket support or crown skin attachment is missing"
    report = dict(source="exact 500 mm Study shell; source-sized offset knee package at (66,12,158)",
                  wall_mm=WALL, clearance_target_mm=MARGIN, angles_deg=ANGLES,
                  topology="two support-mounted side cheeks, fixed front apron, moving 2 mm crown, half-angle posterior shutter",
                  initial_trial_failure="60° rear_shutter × belt penetration: 8.638 mm³ in both operand orders; socket_cheek_B cut into two solids",
                  single_geometry_correction="narrow rear shutter sector from x=38..82 to x=42..68 mm, keeping it medial to the belt",
                  cut_removed_mm3=removed, cut_fixed_solids=solids, fixed_package_check=fixed_check,
                  connection_overlap_both_orders_mm3=connections,
                  rows=rows, first_named_failure=failure,
                  verdict="REJECT" if failure else "GEOMETRY PASS; independent appearance review needed",
                  limits="three sampled poses only; shutter rocker linkage, fasteners, stiffness, sealing, fatigue and whole-leg integration unqualified")
    (HERE/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    for angle in ANGLES:
        make_scene(fixed, moving, shutter, angle).export(HERE/f"socket-{angle}.glb")
        make_scene(fixed, moving, shutter, angle, hardware, moving_hw).export(HERE/f"socket-mechanism-{angle}.glb")
        reference = {n: s for n, s in src_f.items() if n.startswith(("R_thigh_", "R_knee_"))}
        ref_m = {n: src_m[n] for n in ("R_shin_front", "R_shin_back")}
        scene = trimesh.Scene()
        for n, s in reference.items(): mesh(scene, n, s, (225, 217, 199, 255))
        for n, s in ref_m.items(): mesh(scene, n, turn(s, tuple(AXIS), angle), (225, 217, 199, 255))
        scene.export(HERE/f"source-{angle}.glb")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
