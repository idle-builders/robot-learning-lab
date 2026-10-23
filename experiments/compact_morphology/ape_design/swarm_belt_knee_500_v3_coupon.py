"""One bounded exterior fairing study for the unchanged 500 mm belt knee.

The drive and axle geometry come from v2. Coordinates are mm, Y rearward.
"""
from __future__ import annotations

import math
import json
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh

from swarm_belt_knee_coupon import ROOT, box, cyl_x, compound, add, overlap, gap, bbox, volume
from swarm_belt_knee_500_v2_coupon import (build as build_v2, turn, KNEE, SOURCE,
                                           MOTOR_CENTER, angled_capsule)

OUT = ROOT / "artifacts/swarm-belt-knee-500-v3-20261004"


def elliptical_loft(stations):
    return bd.Solid.make_loft([
        bd.Wire.make_ellipse(rx, ry, bd.Plane(origin=(x, y, z)))
        for z, x, y, rx, ry in stations
    ])


def thigh_outer_loft(margins=None):
    source = np.array([
        (87, 42, 3, 16, 18), (98, 43, 3, 20, 22),
        (124, 39, 3, 24, 25), (145, 35, 4, 22, 23),
        (157, 33, 4, 16, 18), (163, 31, 4, 8, 10),
    ], dtype=float) * (500/350)
    if margins is None:
        return elliptical_loft(source)
    stations = []
    for z, margin in margins:
        x, y, rx, ry = [float(np.interp(z, source[:, 0], source[:, i]))
                        for i in range(1, 5)]
        stations.append((z, x, y, rx+margin, ry+margin))
    return elliptical_loft(stations)


def fixed_fairing():
    # A shallow circumferential cowl, tapered into the source thigh skin.
    profile = [(138, 0), (145, 6), (155, 8), (165, 8),
               (175, 7), (180, 5), (185, 0)]
    exterior = thigh_outer_loft(profile)
    interior = thigh_outer_loft([(z, -1.0) for z, _ in profile])
    ring = exterior - interior
    # The static cowl keeps the same deliberate rear channel. A pair of
    # retained side rails offer an attachment edge for a recessed membrane.
    channel = bd.Solid.extrude(bd.Face(bd.Wire.make_polygon([
        (44.5, 19.0, 121), (44.5, 60, 121),
        (44.5, 60, 185), (44.5, 40, 185),
    ], close=True)), (19, 0, 0))
    side_regions = compound([
        box((20, 4, 161.5), (45, 160, 47)),
        box((101, 4, 161.5), (48, 160, 47)),
    ])  # X <= 42.5 and X >= 77; only shallow side cheeks.
    cheeks = (ring - channel) & side_regions
    # Rigid cheek shells have analytic clearance around the supported axle,
    # the two seats and their support ribs. These are static hardware bores.
    cheeks = cheeks - cyl_x(4.8, 29, 87, KNEE[1], KNEE[2])
    for x0, x1 in ((30.5, 38.5), (78.5, 86.5)):
        cheeks = cheeks - cyl_x(10.6, x0, x1, KNEE[1], KNEE[2])
    for x in (34.5, 82.5):
        cheeks = cheeks - box((x, 23, 170), (6, 11, 21))
    cheeks = cheeks - box(MOTOR_CENTER, (37, 26.72, 47.22))
    cheeks = cheeks - angled_capsule(12.5, 61, 80)
    return cheeks, channel, exterior


def moving_fairing(return_outer=False):
    # A closed lower cap overlaps the exact source shin and the yoke pad.
    # The upper shell is open and flares only enough to envelope the yoke.
    outer = elliptical_loft([
        (103, 54, 18, 8.0, 15.5),
        (108, 54, 18, 8.5, 15.5),
        (116, 54, 17, 8.5, 13.0),
        (135, 54, 17, 7.0, 10.5),
        (152, 54, 18, 6.7, 8.5),
        (160, 54, 18, 6.7, 8.5),
        (166, 54, 18, 5.0, 6.0),
    ])
    inner = elliptical_loft([
        (111, 54, 18, 6.5, 13.5),
        (116, 54, 17, 6.5, 11.0),
        (135, 54, 17, 5.0, 8.5),
        (152, 54, 18, 4.7, 6.5),
        (160, 54, 18, 4.7, 6.5),
        (167, 54, 18, 3.0, 4.0),
    ])
    # The shin pad is rectangular, so an elliptical lower section left its
    # corners visible. A shallow rectangular boot covers those corners and
    # meets the tapered loft above the source calf.
    outer = outer.fuse(box((54, 17, 107.75), (17, 36, 9.5)))
    inner = inner.fuse(box((54, 17, 110.25), (14, 32, 8.5)))
    shell = outer - inner
    return (shell, outer) if return_outer else shell


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    g = build_v2()
    cowl, channel, exterior = fixed_fairing()
    shroud, shroud_outer = moving_fairing(return_outer=True)
    thigh = [part - channel for part in g['thigh_fixed']]
    cuff = g['cuff_fixed'] - box((54, 25, 123), (19, 48, 24))
    fixed_shell = compound([*thigh, cuff])
    moving_skin = g['moving_skin']
    fixed_hard = compound([fixed_shell, g['motor'], g['belt'],
                           g['drive_pulley'], g['bearings'], g['seats'],
                           g['supports'], cowl])
    moving_hard = compound([moving_skin, g['yoke'], shroud])
    angles = range(0, 121, 15)
    rows = [{"flex_deg": a,
             "noncontact_hard_overlap_mm3": round(overlap(turn(moving_hard, a), fixed_hard), 4),
             "minimum_noncontact_hard_gap_mm":
                round(gap(turn(moving_hard, a), fixed_hard), 4)
                if a in (0, 60, 120) else None}
            for a in angles]
    for a in (0, 60, 120):
        scene = trimesh.Scene()
        for name, shape, color in (
            ('source_derived_thigh', compound(thigh), (226, 219, 203, 255)),
            ('source_cuff_with_rear_slit', cuff, (103, 98, 87, 255)),
            ('physically_joined_thigh_cheeks', cowl, (235, 227, 210, 255)),
            ('source_shin_and_foot', turn(moving_skin, a), (237, 231, 215, 255)),
            ('shin_attached_yoke_shroud', turn(shroud, a), (217, 204, 182, 255)),
            ('moving_yoke_under_shroud', turn(g['yoke'], a), (199, 116, 68, 255)),
            ('output_shaft', turn(g['shaft'], a), (84, 90, 91, 255)),
            ('output_pulley', turn(g['output_pulley'], a), (215, 155, 68, 255)),
            ('ST3250', g['motor'], (68, 107, 135, 200)),
            ('drive_pulley', g['drive_pulley'], (215, 155, 68, 255)),
            ('Gates_belt', g['belt'], (48, 55, 56, 255)),
            ('bearings', g['bearings'], (126, 128, 122, 255)),
            ('seats', g['seats'], (93, 99, 98, 255)),
            ('support_ribs', g['supports'], (111, 118, 114, 255)),
        ):
            add(scene, name, shape, color)
        scene.export(OUT / f'fairing-{a:03d}.glb')
    source_scene = trimesh.Scene()
    for name, shape, color in (
        ('exact_source_thigh_and_cuff', g['original_fixed'], (226, 219, 203, 255)),
        ('exact_source_shin_and_foot', g['original_moving'], (237, 231, 215, 255)),
    ):
        add(source_scene, name, shape, color)
    source_scene.export(OUT / 'source-standing.glb')
    outer_bb = compound(g['thigh_orig']).bounding_box()
    cheek_bb = cowl.bounding_box()
    shroud_120 = turn(shroud, 120).bounding_box()
    data = {
        'stage': '500 mm unchanged-drive exterior fairing and rear-closure study',
        'source_step': str(SOURCE.relative_to(ROOT)),
        'verdict': 'REJECT AS ENCLOSED EXTERIOR: rigid thigh and shin fairings attach and clear at sampled poses, but the 19 mm wide, 64 mm tall posterior slit has no demonstrated recessed flexible bridge. At 120 degrees the shin fairing reaches 32 mm behind the source thigh posterior envelope; no pleat, attachment, strain or occlusion design has been validated. The sketch silhouette is enlarged at the knee.',
        'drive_unchanged': 'Feetech ST-3250-C001, Gates 123-3MGT-15 belt, 20T:20T 1:1, 31.5 mm centers, independent 8 mm output axle; same axes and hardware as v2.',
        'source_shell_changes': 'Original 500 mm right thigh halves and cuff import, v2 static motor/belt/axle cuts retained; one 19 mm posterior channel from Z121 to185 and 19 mm cuff slit; v2 shin top trim 2.07 mm retained. No moving-pose Boolean cut.',
        'thigh_cheeks': {
            'solid_count': len(cowl.solids()),
            'individual_overlap_with_modified_thigh_and_cuff_mm3':
                [round(overlap(piece, fixed_shell), 3) for piece in cowl.solids()],
            'static_hardware_clearance_mm': {name: round(gap(cowl, g[name]), 4)
                for name in ('motor','belt','shaft','bearings','seats','supports')},
            'source_thigh_bbox_xyz_mm': bbox(compound(g['thigh_orig'])),
            'cheek_bbox_xyz_mm': bbox(cowl),
            'bbox_protrusion_mm': {
                'medial_x': round(max(0, outer_bb.min.X-cheek_bb.min.X), 3),
                'lateral_x': round(max(0, cheek_bb.max.X-outer_bb.max.X), 3),
                'anterior_y': round(max(0, outer_bb.min.Y-cheek_bb.min.Y), 3),
                'posterior_y': round(max(0, cheek_bb.max.Y-outer_bb.max.Y), 3),
            },
        },
        'shin_shroud': {
            'solid_count': len(shroud.solids()),
            'overlap_with_source_shin_mm3': round(overlap(shroud, moving_skin), 3),
            'yoke_outside_shroud_outer_mm3': round(max(0, volume(g['yoke'])-overlap(g['yoke'], shroud_outer)), 3),
            'bbox_standing_xyz_mm': bbox(shroud),
            'bbox_120_xyz_mm': bbox(turn(shroud, 120)),
            'rear_120_beyond_original_thigh_global_posterior_y_mm': round(shroud_120.max.Y-outer_bb.max.Y, 3),
        },
        'motion': {
            'poses': rows,
            'intentional_contacts_excluded': ['Thigh cheeks to thigh skin', 'Shin shroud to source shin and yoke', 'Output shaft to bearings/collars', 'Pulley to belt'],
            'sampled_collision_count': sum(r['noncontact_hard_overlap_mm3'] > .001 for r in rows),
            'continuous_gap_proven': False,
            'gap_scope': 'Exact BRep gap at 0, 60 and 120 degrees; nine 15-degree poses checked for volume overlap. Intersample positive gap and manufacturing tolerance not proven.',
        },
        'closure_proposal_and_failure': {
            'concept': 'A recessed split TPU gaiter would anchor to both static slot edges, carry pleat reserve, and let its free lips slide over the shin-mounted rigid shroud.',
            'status': 'Not modeled as a physical deforming bridge; no material strain, fold path, sealing, retention, or view-occlusion validation.',
            'slot_width_mm': 19, 'slot_height_mm': 64,
            'standing_unfilled_upper_slot_z_mm': [166,185],
            'deep_flex_unfilled_lower_slot_z_mm': [121,144],
            'required_posterior_deployment_approx_mm': 32,
        },
    }
    (OUT/'report.json').write_text(json.dumps(data, indent=2)+'\n')
    (OUT/'index.html').write_text(Path(__file__).with_name('swarm_belt_knee_500_v3_viewer.html').read_text())
    for name, shape in (
        ('thigh-with-widened-channel', compound(thigh)),
        ('cuff-with-widened-slit', cuff),
        ('thigh-side-cheeks', cowl),
        ('shin-attached-shroud', shroud),
        ('fixed-standing-assembly', compound([fixed_shell, cowl, g['motor'],
             g['belt'], g['drive_pulley'], g['bearings'], g['seats'], g['supports']])),
        ('moving-standing-assembly', compound([moving_skin, shroud, g['yoke'],
             g['shaft'], g['output_pulley']])),
    ):
        bd.export_step(compound([shape]), OUT/f'{name}.step')
    print(json.dumps({'artifact':str(OUT),'verdict':data['verdict'],
                      'poses':rows,'cheek_protrusion':data['thigh_cheeks']['bbox_protrusion_mm']}, indent=2))


if __name__ == '__main__':
    run()
