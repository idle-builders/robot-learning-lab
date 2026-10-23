"""Source-shell 350 mm knee coupon with an in-thigh ST3250 and KHK 1:1 gears.

The coupon is a deliberately falsifiable package study, not a gear tooth model.
Gear solids are maximum OD envelopes. Their mutual mesh is deliberate and is
excluded from hard-part clearance. No moving-path Boolean cuts are used.

Run from the repository root:
  .venv/bin/python experiments/compact_morphology/ape_design/swarm_gear_knee_coupon.py
"""
from __future__ import annotations

import json
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[3]
SOURCE_350 = ROOT / "artifacts/sketch-ape-shell/exports/350mm/ape_shell_assembly.step"
SOURCE_500 = ROOT / "artifacts/sketch-ape-shell/exports/500mm/ape_shell_assembly.step"
OUT = ROOT / "artifacts/swarm-gear-knee-20261004"
AXIS = (42.0, 13.0, 95.0)
DRIVE = (42.0, 13.0, 122.0)
GEAR_X0 = 68.0
ANGLES = sorted(set(range(0, 121, 10)) | {5, 35, 65, 95, 115})
VIEWS = (0, 30, 60, 90, 120)


def box(center, dims):
    return bd.Pos(*center) * bd.Box(*dims)


def cyl_x(radius, x0, x1, y, z):
    return bd.Solid.make_cylinder(radius, x1 - x0,
                                  bd.Plane(origin=(x0, y, z), z_dir=(1, 0, 0)))


def ring_x(ro, ri, x0, x1, y, z):
    return cyl_x(ro, x0, x1, y, z) - cyl_x(ri, x0 - 0.1, x1 + 0.1, y, z)


def volume(shape):
    return 0.0 if shape is None else sum(s.volume for s in shape.solids())


def overlap(a, b):
    # OCCT can undercount a compound intersection. Inspect all solid pairs.
    return sum(volume(x & y) for x in a.solids() for y in b.solids())


def distance(a, b):
    return min(x.distance_to(y) for x in a.solids() for y in b.solids())


def bbox_lower_bound(a, b):
    a, b = a.bounding_box(), b.bounding_box()
    delta = [max(0, getattr(b.min, axis) - getattr(a.max, axis),
                 getattr(a.min, axis) - getattr(b.max, axis))
             for axis in 'XYZ']
    return sum(d * d for d in delta) ** .5


def rotate(shape, deg):
    return bd.Pos(*AXIS) * bd.Rot(deg, 0, 0) * bd.Pos(*(-v for v in AXIS)) * shape


def glb_part(scene, name, shape, color):
    if volume(shape) < 0.001:
        return
    verts, faces = shape.tessellate(0.27, 0.27)
    mesh = trimesh.Trimesh(
        vertices=np.asarray([(v.X, v.Y, v.Z) for v in verts]),
        faces=np.asarray(faces), process=False)
    mesh.visual.vertex_colors = color
    scene.add_geometry(mesh, geom_name=name, node_name=name)


def package(parts):
    thigh = bd.Compound(children=[parts['R_thigh_front'], parts['R_thigh_back']])
    cuff = parts['R_knee_cuff']
    shin_raw = bd.Compound(children=[parts['R_shin_front'], parts['R_shin_back']])
    trim = box((42, 3, 24), (150, 150, 102))  # top Z = 75, no pose relief
    shin = bd.Compound(children=[parts[f'R_shin_{side}'] & trim
                                 for side in ('front', 'back')])
    foot = bd.Compound(children=[parts[f'R_{name}'] for name in
        ('foot_dorsal', 'toe_1', 'toe_2', 'toe_3', 'sole')])

    # ST3250-C001 source-size box: X=35, Y=45.22, Z=24.72 mm. Its
    # manufacturer's 12.2 mm output offset is along Y. The source STEP has
    # no actual servo, output spline, or factory tolerance geometry.
    motor = box((42, 0.8, 122), (35, 45.22, 24.72))

    # KHK SSG1-27, module 1, 27 teeth: OD 29, centres 27, face 10, bore 8.
    # Rotationally symmetric OD envelopes deliberately overlap by 2 mm at
    # mesh; actual involute tooth geometry/phase is unavailable in this study.
    drive_gear = ring_x(14.5, 4, GEAR_X0, GEAR_X0 + 10, DRIVE[1], DRIVE[2])
    knee_gear = ring_x(14.5, 4, GEAR_X0, GEAR_X0 + 10, AXIS[1], AXIS[2])

    # A purpose-shaped stationary two-lobe pod: 1 mm radial gear clearance,
    # 2 mm nominal wall, and 2 mm axial clearance either side of the face.
    # Its shape depends only on the fixed axes and OD, not on sampled poses.
    x0, x1 = GEAR_X0 - 2, GEAR_X0 + 12
    outer = cyl_x(17.5, x0, x1, 13, 122).fuse(
        cyl_x(17.5, x0, x1, 13, 95))
    inner = cyl_x(15.5, x0 - .1, x1 + .1, 13, 122).fuse(
        cyl_x(15.5, x0 - .1, x1 + .1, 13, 95))
    pod = outer - inner

    # The 8 mm output axle has two separate 688-sized bearing envelopes.
    # Bearings are support interfaces, so shaft/bearing clearance is neither
    # checked nor claimed. The support webs are explicit fixed hard geometry.
    axle = cyl_x(4, 10, 89, 13, 95)
    bearings = bd.Compound(children=[
        ring_x(8, 4, 16, 21, 13, 95),
        ring_x(8, 4, 82, 87, 13, 95),
    ])
    near_web = box((19.5, 13, 110), (7, 10, 31))  # x 16..23, 2 mm cheek gap
    far_web = box((83, 13, 110), (8, 10, 31))  # x 79..87, meets pod by 1 mm
    support = bd.Compound(children=[bearings, near_web, far_web])

    # Opposed side cheeks and straight crossbars are deliberately simple.
    # They make the axle-to-shin connection physically visible and expose
    # swept interference without sculpting the fixed shell to the path.
    yoke = bd.Compound(children=[
        ring_x(10, 4, 10, 14, 13, 95),
        ring_x(10, 4, 89, 93, 13, 95),
        box((12, 13, 82), (4, 8, 25)),
        box((91, 13, 82), (4, 8, 25)),
        box((20, 13, 72), (16, 8, 8)),
        box((79, 13, 72), (24, 8, 8)),
    ])
    fixed = {'thigh': thigh, 'pod': pod, 'motor': motor, 'support': support}
    moving = {'shin': shin, 'foot': foot, 'yoke': yoke}
    return locals()


def screen_500():
    """Same bought components in the 500 mm source shell; no redesign."""
    if not SOURCE_500.exists():
        return {'available': False}
    p = {c.label: c for c in bd.import_step(SOURCE_500).children}
    thigh = bd.Compound(children=[p['R_thigh_front'], p['R_thigh_back']])
    cuff = p['R_knee_cuff']
    scale = 500 / 350
    cx, cy, kz = (v * scale for v in AXIS)
    mz = kz + 27
    motor = box((cx, cy - 12.2, mz), (35, 45.22, 24.72))
    # An outboard gear with 1 mm lateral separation from the source thigh
    # bounding box at its widest point; not an optimized 500 mm layout.
    gear_x0 = thigh.bounding_box().max.X + 1
    gears = [cyl_x(14.5, gear_x0, gear_x0 + 10, cy, z)
             for z in (kz, mz)]
    return {
        'available': True, 'height_mm': 500,
        'motor_center_mm': [round(cx, 2), round(cy - 12.2, 2), round(mz, 2)],
        'motor_vs_source_thigh_skin_mm3': round(overlap(motor, thigh), 3),
        'motor_vs_source_cuff_mm3': round(overlap(motor, cuff), 3),
        'gear_face_x_range_mm': [round(gear_x0, 2), round(gear_x0 + 10, 2)],
        'gear_pair_vs_source_thigh_skin_mm3': round(sum(overlap(g, thigh) for g in gears), 3),
        'gear_pair_outboard_of_source_cuff_mm': round(gear_x0 + 10 - cuff.bounding_box().max.X, 2),
        'warning': 'Envelope comparison only; no support, cover, motion or strength validation at 500 mm.'
    }


def main():
    if not SOURCE_350.exists():
        raise FileNotFoundError(SOURCE_350)
    OUT.mkdir(parents=True, exist_ok=True)
    p = {c.label: c for c in bd.import_step(SOURCE_350).children}
    g = package(p)
    fixed, moving = g['fixed'], g['moving']
    source_thigh, cuff = g['thigh'], g['cuff']
    motor, pod, drive_gear, knee_gear = (
        g['motor'], g['pod'], g['drive_gear'], g['knee_gear'])
    static = {
        'motor_vs_source_thigh_skin_mm3': round(overlap(motor, source_thigh), 3),
        'drive_gear_vs_source_thigh_skin_mm3': round(overlap(drive_gear, source_thigh), 3),
        'knee_gear_vs_source_thigh_skin_mm3': round(overlap(knee_gear, source_thigh), 3),
        'pod_vs_source_thigh_skin_mm3': round(overlap(pod, source_thigh), 3),
        'pod_vs_source_cuff_mm3': round(overlap(pod, cuff), 3),
        'pod_to_source_thigh_min_distance_mm': round(distance(pod, source_thigh), 4),
        'pod_to_fixed_support_min_distance_mm': round(distance(pod, g['support']), 4),
        'pod_lateral_max_mm': round(pod.bounding_box().max.X, 3),
        'source_cuff_lateral_max_mm': round(cuff.bounding_box().max.X, 3),
        'outboard_pod_extension_from_cuff_mm': round(
            pod.bounding_box().max.X - cuff.bounding_box().max.X, 3),
        'gear_envelope_mutual_overlap_mm3_is_intended_mesh': round(
            overlap(drive_gear, knee_gear), 3),
        'pod_to_gear_envelope_min_distance_mm': round(min(
            distance(pod, drive_gear), distance(pod, knee_gear)), 4),
    }

    rows = []
    for deg in ANGLES:
        moved = {name: rotate(shape, deg) for name, shape in moving.items()}
        candidates = []
        for f_name, f_shape in fixed.items():
            for m_name, m_shape in moved.items():
                for f_solid in f_shape.solids():
                    for m_solid in m_shape.solids():
                        candidates.append((bbox_lower_bound(f_solid, m_solid),
                                           f_name, m_name, f_solid, m_solid))
        candidates.sort(key=lambda item: item[0])
        min_dist = float('inf')
        closest = None
        for bound, f_name, m_name, f_solid, m_solid in candidates:
            if bound >= min_dist:
                break
            exact = f_solid.distance_to(m_solid)
            if exact < min_dist:
                min_dist = exact
                closest = f'{m_name}_vs_{f_name}'
        contacts = {}
        total_overlap = 0.0
        # Only overlapping bounding boxes can have a solid overlap. These
        # pairwise BRep intersections are independent of the distance pruning.
        for bound, f_name, m_name, f_solid, m_solid in candidates:
            if bound > 0:
                break
            v = volume(f_solid & m_solid)
            if v > .00001:
                key = f'{m_name}_vs_{f_name}'
                contacts[key] = round(contacts.get(key, 0) + v, 4)
                total_overlap += v
        rows.append({'flex_deg': deg,
                     'minimum_unintended_fixed_moving_distance_mm': round(min_dist, 5),
                     'total_unintended_fixed_moving_overlap_mm3': round(total_overlap, 4),
                     'closest_pair': closest, 'overlap_by_pair_mm3': contacts})
        if deg not in VIEWS:
            continue
        scene = trimesh.Scene()
        for name, shape, color in (
            ('exact_source_thigh_ghost', source_thigh, (216, 213, 199, 65)),
            ('exact_source_cuff_ghost', cuff, (63, 68, 70, 85)),
            ('source_shin_ghost', rotate(g['shin_raw'], deg), (213, 206, 185, 60)),
            ('ST3250_body', motor, (65, 111, 140, 255)),
            ('motor_gear_OD_envelope', drive_gear, (208, 154, 65, 255)),
            ('knee_gear_OD_envelope', knee_gear, (203, 124, 49, 255)),
            ('fixed_gear_pod', pod, (67, 84, 76, 245)),
            ('fixed_support_and_bearings', g['support'], (89, 95, 94, 255)),
            ('moving_axle', rotate(g['axle'], deg), (142, 99, 54, 255)),
            ('moving_yoke', moved['yoke'], (177, 91, 56, 255)),
            ('trimmed_source_shin', moved['shin'], (237, 226, 206, 255)),
            ('source_foot', moved['foot'], (227, 215, 195, 255)),
        ):
            glb_part(scene, name, shape, color)
        scene.export(OUT / f'knee-{int(deg):03d}.glb')

    report = {
        'verdict': 'REJECT this 350 mm source-shell KHK spur-pair package',
        'reason': 'The bought motor penetrates the exact source thigh; the externally clear gear pod projects far outside the sketch cuff; the trimmed source shin contacts the thigh at deep flex.',
        'source_step': str(SOURCE_350.relative_to(ROOT)),
        'geometry': {
            'coordinates': 'X lateral, Y rearward, Z up, millimetres',
            'knee_axis_mm': AXIS, 'motor_gear_axis_mm': DRIVE,
            'axis_center_distance_mm': 27.0,
            'motor': 'Feetech ST3250-C001: 35 X x 45.22 Y x 24.72 Z mm body, 74.5 g, 1.569 Nm published rated output; output offset 12.2 mm in Y',
            'gear_pair': 'Two KHK SSG1-27 1:1, 27T, module 1, 29 mm OD, 10 mm face, 8 mm bore, 54 g each; catalog surface durability torque 3.60 Nm',
            'gear_model_limit': 'Cylindrical OD and 8 mm bore envelopes; not involute teeth, hub, keyway or mesh backlash. Their 2 mm OD-cylinder overlap is intended mesh and is not assessed as an interference.',
            'support': 'Independent 8 mm axle with two 688-size bearing envelopes in opposed fixed webs. This failed package has no connected pod or outboard support load path to the thigh; bearing seats, serviceability and load rating are unverified.',
            'original_shin_top_z_mm': 83, 'trimmed_shin_top_z_mm': 75,
            'fixed_pod': 'Two-lobe fixed cover designed from gear axes: 1 mm radial OD-envelope gap, 2 mm wall and 2 mm axial face clearance; no inverse moving-path Boolean cuts.',
            'moving_yoke': 'Two opposed axle cheeks and straight lower crossbars linking toward the trimmed source shin; no hidden inverse relief.',
        },
        'static_fit': static,
        'motion': {
            'range_deg': [0, 120], 'checked_angles_deg': ANGLES,
            'minimum_distance_scope': 'Exact minimum BRep surface distance at each checked pose over all 12 named unintended fixed/moving group pairs, evaluated with bounding-box lower-bound pruning. Intended gear mesh and axle/bearing interfaces are separate.',
            'minimum_unintended_distance_mm': min(r['minimum_unintended_fixed_moving_distance_mm'] for r in rows),
            'maximum_unintended_overlap_mm3': max(r['total_unintended_fixed_moving_overlap_mm3'] for r in rows),
            'poses_with_unintended_overlap': sum(r['total_unintended_fixed_moving_overlap_mm3'] > 0.0001 for r in rows),
            'positive_working_gap_verified': False,
            'continuous_motion_note': 'The 10 degree grid includes five held-out half-step positions. Existing penetrations reject the design; no continuous clearance claim is made.',
            'poses': rows,
        },
        'scale_sensitivity_500_mm': screen_500(),
        'unresolved': ['Disconnected outboard gear pod and bearing support',
                       'Actual KHK tooth/hub CAD and gear phase', 'Servo spline and adapter',
                       'Bearing and shaft life', 'Fastening and split assembly',
                       'Cable strain relief', 'Flexible sealed knee cover',
                       'Full-body collision and controls replay',
                       'Gear surface rating is not verified system continuous knee torque.'],
    }
    bd.export_step(bd.Compound(children=[source_thigh, cuff, motor, pod, g['support'], drive_gear]),
                   OUT / 'fixed-source-shell-and-gear-pod.step')
    bd.export_step(bd.Compound(children=[g['shin'], g['foot'], g['yoke'], g['axle'], knee_gear]),
                   OUT / 'moving-shin-yoke-and-knee-gear.step')
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (OUT / 'index.html').write_text(
        Path(__file__).with_name('swarm_gear_knee_viewer.html').read_text())
    print(json.dumps({
        'artifact': str(OUT), 'static_fit': static,
        'motion': {k: v for k, v in report['motion'].items() if k != 'poses'},
        'scale_500': report['scale_sensitivity_500_mm'],
    }, indent=2))


if __name__ == '__main__':
    main()
