"""Bounded 350 mm ST3250 + Gates 123-3MGT-15 internal-knee fit study.

Uses the exact sketch STEP, a 1:1 20T:20T belt, and a supported output shaft.
This is an honest failed source-envelope study, not a production mechanism.
"""
from __future__ import annotations

import json, math
from pathlib import Path
import build123d as bd
import trimesh

from swarm_belt_knee_coupon import (ROOT, SOURCE, box, cyl_x, compound, add,
                                     overlap, gap, bbox, volume)
from swarm_strong_internal_knee_screen import capsule, turn

OUT = ROOT / 'artifacts/swarm-strong-internal-knee-20261004'
KNEE = (42.0, 21.0, 95.0)
DRIVE = (42.0, -6.0, 95.0+math.sqrt(31.5**2-27**2))
MOTOR_CENTER = (33.5, DRIVE[1], DRIVE[2]+12.2)
MOTOR_SIZE = (35.0, 24.72, 45.22)
X0, X1 = 51.0, 66.0
R = 60/(2*math.pi)


def source_outer(pad=0):
    stations = ((87,42,3,16,18),(98,43,3,20,22),(124,39,3,24,25),
                (145,35,4,22,23),(157,33,4,16,18),(163,31,4,8,10))
    return bd.Solid.make_loft([bd.Wire.make_ellipse(rx+pad, ry+pad,
            bd.Plane(origin=(x,y,z))) for z,x,y,rx,ry in stations])


def clip(x0,x1,y0,y1,z0,z1):
    return bd.Solid.make_box(x1-x0,y1-y0,z1-z0,bd.Plane(origin=(x0,y0,z0)))


def rod(a,b,r):
    v=bd.Vector(b)-bd.Vector(a)
    return bd.Solid.make_cylinder(r,v.length,bd.Plane(origin=a,z_dir=v))


def build():
    p={v.label:v for v in bd.import_step(SOURCE).children}
    thigh_orig=[p['R_thigh_front'],p['R_thigh_back']]
    cuff_orig=p['R_knee_cuff']
    shin_orig=[p['R_shin_front'],p['R_shin_back']]
    foot=compound([p[f'R_{n}'] for n in
                   ('foot_dorsal','toe_1','toe_2','toe_3','sole')])
    source_fixed=compound([*thigh_orig,cuff_orig])
    source_moving=compound([*shin_orig,foot])

    motor=box(MOTOR_CENTER,MOTOR_SIZE)
    motor_access=box(MOTOR_CENTER,(37.0,26.72,47.22))
    belt=capsule(R+1.5,X0,X1,KNEE,DRIVE)-capsule(R-1,X0-.1,X1+.1,KNEE,DRIVE)
    belt_access=capsule(12.5,X0-2,X1+2,KNEE,DRIVE)
    drive_pulley=cyl_x(9,X0,X1,DRIVE[1],DRIVE[2])
    output_pulley=cyl_x(9,X0,X1,KNEE[1],KNEE[2])
    shaft=cyl_x(4,23,68,KNEE[1],KNEE[2])
    bearings=compound([cyl_x(8,x,x+5,KNEE[1],KNEE[2])-
        cyl_x(4.1,x-.2,x+5.2,KNEE[1],KNEE[2]) for x in (25,43)])
    seats=compound([cyl_x(10,x,x+7,KNEE[1],KNEE[2])-
        cyl_x(8.1,x-.2,x+7.2,KNEE[1],KNEE[2]) for x in (24,42)])
    supports=compound([box((x,21,106),(5,10,20)) for x in (27.5,45.5)])
    shaft_access=cyl_x(4.7,22,69,KNEE[1],KNEE[2])
    seat_access=compound([cyl_x(8.3,x,x+7,KNEE[1],KNEE[2]) for x in (24,42)])
    thigh=[part-motor_access-belt_access-shaft_access-seat_access
           for part in thigh_orig]
    posterior_slit=box((35,19,86),(17,32,22))
    cuff=cuff_orig-belt_access-posterior_slit
    shin=[part & clip(-50,140,-80,80,-10,78.5) for part in shin_orig]
    skin=compound([*shin,foot])
    pad=box((35,19,76),(13,10,4))
    rods=[rod((x,20,76),(x,21,95),2.3) for x in (31,39)]
    collars=[cyl_x(5,x,x+3,KNEE[1],KNEE[2]) for x in (30,38)]
    yoke=compound([pad,*rods,*collars])
    return locals()


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    g=build()
    fixed_skin=compound([*g['thigh'],g['cuff']])
    moving_skin=g['skin']
    fixed_hard=compound([fixed_skin,g['motor'],g['belt'],g['drive_pulley'],
                         g['bearings'],g['seats'],g['supports']])
    moving_hard=compound([moving_skin,g['yoke']])
    angles=(0,15,30,45,60,75,90,105,120)
    rows=[]
    for a in angles:
        posed=turn(moving_hard,KNEE,a)
        hit=overlap(posed,fixed_hard)
        distance=0.0 if hit>1e-5 else gap(posed,fixed_hard)
        rows.append({'flex_deg':a,'split':'main' if a%30==0 else 'held_out_midpoint',
                     'nonmating_overlap_mm3':round(hit,4),
                     'minimum_nonmating_hard_gap_mm':round(distance,4)})
        if a in (0,60,120):
            scene=trimesh.Scene()
            for name,shape,color in (
                ('exact_thigh_after_local_relief',compound(g['thigh']),(226,219,204,255)),
                ('opened_source_cuff',g['cuff'],(102,98,87,255)),
                ('trimmed_source_shin_and_foot',turn(moving_skin,KNEE,a),(236,230,214,255)),
                ('supported_output_yoke',turn(g['yoke'],KNEE,a),(199,116,68,255)),
                ('output_shaft',turn(g['shaft'],KNEE,a),(87,91,90,255)),
                ('output_20T',turn(g['output_pulley'],KNEE,a),(214,151,65,255)),
                ('ST3250_case',g['motor'],(68,107,135,210)),
                ('drive_20T',g['drive_pulley'],(214,151,65,255)),
                ('Gates_123_3MGT_15',g['belt'],(47,54,55,255)),
                ('bearings',g['bearings'],(122,126,122,255)),
                ('seats',g['seats'],(92,98,95,255)),
                ('support_ribs',g['supports'],(107,113,110,255)),
            ):
                add(scene,name,shape,color)
            scene.export(OUT/f'knee-{a:03d}.glb')
    scene=trimesh.Scene()
    for name,shape,color in (
        ('exact_source_thigh_and_cuff',g['source_fixed'],(226,219,204,255)),
        ('exact_source_shin_and_foot',g['source_moving'],(236,230,214,255)),
    ):
        add(scene,name,shape,color)
    scene.export(OUT/'source-standing.glb')
    outer=source_outer()
    static={
        'motor_vs_original_skin_mm3':round(overlap(g['motor'],g['source_fixed']),3),
        'belt_vs_original_skin_mm3':round(overlap(g['belt'],g['source_fixed']),3),
        'motor_vs_relieved_skin_mm3':round(overlap(g['motor'],fixed_skin),4),
        'belt_vs_relieved_skin_mm3':round(overlap(g['belt'],fixed_skin),4),
        'motor_to_relieved_skin_gap_mm':round(gap(g['motor'],fixed_skin),4),
        'belt_to_relieved_skin_gap_mm':round(gap(g['belt'],fixed_skin),4),
        'motor_outside_exact_thigh_outer_mm3':round(volume(g['motor']-outer),3),
        'belt_outside_exact_thigh_outer_mm3':round(volume(g['belt']-outer),3),
        'pulley_outside_exact_thigh_outer_mm3':round(volume(g['output_pulley']-outer),3),
        'motor_outside_uniform_plus_5mm_loft_mm3':round(volume(g['motor']-source_outer(5)),3),
        'belt_outside_uniform_plus_5mm_loft_mm3':round(volume(g['belt']-source_outer(5)),3),
        'yoke_pad_to_trimmed_source_shin_overlap_mm3':round(overlap(g['pad'],moving_skin),3),
        'shaft_to_relieved_skin_overlap_mm3':round(overlap(g['shaft'],fixed_skin),4),
        'bearing_to_relieved_skin_overlap_mm3':round(overlap(g['bearings'],fixed_skin),4),
        'supports_to_relieved_skin_overlap_mm3':round(overlap(g['supports'],fixed_skin),3),
        'seat_to_support_overlap_mm3':round(overlap(g['seats'],g['supports']),3),
        'each_relieved_thigh_solid_count':[len(v.solids()) for v in g['thigh']],
    }
    contact_audit={}
    for a in (0,60,120):
        link=turn(g['yoke'],KNEE,a)
        contact_audit[str(a)]={name:round(overlap(link,shape),3) for name,shape in (
            ('yoke_vs_fixed_skin_mm3',fixed_skin),
            ('yoke_vs_bearings_mm3',g['bearings']),
            ('yoke_vs_seats_mm3',g['seats']),
            ('yoke_vs_supports_mm3',g['supports']))}
    data={
      'stage':'35 cm high-torque-rated source-shell internal-belt screen',
      'source_step':str(SOURCE.relative_to(ROOT)),
      'verdict':'REJECT: the 35 mm ST3250 case and 15 mm Gates belt protrude outside the exact 350 mm source outer loft. Static relief fragments the rear thigh into three solids, and the yoke has nonmating collisions at all nine checked poses, including held-out midpoints; minimum hard gap is zero. No hidden-motor, enclosed 0–120 degree knee was achieved.',
      'knee_axis_xyz_mm':KNEE,'drive_axis_xyz_mm':DRIVE,
      'motor_size_xyz_mm':MOTOR_SIZE,'motor_center_xyz_mm':MOTOR_CENTER,
      'motor_output_offset_note':'12.2 mm below box center is a placement assumption pending adapter/drawing verification.',
      'belt':'Gates 123-3MGT-15, 41 teeth, 3 mm pitch, 15 mm width; 20T:20T 1:1; 31.5 mm centers',
      'supported_output_path':'8 mm shaft with two 688-class bearing envelopes at X25–30 and 43–48 mm, both medial of the overhung X51–66 mm pulley; rear yoke couples shaft to shin pad. Bearing retention and shaft bending unverified.',
      'ratio_and_torque':'Motor and knee rotate at the same angular speed and transmit approximately equal torque before belt losses. ST3250 1.569 N m is a sourced rating, not a verified gait requirement or proven continuous knee output.',
      'shell_changes':'Exact source thigh halves are cut only by static motor, belt, bearing and shaft keepouts; source cuff receives a posterior slit; source shin top trimmed from Z83 to78.5 mm. No pose-derived Boolean cuts. No exterior pod or enlarged cowl is modeled.',
      'static_geometry':static,
      'contact_audit_selected_poses':contact_audit,
      'motion':{'angles_deg':angles,'poses':rows,
                'intentional_contacts_excluded':['belt/pulley mesh','shaft/bearing bore','shaft/yoke collars','shin/yoke pad'],
                'continuous_gap_proven':False},
      'exterior_blocker':'At Z≈100–120 the adjacent motor case and 15 mm belt consume at least 50 mm laterally before bearing, shell and gap allowances; the original source thigh tapers to roughly 40–48 mm there and the cuff is 30 mm wide. Even a uniformly +5 mm enlarged source loft leaves 750.8 mm³ of motor and 1434.9 mm³ of belt outside the enclosing volume, while a +11 mm uniform loft still does not fully contain the hardware and dramatically swells the small knee.',
      'unverified':['Exact spline adapter, pulley tooth/flange widths and tensioning','Motor mount fasteners, bearing retention and shaft bending','Local wall strength after relief','No matching moving thigh/shin fairing or flexible closure; exterior remains open','Torque, thermal, tolerance, gait and other-leg checks'],
    }
    (OUT/'report.json').write_text(json.dumps(data,indent=2)+'\n')
    (OUT/'index.html').write_text(Path(__file__).with_name('swarm_strong_internal_knee_viewer.html').read_text())
    for name,shape in (
        ('relieved-thigh',compound(g['thigh'])),('opened-cuff',g['cuff']),
        ('trimmed-source-shin-and-foot',moving_skin),('ST3250-case',g['motor']),
        ('Gates-belt-envelope',g['belt']),('drive-20T',g['drive_pulley']),
        ('output-20T',g['output_pulley']),('output-shaft',g['shaft']),
        ('bearings',g['bearings']),('bearing-seats',g['seats']),
        ('bearing-supports',g['supports']),('moving-yoke',g['yoke']),
        ('fixed-standing-assembly',fixed_hard),
        ('moving-standing-assembly',compound([moving_skin,g['yoke'],g['shaft'],g['output_pulley']])),
    ):
        bd.export_step(compound([shape]),OUT/f'{name}.step')
    print(json.dumps({'artifact':str(OUT),'verdict':data['verdict'],
                      'static':static,'motion':rows},indent=2))


if __name__=='__main__':run()
