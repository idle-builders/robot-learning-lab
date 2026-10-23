"""One-sided 500 mm thigh-dominant hip screen; exact BReps, sampled poses.

The first and corrected variants are deliberately finite trials. The fixed
pelvis is a short, newly lofted V; the rounded hip belongs to the moving
thigh. A vertical yaw pivot has two bearing stations and open-sided support
arms. Pitch is a kinematic screen, not a designed pitch bearing.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys

import build123d as b
import trimesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / "artifacts/thigh-dominant-hip-20261006"
sys.path.insert(0, str(ROOT))
from experiments.sketch_ape_shell.build import Study, above_waist
from experiments.compact_morphology.ape_design.swarm_rated_leg_skin_v2.skin_v2_case import integration_shapes
from experiments.compact_morphology.ape_design.swarm_rated_leg_drive.rated_leg_coupon import mesh

K = 500 / 350
PIVOT = (39.0, 0.0, 234.0)
POSES = (("stand", 0, 0), ("yaw_out", -20, 0), ("yaw_in", 20, 0),
         ("stride", 20, -35), ("squat", 0, -45))
MARGIN = 0.5


def clip(x0=-200, x1=200, y0=-200, y1=200, z0=-200, z1=600):
    return b.Solid.make_box(x1-x0, y1-y0, z1-z0,
                            b.Plane(origin=(x0,y0,z0)))


def box_at(x, y, z, sx, sy, sz):
    return b.Pos(x,y,z) * b.Box(sx,sy,sz)


def arm_polygon(points,z):
    edges=[b.Edge.make_line((x,y,z-2.5),(*points[(i+1)%len(points)],z-2.5))
           for i,(x,y) in enumerate(points)]
    return b.Solid.extrude(b.Face(b.Wire(edges)),(0,0,5))


def cyl_z(radius, z0, z1):
    return b.Solid.make_cylinder(radius,z1-z0,
                                 b.Plane(origin=(PIVOT[0],PIVOT[1],z0)))


def volume(shape):
    return sum(s.volume for s in shape.solids()) if shape else 0.0


def bounds(shape):
    q=shape.bounding_box()
    return [[round(v,3) for v in q.min],[round(v,3) for v in q.max]]


def intersection(a, c):
    aa,cc=a.bounding_box(),c.bounding_box()
    if any(getattr(aa.min,p)>=getattr(cc.max,p) or
           getattr(cc.min,p)>=getattr(aa.max,p) for p in "XYZ"):
        return (0.0,0.0)
    return (volume(a & c),volume(c & a))


def posed(shape, yaw, pitch):
    return (b.Pos(*PIVOT)*b.Rot(0,0,yaw)*b.Rot(pitch,0,0)*
            b.Pos(*(-v for v in PIVOT))*shape)


def parts_for(variant):
    with contextlib.redirect_stdout(io.StringIO()):
        study=Study(500,2,.65).build()
    source={p['name']:p['shape'] for p in study.parts}
    knee=integration_shapes(500)
    # This V is a fresh narrow anatomy, not a motion sweep cut into the source
    # pelvis. Its front/back split remains the sketch's removable seam.
    pelvis_stations=[(130,0,5,9,11),(140,0,5,13,17),
                     (150,0,4,16,21),(160,0,3,18,24),
                     (175,0,3,20,27)]
    pelvis,_=study.hollow_loft(pelvis_stations)
    pelvis=(pelvis-above_waist(-study.gap)).scale(K)
    if variant=='corrected':
        pelvis=pelvis & clip(z0=212)
    fixed_skin={'pelvis_front':pelvis & clip(y1=4*K-.65/2),
                'pelvis_back':pelvis & clip(y0=4*K+.65/2),
                'torso_front':source['torso_front'],
                'torso_back':source['torso_back']}
    # The outer proximal thigh is broad and round. Only the inner mechanical
    # neck is narrow; there is no independent external cup or fixed cheek.
    thigh_stations=[(87,42,3,16,18),(98,43,3,20,22),
                    (124,39,3,24,25),(139,39,3,21,24),
                    (151,39,3,19,22),(160,39,3,17,20),
                    (169,39,3,11,14)]
    thigh,_=study.hollow_loft(thigh_stations)
    thigh=thigh.scale(K)
    if variant=='corrected':
        # One bounded correction: take down the front upper rim on one diagonal
        # plane, while retaining the moving rear buttock volume and lower lobe.
        poly=[(-100,224),(-24,224),(24,238),(100,260),
              (100,500),(-100,500)]
        edges=[b.Edge.make_line((-100,*poly[i]),(-100,*poly[(i+1)%len(poly)]))
               for i in range(len(poly))]
        # Above is an X-constant YZ face, swept across the right hip.
        cutter=b.Solid.extrude(b.Face(b.Wire(edges)),(250,0,0))
        thigh=thigh-cutter
    moving_skin={'thigh_front':thigh & clip(y1=3*K-.65/2),
                 'thigh_back':thigh & clip(y0=3*K+.65/2)}

    # Source-sized yaw body and opposed W618/6 bearing stations. Each seat has
    # an annular shaft opening. The two separate front/rear arms avoid the
    # solid bridge that crossed the shaft in the rejected cheek coupon.
    motor=b.Solid.make_cylinder(12.5,35.8,
        b.Plane(origin=(18,-14 if variant=='corrected' else -13,239)))
    bearings={}; seats={}
    for label,z in (('lower',219),('upper',237)):
        bearings[label]=cyl_z(6.5,z,z+3.5)-cyl_z(3.1,z-.1,z+3.6)
        seats[label]=cyl_z(9,z-.8,z+4.3)-cyl_z(6.6,z-.9,z+4.4)
    # Arms terminate on opposite sides of the split pelvis. Their separate
    # attachment volumes are checked instead of assuming a rear connection.
    if variant=='corrected':
        front=[(7,-28),(12,-34),(43,-8),(39,-5)]
        back=[(7,30),(12,36),(43,8),(39,5)]
        arms={f'{side}_arm_{label}':arm_polygon(points,z)
              for side,points in (('front',front),('back',back))
              for label,z in (('lower',221),('upper',239))}
    else:
        arms={
            'front_arm_lower':box_at(22,-15,221,32,5,5),
            'back_arm_lower':box_at(22,18,221,32,5,5),
            'front_arm_upper':box_at(22,-15,239,32,5,5),
            'back_arm_upper':box_at(22,18,239,32,5,5),
        }
    fixed_hw={'RSF_yaw_body':motor,
              **{f'bearing_{k}':v for k,v in bearings.items()},
              **{f'seat_{k}':v for k,v in seats.items()},**arms}
    # A vertical output is shown separately from the kinematic pitch proxy.
    # Its 6 mm diameter clears both bearing bores by 0.1 mm radially.
    moving_hw={'yaw_shaft':cyl_z(3,217,246),
               'output_flange':cyl_z(6,214,217),
               'offset_knee_motor':(knee['fixed_knee_hardware']['motor'] if variant=='corrected'
                                    else b.Pos(0,10,12)*knee['fixed_knee_hardware']['motor'])}
    pack=box_at(0,14,200*K,60,30,30.5)
    return source,fixed_skin,moving_skin,fixed_hw,moving_hw,pack,pelvis_stations,thigh_stations


def measure(variant):
    source,fixed,moving,hardware,moving_hw,pack,pelvis_stations,thigh_stations=parts_for(variant)
    stationary={**fixed,**hardware,'rear_shifted_4S_pack':pack}
    rows=[]; first_failure=None
    for pose,yaw,pitch in POSES:
        pairs=[]
        for mn,shape in {**moving,**moving_hw}.items():
            mm=posed(shape,yaw,0 if mn in ('yaw_shaft','output_flange') else pitch)
            for fn,fs in stationary.items():
                # Shaft/bearing and bearing/seat are intentionally close. All
                # are still logged, with a distinct intended-interface flag.
                designed=(mn=='yaw_shaft' and fn.startswith(('bearing_','seat_')))
                ab,ba=intersection(mm,fs)
                gap=fs.distance_to(mm)
                if max(ab,ba)>.05 or gap<MARGIN:
                    entry=dict(moving=mn,fixed=fn,ab_mm3=round(ab,3),
                               ba_mm3=round(ba,3),gap_mm=round(gap,4),
                               intended_interface=designed)
                    pairs.append(entry)
                    if first_failure is None and not designed:
                        first_failure=dict(pose=pose,**entry)
        rows.append(dict(pose=pose,yaw_deg=yaw,pitch_deg=pitch,pairs=pairs))
    fixed_static=[]
    for name,shape in hardware.items():
        for target in ('torso_front','torso_back'):
            ab,ba=intersection(shape,fixed[target])
            if max(ab,ba)>.05:fixed_static.append(dict(a=name,b=target,ab_mm3=round(ab,3),ba_mm3=round(ba,3)))
        ab,ba=intersection(shape,pack)
        if max(ab,ba)>.05:fixed_static.append(dict(a=name,b='rear_shifted_4S_pack',ab_mm3=round(ab,3),ba_mm3=round(ba,3)))
    attachment={name:{p:round(max(intersection(shape,fixed[p])),3)
                      for p in ('pelvis_front','pelvis_back')}
                for name,shape in hardware.items() if name.endswith('_arm_lower') or name.endswith('_arm_upper')}
    support_connections={name:round(max(intersection(shape,hardware[f"seat_{'lower' if name.endswith('lower') else 'upper'}"])),3)
                         for name,shape in hardware.items() if '_arm_' in name}
    part_ledger={name:dict(solids=len(shape.solids()),volume_mm3=round(volume(shape),3),bounds_mm=bounds(shape))
                 for name,shape in {**fixed,**moving,**hardware,**moving_hw,
                                    'rear_shifted_4S_pack':pack}.items()}
    report=dict(variant=variant,target_height_mm=500,source='sketch-06 Study(500,2,.65)',
                pivot_mm=PIVOT,pack_center_mm=(0,14,200*K),
                topology='short newly lofted fixed V; one rounded moving thigh root; internal two-station yaw support with front/rear arms',
                pelvis_stations_source_mm=pelvis_stations,
                thigh_stations_source_mm=thigh_stations,
                parts=part_ledger,attachment_mm3=attachment,
                support_arm_to_seat_mm3=support_connections,
                fixed_static_contacts=fixed_static,rows=rows,
                first_named_failure=first_failure,
                qualification='Both BRep Boolean operand orders and exact distances at five sampled poses. Pitch is kinematic only; no pitch/roll bearing, continuous sweep, tolerance, fastener, strength, cable or heat qualification.')
    return report,(source,fixed,moving,hardware,moving_hw,pack)


def export(report,shapes,variant):
    source,fixed,moving,hardware,moving_hw,pack=shapes
    OUT.mkdir(parents=True,exist_ok=True)
    (HERE/('report_first.json' if variant=='first' else 'report.json')).write_text(json.dumps(report,indent=2)+'\n')
    for pose,yaw,pitch in POSES:
        for layer in ('source','candidate','service'):
            scene=trimesh.Scene()
            static={n:source[n] for n in ('pelvis_front','pelvis_back','torso_front','torso_back')} if layer=='source' else fixed
            for name,shape in static.items():mesh(scene,name,shape,(224,216,198,255))
            if layer=='service':
                mesh(scene,'rear_shifted_4S_pack',pack,(115,96,80,255))
                for name,shape in hardware.items():mesh(scene,name,shape,(68,88,97,255))
            skins={n:source[n] for n in ('R_thigh_front','R_thigh_back')} if layer=='source' else moving
            for name,shape in skins.items():mesh(scene,name,posed(shape,yaw,pitch),(230,221,203,255))
            if layer=='service':
                for name,shape in moving_hw.items():mesh(scene,name,posed(shape,yaw,0 if name in ('yaw_shaft','output_flange') else pitch),(92,114,128,255))
            suffix='-first' if variant=='first' else ''
            scene.export(OUT/f'{layer}-{pose}{suffix}.glb')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=('first','corrected'),required=True)
    args=parser.parse_args()
    report,shapes=measure(args.variant)
    export(report,shapes,args.variant)
    print(json.dumps({'variant':args.variant,'first_named_failure':report['first_named_failure'],
                      'attachment_mm3':report['attachment_mm3'],
                      'fixed_static_contacts':report['fixed_static_contacts']},indent=2))


if __name__=='__main__':main()
