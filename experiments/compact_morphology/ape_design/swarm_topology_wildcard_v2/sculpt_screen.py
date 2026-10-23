"""V2 smooth deltoid and recessed-aperture screen; mm, exact BRep solids.

Baseline c0c6343 stays untouched. This script compares a visually bounded
2/1 mm lobe with a much larger positive-clearance lobe. It deliberately stops
at static when the smaller lobe fails; no hidden pose pass is generated.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

from build123d import Axis, Compound, Plane, Solid, Vertex, Wire, export_step

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT))
from experiments.sketch_ape_shell.build import Study, box as cut_box  # noqa: E402
from experiments.compact_morphology.ape_design.swarm_topology_wildcard.shoulder_coupon import (  # noqa: E402
    box, cyl, ring, mechanism, outside_volume, positive_intersection,
)

STATIONS=[(199,93,2,14,17),(207,94,2,17,20),(228,89,3,20,22),
          (249,80,4,19,21),(260,73,4,14,17),(265,70,4,6,9)]
TAPER=(0,.3,1,1,.3,0)


def arm_outer(dx=0.,dy=0.):
    wires=[Wire.make_ellipse(rx+dx*f,ry+dy*f,Plane(origin=(x,y,z)))
           for (z,x,y,rx,ry),f in zip(STATIONS,TAPER)]
    return Solid.make_loft(wires+[Vertex(70,4,267.8)])


def arm_inner(wall,dx=0.,dy=0.):
    return Solid.make_loft([
        Wire.make_ellipse(rx+dx*f-wall,ry+dy*f-wall,Plane(origin=(x,y,z)))
        for (z,x,y,rx,ry),f in zip(STATIONS,TAPER)])


def sculpt_halves(wall,dx,dy,side=1):
    skin=arm_outer(dx,dy)-arm_inner(wall,dx,dy)
    if side<0:skin=skin.mirror(Plane.YZ)
    return [skin & cut_box(-180,180,-100,1.675),
            skin & cut_box(-180,180,2.325,100)]


def case_with_clearance(clearance):
    # Supplier STEP-bounding valid box, expanded uniformly for a positive
    # assembly gap before the same arm-axis alignment and datum translation.
    c=clearance
    return box(75-c,95+c,-14-c,20+c,216-c,245+c).rotate(
        Axis((85,3,230.5),(0,1,0)),-20).translate((2.5,0,-.5))


def volume(shape):
    return 0. if shape is None else round(shape.volume,4)


def tessellate(name,shape,group,color):
    verts,faces=shape.tessellate(.65)
    return dict(name=name,group=group,color=color,
                vertices=[[round(v.X,3),round(v.Y,3),round(v.Z,3)] for v in verts],
                faces=[list(f) for f in faces])


def source_study(wall):
    s=Study(350,wall=wall,gap=.65)
    with contextlib.redirect_stdout(io.StringIO()):s.build()
    return s


def scan(wall,s):
    source={p['name']:p['shape'] for p in s.parts}
    original=arm_outer()
    aperture=cyl(3.6,10,(59,4,232),(1,0,0))
    cuff=ring(4.2,3.5,6,(60,4,232),(1,0,0))
    results={}
    for key,(dx,dy) in {'visual_2y_1x':(1,2),'clearance_8y_3x':(3,8)}.items():
        out=arm_outer(dx,dy)
        inner=arm_inner(wall,dx,dy)
        halves=sculpt_halves(wall,dx,dy)
        candidate_skin=[source[n] for n in ('torso_front','torso_back','R_shoulder_trim')]+halves
        case=case_with_clearance(0)
        results[key]={
            'max_lateral_radius_add_mm':dx,
            'max_front_rear_radius_add_mm':dy,
            # Nested lofts share a coincident cap and OCC's subtraction
            # incorrectly returns the whole target. A scalar volume delta
            # avoids presenting that failed boolean as material addition.
            'outer_solid_volume_increase_mm3':round(out.volume-original.volume,4),
            'vendor_case_vs_new_skin_mm3':round(sum(positive_intersection(case,x) for x in halves),4),
            'case_plus_0p5_gap_outside_inner_mm3':volume(case_with_clearance(.5)-inner),
            'case_plus_1_gap_outside_inner_mm3':volume(case_with_clearance(1)-inner),
            'cuff_vs_aperture_cut_candidate_skin_mm3':round(sum(
                positive_intersection(cuff,p-aperture) for p in candidate_skin),4),
        }
    pack=box(-15.25,15.25,-25.5,34.5,210,240)
    hardware=[p for p in mechanism() if p['group'] in
              ('actuator-envelope','transmission','output','support')]
    # The rear shell is removed during withdrawal; the sweep is still checked
    # against all fixed motor envelopes and the front shell.
    sweep=box(-16.25,16.25,-26.5,100,209,241)
    results['battery_service']={
        'closed_4S_pack_skin_mm3':round(sum(positive_intersection(pack,source[n]) for n in ('torso_front','torso_back')),4),
        'closed_4S_pack_vs_hardware_mm3':{p['name']:positive_intersection(pack,p['shape']) for p in hardware},
        'rear_panel_removed_withdrawal_vs_hardware_mm3':{p['name']:positive_intersection(sweep,p['shape']) for p in hardware},
        'rear_panel_removed_withdrawal_vs_front_skin_mm3':positive_intersection(sweep,source['torso_front']),
        'connector_and_wire_bend_reserve':'unresolved: XT30 and balance lead not modeled',
    }
    # 0.6 mm moving radial gap about the 3 mm shaft. Narrow 0.7 mm cuff wall.
    joint_outer=arm_outer().fuse(Solid.make_loft([
        Wire.make_ellipse(ry,rz,Plane(origin=(x,4,245),x_dir=(0,1,0),z_dir=(1,0,0)))
        for x,ry,rz in [(63,17,19),(65.5,17,19),(68,16,18)]])).fuse(Solid.make_loft([
        Wire.make_ellipse(rx,ry,Plane(origin=(x,y,z))) for z,x,y,rx,ry in
        [(160,0,2,42,25),(173,0,2,49,29),(199,0,3,58,34),
         (229,0,4,63,35),(253,0,5,63,30),(264,0,5,60,27),
         (274,0,5,35,22),(278,0,5,23,18)]]))
    results['roll_aperture']={
        'source_skin_cut_mm3':{n:positive_intersection(aperture,source[n]) for n in
            ('torso_front','torso_back','R_upper_arm_front','R_upper_arm_back','R_shoulder_trim')},
        'narrow_cuff_outside_source_outer_mm3':outside_volume(cuff,joint_outer),
        'narrow_cuff_vs_source_skin_mm3':round(sum(positive_intersection(cuff,source[n]) for n in
            ('torso_front','torso_back','R_upper_arm_front','R_upper_arm_back','R_shoulder_trim')),4),
        'narrow_cuff_vs_aperture_cut_skin_mm3':round(sum(positive_intersection(cuff,source[n]-aperture) for n in
            ('torso_front','torso_back','R_upper_arm_front','R_upper_arm_back','R_shoulder_trim')),4),
        'shaft_vs_aperture_cut_skin_mm3':round(sum(positive_intersection(cyl(3,22,(45,4,232),(1,0,0)),source[n]-aperture) for n in
            ('torso_front','torso_back','R_upper_arm_front','R_upper_arm_back','R_shoulder_trim')),4),
        'cover_status':'annular ring remains visible/protruding; continuous opaque flex skin not demonstrated',
    }
    return results


def main():
    s1=source_study(1)
    s2=source_study(2)
    wall1=scan(1,s1)
    wall2=scan(2,s2)
    report={'source_baseline_commit':'c0c6343',
            'status':'static rejection; moving poses skipped',
            'source_shell':'experiments/sketch_ape_shell/build.py Study(350,wall=1|2,gap=.65)',
            'clearance_rule':'0.5 mm minimum case and moving shaft; 1 mm case sensitivity also shown',
            'wall_1_mm':wall1,'wall_2_mm':wall2}
    (HERE/'report.json').write_text(json.dumps(report,indent=2))
    # Full character for matched opaque aesthetic views. Both shoulders get
    # a physical 3.6 mm shaft aperture and the narrow annular cover. The
    # reported cover/skin overlap remains a measured rejection, not a pass.
    right_aperture=cyl(3.6,10,(59,4,232),(1,0,0))
    left_aperture=right_aperture.mirror(Plane.YZ)
    data=[]
    for p in s2.parts:
        name=p['name']
        group='source-arm' if 'upper_arm_' in name else 'source-shell'
        shape=p['shape']
        if name.startswith(('torso_','R_upper_arm_','L_upper_arm_','R_shoulder_trim','L_shoulder_trim')):
            shape=shape-right_aperture-left_aperture
        if shape is not None:data.append(tessellate(name,shape,group,p['color']))
    for key,dx,dy,color in [('visual-arm',1,2,'#dfd8c6'),('clearance-arm',3,8,'#dfd8c6')]:
        for side,label in ((1,'R'),(-1,'L')):
            for i,shape in enumerate(sculpt_halves(2,dx,dy,side)):
                shape=shape-(right_aperture if side>0 else left_aperture)
                data.append(tessellate(f'{key}_{label}_{i}',shape,key,color))
    data.append(tessellate('XL330_case_0p5_clearance',case_with_clearance(.5),'mechanism','#5485a8'))
    right_cuff=ring(4.2,3.5,6,(60,4,232),(1,0,0))
    data.append(tessellate('right_narrow_roll_cuff',right_cuff,'cover','#645f58'))
    data.append(tessellate('left_narrow_roll_cuff',right_cuff.mirror(Plane.YZ),'cover','#645f58'))
    data.append(tessellate('fourS_pack',box(-15.25,15.25,-25.5,34.5,210,240),'mechanism','#c57739'))
    (HERE/'scene.json').write_text(json.dumps(data))
    export_step(Compound(children=[s-right_aperture for s in sculpt_halves(2,1,2)]+[case_with_clearance(.5),right_cuff]),str(HERE/'visual_sculpt_coupon.step'))
    export_step(Compound(children=[s-right_aperture for s in sculpt_halves(2,3,8)]+[case_with_clearance(.5),right_cuff]),str(HERE/'clearance_sculpt_coupon.step'))
    print(json.dumps({'visual_2mm_deficit':wall2['visual_2y_1x']['case_plus_0p5_gap_outside_inner_mm3'],
                      'clearance_8mm_deficit':wall2['clearance_8y_3x']['case_plus_0p5_gap_outside_inner_mm3'],
                      'cover_outside':wall2['roll_aperture']['narrow_cuff_outside_source_outer_mm3']},indent=2))


if __name__=='__main__':main()
