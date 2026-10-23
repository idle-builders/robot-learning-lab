"""Exact-solid static screen for a torso-mounted, orthogonal belt shoulder.

Source shell: experiments/sketch_ape_shell/build.py, Study(350, wall=1/2, gap=.65).
Units mm. This intentionally stops before pose generation if static containment fails.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from build123d import Axis, Compound, Plane, Solid, Wire, export_step, import_step

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from experiments.sketch_ape_shell.build import Study, loft, box as shell_box  # noqa: E402


def box(x0, x1, y0, y1, z0, z1):
    return Solid.make_box(x1-x0, y1-y0, z1-z0, Plane(origin=(x0,y0,z0)))


def cyl(radius, length, start, axis):
    return Solid.make_cylinder(radius, length, Plane(origin=start, z_dir=axis))


def ring(outer, inner, length, start, axis):
    return cyl(outer,length,start,axis)-cyl(inner,length,start,axis)


VENDOR_STEP=ROOT/'artifacts/round2-hip-350/vendor-cad/XL_XC330_official.stp'


def official_motor_solid():
    raw=import_step(str(VENDOR_STEP))
    solids=raw.solids()
    united=solids[0]
    for solid in solids[1:]:
        united=united.fuse(solid)
    return united


def place_step(shape, rotations, xyz_min):
    for axis,angle in rotations:
        shape=shape.rotate(axis,angle)
    b=shape.bounding_box().min
    return shape.translate((xyz_min[0]-b.X,xyz_min[1]-b.Y,xyz_min[2]-b.Z))


def part(name, shape, group, color, zone):
    return dict(name=name,shape=shape,group=group,color=color,zone=zone)


def mechanism():
    p=[]
    add=lambda *args: p.append(part(*args))
    # Official CAD reaches 29 mm in depth including its modeled output feature.
    # Reserve the entire bounding case separately from material intersections.
    # One correction after the common brief: face the pitch output forward,
    # move the drive and joint inboard, and lengthen the belt run.
    add('pitch_XL330_envelope',box(18,38,-3,26,223,257),'actuator-envelope','#2674af','torso')
    add('roll_XL330_envelope',box(10,39,-6,14,174,208),'actuator-envelope','#2c93b5','torso')
    yaw_axis=(85,3,230.5)
    yaw_case=box(75,95,-14,20,216,245).rotate(Axis(yaw_axis,(0,1,0)),-20)
    yaw_case=yaw_case.translate((2.5,0,-.5))
    add('arm_twist_XL330_envelope',yaw_case,'actuator-envelope','#6e9bcb','arm')
    if VENDOR_STEP.exists():
        # Supplier STEP is a visual reference only. Its internal overlapping
        # solids make the aggregate BRep invalid for exact booleans.
        vendor=official_motor_solid()
        add('pitch_XL330_visual',place_step(vendor,[(Axis.X,90)],(18,-3,223)),
            'actuator-visual','#2674af','torso')
        add('roll_XL330_visual',place_step(vendor,[(Axis.X,-90),(Axis.Z,-90)],(10,-6,174)),
            'actuator-visual','#2c93b5','torso')
        yaw_visual=place_step(vendor,[],(75,-14,216)).rotate(Axis(yaw_axis,(0,1,0)),-20).translate((2.5,0,-.5))
        add('arm_twist_XL330_visual',yaw_visual,
            'actuator-visual','#6e9bcb','arm')
    # 2S pack is reserved with a 2 mm all-side service corridor.
    add('battery_2S_450mAh',box(-5.5,5.5,-10.5,20.5,188,244),'reserve','#d68837','torso')
    add('battery_service',box(-7.5,7.5,-12.5,22.5,186,246),'reserve-clearance','#f7c579','torso')
    # Shared comparison pack: Tattu 4S 850 mAh, 60 x 30 x 30.5 mm reserve.
    add('comparison_4S_850mAh',box(-15.25,15.25,-26,34,210,240),
        'comparison-reserve','#ad6830','torso')
    add('comparison_4S_service',box(-17.25,17.25,-28,36,208,242),
        'comparison-clearance','#e3ad72','torso')
    # Pitch: +Y motor shaft -> 10T pulley -> 18T pulley -> 3 mm joint shaft.
    add('pitch_motor_output',cyl(2.5,3,(28,-6,240),(0,1,0)),'output','#afb7bd','torso')
    add('pitch_10T_pulley',cyl(3.18,3,(28,-9,240),(0,1,0)),'transmission','#a3a0a2','torso')
    add('pitch_18T_pulley',cyl(5.73,3,(42,-9,240),(0,1,0)),'transmission','#a3a0a2','torso')
    add('pitch_belt_upper',box(28,42,-9,-6,244.2,245.4),'transmission','#4e4c4b','torso')
    add('pitch_belt_lower',box(28,42,-9,-6,234.6,235.8),'transmission','#4e4c4b','torso')
    add('pitch_joint_shaft',cyl(3,30,(42,-15,240),(0,1,0)),'output','#b6bcc2','joint')
    add('pitch_bearing_front',ring(6,3.1,4,(42,-15,240),(0,1,0)),'support','#8d9698','torso')
    add('pitch_bearing_back',ring(6,3.1,4,(42,11,240),(0,1,0)),'support','#8d9698','torso')
    # Roll: +X motor shaft -> 10T pulley -> 18T pulley -> X-axis shaft.
    add('roll_motor_output',cyl(2.5,10,(39,4,191),(1,0,0)),'output','#afb7bd','torso')
    add('roll_10T_pulley',cyl(3.18,3,(49,4,191),(1,0,0)),'transmission','#a3a0a2','torso')
    add('roll_18T_pulley',cyl(5.73,3,(49,4,232),(1,0,0)),'transmission','#a3a0a2','torso')
    add('roll_belt_front',box(49,52,-1.4,-.2,191,232),'transmission','#4e4c4b','torso')
    add('roll_belt_back',box(49,52,8.2,9.4,191,232),'transmission','#4e4c4b','torso')
    add('roll_joint_shaft',cyl(3,22,(45,4,232),(1,0,0)),'output','#b6bcc2','joint')
    add('roll_bearing_inboard',ring(6,3.1,3,(47,4,232),(1,0,0)),'support','#8d9698','torso')
    add('roll_bearing_outboard',ring(6,3.1,3,(57,4,232),(1,0,0)),'support','#8d9698','joint')
    add('roll_aperture_cuff',ring(6.1,3.5,6,(60,4,232),(1,0,0)),
        'cover','#645f58','joint')
    # Rigid mount ribs hold the bearing races from the inner torso structure.
    # The shell itself is aesthetic and is not taken as a structural bearing.
    add('pitch_bearing_bridge',box(36,42,11,14,231,234),'support','#8d9698','torso')
    add('roll_bearing_bridge',box(40,48,10,13,215,232),'support','#8d9698','torso')
    # A servo-body reaction support for the yaw axis would need an independent
    # bearing/horn study, so the actuator here is only a space reservation.
    return p


def positive_intersection(a,b):
    try:
        common=a & b
        return 0 if common is None else round(common.volume,4)
    except Exception as exc:
        return {'error':str(exc)}


def outside_volume(a,b):
    try:
        difference=a
        for envelope in b if isinstance(b,list) else [b]:
            difference=difference-envelope
            if difference is None:return 0
        return 0 if difference is None else round(difference.volume,4)
    except Exception as exc:
        return {'error':str(exc)}


def bbox_overlap(a,b):
    x,y=a.bounding_box(),b.bounding_box()
    return all(getattr(x.min,i) < getattr(y.max,i) and getattr(y.min,i) < getattr(x.max,i)
               for i in ('X','Y','Z'))


def shell(wall):
    s=Study(350,wall=wall,gap=.65)
    s.build()
    names=('torso_front','torso_back','R_upper_arm_front','R_upper_arm_back','R_shoulder_trim')
    chosen=[part(x['name'],x['shape'],'source-shell',x['color'],'shell') for x in s.parts if x['name'] in names]
    torso_outer=loft([
        (160,0,2,42,25),(173,0,2,49,29),(199,0,3,58,34),
        (229,0,4,63,35),(253,0,5,63,30),(264,0,5,60,27),
        (274,0,5,35,22),(278,0,5,23,18)])
    arm_outer=Solid.make_loft([
        Wire.make_ellipse(rx,ry,Plane(origin=(x,y,z)))
        for z,x,y,rx,ry in [(199,93,2,14,17),(207,94,2,17,20),
            (228,89,3,20,22),(249,80,4,19,21),(260,73,4,14,17),
            (265,70,4,6,9)]])
    shoulder_outer=Solid.make_loft([
        Wire.make_ellipse(ry,rz,Plane(origin=(x,4,245),x_dir=(0,1,0),z_dir=(1,0,0)))
        for x,ry,rz in [(63,17,19),(65.5,17,19),(68,16,18)]])
    joint_outer=torso_outer.fuse(arm_outer).fuse(shoulder_outer)
    return chosen,torso_outer,arm_outer,joint_outer


def screen(parts,shell_parts,torso_outer,arm_outer,joint_outer):
    result={}
    for p in parts:
        solid=p['shape']
        skin={q['name']:positive_intersection(solid,q['shape'])
              for q in shell_parts if bbox_overlap(solid,q['shape'])}
        skin={k:v for k,v in skin.items() if isinstance(v,dict) or v>0.01}
        env=torso_outer if p['zone']=='torso' else arm_outer if p['zone']=='arm' else joint_outer
        result[p['name']]={
            'volume_mm3':round(solid.volume,3),
            'outside_nominal_outer_mm3':outside_volume(solid,env),
            'source_shell_intersections_mm3':skin,
        }
    # Mutually exclusive hardware vs pack corridor; intentionally mated shaft,
    # pulley and race interfaces are omitted from pairwise collision claims.
    for service_name in ('battery_service','comparison_4S_service'):
        service=next(p['shape'] for p in parts if p['name']==service_name)
        result[service_name]['mechanism_intersections_mm3']={
            p['name']:positive_intersection(service,p['shape'])
            for p in parts if p['group'] not in ('reserve','reserve-clearance','comparison-reserve','comparison-clearance','actuator-visual')
            and bbox_overlap(service,p['shape'])
            and positive_intersection(service,p['shape'])>0.01}
    motors=[p for p in parts if p['group']=='actuator-envelope']
    transmissions=[p for p in parts if p['group']=='transmission']
    result['nonmating_check']={'motor_pair_and_cross_belt_overlap_mm3':{
        f"{a['name']} vs {b['name']}":positive_intersection(a['shape'],b['shape'])
        for i,a in enumerate(motors) for b in motors[i+1:]+transmissions
        if bbox_overlap(a['shape'],b['shape'])
        and not (a['name'].split('_')[0]==b['name'].split('_')[0])
        and positive_intersection(a['shape'],b['shape'])>0.01}}
    # A narrow proposed aperture is measured separately from the raw source
    # shell verdict. The dark annular cuff stays inside the original silhouette.
    aperture=cyl(6.3,10,(59,4,232),(1,0,0))
    result['aperture_trial']={'source_torso_skin_removed_mm3':{
        q['name']:positive_intersection(aperture,q['shape'])
        for q in shell_parts if q['name'].startswith('torso_')
        and bbox_overlap(aperture,q['shape'])},
        'source_shell_modified_for_pass':False}
    return result


def viewer_scene(parts):
    out=[]
    for p in parts:
        try:
            verts,faces=p['shape'].tessellate(0.45)
            out.append(dict(name=p['name'],group=p['group'],color=p['color'],
                            vertices=[[round(v.X,3),round(v.Y,3),round(v.Z,3)] for v in verts],
                            faces=[list(f) for f in faces]))
        except Exception as exc:
            out.append(dict(name=p['name'],error=str(exc)))
    return out


def main():
    parts=mechanism()
    collision_parts=[p for p in parts if p['group']!='actuator-visual']
    reports={}
    shell1=[]
    for wall in (1,2):
        shell_parts,torso_outer,arm_outer,joint_outer=shell(wall)
        reports[str(wall)]=screen(collision_parts,shell_parts,torso_outer,arm_outer,joint_outer)
        if wall==1: shell1=shell_parts
    failures=[]
    for wall,measurements in reports.items():
        for name,item in measurements.items():
            if name=='nonmating_check':
                if item['motor_pair_and_cross_belt_overlap_mm3']:failures.append((wall,name))
                continue
            if name=='aperture_trial':continue
            if item['outside_nominal_outer_mm3'] or item['source_shell_intersections_mm3']:
                failures.append((wall,name))
            if item.get('mechanism_intersections_mm3'):
                failures.append((wall,name+' mechanism interference'))
    report={'source_shell':'experiments/sketch_ape_shell/build.py Study(350,wall=1|2,gap=.65)',
            'concept':'torso-mounted orthogonal belt shoulder',
            'motor_collision_basis':'valid conservative 20x34x29 mm STEP bounding cases; invalid vendor BRep is visual only',
            'aperture':'A 6.3 mm radius X-axis aperture and 6.1/3.5 mm annular cuff were trialled inside the source shoulder trim; raw source shell remains the collision gate.',
            'static_pass':not failures,
            'failure_parts':failures,
            'pose_screen':'not run: static clearance must pass first' if failures else 'static pass; poses not yet implemented',
            'wall_mm':reports}
    (HERE/'report.json').write_text(json.dumps(report,indent=2))
    export_step(Compound(children=[p['shape'] for p in shell1+collision_parts]),str(HERE/'coupon.step'))
    (HERE/'scene.json').write_text(json.dumps(viewer_scene(shell1+parts)))
    print(json.dumps({'static_pass':report['static_pass'],'failure_parts':failures},indent=2))


if __name__=='__main__':
    main()
