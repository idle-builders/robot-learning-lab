"""Relocated 3-axis shoulder inside exact 45/50 cm sketch shell.

Cases use the valid 20 x 34 x 29 mm bounding envelope of the official XL330
STEP, referred to its output face. This is an assembly screen, not a torque
rating or mechanism certification. Units are millimetres.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
from pathlib import Path

import build123d as b

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT/'artifacts/scaled-shoulder'
sys.path.insert(0, str(ROOT))
from experiments.sketch_ape_shell.build import Study, loft  # noqa: E402

TORSO = [(160,0,2,42,25),(173,0,2,49,29),(199,0,3,58,34),
         (229,0,4,63,35),(253,0,5,63,30),(264,0,5,60,27),
         (274,0,5,35,22),(278,0,5,23,18)]
ARM = [(199,93,2,14,17),(207,94,2,17,20),(228,89,3,20,22),
       (249,80,4,19,21),(260,73,4,14,17),(265,70,4,6,9)]


def volume(shape):
    return 0. if shape is None else sum(s.volume for s in shape.solids())


def cut_volume(shape, cutters):
    rem=shape
    for cutter in cutters:
        rem=rem-cutter
        if rem is None or volume(rem)<1e-9:return 0.
    return round(volume(rem),4)


def intersect_volume(a,c):
    x=a.bounding_box();y=c.bounding_box()
    if any(getattr(x.min,k)>getattr(y.max,k) or getattr(y.min,k)>getattr(x.max,k)
           for k in ('X','Y','Z')):
        return 0.
    return round(volume(a.intersect(c)),4)


def box(x0,x1,y0,y1,z0,z1):
    return b.Solid.make_box(x1-x0,y1-y0,z1-z0,b.Plane(origin=(x0,y0,z0)))


def cyl(radius,length,start,direction):
    return b.Solid.make_cylinder(radius,length,b.Plane(origin=start,z_dir=direction))


def ring(ro,ri,length,start,direction):
    return cyl(ro,length,start,direction)-cyl(ri,length,start,direction)


def beam(radius,start,end):
    a=b.Vector(*start);d=b.Vector(*end)-a
    return cyl(radius,d.length,start,d.normalized())


def vendor_case(output,rotation,clearance=0):
    # Output datum normalized from raw official STEP bounds X[-10,10],
    # Y[-24.5,9.5], Z[-29,0] after its -6.5 mm face adjustment.
    c=clearance
    case=box(-10-c,10+c,-24.5-c,9.5+c,-29-c,c)
    return b.Pos(*output)*rotation*case


def source_geometry(height,wall):
    scale=height/350
    study=Study(height,wall,.65)
    with contextlib.redirect_stdout(io.StringIO()):study.build()
    source={p['name']:p['shape'] for p in study.parts}
    t=study.t
    torso_outer=loft(TORSO).scale(scale)
    torso_inner=loft([(z,x,y,rx-t,ry-t) for z,x,y,rx,ry in TORSO]).scale(scale)
    arm_outer=b.Solid.make_loft([
        b.Wire.make_ellipse(rx,ry,b.Plane(origin=(x,y,z)))
        for z,x,y,rx,ry in ARM]+[b.Vertex(70,4,267.8)]).scale(scale)
    arm_inner=loft([(z,x,y,rx-t,ry-t) for z,x,y,rx,ry in ARM]).scale(scale)
    shoulder_outer=b.Solid.make_loft([
        b.Wire.make_ellipse(ry,rz,b.Plane(origin=(x,4,245),x_dir=(0,1,0),z_dir=(1,0,0)))
        for x,ry,rz in [(63,17,19),(65.5,17,19),(68,16,18)]]).scale(scale)
    return source,torso_outer,torso_inner,arm_outer,arm_inner,shoulder_outer


def part(name,shape,kind,parent,clearance=None):
    return dict(name=name,shape=shape,kind=kind,parent=parent,clearance=clearance)


def mechanism(height):
    k=height/500
    scale=height/350
    py=(73*k,4*scale,245*scale)
    mx=(54*k-(1.5 if height==450 else 0),5*k,320*k)
    pp=(54*k,18*k,355*k)
    # The roll +Z output points inboard/up the source arm centerline.
    rp=(122*k,11,332*k)
    pitch_rot=b.Rot(-90,0,0)
    roll_rot=b.Rot(0,-20,0)
    axis=(-math.sin(math.radians(20)),0,math.cos(math.radians(20)))
    at=lambda dist: tuple(rp[i]+axis[i]*dist*k for i in range(3))
    px,cy,pz=py
    parts=[]
    add=lambda *args,**kwargs:parts.append(part(*args,**kwargs))
    add('yaw_XL330_case',vendor_case(mx,b.Rot(0,0,0)),'case','fixed','torso')
    add('pitch_XL330_case',vendor_case(pp,pitch_rot),'case','yaw','torso')
    add('roll_XL330_case',vendor_case(rp,roll_rot),'case','pitch','arm')

    # Yaw: fixed torso motor, 10T:18T 2M belt, 3 mm vertical output shaft,
    # two bearing seats and independent ribs into an inner chest frame.
    add('yaw_motor_pulley_10T',cyl(3.18,3,(mx[0],mx[1],mx[2]+2*k),(0,0,1)),'drive','fixed')
    add('yaw_motor_output_stub',cyl(2.5,2*k,mx,(0,0,1)),'shaft','fixed')
    add('yaw_joint_pulley_18T',cyl(5.73,3,(px,cy,mx[2]+2*k),(0,0,1)),'drive','fixed')
    for label,y0,y1 in [('front',cy-5.73,cy-4.53),('back',cy+4.53,cy+5.73)]:
        add('yaw_belt_'+label,box(mx[0],px,y0,y1,mx[2]+2*k,mx[2]+5*k),'drive','fixed')
    add('yaw_output_shaft',cyl(3,28*k,(px,cy,314*k),(0,0,1)),'shaft','fixed')
    for n,z in [('low',316*k),('high',334*k)]:
        add('yaw_bearing_'+n,ring(6.5,3.1,3.5,(px,cy,z),(0,0,1)),'seat','fixed')
        add('yaw_bearing_housing_'+n,ring(8.5,6.5,3.5,(px,cy,z),(0,0,1)),'bridge','fixed')
        add('yaw_bridge_'+n,box(px-8.5,px,cy+8,cy+11,z,z+3.5),'bridge','fixed')

    # Pitch motor is carried by yaw yoke. It drives a 26-mm, two-bearing
    # pitch shaft with a short XZ-plane belt; this whole subassembly yaws.
    add('pitch_motor_pulley_10T',cyl(3.18,3,(pp[0],20*k,pp[2]),(0,1,0)),'drive','yaw')
    add('pitch_motor_output_stub',cyl(2.5,2*k,pp,(0,1,0)),'shaft','yaw')
    add('pitch_joint_pulley_18T',cyl(5.73,3,(px,20*k,pz),(0,1,0)),'drive','yaw')
    for label,z0,z1 in [('upper',max(pp[2],pz)+4.53,max(pp[2],pz)+5.73),
                        ('lower',min(pp[2],pz)-5.73,min(pp[2],pz)-4.53)]:
        add('pitch_belt_'+label,box(pp[0],px,20*k,20*k+3,z0,z1),'drive','yaw')
    half=13*k
    add('pitch_output_shaft',cyl(3,36*k,(px,-half,pz),(0,1,0)),'shaft','yaw')
    # The bearing outer races touch cylindrical carrier bores.  The carrier
    # rings, rather than solid cheeks through the shaft, transfer pitch load
    # into the yoke crossbar and its internal spine.
    yaw_shaft_clearance=cyl(3.5,30*k,(px,cy,313*k),(0,0,1))
    for label,y in [('front',-9.25*k),('back',7.25*k)]:
        add('pitch_bearing_'+label,ring(6.5,3.1,3.5,(px,y,pz),(0,1,0)),'seat','yaw')
        housing=ring(8.5,6.5,5,(px,y-.75,pz),(0,1,0))
        if label=='back':housing=housing-yaw_shaft_clearance
        add('pitch_bearing_housing_'+label,housing,'bridge','yaw')
    yaw_bore=cyl(3.1,10,(px,cy,pz-14),(0,0,1))
    add('yaw_yoke_clamp_hub',ring(7,3.1,4,(px,cy,pz-12),(0,0,1)),'bridge','yaw')
    add('yaw_pitch_yoke_crossbar',box(px-4,px+4,-8*k,8*k,pz-11,pz-7)-yaw_bore,'bridge','yaw')
    for label,y0,y1 in [('front',-10*k,-5*k),('back',6.5*k,11.5*k)]:
        add('yaw_pitch_yoke_bearing_tie_'+label,
            box(px-4,px+4,y0,y1,pz-10,pz-8)-yaw_shaft_clearance,'bridge','yaw')

    # The fixed yoke spine supports the bearing carrier. A separate keyed
    # output hub and rotating spine pick up the 6 mm arm-root rail; the two
    # frames have an axial/vertical gap and cannot be silently joined.
    add('pitch_yoke_spine_horizontal',box(px,px+8*k,3*k,9*k,pz-11,pz-7)-yaw_bore,'bridge','yaw')
    add('pitch_output_hub',ring(7,3.1,5,(px,0,pz),(0,1,0)),'bridge','pitch')
    add('pitch_output_hub_key',box(px+2.6,px+3.4,1,4,pz-1,pz+1),'bridge','pitch')
    add('pitch_output_spine_vertical',
        box(px+4*k,px+8*k,3*k,6*k,pz-6.5,pz+8),'bridge','pitch')

    # Roll motor is inside the arm and turns a coaxial 40-mm output shaft.
    # Paired seats react torque to an internal arm carrier, not the skin.
    add('roll_output_shaft',cyl(3,40*k,rp,axis),'shaft','pitch')
    for label,d in [('distal',19),('proximal',34)]:
        add('roll_bearing_'+label,ring(7,3.1,4,at(d),axis),'seat','pitch')
    add('roll_carrier_tube',ring(8,7.1,15*k,at(19),axis),'bridge','pitch')
    rail_start=(px+7*k,6*k,pz+6*k)
    near=at(34)
    rail_end=(near[0]-3*k,6*k,near[2])
    rail=beam(3,rail_start,rail_end)-cyl(3.5,40*k,rp,axis)
    rail=rail-cyl(7.5,4,at(34),axis)
    add('pitch_roll_yoke_rail',rail,'bridge','pitch')
    # Exact 0.5 mm channel clearance through only the local overlapping skins.
    rail_channel=beam(3.5,rail_start,rail_end)
    return parts,dict(yaw_motor=mx,pitch_motor=pp,roll_motor=rp,
                      yaw_pitch_axis=(px,cy,pz),roll_axis=axis,
                      rail_channel=rail_channel)


def power_reserve(height,source_z,orientation='x'):
    # Shared pack and service units remain physical dimensions at anatomical
    # centers scaled from the 35 cm sketch; the pack is never itself scaled.
    s=height/350
    dims=(60,30,30.5) if orientation=='x' else (30,60,30.5)
    pack=b.Pos(0,0,source_z*s)*b.Box(*dims)
    # One millimetre insertion envelope, swept through the rear open panel.
    bb=pack.bounding_box();gap=1
    sweep=box(bb.min.X-gap,bb.max.X+gap,bb.min.Y-gap,100*s,
              bb.min.Z-gap,bb.max.Z+gap)
    # Shared hip-yaw/serial-scale front-center reserve. These are conceptual
    # converter and connector envelopes, not sourced component CAD.
    converter=b.Pos(0,-30*s,194*s)*b.Box(24,12,12)
    connector=b.Pos(0,-30*s,207*s)*b.Box(24,12,8)
    midline=b.Pos(0,0,225*s)*b.Box(25,20,14)
    # Separate fallback bay high in the chest, away from both shoulder sides.
    upper=b.Pos(0,0,255*s)*b.Box(25,20,14)
    return pack,sweep,dict(converter=converter,connector=connector,
                           midline_shared=midline,upper_fallback=upper)


def measure(height,wall):
    source,torso_outer,torso_inner,arm_outer,arm_inner,shoulder_outer=source_geometry(height,wall)
    pieces,datums=mechanism(height)
    relevant=('torso_front','torso_back','R_upper_arm_front','R_upper_arm_back','R_shoulder_trim')
    channel=datums['rail_channel']
    modified=source.copy()
    cut_names=('torso_front','torso_back','R_upper_arm_front','R_upper_arm_back')
    for n in cut_names:modified[n]=source[n]-channel
    for n in ('L_upper_arm_front','L_upper_arm_back'):
        modified[n]=source[n]-channel.mirror(b.Plane.YZ)
    for n in ('torso_front','torso_back'):
        modified[n]=modified[n]-channel.mirror(b.Plane.YZ)
    rail_cut={n:intersect_volume(channel,source[n]) for n in cut_names}
    rail_exposed={}
    for n in cut_names:
        intersection=channel.intersect(source[n])
        if volume(intersection)<.01:continue
        covered=(arm_outer,shoulder_outer) if n.startswith('torso') else (torso_outer,shoulder_outer)
        rail_exposed[n]=cut_volume(b.Compound(children=intersection.solids()),covered)
    outers=(torso_outer,arm_outer,shoulder_outer)
    skin={}
    for p in pieces:
        skin[p['name']]={n:intersect_volume(p['shape'],modified[n]) for n in relevant
                         if intersect_volume(p['shape'],modified[n])>.01}
    cases=[p for p in pieces if p['kind']=='case']
    case_clearance={}
    for p in cases:
        output=datums[p['name'].split('_')[0]+'_motor']
        rot={'yaw':b.Rot(0,0,0),'pitch':b.Rot(-90,0,0),'roll':b.Rot(0,-20,0)}[p['name'].split('_')[0]]
        inner=arm_inner if p['clearance']=='arm' else torso_inner
        case_clearance[p['name']]={str(c):cut_volume(vendor_case(output,rot,c),[inner]) for c in (0,.5,1)}
    outer={}
    for p in pieces:
        try:outer[p['name']]=cut_volume(p['shape'],outers)
        except Exception as exc:raise RuntimeError(f"outer screen {p['name']}: {exc}") from exc
    # Case collisions with other drive groups, omitting its own mated output.
    cross={}
    for i,p in enumerate(pieces):
        if p['kind']!='case':continue
        for q in pieces[i+1:]:
            if q['name'].startswith(p['name'].split('_')[0]+'_'):continue
            v=intersect_volume(p['shape'],q['shape'])
            if v>.01:cross[p['name']+' vs '+q['name']]=v
    power={}
    for z,orientation in ((200,'x'),(220,'x'),(250,'x'),(250,'y')):
        pack,sweep,services=power_reserve(height,z,orientation)
        # Both shoulders mirror about YZ. A central reserve must clear both.
        bilateral=[(p['name']+'_R',p['shape']) for p in pieces]
        bilateral +=[(p['name']+'_L',p['shape'].mirror(b.Plane.YZ)) for p in pieces]
        bb=pack.bounding_box();expanded_pack=box(bb.min.X-.5,bb.max.X+.5,
                                                   bb.min.Y-.5,bb.max.Y+.5,
                                                   bb.min.Z-.5,bb.max.Z+.5)
        head_case=vendor_case((0,4*height/350,276*height/350),b.Rot(0,0,0))
        power[f'{z}_{orientation}']={
            'pack_torso_skin_mm3':round(sum(intersect_volume(pack,source[n]) for n in ('torso_front','torso_back')),4),
            'pack_hardware_hits_mm3':{n:intersect_volume(pack,s) for n,s in bilateral if intersect_volume(pack,s)>.01},
            'pack_plus_0p5_vs_cases_plus_0p5_mm3':{
                p['name']+'_'+side:intersect_volume(expanded_pack,vendor_case(
                    datums[p['name'].split('_')[0]+'_motor'],
                    {'yaw':b.Rot(0,0,0),'pitch':b.Rot(-90,0,0),'roll':b.Rot(0,-20,0)}[p['name'].split('_')[0]],.5).mirror(b.Plane.YZ) if side=='L' else
                    vendor_case(datums[p['name'].split('_')[0]+'_motor'],
                    {'yaw':b.Rot(0,0,0),'pitch':b.Rot(-90,0,0),'roll':b.Rot(0,-20,0)}[p['name'].split('_')[0]],.5))
                for p in cases for side in ('R','L')},
            'withdrawal_front_skin_mm3':intersect_volume(sweep,source['torso_front']),
            'withdrawal_hardware_hits_mm3':{n:intersect_volume(sweep,s) for n,s in bilateral if intersect_volume(sweep,s)>.01},
            'pack_vs_canonical_head_yaw_XL330_mm3':intersect_volume(pack,head_case),
            'service':{name:{
                'torso_skin_mm3':round(sum(intersect_volume(service,source[n]) for n in ('torso_front','torso_back')),4),
                'outside_torso_outer_mm3':cut_volume(service,[torso_outer]),
                'pack_mm3':intersect_volume(service,pack),
                'hardware_hits_mm3':{n:intersect_volume(service,s) for n,s in bilateral if intersect_volume(service,s)>.01},
                'bbox_mm':str(service.bounding_box()),
            } for name,service in services.items()},
            'pack_bbox_mm':str(pack.bounding_box()),
        }
    return dict(height_mm=height,wall_mm=wall,
                datums={k:v for k,v in datums.items() if k!='rail_channel'},
                rail_channel_source_skin_cut_mm3=rail_cut,
                rail_channel_exposed_after_other_outer_mm3=rail_exposed,
                case_outside_inner_mm3=case_clearance,
                hardware_skin_hits_mm3={k:v for k,v in skin.items() if v},
                hardware_outside_source_outer_mm3={k:v for k,v in outer.items() if v>.01},
                cross_case_hardware_hits_mm3=cross,power=power),source,modified,pieces


def pose_sweep(height,source,modified,pieces):
    """Rigid coupled shoulders against fixed skin and their moving arm skins.

    The source right arm is mirrored after articulation for a symmetric left
    arm.  These are envelope poses, not an inverse-kinematics or cable model.
    """
    _,d=mechanism(height)
    py=d['yaw_pitch_axis'];rp=d['roll_motor'];axis=d['roll_axis']
    skin_names=('torso_front','torso_back','R_shoulder_trim','neck_collar',
                'head_cranium_front','head_cranium_back')
    fixed_skin={n:modified[n] for n in skin_names}
    arm_names=('R_upper_arm_front','R_upper_arm_back','R_forearm_front',
               'R_forearm_back','R_elbow_cuff','R_wrist_cuff','R_palm_front',
               'R_palm_back','R_finger_1_cap','R_finger_2_cap',
               'R_finger_3_cap','R_finger_4_cap','R_thumb_cap')
    poses={'down':(0,0,0),'forward':(-90,-75,0),
           'out':(0,-75,0),'floor_reach':(-45,-35,30)}
    def moved(shape,parent,angles):
        yaw,pitch,roll=angles
        if parent=='arm' and roll:
            shape=shape.rotate(b.Axis(rp,axis),roll)
        if parent in ('arm','pitch') and pitch:
            shape=shape.rotate(b.Axis(py,(0,1,0)),pitch)
        if parent in ('arm','pitch','yaw') and yaw:
            shape=shape.rotate(b.Axis(py,(0,0,1)),yaw)
        return shape
    out={}
    for label,angles in poses.items():
        print('screening pose',label,flush=True)
        arm={n:moved(modified[n],'arm',angles) for n in arm_names}
        active={p['name']:moved(p['shape'],p['parent'],angles) for p in pieces}
        impacts={}
        for p in pieces:
            n=p['name'];shape=active[n]
            if p['kind']=='case':
                key=n.split('_')[0]
                rot={'yaw':b.Rot(0,0,0),'pitch':b.Rot(-90,0,0),
                     'roll':b.Rot(0,-20,0)}[key]
                shape=moved(vendor_case(d[key+'_motor'],rot,.5),p['parent'],angles)
            elif n=='pitch_roll_yoke_rail':
                # Radius 3.5 instead of 3 mm gives a conservative 0.5 mm
                # moving keep-out, including the un-notched tube endpoint.
                shape=moved(d['rail_channel'],p['parent'],angles)
            targets=dict(fixed_skin)
            targets.update({k:v for k,v in arm.items() if k.startswith('R_upper_arm')})
            hits={k:q for k,v in targets.items()
                  if (q:=intersect_volume(shape,v))>.01}
            if hits:impacts[n]=hits
        arm_torso={}
        for n,shape in arm.items():
            v=round(sum(intersect_volume(shape,fixed_skin[t])
                        for t in ('torso_front','torso_back')),3)
            if v>.01:arm_torso[n]=v
        # Mirror the entire right assembly; ask whether any active right hard
        # part physically reaches the opposite arm's opaque source skin.
        left_arm={n:v.mirror(b.Plane.YZ) for n,v in arm.items()
                  if n.startswith('R_upper_arm')}
        cross_arm={}
        for n,shape in active.items():
            v=round(sum(intersect_volume(shape,other)
                        for other in left_arm.values()),3)
            if v>.01:cross_arm[n]=v
        out[label]=dict(angles_deg=dict(yaw=angles[0],pitch=angles[1],roll=angles[2]),
                        hardware_skin_hits_mm3=impacts,
                        case_and_rail_inflation_mm=.5,
                        arm_to_fixed_torso_skin_mm3=arm_torso,
                        hardware_to_opposite_upper_arm_mm3=cross_arm)
    return out


def tessellate(name,shape,group,color):
    vv,ff=shape.tessellate(.8)
    return dict(name=name,group=group,color=color,
                vertices=[[round(v.X,3),round(v.Y,3),round(v.Z,3)] for v in vv],
                faces=[list(f) for f in ff])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--height',type=int,choices=[350,450,500],action='append')
    ap.add_argument('--wall',type=int,choices=[1,2],action='append')
    args=ap.parse_args();heights=args.height or [450,500];walls=args.wall or [1,2]
    reports={};OUT.mkdir(parents=True,exist_ok=True)
    for h in heights:
        for w in walls:
            result,source,modified,pieces=measure(h,w)
            if h==500 and w==2:
                result['coupled_poses']=pose_sweep(h,source,modified,pieces)
            reports[f'{h}_{w}']=result
            print(h,w,'case outside +1',result['case_outside_inner_mm3'],
                  'hardware skin',len(result['hardware_skin_hits_mm3']),
                  'outside',len(result['hardware_outside_source_outer_mm3']),flush=True)
            if w==2:
                def color(name):
                    return ('#24231f' if 'eye_lens' in name or 'nostril_inlay' in name else
                            '#a79d89' if 'eye_bezel' in name or 'mouth_inlay' in name else
                            '#c9794c' if 'chest_indicator' in name else '#dfd8c6')
                changed={n for n in source if modified[n] is not source[n]}
                scene=[tessellate(name,shape,'fixed-shell',color(name))
                       for name,shape in source.items() if name not in changed]
                scene +=[tessellate(name,source[name],'source-cut',color(name)) for name in sorted(changed)]
                scene +=[tessellate(name,modified[name],'candidate-cut',color(name)) for name in sorted(changed)]
                scene +=[tessellate(p['name'],p['shape'],p['kind'],
                                   {'case':'#3978aa','drive':'#52575c','shaft':'#9ca6ab',
                                    'seat':'#b68d53','bridge':'#937149'}[p['kind']]) for p in pieces]
                pack,sweep,services=power_reserve(h,200,'x')
                scene.extend([tessellate('pack_source_z200_x',pack,'power','#cb8341'),
                              tessellate('midline_service_source_z225',services['midline_shared'],'power','#c3ab68')])
                (OUT/f'scene_{h}.json').write_text(json.dumps(scene))
    report={'status':'REJECTED: exact coupled forward/out/floor poses collide despite neutral source-shell fit',
            'source':'experiments/sketch_ape_shell/build.py Study(height,wall,gap=.65)',
            'component_basis':'valid official XL330 20x34x29 mm output-datum box; physical material STEP is not used for booleans',
            'topology':'torso-fixed yaw belt; pitch motor and belt on yaw yoke; coaxial arm-long-axis roll motor/shaft with paired internal supports',
            'screens':reports}
    (OUT/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
