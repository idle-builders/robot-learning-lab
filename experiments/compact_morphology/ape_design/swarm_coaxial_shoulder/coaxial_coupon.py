"""Bounded torso-fixed coaxial yaw/pitch shoulder inside Study(500,2,.65).

Conservative valid solids represent vendor-sized packages. The KHK gear is an
OD/axial bounding envelope rather than tooth CAD; meshing volumes are allowed
only for the gear pair. Units are millimetres.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import build123d as b

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OUT=ROOT/'artifacts/coaxial-shoulder'
sys.path.insert(0,str(ROOT/'experiments/compact_morphology/ape_design/swarm_scaled_shoulder'))
import scaled_shoulder_coupon as base  # noqa: E402

PX,CY,PZ=67.,5.714285714,350.
YAW_M=(43.,5.,320.)
PITCH_M=(43.,5.,360.)
ROLL_M=(130.,11.,320.)
ROLL_AXIS=(-math.sin(math.radians(20)),0,math.cos(math.radians(20)))
PIVOT=(PX,CY,PZ)


def add(parts,name,shape,kind,parent):
    parts.append(dict(name=name,shape=shape,kind=kind,parent=parent))


def mechanism():
    p=[]
    c=base.cyl;r=base.ring;box=base.box;beam=base.beam
    at=lambda d:tuple(ROLL_M[i]+ROLL_AXIS[i]*d for i in range(3))
    add(p,'yaw_XL330_case',base.vendor_case(YAW_M,b.Rot(0,0,0)),'case','fixed')
    add(p,'pitch_XL330_case',base.vendor_case(PITCH_M,b.Rot(0,0,0)),'case','fixed')
    add(p,'roll_XL330_case',base.vendor_case(ROLL_M,b.Rot(0,-20,0)),'case','pitch')

    # Two distinct fixed-motor inputs: 14T:24T GT2 XY belts. The outer tube
    # defines yaw, the inner shaft remains independent; pitch is their angular
    # difference after the one-to-one spiral miter turn in the yaw carrier.
    add(p,'yaw_motor_output_stub',c(2.5,2,YAW_M,(0,0,1)),'shaft','fixed')
    add(p,'pitch_motor_output_stub',c(2.5,2,PITCH_M,(0,0,1)),'shaft','fixed')
    for label,m,z in [('yaw',YAW_M,322),('pitch',PITCH_M,362)]:
        add(p,label+'_motor_14T',c(4.46,3,(m[0],m[1],z),(0,0,1)),'drive','fixed')
        add(p,label+'_axis_24T',r(7.64,5 if label=='yaw' else 4,3,
                                  (PX,CY,z),(0,0,1)),'drive','fixed')
        for side,y0,y1 in [('front',CY-6.7,CY-5.5),('back',CY+5.5,CY+6.7)]:
            add(p,label+'_belt_'+side,box(m[0],PX,y0,y1,z,z+3),'drive','fixed')
    add(p,'inner_pitch_shaft',c(4,65,(PX,CY,300),(0,0,1)),'shaft','fixed')
    add(p,'outer_yaw_tube',r(5,4.2,26,(PX,CY,306),(0,0,1)),'shaft','fixed')
    for label,z in [('low',308),('high',317)]:
        add(p,'yaw_bearing_'+label,r(9,5.1,4,(PX,CY,z),(0,0,1)),'seat','fixed')
        add(p,'yaw_bearing_housing_'+label,r(10.5,9,4,(PX,CY,z),(0,0,1)),'bridge','fixed')
        add(p,'yaw_chest_anchor_'+label,box(PX-12,PX-9,CY+9,CY+14,z,z+4),'bridge','fixed')
    add(p,'yaw_yoke_hub',r(7.5,5,4,(PX,CY,328),(0,0,1)),'bridge','yaw')

    # KHK MMSA1-20R/MMSB1-20L: actual OD 21.29, total length 13.53.
    # Inflate each by 0.5 radially and axially for the package screen.
    # The cylinders conservatively occupy the catalog axial length; their
    # mutual intersection stands for meshing tooth volumes, not metal overlap.
    add(p,'vertical_miter_envelope',r(11.145,4.1,14.53,
                                     (PX,CY,334.5),(0,0,1)),'gear','fixed')
    add(p,'horizontal_miter_envelope',r(11.145,4.1,14.53,
                                       (PX,6.5,PZ),(0,1,0)),'gear','yaw')
    add(p,'pitch_output_shaft',c(4,44,(PX,-16,PZ),(0,1,0)),'shaft','yaw')
    for label,y in [('front',-12),('back',22)]:
        add(p,'pitch_bearing_'+label,r(7,4.1,4,(PX,y,PZ),(0,1,0)),'seat','yaw')
        add(p,'pitch_bearing_housing_'+label,r(9,7,4,(PX,y,PZ),(0,1,0)),'bridge','yaw')

    # Rotating yaw yoke bypasses the miter envelopes. It supports both pitch
    # bearing outer races via lower crossbar, offset posts and short ties.
    add(p,'yaw_yoke_lower_crossbar',box(54,PX,-14,26,328,331.5),'bridge','yaw')
    for label,y0,y1 in [('front',-14,-8),('back',21,27)]:
        add(p,'yaw_yoke_post_'+label,box(54,55.8,y0,y1,331,354),'bridge','yaw')
        add(p,'yaw_yoke_tie_'+label,box(55.7,60,y0,y1,348,352),'bridge','yaw')

    # The pitch hub rotates on its own supported shaft. A small key provides
    # torque continuity; an arm-root rail joins the hub to the roll carrier.
    add(p,'pitch_output_hub',r(7,4.1,6,(PX,-2,PZ),(0,1,0)),'bridge','pitch')
    add(p,'pitch_hub_key',box(PX+3.6,PX+4.4,-1,3,PZ-1,PZ+1),'bridge','pitch')
    add(p,'pitch_output_spine',box(PX+4,PX+8,3,6,PZ-6.5,PZ+8),'bridge','pitch')
    add(p,'roll_output_shaft',c(3,40,ROLL_M,ROLL_AXIS),'shaft','pitch')
    for label,d in [('distal',19),('proximal',34)]:
        add(p,'roll_bearing_'+label,r(7,3.1,4,at(d),ROLL_AXIS),'seat','pitch')
    add(p,'roll_carrier_tube',r(8,7.1,15,at(19),ROLL_AXIS),'bridge','pitch')
    rail_start=(PX+7,6,PZ+8)
    near=at(34)
    rail_end=(near[0]-3,6,near[2])
    rail=beam(3,rail_start,rail_end)-c(3.5,40,ROLL_M,ROLL_AXIS)
    rail=rail-c(7.5,4,at(34),ROLL_AXIS)
    add(p,'pitch_roll_rail',rail,'bridge','pitch')
    return p,beam(3.5,rail_start,rail_end)


def stock_pulley_envelopes():
    # SDP/SI catalog A 6A51M014DF0303 and A 6A51M024DF0306.
    # The flanges set the maximum diameter (13.5/20.1); the whole axial
    # length is reserved even though only 3 mm carries belt teeth.
    c=base.cyl
    return {f'{label}_motor_14T_stock':c(6.75,11.1,(m[0],m[1],z),(0,0,1))
            for label,m,z in [('yaw',YAW_M,322),('pitch',PITCH_M,362)]} | {
           f'{label}_axis_24T_stock':c(10.05,13.1,(PX,CY,z),(0,0,1))
           for label,z in [('yaw',322),('pitch',362)]}


def moved(shape,parent,angles):
    yaw,pitch,roll=angles
    if parent=='arm' and roll:
        shape=shape.rotate(b.Axis(ROLL_M,ROLL_AXIS),roll)
    if parent in ('arm','pitch') and pitch:
        shape=shape.rotate(b.Axis(PIVOT,(0,1,0)),pitch)
    if parent in ('arm','pitch','yaw') and yaw:
        shape=shape.rotate(b.Axis(PIVOT,(0,0,1)),yaw)
    return shape


def screen():
    source,to,ti,ao,ai,so=base.source_geometry(500,2)
    parts,channel=mechanism()
    modified=source.copy()
    names=('torso_front','torso_back','R_upper_arm_front','R_upper_arm_back')
    for n in names:
        modified[n]=source[n]-channel
    for n in ('torso_front','torso_back'):
        modified[n]=modified[n]-channel.mirror(b.Plane.YZ)
    for n in ('L_upper_arm_front','L_upper_arm_back'):
        modified[n]=source[n]-channel.mirror(b.Plane.YZ)
    rail_cut={n:base.intersect_volume(channel,source[n]) for n in names}
    rail_exposed={}
    for n in names:
        x=channel.intersect(source[n])
        if base.volume(x)<.01:continue
        covers=(ao,so) if n.startswith('torso') else (to,so)
        rail_exposed[n]=base.cut_volume(b.Compound(children=x.solids()),covers)
    assigned={'yaw_XL330_case':ti,'pitch_XL330_case':ti,'roll_XL330_case':ai}
    case_outer={}
    for n,out,rot in [('yaw_XL330_case',YAW_M,b.Rot(0,0,0)),
                      ('pitch_XL330_case',PITCH_M,b.Rot(0,0,0)),
                      ('roll_XL330_case',ROLL_M,b.Rot(0,-20,0))]:
        case_outer[n]={str(c):base.cut_volume(base.vendor_case(out,rot,c),[assigned[n]])
                       for c in (0,.5,1)}
    skin_names=('torso_front','torso_back','R_upper_arm_front',
                'R_upper_arm_back','R_shoulder_trim')
    skin={};outside={}
    for p in parts:
        v={n:q for n in skin_names if (q:=base.intersect_volume(p['shape'],modified[n]))>.01}
        if v:skin[p['name']]=v
        q=base.cut_volume(p['shape'],(to,ao,so))
        if q>.01:outside[p['name']]=q
    stock=stock_pulley_envelopes()
    stock_skin={n:{k:q for k in skin_names
                   if (q:=base.intersect_volume(v,modified[k]))>.01}
                for n,v in stock.items()}
    stock_outside={n:base.cut_volume(v,(to,ao,so)) for n,v in stock.items()}
    stock_pairs={n:{p['name']:q for p in parts
                    if p['kind'] in ('case','seat','bridge','gear')
                    and (q:=base.intersect_volume(v,p['shape']))>.01}
                 for n,v in stock.items()}
    power={}
    pack,sweep,svc=base.power_reserve(500,200,'x')
    bilateral=[(p['name']+'_R',p['shape']) for p in parts]
    bilateral +=[(p['name']+'_L',p['shape'].mirror(b.Plane.YZ)) for p in parts]
    for label,shape in [('pack',pack),('withdrawal',sweep),('service',svc['midline_shared'])]:
        power[label]=dict(torso_front_skin_mm3=base.intersect_volume(shape,source['torso_front']),
                          torso_back_skin_mm3=base.intersect_volume(shape,source['torso_back']),
                          hardware_hits_mm3={n:q for n,v in bilateral
                                             if (q:=base.intersect_volume(shape,v))>.01},
                          pack_overlap_mm3=base.intersect_volume(shape,pack)
                          if label=='service' else 0)
    report=dict(source='Study(500,wall=2,gap=.65)',
                topology='fixed yaw/pitch motors; nested yaw tube/inner shaft; 1:1 spiral miter differential; arm roll motor',
                kinematics='yaw = outer tube; pitch = inner shaft minus outer tube (equal 1:1 miter ratio); motor angle scales by belt ratios',
                case_outside_assigned_inner_mm3=case_outer,
                static_skin_hits_mm3=skin,
                static_outside_combined_outer_mm3=outside,
                stock_pulley_skin_hits_mm3=stock_skin,
                stock_pulley_outside_combined_outer_mm3=stock_outside,
                stock_pulley_hardware_hits_mm3=stock_pairs,
                rail_channel_source_cut_mm3=rail_cut,
                rail_channel_exposed_mm3=rail_exposed,power=power)
    return report,source,modified,parts,channel


def pose_screen(source,modified,parts,channel):
    fixed_names=('torso_front','torso_back','R_shoulder_trim',
                 'neck_collar','head_cranium_front','head_cranium_back')
    fixed={n:modified[n] for n in fixed_names}
    arms=('R_upper_arm_front','R_upper_arm_back','R_forearm_front',
          'R_forearm_back','R_elbow_cuff','R_wrist_cuff','R_palm_front',
          'R_palm_back','R_finger_1_cap','R_finger_2_cap','R_finger_3_cap',
          'R_finger_4_cap','R_thumb_cap')
    poses={'down':(0,0,0),'forward':(-90,-75,0),
           'out':(0,-75,0),'floor_reach':(-45,-35,30)}
    result={}
    for name,angles in poses.items():
        print('pose',name,flush=True)
        arm={n:moved(modified[n],'arm',angles) for n in arms}
        targets=dict(fixed)
        targets.update({n:v for n,v in arm.items() if n.startswith('R_upper_arm')})
        hits={}
        for p in parts:
            n=p['name'];shape=p['shape']
            if p['kind']=='case':
                out,rot={'yaw_XL330_case':(YAW_M,b.Rot(0,0,0)),
                         'pitch_XL330_case':(PITCH_M,b.Rot(0,0,0)),
                         'roll_XL330_case':(ROLL_M,b.Rot(0,-20,0))}[n]
                shape=base.vendor_case(out,rot,.5)
            elif n=='pitch_roll_rail':shape=channel
            shape=moved(shape,p['parent'],angles)
            h={t:q for t,v in targets.items()
               if (q:=base.intersect_volume(shape,v))>.01}
            if h:hits[n]=h
        result[name]=dict(angles_deg=angles,case_and_rail_allowance_mm=.5,
                          hardware_skin_hits_mm3=hits)
    return result


def main():
    scenes_only='--scenes-only' in sys.argv
    if scenes_only:
        source,*_=base.source_geometry(500,2)
        parts,channel=mechanism()
        modified=source.copy()
        for n in ('torso_front','torso_back','R_upper_arm_front','R_upper_arm_back'):
            modified[n]=source[n]-channel
        for n in ('torso_front','torso_back'):
            modified[n]=modified[n]-channel.mirror(b.Plane.YZ)
        for n in ('L_upper_arm_front','L_upper_arm_back'):
            modified[n]=source[n]-channel.mirror(b.Plane.YZ)
        report={'status':'scenes regenerated from existing exact report'}
    else:
        report,source,modified,parts,channel=screen()
        print('static',report['static_skin_hits_mm3'],report['static_outside_combined_outer_mm3'],flush=True)
        proxy_static_ok=(not report['static_skin_hits_mm3'] and
               not report['static_outside_combined_outer_mm3'] and
               not any(rows['0.5']>.01 for rows in
                       report['case_outside_assigned_inner_mm3'].values()) and
               not any(report['rail_channel_exposed_mm3'].values()))
        stock_ok=(not any(report['stock_pulley_skin_hits_mm3'].values()) and
              not any(v>.01 for v in report['stock_pulley_outside_combined_outer_mm3'].values()) and
              not any(report['stock_pulley_hardware_hits_mm3'].values()))
        if proxy_static_ok:
            report['proxy_poses']=pose_screen(source,modified,parts,channel)
        report['status']=('REJECTED: source-size stock pulley interference'
                      if not stock_ok else
                      'REJECTED: proxy moving hard-part/skin collision'
                      if proxy_static_ok and any(p['hardware_skin_hits_mm3']
                          for p in report['proxy_poses'].values()) else
                      'REJECTED: proxy static collision'
                      if not proxy_static_ok else 'UNQUALIFIED: proxy poses clear')
    OUT.mkdir(parents=True,exist_ok=True)
    if not scenes_only:(OUT/'report.json').write_text(json.dumps(report,indent=2))
    def color(n):return '#24231f' if 'eye_lens' in n else '#dfd8c6'
    changed={n for n in source if modified[n] is not source[n]}
    pack,sweep,svc=base.power_reserve(500,200,'x')
    for pose,angles in [('down',(0,0,0)),('forward',(-90,-75,0)),
                        ('out',(0,-75,0)),('floor_reach',(-45,-35,30))]:
        arm_prefixes=('R_upper_arm','R_forearm','R_elbow_cuff','R_wrist_cuff',
                      'R_palm','R_finger','R_thumb')
        pose_shape=lambda n,s: moved(s,'arm',angles) if n.startswith(arm_prefixes) else s
        scene=[base.tessellate(n,pose_shape(n,v),'fixed-shell',color(n))
               for n,v in source.items() if n not in changed]
        scene +=[base.tessellate(n,pose_shape(n,source[n]),'source-cut',color(n))
                for n in sorted(changed)]
        scene +=[base.tessellate(n,pose_shape(n,modified[n]),'candidate-cut',color(n))
                for n in sorted(changed)]
        scene +=[base.tessellate(p['name'],moved(p['shape'],p['parent'],angles),p['kind'],
                                {'case':'#3978aa','drive':'#52575c','shaft':'#9ca6ab',
                                 'gear':'#bb945b','seat':'#b68d53','bridge':'#937149'}[p['kind']])
                 for p in parts]
        scene +=[base.tessellate(n,v,'stock-envelope','#b8564a')
                 for n,v in stock_pulley_envelopes().items()]
        scene +=[base.tessellate('4S_pack',pack,'power','#cb8341'),
                 base.tessellate('5V_CAN_service',svc['midline_shared'],'power','#c3ab68')]
        (OUT/('scene.json' if pose=='down' else f'scene_{pose}.json')).write_text(json.dumps(scene))
    print(report['status'],flush=True)


if __name__=='__main__':main()
