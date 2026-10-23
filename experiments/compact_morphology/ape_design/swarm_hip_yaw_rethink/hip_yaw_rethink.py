"""Exact 350/500 mm hip-yaw topology screen against the sketch-06 shell.

This is a package/support coupon, not a qualified transmission design. It keeps
the round motor's manufacturer envelope separate from proposed transfer parts.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import build123d as b
import numpy as np
import trimesh

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'experiments/sketch_ape_shell'))
from build import Study
OUT=ROOT/'artifacts/hip-yaw-rethink'
OUT.mkdir(parents=True,exist_ok=True)
PACK=(0,0,200); PACK_SIZE=(60,30,30.5)
HIP_X=26.5; HIP_Z=160.0
SHELL_CACHE={}

def box(c,sz): return b.Pos(*c)*b.Box(*sz)
def cyl(axis,r,h,p):
    rot={'x':b.Rot(0,90,0),'y':b.Rot(90,0,0),'z':b.Rot(0,0,0)}[axis]
    return b.Pos(*p)*rot*b.Cylinder(r,h)
def vv(sh): return round(sum(s.volume for s in sh.solids()),3) if sh else 0.0
def bounds(sh):
    q=sh.bounding_box();return [[round(v,3) for v in q.min],[round(v,3) for v in q.max]]
def scaled(sh,k): return sh.scale(k)
def slot_belt(p1,p2,z,width=2.0,clearance=3.0):
    dx=p2[0]-p1[0];dy=p2[1]-p1[1];d=(dx*dx+dy*dy)**.5
    if d<=0: raise ValueError('belt centers must be distinct')
    mid=((p1[0]+p2[0])/2,(p1[1]+p2[1])/2,z)
    plane=b.Plane(origin=mid,x_dir=(dx/d,dy/d,0),z_dir=(0,0,1))
    outer=b.extrude(b.SlotCenterToCenter(d,10+2*clearance),amount=width/2,both=True)
    inner=b.extrude(b.SlotCenterToCenter(d,10),amount=width/2+.1,both=True)
    return plane.location*(outer-inner)
def overlaps(a,c,margin=0.0):
    A=a.bounding_box();C=c.bounding_box()
    amin=(A.min.X,A.min.Y,A.min.Z);amax=(A.max.X,A.max.Y,A.max.Z)
    cmin=(C.min.X,C.min.Y,C.min.Z);cmax=(C.max.X,C.max.Y,C.max.Z)
    if not all(amin[i]-margin<=cmax[i]+margin and cmin[i]-margin<=amax[i]+margin for i in range(3)):
        return 0.0
    # OCCT can return zero in one operand order for near-tangent compound
    # shells while the reverse order contains a finite solid. Keep the larger
    # exact result so a nominal clearance is never inferred from that asymmetry.
    return max(vv(a.intersect(c)), vv(c.intersect(a)))
def shell(height,wall):
    if (height,wall) in SHELL_CACHE:return SHELL_CACHE[(height,wall)]
    st=Study(height,wall,.65);st.build();k=height/350
    parts={x['name']:x['shape'] for x in st.parts}
    named={n:s for n,s in parts.items() if n in {'torso_front','torso_back','pelvis_front','pelvis_back'}}
    outer={}
    for n,stations in {
      'torso':[(160,0,2,42,25),(173,0,2,49,29),(199,0,3,58,34),(229,0,4,63,35),(253,0,5,63,30),(264,0,5,60,27),(274,0,5,35,22),(278,0,5,23,18)],
      'pelvis':[(130,0,5,9,11),(135,0,5,20,17),(145,0,5,34,23),(155,0,4,43,26),(164,0,3,46,26),(178,0,3,49,29)]}.items():
        outer[n]=st.hollow_loft(stations)[1]
    result=(k,named,{n:s.scale(k) for n,s in outer.items()})
    SHELL_CACHE[(height,wall)]=result
    return result
def rsf(point,k):
    # Official drawing: Ø25 × 35.8 axial. Output face is local z=0; body lies +Z.
    x,y,z=point
    return cyl('z',12.5,35.8,(x*k,y*k,z*k+17.9))
def service_parts(k):
    # 12/5 V conversion, CAN/interface and connector pull space; assumed envelope.
    return [('central regulated-power and CAN service reserve',box((0,0,225*k),(25,20,14)))]

def assembly_solids(height=500, layout='remote', pack_z_source=200,
                   service_z_source=225):
    """Return an importable, source-frame hip-yaw coupon for assembly checks.

    ``fixed`` includes motor cases, bearing/support envelopes, the belt
    envelope, battery, and service reservation. ``moving`` contains the
    supported yaw shafts and femur flanges. These are geometry proposals, not
    qualified bearing fits or a completed belt transmission.
    """
    if layout not in {'remote','coaxial'}:
        raise ValueError("layout must be 'remote' or 'coaxial'")
    k=height/350
    fixed=[]; moving=[]; axes=[]
    for side in (-1,1):
        hip=(side*HIP_X*k,0,HIP_Z*k)
        motor_x=side*(13.5 if layout=='remote' else HIP_X)
        motor_y=-10 if layout=='remote' else 0
        fixed.append((f'{side:+} RSF-5B-50-IDT motor case',rsf((motor_x,motor_y,HIP_Z),k)))
        axes.append({'name':f'{side:+} hip yaw','origin_mm':hip,'axis':[0,0,1]})
        moving.append((f'{side:+} supported yaw shaft',cyl('z',3,16,(hip[0],0,hip[2]-8))))
        moving.append((f'{side:+} femur yaw flange',cyl('z',5,4,(hip[0],0,hip[2]-14))))
        for dz in (-10,-2):
            bearing=cyl('z',6.5,3.5,(hip[0],0,hip[2]+dz))-cyl('z',3,4,(hip[0],0,hip[2]+dz))
            fixed.append((f'{side:+} W618/6 bearing envelope z{dz}',bearing))
        fixed.append((f'{side:+} belt-side support cheek',box((side*32*k,0,HIP_Z*k-5),(4,8,8))))
        if layout=='remote':
            fixed.append((f'{side:+} belt envelope',slot_belt((motor_x*k,motor_y*k),(hip[0],0),HIP_Z*k-2,2,3)))
    fixed.append(('fixed transverse support bridge',box((0,0,HIP_Z*k-10),(66,8,3.5))))
    fixed.append(('4S battery envelope',box((0,0,pack_z_source*k),(60,30,30.5))))
    fixed.append(('power/CAN service reserve',box((0,0,service_z_source*k),(25,20,14))))
    return {'height_mm':height,'layout':layout,'fixed':fixed,'moving':moving,
            'joint_axes':axes,'scale':k}
def candidate(height,wall,layout):
    k,named,outer=shell(height,wall); comps=[]; motors=[];supports=[];belts=[];shafts=[]
    # Mirrored pair. local: actuators coaxial with hip axes. remote: bodies near
    # centerline with parallel vertical outputs, short 1:1 belt to lateral shafts.
    for side in (-1,1):
        hip=(side*HIP_X,0,HIP_Z)
        mpoint=hip if layout=='coaxial' else (side*13.5,-10,HIP_Z)
        motor=rsf(mpoint,k); motors.append(motor);comps.append((f'{side:+} RSF-5B-50-IDT body',motor))
        # Manufacturer-dimensioned output shaft datum; output is at hip z=160.
        output=cyl('z',3,16,(hip[0]*k,0,HIP_Z*k-8));shafts.append(output)
        for dz in (-10,-2):
            ring=cyl('z',6.5,3.5,(hip[0]*k,0,HIP_Z*k+dz))-cyl('z',3,4,(hip[0]*k,0,HIP_Z*k+dz))
            supports.append(ring)
        # Compact yoke/rotating carrier connects shaft to proximal femur clevis.
        yoke=cyl('z',5,4,(hip[0]*k,0,HIP_Z*k-14))
        supports.append(yoke)
        if layout=='remote':
            belt=slot_belt((mpoint[0]*k,mpoint[1]*k),(hip[0]*k,0),HIP_Z*k-2,2,3)
            belts.append(belt)
    # Fixed cross-bridge supports each lower radial bearing; paired cheeks carry
    # the upper bearings. These are proposed structure envelopes, not rated parts.
    supports.append(box((0,0,HIP_Z*k-10),(66,8,3.5)))
    for side in (-1,1):supports.append(box((side*32*k,0,HIP_Z*k-5),(4,8,8)))
    comps += [(f'{layout} paired supported shafts',x) for x in shafts]
    comps += [(f'{layout} support or carrier {i}',x) for i,x in enumerate(supports)]
    comps += [(f'{layout} belt {i}',x) for i,x in enumerate(belts)]
    pack=box((PACK[0]*k,PACK[1]*k,PACK[2]*k),PACK_SIZE);comps.append(('Tattu 4S850 candidate pack',pack))
    serv=service_parts(k);comps+=serv
    # Shoulder yaw and pitch XL330 cases from the other team's reported 500 mm
    # bounds, represented by conservative valid 20×34×29/34 mm box envelopes.
    # Scale these reference cases from 500 mm, independently of source coords.
    shoulder=[]
    for side in (-1,1):
        q=height/500
        shoulder.append((f'{side:+} shoulder yaw case',box((side*55*q,5*q,305.5*q),(20,34,29))))
        shoulder.append((f'{side:+} shoulder pitch case',box((side*55*q,-12*q,362.5*q),(20,34,34))))
    # 18×18×14 service reserve includes regulators, CAN transceiver and service loop.
    skinhits=[];outside=[]
    for name,shape in comps:
        if 'shoulder' in name: continue
        possible=['torso','pelvis'] if ('pack' in name or 'reserve' in name) else ['pelvis','torso']
        for sn,s in named.items():
            if any(sn.startswith(x) for x in possible):
                v=overlaps(shape,s)
                if v>0.001: skinhits.append({'component':name,'skin':sn,'mm3':v})
        # Component material outside any relevant *outer* lobe is a placement warning.
        refs=[outer['torso'],outer['pelvis']]
        remaining=shape
        for reg in refs:
            if remaining and overlaps(remaining,reg)>0:
                remaining=remaining.cut(reg)
        ext=vv(remaining)
        if ext>0.01: outside.append({'component':name,'outside_torso_plus_pelvis_mm3':ext,'fraction':round(ext/max(vv(shape),.001),5)})
    pairhits=[]
    special=[('pack',pack),*serv,*shoulder]
    motors_named=[(f'{layout} motor {i}',m) for i,m in enumerate(motors)]
    for na,a in motors_named:
        for nb,c in special:
            v=overlaps(a,c,margin=.5*k)
            if v>0.001:pairhits.append({'a':na,'b':nb,'mm3':v})
    for i,(na,a) in enumerate(shoulder):
        for nb,c in [('pack',pack),*serv]:
            v=overlaps(a,c,margin=.5*k)
            if v>0.001:pairhits.append({'a':na,'b':nb,'mm3':v})
    expanded_pack=box((PACK[0]*k,PACK[1]*k,PACK[2]*k),tuple(x+1.0 for x in PACK_SIZE))
    pack_shoulders=[]
    for name,should in shoulder:
        pack_shoulders.append({'shoulder_case':name,'pack_case_common_mm3':overlaps(pack,should),
          'expanded_pack_0p5mm_common_mm3':overlaps(expanded_pack,should),
          'exact_surface_gap_mm':round(pack.distance_to(should),4)})
    # 0.5 mm hard-clearance test via expanded AABB prefilter + exact dilated shape is not
    # supported here; conservatively report near-gap from bounding intervals for critical pairs.
    def aabb_gap(a,c):
        A=a.bounding_box();C=c.bounding_box()
        amin=(A.min.X,A.min.Y,A.min.Z);amax=(A.max.X,A.max.Y,A.max.Z)
        cmin=(C.min.X,C.min.Y,C.min.Z);cmax=(C.max.X,C.max.Y,C.max.Z)
        return round(min(max(cmin[i]-amax[i],amin[i]-cmax[i],0) for i in range(3)),3)
    gaps=[]
    for n,s in motors_named:
        for obname,ob in [('pack',pack),*serv,*shoulder]:gaps.append({'a':n,'b':obname,'aabb_min_axis_gap_mm':aabb_gap(s,ob)})
    exact_gaps=[]
    if height==500 and wall==1:
      for n,s in motors_named:
        for obname,ob in [('pack',pack),*serv,*shoulder]:exact_gaps.append({'a':n,'b':obname,'distance_mm':round(s.distance_to(ob),4)})
    return {'height_mm':height,'wall_mm':wall,'layout':layout,'scale':k,'motor_mass_g_each':77,'yaw_allowable_continuous_Nm_each':.29,'yaw_max_Nm_each':.9,
      'motor_case_bounds':[(n,bounds(s)) for n,s in motors_named],'pack_bounds':bounds(pack),'service_bounds':[(n,bounds(s)) for n,s in serv],
      'shoulder_bounds':[(n,bounds(s)) for n,s in shoulder],'shell_intersections':skinhits,'outside_combined_lobes':outside,'critical_case_intersections':pairhits,'critical_aabb_gaps':gaps,
      'shaft_bounds':[bounds(x) for x in shafts],'supports_volumes_mm3':[vv(x) for x in supports],'belt_volumes_mm3':[vv(x) for x in belts],
      'motor_pair_overlap_mm3':overlaps(motors[0],motors[1]),'pack_shoulder_clearance':pack_shoulders,'exact_motor_clearances_500_wall1':exact_gaps,
      'mass_kg_known_pair_pack_and_four_bearings':round((154+107+4*1.9)/1000,4)}

def main():
    rows=[]
    for h in (350,500):
      for wall in (1,2):
        for layout in ('coaxial','remote'):
          rows.append(candidate(h,wall,layout))
          print(h,wall,layout,'wall hits',len(rows[-1]['shell_intersections']),'outside',len(rows[-1]['outside_combined_lobes']),'hard pairs',len(rows[-1]['critical_case_intersections']),flush=True)
    (OUT/'report.json').write_text(json.dumps({'source_frame':'X lateral, Y rear, Z up','rows':rows},indent=2))
    (OUT/'scale-summary.csv').write_text('height_mm,wall_mm,layout,wall_hit_count,package_outside_count,critical_overlap_count,max_wall_overlap_mm3,motor_pair_overlap_mm3\n'+''.join(
      f"{r['height_mm']},{r['wall_mm']},{r['layout']},{len(r['shell_intersections'])},{len(r['outside_combined_lobes'])},{len(r['critical_case_intersections'])},{max([x['mm3'] for x in r['shell_intersections']] or [0])},{r['motor_pair_overlap_mm3']}\n" for r in rows))
if __name__=='__main__':main()
