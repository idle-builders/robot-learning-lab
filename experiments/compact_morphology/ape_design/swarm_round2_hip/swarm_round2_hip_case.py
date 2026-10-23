"""One-right-leg source-shell packaging gate. Geometry is exact BRep; ratings remain vendor-limited."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import build123d as b
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'experiments/sketch_ape_shell'))
from build import Study
OUT=ROOT/'artifacts/round2-hip-350'
STEP=OUT/'vendor-cad/XL_XC330_official.stp'
PACK=(0,0,200); PACK_SIZE=(60,30,30.5)
HIP=(32,4,160); KNEE=(42,4,86); ANKLE=(46,3,24)
YAW_Z=160.0
OUTER_CACHE={}

def vol(s): return round(sum(x.volume for x in s.solids()),3) if s else 0.0
def box(c,sz): return b.Pos(*c)*b.Box(*sz)
def axis_cyl(axis,r,h,p):
    rot={'x':b.Rot(0,90,0),'y':b.Rot(90,0,0),'z':b.Rot(0,0,0)}[axis]
    return b.Pos(*p)*rot*b.Cylinder(r,h)
def turn(shape,axis,deg,p):
    r={'x':b.Rot(deg,0,0),'y':b.Rot(0,deg,0),'z':b.Rot(0,0,deg)}[axis]
    return b.Pos(*p)*r*b.Pos(*(-v for v in p))*shape
def rotpt(p,axis,deg,o):
    q=np.array(p,float)-np.array(o,float); t=math.radians(deg); c,s=math.cos(t),math.sin(t)
    R={'x':np.array([[1,0,0],[0,c,-s],[0,s,c]]),'y':np.array([[c,0,s],[0,1,0],[-s,0,c]]),'z':np.array([[c,-s,0],[s,c,0],[0,0,1]])}[axis]
    return tuple(R@q+np.array(o,float))
def exact_step():
    sh=b.import_step(str(STEP))
    # Source datum is output-axis centered, with the output hub face at local z=6.5.
    return sh
def placed_motor(shape,axis,point):
    # Local CAD output datum is (0,0,6.5); place that datum at the supported joint.
    r={'x':b.Rot(0,90,0),'y':b.Rot(-90,0,0),'z':b.Rot(0,0,0)}[axis]
    return b.Pos(*point)*r*b.Pos(0,0,-6.5)*shape
def belt_envelope(axis,p1,p2,pulley_d=10.0,belt_w=1.2,axial_w=4.0):
    """Closed, dimensioned 1:1 belt envelope with pulley centerline datums."""
    v=np.array(p2,float)-np.array(p1,float)
    n=np.array({'x':(1,0,0),'y':(0,1,0),'z':(0,0,1)}[axis],float)
    v=v-np.dot(v,n)*n; center=float(np.linalg.norm(v)); u=v/center
    mid=tuple((np.array(p1,float)+np.array(p2,float))/2)
    plane=b.Plane(origin=mid,x_dir=tuple(u),z_dir=tuple(n))
    outer=b.extrude(b.SlotCenterToCenter(center,pulley_d+2*belt_w),amount=axial_w/2,both=True)
    inner=b.extrude(b.SlotCenterToCenter(center,pulley_d),amount=axial_w/2+.1,both=True)
    return plane.location*(outer-inner)
def shell_outer_regions(wall):
    s=Study(350,wall,.65)
    torso=s.hollow_loft([(160,0,2,42,25),(173,0,2,49,29),(199,0,3,58,34),(229,0,4,63,35),(253,0,5,63,30),(264,0,5,60,27),(274,0,5,35,22),(278,0,5,23,18)])[1]
    pelvis=s.hollow_loft([(130,0,5,9,11),(135,0,5,20,17),(145,0,5,34,23),(155,0,4,43,26),(164,0,3,46,26),(178,0,3,49,29)])[1]
    thigh=s.hollow_loft([(87,42,3,16,18),(98,43,3,20,22),(124,39,3,24,25),(145,35,4,22,23),(157,33,4,16,18),(163,31,4,8,10)])[1]
    shin=s.hollow_loft([(23,46,3,21,20),(36,46,3,23,23),(57,45,4,22,23),(75,43,4,19,20),(83,42,3,16,17)])[1]
    boot=s.hollow_loft([(2,46,-8,30,34),(5,46,-8,31,35),(12,46,-7,29,33),(20,46,-1,24,25),(26,46,3,19,19)])[1]
    return {'torso':torso,'pelvis':pelvis,'thigh':thigh,'shin':shin,'boot':boot}
def pulley(axis,point,od=10.0,width=4.0,bore=5.0):
    return axis_cyl(axis,od/2,width,point)-axis_cyl(axis,bore/2,width+.4,point)

def build(wall,layout):
    study=Study(350,wall,.65);study.build(); p={x['name']:x['shape'] for x in study.parts}
    torso=p['torso_front'].fuse(p['torso_back']); pelvis=p['pelvis_front'].fuse(p['pelvis_back'])
    thigh=p['R_thigh_front'].fuse(p['R_thigh_back']); shin=p['R_shin_front'].fuse(p['R_shin_back'])
    boot=p['R_foot_dorsal'].fuse(p['R_toe_1']).fuse(p['R_toe_2']).fuse(p['R_toe_3']).fuse(p['R_sole'])
    # RSF-5B-50-IDT envelope from vendor outline drawing, selected rear-cable PT1.
    # Manufacturer rated package: Ø25 x 35.8 axial; hollow output details and cable model below.
    yaw=axis_cyl('z',12.5,35.8,(0,0,YAW_Z))
    # Explicit 12x8mm rear connector pocket/exit with 15mm cable bend sweep reservation.
    rsf_exit=box((0,14.5,YAW_Z),(10,8,12))
    # Conservative Ø6 PT1 internal lead bend; the external feedthrough is still a shell-design item.
    cable_edge=b.Edge.make_spline([(0,18,YAW_Z),(0,21,YAW_Z),(5,22,YAW_Z-2),(12,20,YAW_Z-5)])
    cable=b.Solid.sweep(b.Wire.make_circle(3,b.Plane(origin=(0,18,YAW_Z),z_dir=(0,1,0))),cable_edge)
    # XL330 source STEP actual 15-solid manufacturer package. Build orientation envelopes from exact bounding box;
    # mounted output points use vendor STEP origin convention documented by inspection.
    xl=exact_step()
    # Output-supported joints: two SKF W618/6 rings at 6mm shaft, 20mm apart; box yokes connect respective links.
    def support(axis,pt,sep=14):
        rings=[]
        for d in (-sep/2,sep/2):
            q=list(pt);q['xyz'.index(axis)]+=d
            rings.append(axis_cyl(axis,6.5,3.5,q)-axis_cyl(axis,3,4,q))
        shaft=axis_cyl(axis,3,sep+8,pt)
        return rings,shaft
    supports=[]; shafts=[]
    # Orthogonal hip trunnion pairs are separated along their own axes; knee/ankle use compact paired seats.
    for ax,pt,sep in [('z',HIP,18),('y',HIP,18),('x',HIP,18),('x',KNEE,14),('x',ANKLE,14)]:
        rr,ss=support(ax,pt,sep); supports+=rr;shafts.append(ss)
    # Narrow carrier spine follows the hollow link centerlines; this replaces the shell-cutting bulky block.
    carrier=box((32,4,160),(8,8,8)).fuse(box((37,4,123),(5,6,66)))
    knee_carrier=box((42,4,86),(8,8,8)).fuse(box((44,4,55),(5,6,52)))
    ankle_carrier=box((46,3,24),(8,8,8))
    frame=carrier.fuse(knee_carrier).fuse(ankle_carrier)
    for x in supports: frame=frame.fuse(x)
    frame=frame.fuse(*shafts)
    # XL330 bodies represented using exact STEP transformed to package-size envelope placement; axial hub faces are stated.
    # Mount axes: 20x34x26 source geometry transformed to measured motor bounding box and shown in GLB.
    placements={
      'local':[(49,4,160,'y'),(32,4,160,'x'),(42,4,108,'x'),(46,3,48,'x')],
      'remote':[(49,4,160,'y'),(20,4,160,'x'),(42,4,108,'x'),(46,3,48,'x')]
    }[layout]
    # source STEP imported shape is normalized to its actual bbox, reused per XL package.
    motors=[placed_motor(xl,item[3],item[:3]) for item in placements]
    # Belt/coupling transmission: each sourced output hub has a pulley/shaft interface to its supported output.
    yaw_plane=YAW_Z-34.0
    yaw_pulleys=[pulley('z',(0,0,yaw_plane)),pulley('z',(HIP[0],HIP[1],yaw_plane))]
    yaw_belt=belt_envelope('z',(0,0,yaw_plane),(HIP[0],HIP[1],yaw_plane))
    roll_plane_y=-12.0
    roll_mpoint=(placements[0][0],roll_plane_y,placements[0][2]); roll_opoint=(HIP[0],roll_plane_y,HIP[2])
    roll_pulleys=[pulley('y',roll_mpoint),pulley('y',roll_opoint)]
    roll_belt=belt_envelope('y',roll_mpoint,roll_opoint)
    roll_motor_ext=axis_cyl('y',2.5,16,(placements[0][0],-4,placements[0][2]))
    roll_output_ext=axis_cyl('y',2.5,16,(HIP[0],-4,HIP[2]))
    pitch_drive_x=placements[1][0]
    pitch_shaft=axis_cyl('x',2.5,44-pitch_drive_x,((pitch_drive_x+44)/2,4,160))
    knee_plane_x=50.0
    knee_mpoint=(knee_plane_x,KNEE[1],placements[2][2]);knee_opoint=(knee_plane_x,KNEE[1],KNEE[2])
    knee_pulleys=[pulley('x',knee_mpoint),pulley('x',knee_opoint)]
    knee_belt=belt_envelope('x',knee_mpoint,knee_opoint)
    knee_motor_ext=axis_cyl('x',2.5,8,(46,KNEE[1],placements[2][2]))
    knee_output_ext=axis_cyl('x',2.5,8,(46,KNEE[1],KNEE[2]))
    ankle_plane_x=60.0
    ankle_mpoint=(ankle_plane_x,ANKLE[1],placements[3][2]);ankle_opoint=(ankle_plane_x,ANKLE[1],ANKLE[2])
    ankle_pulleys=[pulley('x',ankle_mpoint),pulley('x',ankle_opoint)]
    ankle_belt=belt_envelope('x',ankle_mpoint,ankle_opoint)
    ankle_motor_ext=axis_cyl('x',2.5,14,(53,ANKLE[1],placements[3][2]))
    ankle_output_ext=axis_cyl('x',2.5,14,(53,ANKLE[1],ANKLE[2]))
    yaw_input_shaft=axis_cyl('z',2.5,18,(0,0,yaw_plane+8))
    yaw_shaft=axis_cyl('z',2.5,46,(32,4,145))
    trains=[('yaw transfer',[('yaw motor pulley',yaw_pulleys[0]),('yaw output pulley',yaw_pulleys[1]),('yaw belt',yaw_belt),('yaw input shaft',yaw_input_shaft),('yaw supported output shaft',yaw_shaft)]),
      ('roll transfer',[('roll motor pulley',roll_pulleys[0]),('roll output pulley',roll_pulleys[1]),('roll belt',roll_belt),('roll motor shaft extension',roll_motor_ext),('roll output shaft extension',roll_output_ext)]),
      ('pitch transfer',[('pitch shaft coupling',pitch_shaft)]),
      ('knee transfer',[('knee motor pulley',knee_pulleys[0]),('knee output pulley',knee_pulleys[1]),('knee belt',knee_belt),('knee motor shaft extension',knee_motor_ext),('knee output shaft extension',knee_output_ext)]),
      ('ankle transfer',[('ankle motor pulley',ankle_pulleys[0]),('ankle output pulley',ankle_pulleys[1]),('ankle belt',ankle_belt),('ankle motor shaft extension',ankle_motor_ext),('ankle output shaft extension',ankle_output_ext)])]
    battery=box(PACK,PACK_SIZE)
    # Separate service volumes in a lateral bay, below the shoulder team's Z≈220 start.
    buck=box((42,0,200),(16,18,8));can=box((42,0,210),(16,18,8));connect=box((42,0,219),(16,14,6))
    service=buck.fuse(can).fuse(connect)
    fixed=[('torso',torso),('pelvis',pelvis),('thigh',thigh),('shin',shin),('boot',boot)]
    components=[('RSF yaw body',yaw),*[(v,m) for v,m in zip(['hip roll XL330','hip pitch XL330','knee XL330','ankle XL330'],motors)],('supported yokes and SKF rings',frame),*[(n,s) for _,ss in trains for n,s in ss],('4S battery',battery),('5V buck reserve',buck),('CAN/TTL reserve',can),('connectors',connect),('RSF PT1 cable exit',rsf_exit),('RSF lead bend',cable)]
    candidate_shells={
      'RSF yaw body':['pelvis'],'RSF PT1 cable exit':['pelvis'],'RSF lead bend':['pelvis'],
      'hip roll XL330':['pelvis','thigh'],'hip pitch XL330':['pelvis','thigh'],
      'supported yokes and SKF rings':['pelvis','thigh','shin'],
      '4S battery':['torso'],'5V buck reserve':['torso'],'CAN/TTL reserve':['torso'],'connectors':['torso']}
    for label in ('yaw','roll','pitch'):
      for n,_ in next(ss for key,ss in trains if key.startswith(label)):
        candidate_shells[n]=['pelvis','thigh']
    for label in ('knee',):
      for n,_ in next(ss for key,ss in trains if key.startswith(label)):
        candidate_shells[n]=['thigh','shin']
    for label in ('ankle',):
      for n,_ in next(ss for key,ss in trains if key.startswith(label)):
        candidate_shells[n]=['shin','boot']
    intersections=[]
    for name,s in components:
      for sn in candidate_shells.get(name,[]):
        skin=dict(fixed)[sn]
        a=s.bounding_box();bb=skin.bounding_box(); amin=list(a.min);amax=list(a.max);bmin=list(bb.min);bmax=list(bb.max)
        if all(amin[i]<=bmax[i] and bmin[i]<=amax[i] for i in range(3)):
          v=vol(s.intersect(skin))
          if v>.01: intersections.append({'a':name,'b':'source '+sn,'mm3':v})
    print(f'wall={wall} layout={layout} source shell pairs done ({len(intersections)} hits)',flush=True)
    # Assembly self-pairs excluding deliberate touching support/motor packages and vendor mating output pieces.
    selfhits=[]; mating_hits=[]
    def intended_mating(a,b):
        pair={a,b}
        if 'supported yokes and SKF rings' in pair and any(('shaft' in n or 'pulley' in n) for n in pair-{'supported yokes and SKF rings'}): return True
        for drive in ('yaw','roll','knee','ankle'):
            related={'RSF yaw body':'yaw','hip roll XL330':'roll','hip pitch XL330':'pitch','knee XL330':'knee','ankle XL330':'ankle'}.get(a)
            related_b={'RSF yaw body':'yaw','hip roll XL330':'roll','hip pitch XL330':'pitch','knee XL330':'knee','ankle XL330':'ankle'}.get(b)
            if related==drive and any(drive in n and ('pulley' in n or 'shaft' in n) for n in pair-{a}): return True
            if related_b==drive and any(drive in n and ('pulley' in n or 'shaft' in n) for n in pair-{b}): return True
            belt=f'{drive} belt'
            if belt in pair and any(drive in n and 'pulley' in n for n in pair-{belt}): return True
            if any(drive in n and 'shaft' in n for n in pair) and any(drive in n and 'pulley' in n for n in pair): return True
        return False
    for i,(na,a) in enumerate(components):
      for nb,bb in components[i+1:]:
        # Keep the exact pair screen bounded to plausible non-mating contacts:
        # motor packages, belts, cross-axis hubs, and service envelopes.
        motorset={'RSF yaw body','hip roll XL330','hip pitch XL330','knee XL330','ankle XL330'}
        critical=(na in motorset and nb in motorset) or ('belt' in na and 'belt' in nb) or \
          (na in motorset and (('belt' in nb and not intended_mating(na,nb)) or nb in {'4S battery','5V buck reserve','CAN/TTL reserve','connectors','RSF PT1 cable exit','RSF lead bend'})) or \
          (nb in motorset and na in {'4S battery','5V buck reserve','CAN/TTL reserve','connectors','RSF PT1 cable exit','RSF lead bend'}) or \
          (('pulley' in na and 'pulley' in nb) and na.split()[0]!=nb.split()[0]) or \
          (('supported yokes' in na or 'supported yokes' in nb) and ('belt' in na or 'belt' in nb or 'connector' in na or 'connector' in nb)) or \
          (na in {'4S battery','5V buck reserve','CAN/TTL reserve','connectors'} and nb in {'4S battery','5V buck reserve','CAN/TTL reserve','connectors'})
        if not critical: continue
        if na in ('RSF yaw body','RSF PT1 cable exit','RSF lead bend') and nb in ('RSF yaw body','RSF PT1 cable exit','RSF lead bend'): continue
        aa=a.bounding_box();cc=bb.bounding_box(); amin=list(aa.min);amax=list(aa.max);cmin=list(cc.min);cmax=list(cc.max)
        if all(amin[k]<=cmax[k] and cmin[k]<=amax[k] for k in range(3)):
          v=vol(a.intersect(bb))
          if v>.01:
            row={'a':na,'b':nb,'mm3':v}
            (mating_hits if intended_mating(na,nb) else selfhits).append(row)
    print(f'wall={wall} layout={layout} selected component pairs done ({len(selfhits)} non-mating)',flush=True)
    if wall not in OUTER_CACHE: OUTER_CACHE[wall]=shell_outer_regions(wall)
    outer=OUTER_CACHE[wall]
    protrusions=[]
    references={
      'RSF yaw body':['torso','pelvis'],'hip roll XL330':['torso','pelvis','thigh'],'hip pitch XL330':['torso','pelvis','thigh'],
      'knee XL330':['thigh','shin'],'ankle XL330':['shin','boot'],'supported yokes and SKF rings':['pelvis','thigh','shin'],
      'RSF PT1 cable exit':['torso','pelvis'],'RSF lead bend':['torso','pelvis']}
    component_map=dict(components)
    for name in references:
      shape=component_map[name];cb=shape.bounding_box();cmin=list(cb.min);cmax=list(cb.max)
      measures=[]
      for ref in references[name]:
        env=outer[ref];eb=env.bounding_box();emin=list(eb.min);emax=list(eb.max)
        if not all(cmin[k]<=emax[k] and emin[k]<=cmax[k] for k in range(3)):
            outside=vol(shape)
        else:
            outside=max(0.0,round(vol(shape)-vol(shape.intersect(env)),3))
        measures.append({'reference_outer_lobe':ref,'outside_volume_mm3':outside})
      protrusions.append({'component':name,'outside_source_outer_envelopes':measures})
    print(f'wall={wall} layout={layout} selected outer-lobe audits done',flush=True)
    batteryhit=vol(battery.intersect(torso)); servicehit=vol(service.intersect(torso))
    return dict(wall=wall,layout=layout,p=p,parts=(torso,pelvis,thigh,shin,boot),components=components,intersections=intersections,selfhits=selfhits,mating_hits=mating_hits,protrusions=protrusions,motor_datums=placements,
      batteryhit=batteryhit,servicehit=servicehit,hip=HIP,knee=KNEE,ankle=ANKLE,motors=motors,frame=frame,supports=supports,shafts=shafts,
      assets={'xl_step':str(STEP),'rsf_exit':rsf_exit,'cable':cable,'battery':battery,'buck':buck,'can':can,'connect':connect},
      source_shell_mass_g=round(sum(x.volume for x in [thigh,shin,boot])*.00124,1))
def pose_screen(result):
    torso,pelvis,thigh,shin,boot=result['parts']; out=[]
    for name,hip_pitch,knee,ankle in [('stand',0,0,0),('leg_lift',-25,55,-30),('squat',-45,100,-20),('floor_reach',-54,109,-20)]:
      # hip pitch rotates thigh and all distal parts, knee then ankle are relative link pivots.
      t=turn(thigh,'x',hip_pitch,HIP); s=turn(turn(shin,'x',hip_pitch,HIP),'x',knee,rotpt(KNEE,'x',hip_pitch,HIP))
      btm0=turn(boot,'x',hip_pitch,HIP); btm=turn(turn(btm0,'x',knee,rotpt(KNEE,'x',hip_pitch,HIP)),'x',ankle,rotpt(ANKLE,'x',hip_pitch+ knee,HIP))
      def inter(x,y): return vol(x.intersect(y))
      out.append({'pose':name,'angles_deg':{'hip_pitch':hip_pitch,'knee':knee,'ankle':ankle},'thigh_torso_mm3':inter(t,torso),'shin_torso_mm3':inter(s,torso),'boot_torso_mm3':inter(btm,torso),'thigh_shin_mm3':inter(t,s),'shin_boot_mm3':inter(s,btm)})
    return out
def add_mesh(scene,name,shape,color):
    if shape is None:return
    vv,ff=shape.tessellate(.35,.35)
    m=trimesh.Trimesh(np.array([[v.X,v.Y,v.Z] for v in vv]),np.asarray(ff),process=False)
    m.visual.vertex_colors=color;scene.add_geometry(m,geom_name=name,node_name=name)
def export_inspector(r):
    torso,pelvis,thigh,shin,boot=r['parts'];scene=trimesh.Scene()
    for n,s in [('torso',torso),('pelvis',pelvis),('R_thigh',thigh),('R_shin',shin),('R_full_boot',boot)]:
      add_mesh(scene,n,s,(225,218,202,170))
    for name,s in r['components']:
      col=(70,111,151,255)
      if 'battery' in name:col=(211,147,58,210)
      if 'buck' in name or 'CAN' in name or 'connector' in name:col=(169,110,87,200)
      if 'yokes' in name:col=(177,129,53,240)
      add_mesh(scene,name,s,col)
    scene.export(OUT/'hip-remote-stand.glb')

def main():
  ap=argparse.ArgumentParser();ap.add_argument('--layout',choices=['local','remote'],action='append');ap.add_argument('--wall',type=int,choices=[1,2],action='append');args=ap.parse_args()
  layouts=args.layout or ['local','remote'];walls=args.wall or [1,2]
  OUT.mkdir(parents=True,exist_ok=True); reports=[]
  for wall in walls:
   for layout in layouts:
    r=build(wall,layout); print(json.dumps({'shell':r['intersections'],'protrusions':r['protrusions'],'nonmating':r['selfhits'],'mating':r['mating_hits']},default=str),flush=True)
    report={'wall_mm':wall,'layout':layout,'source_shell_mass_g':r['source_shell_mass_g'],'source_component_intersections':r['intersections'],'outside_source_outer_envelopes':r['protrusions'],'assembly_nonmating_intersections':r['selfhits'],'intentional_mating_region_intersections':r['mating_hits'],'battery_torso_mm3':r['batteryhit'],'power_service_torso_mm3':r['servicehit'],'poses':pose_screen(r) if wall==2 else None,
      'motor_output_datums_mm':r['motor_datums'],'pack':{'center_mm':PACK,'size_mm':PACK_SIZE,'mass_g':107,'nominal_v':14.8},'motor_specs':{'RSF-5B-50-IDT':{'mass_g':77,'ratio':50,'continuous_torque_Nm':0.29,'max_torque_Nm':0.9,'continuous_speed_rpm':90,'max_speed_rpm':200,'bus':'CANopen integrated','supply':'7-30V; 24V nominal'},'XL330-M288-T':{'mass_g':18,'stall_torque_Nm_at_5V':0.52,'continuous_torque':'unpublished','operating_voltage_V':[3.7,6.0],'bus':'TTL; requires regulated 5V for this package'}},'axis_screen':{'hip_yaw':'1:1 belt; <=0.29 Nm published allowable continuous before belt losses; duty demand uncomputed','hip_roll':'1:1 belt; 0.52 Nm stall only and continuous unpublished; a prior 0.235 Nm gravity-only illustrative demand belongs to this roll axis, not yaw','hip_pitch':'direct shaft extension, 1:1; XL330 continuous torque and required dynamic torque unresolved','knee':'1:1 belt; XL330 continuous torque and required dynamic torque unresolved','ankle':'1:1 belt; XL330 continuous torque and required dynamic torque unresolved'},'mass_g':{'motors':149,'pack':107,'bearings':19.0,'frame_allowance':34,'transmission_allowance':32,'regulator_bus_connectors_reserve':22,'total':363.0},'mass_centres_mm':{'RSF output datum':(0,0,YAW_Z-17.9),'XL roll output datum':r['motor_datums'][0][:3],'XL pitch output datum':r['motor_datums'][1][:3],'XL knee output datum':r['motor_datums'][2][:3],'XL ankle output datum':r['motor_datums'][3][:3],'battery':PACK,'unresolved structure/transmissions':'mass center not assigned'},'center_of_mass_mm':'Assembly COM not solved; carrier/transmission/service materials and component CGs remain unspecified.'}
    reports.append(report)
    if wall==2 and layout=='remote':export_inspector(r)
    print(json.dumps({k:report[k] for k in ['wall_mm','layout','source_component_intersections','outside_source_outer_envelopes','assembly_nonmating_intersections','intentional_mating_region_intersections','battery_torso_mm3','power_service_torso_mm3','poses']},default=str),flush=True)
  (OUT/'report.json').write_text(json.dumps(reports,indent=2)+'\n')
  viewer=(Path(__file__).with_name('viewer.html').read_text()
          .replace('../../../../artifacts/round2-hip-350/','./'))
  (OUT/'index.html').write_text(viewer)
if __name__=='__main__':main()
