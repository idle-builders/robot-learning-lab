"""Fast exact-source falsifier for a distributed three-drive shoulder.

One source-sized XL330 actuates a torso-supported yaw ring by a long belt;
one XL330 sits at the medial upper-arm roll root; one ST3250 sits in the upper
chest and drives a supported pitch shaft across to the upper-arm pivot. This
file stops at static packaging if cases/outputs/services clash; it is not a
load-rated assembly or actuator-duty claim.
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
OUT=ROOT/'artifacts/swarm-round3-shoulder-split-350'
XL_STEP=ROOT/'artifacts/round2-hip-350/vendor-cad/XL_XC330_official.stp'
PIVOT_X=66.0
PACK_CENTER=(0.0,0.0,200.0); PACK_SIZE=(60.0,30.0,30.5)
SERVICE=[('buck',(42,0,200),(16,18,8)),('can',(42,0,210),(16,18,8)),('connectors',(42,0,219),(16,14,6))]
ST_SIZE=(45.22,35.0,24.72)

def vol(s): return round(sum(x.volume for x in s.solids()),3) if s else 0.0
def box(c,size): return b.Pos(*c)*b.Box(*size)
def axial(axis,r,h,p):
    rt={'x':b.Rot(0,90,0),'y':b.Rot(90,0,0),'z':b.Rot(0,0,0)}[axis]
    return b.Pos(*p)*rt*b.Cylinder(r,h)
def ring(axis,p,sep=3.5,od=13, bore=6):
    return axial(axis,od/2,sep,p)-axial(axis,bore/2,sep+.2,p)
def place_xl(shape,axis,p,spin_z=0):
    # Manufacturer STEP datum inspected in the shared hip study: output axis at
    # local (0,0,6.5). Keep actual 15-solid package and boss, not an ideal box.
    axis_rot={'z':b.Rot(0,0,spin_z),'y':b.Rot(-90,0,spin_z)}[axis]
    return b.Pos(*p)*axis_rot*b.Pos(0,0,-6.5)*shape
def fast_intersection(a,bb):
    x=a.bounding_box();y=bb.bounding_box()
    xm=[x.min.X,x.min.Y,x.min.Z];xM=[x.max.X,x.max.Y,x.max.Z]
    ym=[y.min.X,y.min.Y,y.min.Z];yM=[y.max.X,y.max.Y,y.max.Z]
    if any(xM[i]<ym[i] or yM[i]<xm[i] for i in range(3)):return 0.0
    return vol(a.intersect(bb))
def add(scene,n,s,c):
    vv,ff=s.tessellate(.35,.35)
    m=trimesh.Trimesh(np.array([[v.X,v.Y,v.Z] for v in vv]),np.asarray(ff),process=False)
    m.visual.vertex_colors=c;scene.add_geometry(m,geom_name=n,node_name=n)

def build(wall,layout):
    study=Study(height=350,wall=wall,gap=.65);study.build()
    parts={p['name']:p['shape'] for p in study.parts}
    torso=parts['torso_front'].fuse(parts['torso_back'])
    upper=parts['R_upper_arm_front'].fuse(parts['R_upper_arm_back'])
    xl=b.import_step(str(XL_STEP))
    yaw_x=-24.0
    # The one bounded alternate moves the entire shoulder axis rearward 7.2 mm.
    # It preserves the ST3250's documented 12.2 mm axis offset and does not add
    # an external cowl or enlarge the source shoulder trim.
    pivot_y={'primary':5.0,'rear_axis_alt':12.2}[layout]
    pivot=(PIVOT_X,pivot_y,246.0)
    # Yaw input output is vertical and rotated in its own plane so the case's
    # 34 mm length consumes torso lateral width rather than rear depth.
    yaw=place_xl(xl,'z',(yaw_x,pivot_y,246.0),90)
    # Roll motor's +Z source output datum is mapped to +Y. Both roll shafts are
    # parallel and 20 mm apart in X; this is a 1:1 belt/sector envelope.
    roll=place_xl(xl,'y',(86.0,pivot_y,246.0))
    # Feetech output is along local Y, offset 12.2 mm from case centre in local
    # X. Rz(+90) maps shaft to source lateral +X and offset to rearward +Y.
    # Case center sits 12.2 mm medial to its output along the mapped Y axis.
    st_center=(25.0,pivot_y-12.2,246.0)
    st_case=b.Pos(*st_center)*b.Box(35.0,45.22,24.72)
    pitch_output=(42.5,pivot_y,246.0)
    pitch_shaft=b.Pos(59.25,pivot_y,246)*b.Rot(0,90,0)*b.Cylinder(3,33.5)
    pitch_supports=[ring('x',(58,pivot_y,246)),ring('x',(70,pivot_y,246))]
    yaw_output=axial('z',3,36,pivot)
    yaw_supports=[ring('z',(66,pivot_y,237)),ring('z',(66,pivot_y,255))]
    yaw_input_shaft=axial('z',3,25,(yaw_x,pivot_y,253.5))
    roll_output=axial('y',3,32,pivot)
    roll_supports=[ring('y',(66,pivot_y-11,246)),ring('y',(66,pivot_y+11,246))]
    # Three carrier webs close reactions into the chest spine/upper-arm root.
    fixed_frame=box((62,pivot_y,246),(5,20,30)).fuse(box((52,pivot_y+2,262),(34,5,5)))
    moving_frame=box((76,pivot_y,246),(20,5,24)).fuse(box((87,pivot_y,239),(14,5,8)))
    # Conservative service envelopes copied from hip team's current reserve.
    battery=box(PACK_CENTER,PACK_SIZE)
    services=[(n,box(c,s)) for n,c,s in SERVICE]
    # Belt routes are keep-out boxes at the stated planes; tooth path is absent.
    yaw_belt=box((21,pivot_y,266),(90,6,6))
    roll_belt=box((76,pivot_y+3,246),(24,6,6))
    objects=[('yaw_XL330',yaw),('roll_XL330',roll),('pitch_ST3250',st_case),
      ('yaw_supported_output',yaw_output),('yaw_case_output_extension',yaw_input_shaft),
      ('roll_supported_output',roll_output),('pitch_cross_shaft',pitch_shaft),
      ('yaw_bearing_lower',yaw_supports[0]),('yaw_bearing_upper',yaw_supports[1]),
      ('roll_bearing_inner',roll_supports[0]),('roll_bearing_outer',roll_supports[1]),
      ('pitch_bearing_inboard',pitch_supports[0]),('pitch_bearing_outboard',pitch_supports[1]),
      ('fixed_chest_frame',fixed_frame),('moving_arm_root_frame',moving_frame),
      ('yaw_belt_reserve',yaw_belt),('roll_belt_reserve',roll_belt),('4S_pack',battery),*services]
    source=[('torso',torso),('upper_arm',upper)]
    skin_hits=[]
    for n,o in objects:
      if n in ('4S_pack','buck','can','connectors'): regions=source[:1]
      elif n.startswith('roll_') or n=='moving_arm_root_frame':regions=source[1:]
      else: regions=source
      for sn,s in regions:
        v=fast_intersection(o,s)
        if v>.01:skin_hits.append({'object':n,'source':sn,'mm3':v})
    pair_hits=[]
    allow_touch={frozenset(('pitch_cross_shaft','pitch_ST3250')),
       frozenset(('yaw_case_output_extension','yaw_XL330')),
       frozenset(('yaw_case_output_extension','yaw_belt_reserve')),
       frozenset(('yaw_supported_output','yaw_belt_reserve')),
       frozenset(('roll_supported_output','roll_belt_reserve'))}
    for i,(na,a) in enumerate(objects):
      for nb,bb in objects[i+1:]:
        if frozenset((na,nb)) in allow_touch:continue
        v=fast_intersection(a,bb)
        if v>.01:pair_hits.append({'a':na,'b':nb,'mm3':v})
    battery_hit=fast_intersection(battery,torso)
    service_hit={n:fast_intersection(o,torso) for n,o in services}
    report={'layout':layout,'wall_mm':wall,'source':'Study(height=350,wall=1/2,gap=.65) exact BRep from sketch_ape_shell/build.py',
      'axes':{'yaw_Z':pivot,'roll_Y':pivot,'pitch_X':pivot},
      'actuators':{'yaw':{'source':'ROBOTIS XL330-M288-T official STEP','case_bbox_mm':[20,34,29],'mass_g':18,'rated_working_torque':'unpublished; 0.52 N m stall only','output_datum_local_mm':[0,0,6.5],'output_axis':'+Z','case_output_point_mm':[yaw_x,pivot_y,246]},
       'roll':{'source':'ROBOTIS XL330-M288-T official STEP','case_bbox_mm':[20,34,29],'mass_g':18,'rated_working_torque':'unpublished; 0.52 N m stall only','output_axis':'+Y','case_output_point_mm':[86,pivot_y,246]},
       'pitch':{'source':'Feetech ST3250-C001 case envelope','case_mm':[35,45.22,24.72],'mass_g':74.5,'rated_torque_Nm_at_12V':1.569,'output_offset_local_mm':12.2,'output_axis':'+X','case_center_mm':st_center,'output_start_mm':pitch_output}},
      'transmission':{'yaw':'1:1 belt reserve, motor-to-axis centre distance mm', 'yaw_center_distance_mm':round(66-yaw_x,2),'roll':'1:1 belt reserve','roll_center_distance_mm':20.62,'pitch':'direct cross-shaft, 23.5 mm from output datum to shoulder centre'},
      'shared_pack':{'center_mm':PACK_CENTER,'size_mm':PACK_SIZE,'torso_intersection_mm3':battery_hit},
      'hip_service':{n:{'center_mm':c,'size_mm':s,'torso_intersection_mm3':service_hit[n]} for n,c,s in SERVICE},
      'exact_case_output_service_skin_hits':skin_hits,'exact_nonmating_component_overlaps':pair_hits,
      'verdict':'pass_static_candidate' if not skin_hits and not pair_hits and battery_hit==0 and all(v==0 for v in service_hit.values()) else 'reject_at_static_package_gate',
      'limits':['ST3250 box placement and 12.2 mm output offset use prior source drawing evidence; no native vendor STEP was available.','XL330 is the manufacturer STEP; cable exits/connectors are not included.','Belt volumes are keep-out envelopes, not tooth geometry or a verified closed path.','Support rings and frame webs are CAD reaction-path proxies, not bearing-life or stiffness qualification.','No moving pose screen is run when the static package gate fails.']}
    return report,source,objects

def main():
    OUT.mkdir(parents=True,exist_ok=True); reports=[]
    for wall in (1,2):
      for layout in ('primary','rear_axis_alt'):
        report,source,objects=build(wall,layout);reports.append(report)
        print(wall,layout,report['verdict'],'skin',report['exact_case_output_service_skin_hits'],'pairs',report['exact_nonmating_component_overlaps'],flush=True)
        if wall==2:
          scene=trimesh.Scene()
          for n,s in source:add(scene,'source_'+n,s,(226,219,203,190))
          colors={'yaw_XL330':(58,108,157,255),'roll_XL330':(76,130,168,255),'pitch_ST3250':(93,110,132,255),'4S_pack':(207,146,59,120),'buck':(173,106,80,160),'can':(173,106,80,160),'connectors':(173,106,80,160)}
          for n,s in objects:add(scene,n,s,colors.get(n,(171,132,61,190)))
          scene.export(OUT/f'shoulder-{layout}-static.glb')
    (OUT/'report.json').write_text(json.dumps(reports,indent=2)+'\n')
    (OUT/'index.html').write_text(Path(__file__).with_name('round3_shoulder_split_viewer.html').read_text())
if __name__=='__main__':main()
