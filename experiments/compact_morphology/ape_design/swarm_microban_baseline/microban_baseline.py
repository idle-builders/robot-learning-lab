"""Compare Microban's documented 19-axis block layout to the 500 mm ape shell.

The upstream CC BY-NC-SA CAD stays in a separate checkout. This script reads
its URDF/MJCF/STEP scalar frames and bounding dimensions and makes only simple
box envelopes; it does not export or copy upstream geometry.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import build123d as b
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'experiments/sketch_ape_shell'))
from build import Study
OUT=ROOT/'artifacts/microban-baseline'; OUT.mkdir(parents=True,exist_ok=True)
MICROBAN_URL='https://github.com/Rhoban/microban'
MICROBAN_COMMIT='d0a128d180ecfce69eaaf31b295a0fce31280bfd'
XL_CASE=(20.0,34.0,29.0) # Conservative box from official XL330 CAD/output datum.
ANKLE_ROLL={'right_ankle_roll','left_ankle_roll'}
SOURCE_MAP=np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]]) # Microban XYZ -> ape X,-Y,Z
STEP_FILES={'hip_dbl_block__configuration_right':'hip_dbl_block_right.step',
            'hip_dbl_block__configuration_left':'hip_dbl_block_left.step',
            'ankle_dbl_block__configuration_right':'ankle_dbl_block.step',
            'ankle_dbl_block__configuration_left':'ankle_dbl_block.step',
            'arm_bearing_support__configuration_right':'arm_bearing_support.step'}

def xyz(s,default=(0,0,0)):
    return np.array([float(x) for x in s.split()]) if s else np.array(default,dtype=float)
def quat(s):
    q=xyz(s,(1,0,0,0))
    return Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
def tf(pos,rot):
    a=np.eye(4);a[:3,:3]=rot;a[:3,3]=pos;return a
def geom_tf(pos,rotation):return tf(xyz(pos),quat(rotation))
def target_point(p,scale,zmin):
    q=SOURCE_MAP@np.asarray(p,dtype=float)
    q[2]-=zmin
    return q*scale*1000
def target_rotation(r):return SOURCE_MAP@r@SOURCE_MAP.T
def plane_shape(center,rotation,dims):
    return b.Plane(origin=tuple(center),x_dir=tuple(rotation[:,0]),z_dir=tuple(rotation[:,2])).location*b.Box(*dims)
def aabb_overlap(a,c):
    A=a.bounding_box();C=c.bounding_box()
    amin=(A.min.X,A.min.Y,A.min.Z);amax=(A.max.X,A.max.Y,A.max.Z)
    cmin=(C.min.X,C.min.Y,C.min.Z);cmax=(C.max.X,C.max.Y,C.max.Z)
    return all(amin[i]<=cmax[i] and cmin[i]<=amax[i] for i in range(3))
def common_mm3(a,c):
    if not aabb_overlap(a,c):return 0.0
    q=a.intersect(c)
    return round(sum(s.volume for s in q.solids()),3) if q else 0.0
def bnd(s):
    bb=s.bounding_box();return [[round(v,3) for v in bb.min],[round(v,3) for v in bb.max]]

def source_geometry(mroot):
    step=mroot/'cad/microban.step'
    a=b.import_step(str(step));bb=a.bounding_box()
    ext=np.array([bb.max.X-bb.min.X,bb.max.Y-bb.min.Y,bb.max.Z-bb.min.Z])
    zmin=float(bb.min.Z);zmax=float(bb.max.Z)
    blocks={}
    for mesh,filename in STEP_FILES.items():
        p=mroot/'cad/step'/filename
        if not p.exists():
            if 'left' in mesh and filename=='hip_dbl_block_left.step':
                p=mroot/'cad/step/hip_dbl_block_right.step'
            elif 'left' in mesh and filename=='ankle_dbl_block.step':
                p=mroot/'cad/step/ankle_dbl_block.step'
        s=b.import_step(str(p));q=s.bounding_box()
        blocks[mesh]={'center_mm':[float((q.min.X+q.max.X)/2),float((q.min.Y+q.max.Y)/2),float((q.min.Z+q.max.Z)/2)],
          'size_mm':[float(q.max.X-q.min.X),float(q.max.Y-q.min.Y),float(q.max.Z-q.min.Z)]}
    return {'extent_mm':ext.tolist(),'z_min_mm':zmin,'z_max_mm':zmax,
            'scale_to_500':500/float(ext[2]),'blocks':blocks}

def read_mjcf(mroot):
    root=ET.parse(mroot/'src/model/mjcf/robot.xml').getroot()
    world=root.find('./worldbody'); bodies={}; joints={}; geoms=[]; masses=[]
    def walk(body,parent):
        here=parent@geom_tf(body.get('pos'),body.get('quat'))
        name=body.get('name');bodies[name]=here
        inert=body.find('inertial')
        if inert is not None:
            masses.append({'link':name,'kg':float(inert.get('mass')),
              'com_source_m':(here@np.r_[xyz(inert.get('pos')),1])[:3].tolist()})
        for j in list(body.findall('joint'))+list(body.findall('freejoint')):
            if j.get('name')=='trunk_freejoint':continue
            jp=here@np.r_[xyz(j.get('pos')),1]
            ja=here[:3,:3]@xyz(j.get('axis'),(0,0,1));joints[j.get('name')]={'point_m':jp[:3], 'axis':ja, 'body':name}
        for g in body.findall('geom'):
            if g.get('mesh'):
                geoms.append({'mesh':g.get('mesh'),'body':name,'T':here@geom_tf(g.get('pos'),g.get('quat'))})
        for child in body.findall('body'):walk(child,here)
    for body in world.findall('body'):
        walk(body,np.eye(4))
    return joints,geoms,masses

def axis_case(name,frame,scale,zmin,dims=XL_CASE):
    p=frame['point_m'];axis=frame['axis'];rot=frame['rotation']
    # Axis is mapped into ape coordinates. Output datum is at the joint; case
    # envelope extends 29 mm behind that face along the opposed axis.
    out=target_point(p,scale,zmin);a=SOURCE_MAP@axis;a=a/np.linalg.norm(a)
    x=rot[:,0];x=SOURCE_MAP@x
    x=x-a*np.dot(x,a)
    if np.linalg.norm(x)<1e-6:
        x=np.array([1.,0,0])-a*a[0]
    x=x/np.linalg.norm(x);y=np.cross(a,x);R=np.column_stack((x,y,a))
    center=out-a*dims[2]/2
    shape=plane_shape(center,R,dims)
    return shape,{'name':name,'kind':'XL330 envelope','origin_mm':out.tolist(),
      'axis':a.tolist(),'center_mm':center.tolist(),'rotation':R.tolist(),'size_mm':list(dims)}

def shell_outer(st):
    # Source station data from sketch-06, scaled by Study's 500/350 factor.
    k=500/350
    regions={}
    torso=[(160,0,2,42,25),(173,0,2,49,29),(199,0,3,58,34),(229,0,4,63,35),(253,0,5,63,30),(264,0,5,60,27),(274,0,5,35,22),(278,0,5,23,18)]
    pelvis=[(130,0,5,9,11),(135,0,5,20,17),(145,0,5,34,23),(155,0,4,43,26),(164,0,3,46,26),(178,0,3,49,29)]
    thigh=[(87,42,3,16,18),(98,43,3,20,22),(124,39,3,24,25),(145,35,4,22,23),(157,33,4,16,18),(163,31,4,8,10)]
    for n,rows in [('torso',torso),('pelvis',pelvis)]:
        regions[n]=st.hollow_loft([tuple(v*k for v in row) for row in rows])[1]
    thigh_regs=[]
    for side in (-1,1):
        s=st.hollow_loft([tuple(v*k for v in row) for row in thigh])[1]
        if side<0:s=s.mirror(b.Plane.YZ)
        thigh_regs.append(s)
    regions['thighs']=thigh_regs
    return regions

def block_envelopes(geoms,dimdata,scale,zmin):
    out=[];byname={x['mesh']:x for x in geoms}
    for mesh,bb in dimdata.items():
        sources=[x for x in geoms if x['mesh']==mesh]
        for ix,src in enumerate(sources):
            Rsrc=src['T'][:3,:3];R=target_rotation(Rsrc)
            local=np.array(bb['center_mm'])/1000
            center_src=(src['T']@np.r_[local,1])[:3]
            center=target_point(center_src,scale,zmin)
            dims=np.asarray(bb['size_mm'])*scale
            name=mesh if len(sources)==1 else f'{mesh}_{ix+1}'
            out.append((name,plane_shape(center,R,dims),{'name':name,'kind':'printed block BBox','source_asset':mesh,'center_mm':center.tolist(),
              'rotation':R.tolist(),'size_mm':dims.tolist()}))
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--microban-root',default=os.environ.get('MICROBAN_SOURCE_ROOT','/tmp/microban-reference'))
    a=ap.parse_args();mroot=Path(a.microban_root)
    if not (mroot/'src/model/mjcf/robot.xml').exists():raise SystemExit('Clone https://github.com/Rhoban/microban at '+MICROBAN_COMMIT+' outside this repo and pass --microban-root')
    src=source_geometry(mroot);scale=src['scale_to_500'];zmin=src['z_min_mm']/1000
    joints,geoms,masses=read_mjcf(mroot)
    # MJCF frame positions are source metres; the full source STEP gives the
    # exact 326.56 mm vertical extent used for uniform coordinate scaling.
    for n,j in joints.items():j['rotation']=np.eye(3) # overwritten from its body frame below
    # Reconstruct body rotations for joint axis orientation from a second tree walk.
    root=ET.parse(mroot/'src/model/mjcf/robot.xml').getroot();body_rot={}
    def walkrot(body,parentR):
        here=parentR@quat(body.get('quat'));body_rot[body.get('name')]=here
        for c in body.findall('body'):walkrot(c,here)
    for body in root.find('./worldbody').findall('body'):walkrot(body,np.eye(3))
    for j in joints.values():j['rotation']=body_rot[j['body']]
    st=Study(500,2,.65);st.build();parts={x['name']:x['shape'] for x in st.parts}
    skins={n:s for n,s in parts.items() if n not in {'chest_indicator'}}
    regions=shell_outer(st)
    blocks=block_envelopes(geoms,src['blocks'],scale,zmin)
    # Source motor set and explicit ape re-mapping: remove ankle-roll pair,
    # keep shoulder pitch/roll/elbow/head yaw, add a powered closing axis to
    # each ape paw at the center of the source-500 palm shells.
    variants={}
    for label,excluded,add_jaw,add_yaw in [('microban_19_original',set(),False,False),
      ('ape_19_mapped',ANKLE_ROLL,True,False),('ape_21_plus_shoulder_yaw',ANKLE_ROLL,True,True)]:
        items=[]
        for name,frame in joints.items():
            if name in excluded:continue
            shape,rec=axis_case(name,frame,scale,zmin)
            items.append((name,shape,rec))
        if add_jaw:
            for side,labelside in [(-1,'right'),(1,'left')]:
                limb='R' if side<0 else 'L';sh=parts[f'{limb}_palm_front']
                bb=sh.bounding_box();center=np.array([(bb.min.X+bb.max.X)/2,(bb.min.Y+bb.max.Y)/2,(bb.min.Z+bb.max.Z)/2])
                frame={'point_m':np.array([0.,0.,0.]),'axis':np.array([1.,0.,0.]),'rotation':np.eye(3)}
                # Use the source skin's measured target-space palm datum directly.
                aa=np.array([side,0.,0.]);xx=np.array([0.,1.,0.]);yy=np.cross(aa,xx);R=np.column_stack((xx,yy,aa))
                c=center-aa*14.5;shape=plane_shape(c,R,XL_CASE)
                rec={'name':f'{labelside}_paw_closure','kind':'ape-specific XL330 envelope','origin_mm':center.tolist(),'axis':aa.tolist(),'center_mm':c.tolist(),'rotation':R.tolist(),'size_mm':list(XL_CASE)}
                items.append((rec['name'],shape,rec))
        # Conservative envelopes of the original printed block/support families,
        # from the pinned STEP BBoxes transformed by the MJCF mesh poses.
        # Dimensions scale with the 30→50 cm skeleton; servo cavities do not.
        items.extend(blocks)
        if add_yaw:
            # Trial a real serial shoulder stack: put the yaw case at the pitch
            # datum, move its pitch motor 31 mm laterally, and carry the source
            # arm motors by the same offset. A simple bridge bounds the gap.
            shift={}
            for side,name in [(-1,'right_shoulder_pitch'),(1,'left_shoulder_pitch')]:
                pitch=next(x for x in items if x[0]==name);rec=pitch[2]
                origin=np.asarray(rec['origin_mm']);axis=np.array([0.,0.,1.])
                R=np.column_stack((np.array([1.,0,0]),np.array([0.,1,0]),axis));center=origin-axis*14.5
                ys=plane_shape(center,R,XL_CASE)
                items.append((f'{name.replace("pitch","yaw")}_added',ys,{'name':f'{name.replace("pitch","yaw")}_added','kind':'added shoulder-yaw XL330','origin_mm':origin.tolist(),'axis':axis.tolist(),'center_mm':center.tolist(),'rotation':R.tolist(),'size_mm':list(XL_CASE)}))
                shift[side]=31.0
            for i,(name,shape,rec) in enumerate(list(items)):
                if 'shoulder_pitch' in name or 'shoulder_roll' in name or 'elbow' in name:
                    side=-1 if 'right' in name else 1;d=np.array([side*shift[side],0,0])
                    if np.any(d):
                        rec2=dict(rec);rec2['origin_mm']=(np.array(rec['origin_mm'])+d).tolist();rec2['center_mm']=(np.array(rec['center_mm'])+d).tolist()
                        shape2=plane_shape(rec2['center_mm'],np.array(rec['rotation']),rec['size_mm'])
                        items[i]=(name,shape2,rec2)
            # The carrier is an envelope only, derived from the required offset.
            for side,name in [(-1,'right'),(1,'left')]:
                origin=np.array(next(x[2]['origin_mm'] for x in items if x[0]==f'{name}_shoulder_yaw_added'))
                center=origin+np.array([side*15.5,0,0]);bridge=b.Pos(*center)*b.Box(31,12,12)
                items.append((f'{name}_yaw_pitch_carrier',bridge,{'name':f'{name}_yaw_pitch_carrier','kind':'unweighted carrier envelope','center_mm':center.tolist(),'size_mm':[31,12,12]}))
        rows=[]
        for name,shape,rec in items:
            hits=[]
            for sn,skin in skins.items():
                v=common_mm3(shape,skin)
                if v>.01:hits.append({'skin':sn,'mm3':v})
            outer_hits=[]
            relevant=['torso','pelvis'] if ('shoulder' in name or 'head' in name or 'hip' in name) else ['thighs','pelvis'] if ('hip' in name or 'knee' in name or 'ankle' in name) else ['L_upper_arm_front','R_upper_arm_front','L_forearm_front','R_forearm_front','L_palm_front','R_palm_front']
            bb=shape.bounding_box();outside_v=0
            # Screen exact skin wall contacts; containment is evaluated against
            # local outer lobes where a matching sketch loft exists.
            regs=[]
            if 'torso' in name or 'shoulder' in name or name.startswith('head'):
                regs=[regions['torso']]
            elif 'hip' in name or 'knee' in name or 'ankle' in name:
                regs=[regions['pelvis'],*regions['thighs']]
            rem=shape
            for reg in regs:
                if aabb_overlap(rem,reg):rem=rem.cut(reg)
            outside_v=round(sum(s.volume for s in rem.solids()),3) if rem else 0.
            rows.append({'component':name,'geometry':rec,'shell_intersections':hits,'outside_assigned_outer_lobes_mm3':outside_v if regs else None})
        axis_count=sum(x[2]['kind'] in {'XL330 envelope','ape-specific XL330 envelope','added shoulder-yaw XL330'} for x in items)
        variants[label]={'actuated_axis_count':axis_count,
          'component_rows':rows,'components':[x[2] for x in items]}
    # Microban's assembly guide removes the trunk top to slide the battery out.
    # Measure its battery mesh bounds, place it at the MJCF's actual trunk pose,
    # and check a conservative straight-up 100 mm withdrawal corridor.
    bg=next(x for x in geoms if x['mesh']=='battery')
    battery_size=np.array([41.,76.3,20.5])
    battery_local=np.array([0.,0.,.01025])
    battery_center_source=(bg['T']@np.r_[battery_local,1])[:3]
    battery_center=target_point(battery_center_source,scale,zmin)
    battery_R=target_rotation(bg['T'][:3,:3])
    battery_shape=plane_shape(battery_center,battery_R,battery_size)
    battery_skin=[]
    for sn,skin in skins.items():
        v=common_mm3(battery_shape,skin)
        if v>.01:battery_skin.append({'skin':sn,'mm3':v})
    battery_aabb=np.sum(np.abs(battery_R)*battery_size,axis=1)
    path=b.Pos(*(battery_center+np.array([0,0,50])))*b.Box(battery_aabb[0],battery_aabb[1],battery_aabb[2]+100)
    path_skin=[]
    for sn,skin in skins.items():
        v=common_mm3(path,skin)
        if v>.01:path_skin.append({'skin':sn,'mm3':v})
    battery={'size_mm':battery_size.tolist(),'center_mm':battery_center.tolist(),
      'rotation':battery_R.tolist(),'shell_intersections':battery_skin,
      'vertical_100mm_withdrawal_envelope_bounds_mm':bnd(path),'withdrawal_shell_intersections':path_skin,
      'hatch_and_latch':'not modeled; original Microban guide specifies removable trunk top and XT30 disconnection'}
    data={'source':{'url':MICROBAN_URL,'commit':MICROBAN_COMMIT,'hardware_license':'CC BY-NC-SA 4.0; no source CAD exported',
      'source_height_mm':src['extent_mm'][2],'source_bbox_mm':src['extent_mm'],'scale_to_500':scale,
      'block_bounds_mm':{k:v['size_mm'] for k,v in src['blocks'].items()}},
      'mass_tree':masses,'source_mass_kg':round(sum(x['kg'] for x in masses),6),
      'xl330':{'mass_g':18,'count_source':19,'mass_g_source':342,'official_case_mm':[20,34,26],
        'screen_envelope_mm':list(XL_CASE),'official_stall_torque_Nm_at_5V':.52,
        'continuous_torque':'manufacturer does not state a continuous rating in this manual'},
      'bom_reference':{'source_robot_cost_usd':567,'listed_xl330_price_usd':23.9,'two_added_yaw_motor_cost_usd':47.8},
      'service_module':battery,'variants':variants,
      'shared_reach_evidence':{'no_shoulder_yaw_cross_body_target_miss_mm':41.88,
        'source':'experiments/compact_morphology/ape_design/swarm_round4_2axis_shoulder/output/report.json',
        'interpretation':'one sampled target only; not a task-success metric'},
      'limitations':['Source 30 cm block dimensions are scaled as structural envelopes; motor envelopes remain source-sized. Restoring source-sized motor cavities and designing carriers is required, so this is not a print-ready copy of Microban.',
        'Motor package boxes use the official XL330 envelope, not a STEP BRep. Source joint frames come from pinned MJCF at q=0.',
        'Exact skin intersections do not certify a complete link load path, moving clearance, torque duty, wiring or service extraction.']}
    (OUT/'report.json').write_text(json.dumps(data,indent=2));(OUT/'variants.json').write_text(json.dumps(variants,indent=2))
    print('source height',src['extent_mm'][2],'scale',scale,'mass kg',data['source_mass_kg'])
    for name,v in variants.items():
        print(name,'axis count',v['actuated_axis_count'],'intersecting packages',sum(bool(x['shell_intersections']) for x in v['component_rows']))

if __name__=='__main__':main()
