"""Generate portable URDF and native rigid/compliant MuJoCo descriptions."""
import collections
import copy
import re
import xml.etree.ElementTree as ET
import trimesh
from scipy.spatial.transform import Rotation
from common import *

def element(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k:str(v) for k,v in attrs.items()})

def save_xml(path, node):
    ET.indent(node, space='  ')
    ET.ElementTree(node).write(path,encoding='utf-8',xml_declaration=True)

def mass_properties(rows, origin):
    total=sum(p['mass_kg']for p in rows)
    com=sum((p['mass_kg']*(R@np.array(p['com_CAD_m']))for p in rows),np.zeros(3))/total
    inertia=np.zeros((3,3))
    for p in rows:
        offset=R@np.array(p['com_CAD_m'])-com
        inertia += p['mass_kg']*(R@np.array(p['inertia_per_kg_CAD_m2'])@R.T + np.eye(3)*(offset@offset)-np.outer(offset,offset))
    return dict(mass_kg=total,com_link_m=(com-origin).tolist(),com_model_m=com.tolist(),
                inertia_kg_m2=inertia.tolist(),eigenvalues_kg_m2=np.linalg.eigvalsh(inertia).tolist(),parts=[p['name']for p in rows])

def allowances(o):
    # Separate from all represented hardware, includes unmodeled harness/power
    # completion mass. This explicit scenario is neither BOM nor measured mass.
    allocation={'torso':.09,'head':.02,'R_shin':.01,'L_shin':.01,'R_forearm':.01,'L_forearm':.01}
    return [dict(name='unrepresented_installation_allowance_'+link,owning_link=link,
        mass_kg=mass,sensitivity_interval_kg=[mass/3,mass*2],mass_category='unrepresented_allowance',
        mass_basis='Unmeasured wiring/connectors/remaining installation allowance, optional sensitivity; no new installed CAD part',
        com_CAD_m=(R.T@o[link]).tolist(),inertia_per_kg_CAD_m2=(np.eye(3)*.02**2/6).tolist())for link,mass in allocation.items()]

def main():
    manifest=read(CAD/'assembly-parts.json'); o=origins(manifest)
    rows=read(OUT/'part-mass-ledger.json')['parts']
    row_by_name={p['name']:p for p in rows}
    parts={p['name']:p for p in manifest['parts']}
    all_rows=rows+allowances(o)
    link_rows={link:[p for p in all_rows if p['owning_link']==link]for link in o}
    inertias={link:mass_properties(rs,o[link])for link,rs in link_rows.items()}
    write(OUT/'link-inertias.json',dict(links=inertias,unrepresented_installation_allowance=allowances(o),
        total_mass_kg=sum(p['mass_kg']for p in all_rows),represented_mass_kg=sum(p['mass_kg']for p in rows),
        total_engineering_sensitivity_interval_kg=np.sum([p['sensitivity_interval_kg']for p in all_rows],axis=0).tolist(),
        reference_skin_mass_kg=sum(p['mass_kg']for p in rows if p['source_role']in['original_aesthetic_reference','ape_shell_reference']),
        mass_by_category_kg={c:sum(p['mass_kg']for p in all_rows if p['mass_category']==c)for c in sorted({p['mass_category']for p in all_rows})},
        interpretation='Total is a conditional current-geometry simulation estimate incl explicitly assumed skins and150g unfinished-installation allowance, not measured completed robot mass. Native shape moments use uniform material; motor/PCB internals remain approximated.'))
    meshdir=OUT/'meshes';meshdir.mkdir(exist_ok=True)
    audit_path=OUT/'mesh-transfer.json'
    if audit_path.exists():
        meshmeta=read(audit_path)['parts']; collision=read(OUT/'collision-model.json')['links']
    else:
        scene=trimesh.load(CAD/'assembly.glb',force='scene')
        groups=collections.defaultdict(list); meshmeta=[]
        for index,p in enumerate(manifest['parts']):
            name=p['name']; row=row_by_name[name];link=row['owning_link']
            T,key=scene.graph[name];mesh=scene.geometry[key].copy();mesh.apply_transform(T)
            rgba=np.array(mesh.visual.face_colors[0])/255 if len(mesh.faces) else np.array([.5,.5,.5,1.])
            mesh.vertices=(mesh.vertices@R.T)-o[link]
            original=mesh.copy(); reduced=False
            target=20000 if ('skin'in name or 'shell__torso'in name)else 4000
            if len(mesh.faces)>target:
                candidate=mesh.simplify_quadric_decimation(face_count=target)
                if np.max(abs(candidate.bounds-mesh.bounds)) <= .00025:
                    mesh=candidate;reduced=True
            filename=f'p{index:04d}'+('.obj' if len(mesh.faces)>200000 else '.stl');mesh.export(meshdir/filename)
            meshmeta.append(dict(name=name,owning_link=link,mesh='meshes/'+filename,rgba=rgba.tolist(),
                source_faces=len(original.faces),exported_faces=len(mesh.faces),display_reduced=reduced,
                maximum_bounds_deviation_m=float(np.max(abs(mesh.bounds-original.bounds))),
                mesh_sha256=sha(meshdir/filename)))
            # Skin and mechanisms participate in per-link convex contact
            # envelopes; internal wiring and individual screw heads do not.
            if p.get('role') not in ['installation_size_scenario_not_additive_material']:
                groups[link].append(original)
            if index%100==0:print('mesh',index,flush=True)
        collision={}
        for link,meshes in groups.items():
            hull=trimesh.util.concatenate(meshes).convex_hull
            filename='collision_'+link+'.stl';hull.export(meshdir/filename)
            collision[link]=dict(mesh='meshes/'+filename,vertices=len(hull.vertices),faces=len(hull.faces),
                bounds_link_m=hull.bounds.tolist(),source_parts=[p['name']for p in rows if p['owning_link']==link])
        write(audit_path,dict(source_scene=ref(CAD/'assembly.glb'),parts=meshmeta,
            interpretation='Rigid source transforms and sidecar ownership retained. Optional mesh reduction is for visual display only; full native source retained. Collision envelopes exported separately.'))
        write(OUT/'collision-model.json',dict(links=collision,
            method='One conservative convex hull per source link, including shell reference. Hulls fill gaps and do not reproduce concave apertures. Ground interaction uses these envelopes, not exact skin contact.',
            friction=dict(sliding=.8,torsional=.005,rolling=.0001),friction_basis='Unmeasured contact assumptions; sliding0.4..1.2 sensitivity',
            contact_softness='MuJoCo solref0.005s/1, solimp0.9/0.95/0.001; not FEM connector stiffness or measured sole material',
            exclusions='Within two links along source joint tree: assembly mating overlap excluded. Nonlocal shell/limb contacts retained. Also disclose adjacent range-collision blind spots.'))
    # MuJoCo's STL decoder caps individual files at200k triangles. OBJ retains
    # any full-detail mesh whose reduction would move the original bounds.
    for p in meshmeta:
        path=OUT/p['mesh']
        if p['exported_faces']>200000 and path.suffix=='.stl':
            new=path.with_suffix('.obj');trimesh.load(path,force='mesh').export(new)
            p.update(mesh=str(new.relative_to(OUT)),mesh_sha256=sha(new));path.unlink()
    write(audit_path,dict(source_scene=ref(CAD/'assembly.glb'),parts=meshmeta,
        interpretation='Rigid source transforms and sidecar ownership retained. Visual-only reduction; meshes over200k triangles use OBJ because native STL decoder is capped. Full native source retained.'))
    if (OUT/'collision-decomposition.json').exists():
        decomposition=read(OUT/'collision-decomposition.json')['links']
        if set(decomposition)!=set(o):raise ValueError('Incomplete collision decomposition')
        for link in collision:collision[link]['hulls']=decomposition[link]['hulls']
        for link in collision:collision[link]['source_parts']=decomposition[link]['source_parts']
        write(OUT/'collision-model.json',dict(links=collision,
            method='Current V-HACD convex decomposition; legacy mesh entry is the retained initial coarse-hull witness only. Native models use hulls.',
            source=ref(OUT/'collision-decomposition.json'),
            friction=dict(sliding=.8,torsional=.005,rolling=.0001),
            limitations='Nonmaterial installation reservation excluded. Thin-wall/contact voxelization remains approximate. In compliant variant receiving case/horn visuals and mass move; their decomposed collision hulls remain on source parent links. Adjacent two-hop exclusions omit intended mount contact and can hide adjacent-range collisions.'))
    acts=read(OUT/'actuators.json')['actuators'];sensors=read(OUT/'sensors.json')['sensors']
    flex=read(OUT/'compliance.json')['blocks']
    j_by_name={j['name']:j for j in manifest['joints']}
    children=collections.defaultdict(list)
    for j in manifest['joints']:children[j['parent']].append(j)
    by_link_mesh=collections.defaultdict(list)
    for p in meshmeta:by_link_mesh[p['owning_link']].append(p)
    # URDF remains a rigid tree. General six-dimensional elasticity is carried
    # in the native model and sidecar; URDF joint damping cannot encode it.
    urdf=ET.Element('robot',name='cove50_selected_electronics_r1')
    compiler=element(element(urdf,'mujoco'),'compiler',fusestatic='false',discardvisual='false',balanceinertia='false')
    for link in o:
        body=element(urdf,'link',name=link);inert=inertias[link]
        ie=element(body,'inertial');element(ie,'origin',xyz=vec(inert['com_link_m']),rpy='0 0 0');element(ie,'mass',value=inert['mass_kg'])
        I=np.array(inert['inertia_kg_m2']);element(ie,'inertia',ixx=I[0,0],iyy=I[1,1],izz=I[2,2],ixy=I[0,1],ixz=I[0,2],iyz=I[1,2])
        for p in by_link_mesh[link]:
            v=element(body,'visual',name=p['name']);element(v,'geometry');element(v.find('geometry'),'mesh',filename=p['mesh'])
            mat=element(v,'material',name='mat_'+p['mesh'].split('/')[-1][:-4]);element(mat,'color',rgba=vec(p['rgba']))
        for index,c in enumerate(collision[link].get('hulls',[collision[link]])):
            ce=element(body,'collision',name='convex_'+link+'_'+str(index));element(element(ce,'geometry'),'mesh',filename=c['mesh'])
    for j,a in zip(manifest['joints'],acts):
        limit=j['range_deg'];q=element(urdf,'joint',name=j['name'],type='revolute' if limit is not None else 'continuous')
        element(q,'parent',link=j['parent']);element(q,'child',link=j['child']);element(q,'origin',xyz=vec(o[j['child']]-o[j['parent']]),rpy='0 0 0')
        element(q,'axis',xyz=vec(R@np.array(j['axis_world_neutral'])))
        attrs=dict(effort=.980665 if a['model']=='c047' else .2,velocity=np.pi/3/.222 if a['model']=='c047' else 57*2*np.pi/60)
        if limit is not None:attrs.update(lower=np.deg2rad(limit[0]),upper=np.deg2rad(limit[1]))
        element(q,'limit',**attrs);element(q,'dynamics',damping=.001,friction=0.)
    for s in sensors:
        element(urdf,'link',name=s['name']);j=element(urdf,'joint',name=s['name']+'_mount',type='fixed')
        element(j,'parent',link=s['owning_link']);element(j,'child',link=s['name'])
        element(j,'origin',xyz=vec(s['origin_link_m']),rpy=vec(Rotation.from_matrix(s['rotation_link_sensor']).as_euler('xyz')))
    save_xml(OUT/'cove50.urdf',urdf)
    ros=copy.deepcopy(urdf);ros.remove(ros.find('mujoco'))
    for node in ros.iter('mesh'):node.set('filename','package://cove50_description/'+node.get('filename'))
    save_xml(OUT/'cove50.ros.urdf',ros)
    srdf=ET.Element('robot',name='cove50_selected_electronics_r1')
    parent_of={j['child']:j['parent']for j in manifest['joints']}
    for link in o:
        ancestor=link
        for hop in range(2):
            ancestor=parent_of.get(ancestor)
            if ancestor is None:break
            element(srdf,'disable_collisions',link1=link,link2=ancestor,reason='Adjacent'if hop==0 else'Source assembly mating neighborhood')
    save_xml(OUT/'cove50.srdf',srdf)
    for variant in ['rigid','compliant']:
        model=ET.Element('mujoco',model='cove50_selected_'+variant)
        element(model,'compiler',angle='radian',inertiafromgeom='false',autolimits='true',fusestatic='false')
        element(model,'size',nuser_actuator='4')
        dt=.001 if variant=='rigid'else.000002
        element(model,'option',timestep=dt,gravity='0 0 -9.80665',integrator='implicitfast',solver='Newton',iterations='60',cone='elliptic')
        visual=element(model,'visual');element(visual,'global',offwidth='1280',offheight='960');element(visual,'map',znear='.01',zfar='20')
        assets=element(model,'asset')
        for p in meshmeta:element(assets,'mesh',name='v_'+p['mesh'].split('/')[-1][:-4],file=p['mesh'])
        for link,cs in collision.items():
            for index,c in enumerate(cs.get('hulls',[cs])):element(assets,'mesh',name='c_'+link+'_'+str(index),file=c['mesh'])
        world=element(model,'worldbody');element(world,'light',pos='1 -1 2',dir='-1 1 -2',diffuse='.8 .8 .8')
        element(world,'geom',name='floor',type='plane',size='3 3 .1',rgba='.25 .28 .30 1',friction='.8 .005 .0001')
        tendon=element(model,'tendon')
        flexmap={b['receiving_joint']:b for b in flex if b['placement']!='sidecar_only'} if variant=='compliant'else{}
        moved={b['moved_parent_part']:b for b in flexmap.values()if b['moved_parent_part']}
        native_owner={p['name']:p['owning_link']for p in rows}
        for p,b in moved.items():native_owner[p]='flex_'+b['id']
        native_rows=collections.defaultdict(list)
        for p in copy.deepcopy(all_rows):native_rows[native_owner.get(p['name'],p['owning_link'])].append(p)
        numerical_splits=[]
        def split_tiny_mass(source_link,target_link):
            source=next(p for p in native_rows[source_link] if p['mass_kg']>1e-5)
            tiny=copy.deepcopy(source);tiny['mass_kg']=1e-6;tiny['name']=source['name']+'__numerical_split'
            source['mass_kg']-=tiny['mass_kg'];native_rows[target_link].append(tiny)
            numerical_splits.append(dict(source=source['name'],source_link=source_link,target_link=target_link,mass_kg=1e-6,
                reason='Mass-conserving numerical allocation for a moving intermediate body. No invented extra robot mass.'))
        body_elements={}
        def add_inertial(body, link, origin):
            rs=native_rows[link]
            if not rs:return
            inert=mass_properties(rs,origin);I=np.array(inert['inertia_kg_m2'])
            element(body,'inertial',pos=vec(inert['com_link_m']),mass=inert['mass_kg'],fullinertia=vec([I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]]))
        def add_visuals(body, link, origin):
            for p in meshmeta:
                if native_owner[p['name']]!=link:continue
                old_origin=o[p['owning_link']]
                element(body,'geom',name=p['name'],type='mesh',mesh='v_'+p['mesh'].split('/')[-1][:-4],pos=vec(old_origin-origin),
                    rgba=vec(p['rgba']),contype='0',conaffinity='0',group='1',density='0')
        def add_flex(parent, block, parent_origin):
            origin=np.array(block['model_reference_origin_m']);link='flex_'+block['id']
            body=element(parent,'body',name=link,pos=vec(origin-parent_origin));body_elements[link]=body
            names=[]
            for i in range(6):
                name=link+'_'+str(i);names.append(name)
                element(body,'joint',name=name,type='slide'if i<3 else'hinge',axis=vec(np.eye(3)[i%3]),limited='false',damping='0',armature='.000001'if i>=3 else'.00001')
            # An after-joint foot stage owns the physical foot body downstream;
            # the massless flex frame gets no artificial lumped mass.
            if block['placement']=='before' and sum(p['mass_kg']for p in native_rows[link])<=1e-10:
                split_tiny_mass(j_by_name[block['receiving_joint']]['child'],link)
            add_inertial(body,link,origin);add_visuals(body,link,origin)
            for mode,(lam,coef)in enumerate(zip(block['modal_stiffness_N_m'],block['modal_coefficients'])):
                fixed=element(tendon,'fixed',name=link+'_mode_'+str(mode),stiffness=lam,damping=2*.1*np.sqrt(lam*.055),springlength='0')
                for name,c in zip(names,coef):element(fixed,'joint',joint=name,coef=c)
            return body,origin
        def add_body(parent,link,parent_origin,incoming=None):
            block=flexmap.get(incoming['name']) if incoming else None
            before=block and block['placement']=='before'
            if before:parent,parent_origin=add_flex(parent,block,parent_origin)
            body=element(parent,'body',name=link,pos=vec(o[link]-parent_origin));body_elements[link]=body
            if incoming:
                attrs=dict(name=incoming['name'],type='hinge',axis=vec(R@np.array(incoming['axis_world_neutral'])),damping='.001',armature='.000001')
                if incoming['range_deg'] is not None:attrs['range']=vec(np.deg2rad(incoming['range_deg']))
                else:attrs['limited']='false'
                element(body,'joint',**attrs)
            else:element(body,'freejoint',name='root')
            current=body
            if block and block['placement']=='after':
                current,current_origin=add_flex(body,block,o[link])
                # Physical foot mass/visual/collision move after the drive.
                for rs in list(native_rows[link]):native_rows['flex_'+block['id']].append(rs)
                native_rows[link]=[]
                for name,owner in native_owner.items():
                    if owner==link:native_owner[name]='flex_'+block['id']
                split_tiny_mass('flex_'+block['id'],link)
                add_inertial(body,link,o[link])
                add_inertial(current,'flex_'+block['id'],o[link]);add_visuals(current,'flex_'+block['id'],o[link])
            else:
                add_inertial(body,link,o[link]);add_visuals(body,link,o[link])
            for index,c in enumerate(collision[link].get('hulls',[collision[link]])):
                element(current,'geom',name='collision_'+link+'_'+str(index),type='mesh',mesh='c_'+link+'_'+str(index),group='3',rgba='.4 .5 .6 .18',
                    friction='.8 .005 .0001',solref='.005 1',solimp='.9 .95 .001',condim='3',density='0')
            for s in sensors:
                if s['owning_link']!=link:continue
                mat=np.array(s['rotation_link_sensor']);q=Rotation.from_matrix(mat).as_quat();q=np.r_[q[3],q[:3]]
                element(current,'site',name=s['name'],pos=vec(s['origin_link_m']),quat=vec(q),size='.002',rgba='0 .7 .8 .7',group='4')
                if s['kind']in['rgb','depth']:
                    # MuJoCo camera looks -Z with Y up; optical is +Z/Y down.
                    cammat=mat@np.diag([1.,-1.,-1.]);cq=Rotation.from_matrix(cammat).as_quat();cq=np.r_[cq[3],cq[:3]]
                    K=np.array(s['nominal_K']);w,h=s['resolution']
                    element(current,'camera',name=s['name']+'_camera',pos=vec(s['origin_link_m']),quat=vec(cq),
                        sensorsize='.004 .003',resolution=f'{w} {h}',focalpixel=vec([K[0,0],K[1,1]]),principalpixel='0 0')
            for j in children[link]:add_body(current,j['child'],o[link],j)
        add_body(world,'torso',np.zeros(3))
        contact=element(model,'contact')
        parent_of={j['child']:j['parent']for j in manifest['joints']}
        for link in o:
            ancestor=link
            for i in range(2):
                ancestor=parent_of.get(ancestor)
                if ancestor is None:break
                # Contact exclusions refer actual colliding native body owner.
                owner1='flex_'+flexmap[next(j['name']for j in manifest['joints']if j['child']==link)]['id'] if any(j['child']==link and j['name']in flexmap and flexmap[j['name']]['placement']=='after'for j in manifest['joints'])else link
                owner2='flex_'+flexmap[next(j['name']for j in manifest['joints']if j['child']==ancestor)]['id'] if any(j['child']==ancestor and j['name']in flexmap and flexmap[j['name']]['placement']=='after'for j in manifest['joints'])else ancestor
                element(contact,'exclude',body1=owner1,body2=owner2)
        actuator=element(model,'actuator')
        for a in acts:
            peak=2.941995 if a['model']=='c047'else 1.12
            element(actuator,'motor',name=a['joint']+'_motor',joint=a['joint'],gear='1',ctrllimited='true',ctrlrange=vec([-peak,peak]),user=vec([peak,4.717106 if a['model']=='c047'else 5.969026,.980665 if a['model']=='c047'else.2,12.]))
        sens=element(model,'sensor')
        element(sens,'gyro',name='imu_gyro',site='imu_board');element(sens,'accelerometer',name='imu_accel',site='imu_board')
        element(sens,'framequat',name='evaluator_torso_quat',objtype='body',objname='torso')
        for a in acts:element(sens,'jointpos',name=a['joint']+'_position',joint=a['joint']);element(sens,'jointvel',name=a['joint']+'_velocity',joint=a['joint'])
        for side in ['R','L']:element(sens,'touch',name=side+'_sole_touch_oracle',site=side+'_sole_contact_frame')
        save_xml(OUT/('cove50-'+variant+'.xml'),model)
        if variant=='compliant':write(OUT/'compliance-mass-splits.json',numerical_splits)
    write(OUT/'model-build.json',dict(version=VERSION,source_scene=ref(CAD/'assembly.glb'),
        authored_inputs=[ref(Path(__file__)),ref(OUT/'part-mass-ledger.json'),ref(OUT/'compliance.json'),ref(OUT/'actuators.json'),ref(OUT/'sensors.json')],
        active_joints=[j['name']for j in manifest['joints']],unknown_source_limits=[j['name']for j in manifest['joints']if j['range_deg']is None],
        source_neutral_axes='All22 preserved. Six unknown hip ranges exported unlimited/continuous only as unspecified API representation, not evidence of unrestricted physical travel.',
        variants=['cove50.urdf','cove50.ros.urdf','cove50-rigid.xml','cove50-compliant.xml']))
    print(dict(total_mass_kg=sum(p['mass_kg']for p in all_rows),parts=len(rows),joints=len(acts)),flush=True)

if __name__=='__main__':main()
