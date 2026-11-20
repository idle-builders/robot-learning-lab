"""Exercise the actual native consumers and retain numerical evidence."""
import copy
import sys
import time
import xml.etree.ElementTree as ET
import mujoco
from scipy.spatial.transform import Rotation
from common import *
from build import element

def main():
    report=dict(engine=mujoco.__version__,source_inputs=[],models={},kinematics=[],spring_reconstructions=[])
    for source in read(OUT/'source-lock.json')['sources']:
        actual=ref(ROOT/source['path']);report['source_inputs'].append(dict(**actual,expected_sha256=source['sha256']))
        if actual['sha256']!=source['sha256']:raise ValueError('Frozen source drift: '+source['path'])
    manifest=read(CAD/'assembly-parts.json');joints=manifest['joints'];o=origins(manifest)
    models={}
    for filename in ['cove50.urdf','cove50-rigid.xml','cove50-compliant.xml']:
        m=mujoco.MjModel.from_xml_path(str(OUT/filename));models[filename]=m
        d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,int(m.actuator_trnid[i,0]))for i in range(m.nu)]
        contacts=[dict(geom1=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(c.geom1)),geom2=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(c.geom2)),distance_m=float(c.dist))for c in d.contact]
        report['models'][filename]=dict(source=ref(OUT/filename),nq=m.nq,nv=m.nv,actuators=m.nu,
            actuator_joints=names,total_mass_kg=float(m.body_mass.sum()),bodies=m.nbody,geoms=m.ngeom,
            neutral_contacts=contacts,most_negative_neutral_contact_m=min([c['distance_m']for c in contacts],default=0.),
            timestep_s=float(m.opt.timestep))
    rigid=models['cove50-rigid.xml']; urdf=models['cove50.urdf']; flexible=models['cove50-compliant.xml']
    rng=np.random.default_rng(3753)
    for station in range(13):
        commands={j['name']:0. if station==0 else float(rng.uniform(-.08,.08))for j in joints}
        for j in joints:
            if j['range_deg'] is not None:commands[j['name']]=float(np.clip(commands[j['name']],*np.deg2rad(j['range_deg'])))
        data={key:mujoco.MjData(m)for key,m in models.items()}
        for key,m in models.items():
            d=data[key]
            for name,q in commands.items():d.qpos[m.jnt_qposadr[m.joint(name).id]]=q
            mujoco.mj_forward(m,d)
        u=data['cove50.urdf'];d=data['cove50-rigid.xml'];f=data['cove50-compliant.xml']
        errors=[];flexerr=[];rotation=[]
        for link in o:
            pu=u.xpos[urdf.body(link).id]+o['torso'];pd=d.xpos[rigid.body(link).id]
            errors.append(float(np.linalg.norm(pu-pd)))
            flexerr.append(float(np.linalg.norm(f.xpos[flexible.body(link).id]-pd)))
            rotation.append(float(np.max(abs(u.xmat[urdf.body(link).id]-d.xmat[rigid.body(link).id]))))
        report['kinematics'].append(dict(station=station,commands_rad=commands,max_URDF_MJCF_origin_error_m=max(errors),
            max_rigid_flexible_origin_error_m=max(flexerr),max_URDF_MJCF_rotation_element_error=max(rotation)))
    for block in read(OUT/'compliance.json')['blocks']:
        # Independent native coupon checks all coupled matrix terms, with no
        # whole-body contact/gravity or controller obscuring the spring law.
        xml=ET.Element('mujoco');element(xml,'compiler',angle='radian');world=element(xml,'worldbody');body=element(world,'body',name='receiver')
        for i in range(6):element(body,'joint',name='j'+str(i),type='slide'if i<3 else'hinge',axis=vec(np.eye(3)[i%3]),limited='false')
        element(body,'inertial',mass='.055',pos='0 0 0',diaginertia='.00002 .00002 .00002')
        tendon=element(xml,'tendon')
        for mode,(k,coef)in enumerate(zip(block['modal_stiffness_N_m'],block['modal_coefficients'])):
            t=element(tendon,'fixed',name='m'+str(mode),stiffness=k,springlength='0')
            for i,c in enumerate(coef):element(t,'joint',joint='j'+str(i),coef=c)
        mm=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'));dd=mujoco.MjData(mm)
        dd.qpos[:]=[.00001,-.00002,.00001,.0001,-.0001,.0002]
        mujoco.mj_forward(mm,dd);expected=-np.array(block['stiffness_SI'])@dd.qpos
        report['spring_reconstructions'].append(dict(block=block['id'],configuration_m_rad=dd.qpos.tolist(),
            expected_N_Nm=expected.tolist(),native_passive_force_N_Nm=dd.qfrc_passive.tolist(),
            relative_force_error=float(np.linalg.norm(dd.qfrc_passive-expected)/np.linalg.norm(expected)),
            source_reciprocity_relative_error=block['source_reciprocity_relative_error']))
    mass=read(OUT/'link-inertias.json');report['mass']=mass
    report['frame_rotation_determinant']=float(np.linalg.det(R))
    report['mesh_transfer']=dict(parts=len(read(OUT/'mesh-transfer.json')['parts']),
        maximum_display_bounds_deviation_m=max(p['maximum_bounds_deviation_m']for p in read(OUT/'mesh-transfer.json')['parts']))
    # Old actor metadata is retained, rather than pretending to load it into the
    # new75-input22-output ABI. The separate trace adapter is explicitly lossy.
    report['policy_interface']=dict(current=dict(actor_observations=75,actions=22,joint_order=[j['name']for j in joints]),
        historical=dict(actor_observations=72,actions=21,checkpoint='artifacts/policies/checkpoints/frame50_catalog_motor_walk.pt',
            sha256=sha(ROOT/'artifacts/policies/checkpoints/frame50_catalog_motor_walk.pt')),
        compatibility='Different body, topology, observation/action dimensions and neutral frames. No direct checkpoint inference or channel deletion. Diagnostic trajectory mapping is separately labeled and does not establish actor compatibility.')
    report['interpretation']='Native model loading, FK and spring-law measurements; no engineering binary score or hardware qualification.'
    write(OUT/'native-validation.json',report)
    print(dict(models={k:(v['nq'],v['nv'],v['actuators'])for k,v in report['models'].items()},
        max_FK_error_m=max(r['max_URDF_MJCF_origin_error_m']for r in report['kinematics']),
        max_spring_relative_error=max(r['relative_force_error']for r in report['spring_reconstructions'])),flush=True)

if __name__=='__main__':main()
