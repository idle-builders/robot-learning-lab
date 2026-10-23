"""Attach retained assembly CAD visuals to a sealed dynamics bundle, without changing physics."""
import argparse, hashlib, json, sys, struct, xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import mujoco
from build123d import import_step, export_stl
from lab.artifact_store import snapshot_file
from lab.config import RobotConfig
from lab.robot import make_robot
from lab.leg_mechanism import transform_shape
from tools.check_arm_episode import parts_from_step

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build(bundle,assembly,output):
    output.mkdir(parents=True,exist_ok=False)
    seal=json.loads((bundle/'manifest.json').read_text())
    for name,expected in seal['files'].items():
        if sha(bundle/name)!=expected:raise ValueError('Changed input '+name)
        if not name.startswith('source/') and name not in ('model/robot.xml','model/supported.xml','model/report.json'):
            snapshot_file(bundle/name,output/name)
    report=json.loads((assembly/'report.json').read_text())
    if report['assembly_step_roundtrip']!='pass':raise ValueError('Unverified STEP roundtrip')
    if sha(assembly/'assembly.step')!=report['output_hashes']['assembly.step']:raise ValueError('Changed CAD')
    cfg=json.loads((bundle/'model/resolved-config.json').read_text())
    if cfg!=json.loads((assembly/'resolved-config.json').read_text()):raise ValueError('CAD/physics configuration mismatch')
    frames=make_robot(RobotConfig.model_validate(cfg)).forward_kinematics()
    print('Loading retained CAD assembly',flush=True)
    parts=parts_from_step(list(import_step(assembly/'assembly.step').children),report['parts'])
    metadata={p['name']:p for p in report['parts']}
    inventory=[];meshes=output/'model/meshes';meshes.mkdir()
    for i,part in enumerate(parts):
        local=transform_shape(part.shape,np.linalg.inv(frames[part.owner]))
        path=meshes/(part.name+'.stl')
        export_stl(local,path,tolerance=.1,angular_tolerance=.2)
        if struct.unpack('<I',path.read_bytes()[80:84])[0]>200000:
            # MuJoCo's STL decoder has a face cap; OBJ preserves these same
            # CAD triangles without decimation. Retain the original as evidence.
            import trimesh
            obj=path.with_suffix('.obj');trimesh.load_mesh(path).export(obj);path=obj
        color=list(metadata[part.name]['color'])
        if len(color)==3:color.append(1.0)
        inventory.append(dict(name=part.name,owner=part.owner,role=part.role,color=color,file=str(path.relative_to(output)),sha256=sha(path),bytes=path.stat().st_size))
        if i%30==0:print('Tessellated',i+1,'of',len(parts),flush=True)
    checks=[]
    for name in ('robot.xml','supported.xml'):
        tree=ET.parse(bundle/'model'/name);root=tree.getroot();asset=root.find('asset')
        bodies={b.get('name'):b for b in root.findall('.//body')}
        for body in bodies.values():
            for geom in body.findall('geom'):
                rgba=geom.get('rgba','1 1 1 1').split();geom.set('rgba',' '.join([*rgba[:3],'0']))
        for part in inventory:
            meshname='cad_'+part['name']
            ET.SubElement(asset,'mesh',name=meshname,file='meshes/'+Path(part['file']).name,scale='0.001 0.001 0.001')
            ET.SubElement(bodies[part['owner']],'geom',name=meshname,type='mesh',mesh=meshname,rgba=' '.join(map(str,part['color'])),contype='0',conaffinity='0',mass='0',group='2')
        ET.indent(tree);tree.write(output/'model'/name,encoding='unicode')
        before=mujoco.MjModel.from_xml_path(str(bundle/'model'/name));after=mujoco.MjModel.from_xml_path(str(output/'model'/name))
        props=['body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat','jnt_range','jnt_axis','jnt_pos','dof_armature','dof_damping','actuator_gainprm','actuator_biasprm']
        errors={p:float(np.max(np.abs(getattr(before,p)-getattr(after,p)),initial=0)) for p in props}
        if any(v>1e-10 for v in errors.values()):raise ValueError('Visual attachment changed dynamics: '+str(errors))
        # Check all previous collision shapes by name despite shifted geom indices.
        for i in range(before.ngeom):
            if not (before.geom_contype[i] or before.geom_conaffinity[i]):continue
            j=mujoco.mj_name2id(after,mujoco.mjtObj.mjOBJ_GEOM,mujoco.mj_id2name(before,mujoco.mjtObj.mjOBJ_GEOM,i))
            for prop in ['geom_contype','geom_conaffinity','geom_type','geom_size','geom_pos','geom_quat','geom_friction','geom_condim','geom_solref','geom_solimp']:
                if not np.array_equal(getattr(before,prop)[i],getattr(after,prop)[j]):raise ValueError('Changed collision '+prop)
        checks.append(dict(model=name,dynamic_parameter_max_errors=errors,mesh_count=after.nmesh,collision_parity='pass',mass_kg=float(after.body_mass.sum())))
    binding=dict(status='pass',scope='Actual retained CAD tessellation; unchanged primitive collisions and assembly mass. Not physical or clearance qualification.',assembly=str(assembly),assembly_sha256=sha(assembly/'assembly.step'),assembly_report_sha256=sha(assembly/'report.json'),assembly_status=report['status'],source_bundle_sha256=sha(bundle/'manifest.json'),tessellation=dict(linear_deflection_mm=.1,angular_deflection_rad=.2),parts=inventory,checks=checks)
    (output/'cad-manifest.json').write_text(json.dumps(binding,indent=2))
    (output/'model/report.json').write_text(json.dumps(dict(status='pass',scope=binding['scope'],checks=checks),indent=2))
    snapshot_file(Path(__file__),output/'source/build_cad_gym_bundle.py')
    manifest=dict(seal);manifest['files']={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}
    manifest['cad_binding_sha256']=sha(output/'cad-manifest.json')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(status='pass',parts=len(parts),mesh_bytes=sum(p['bytes'] for p in inventory),bundle_sha256=sha(output/'manifest.json'))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--assembly',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.bundle,a.assembly,a.output)
