"""Source-bound offline projection of native Isaac telemetry to existing Rerun geometry."""
import argparse,json,sys,subprocess,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lab.rerun_scene import export_scene
from lab.artifact_store import snapshot_file
import mujoco,numpy as np
p=argparse.ArgumentParser();p.add_argument('evaluation',type=Path);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--skills',action='store_true');a=p.parse_args()
a.output.mkdir(exist_ok=False,parents=True)
source=a.output/'source';source.mkdir()
snapshot_file(a.evaluation,source/'evaluation.json')
snapshot_file(a.bundle/'manifest.json',source/'bundle-manifest.json')
snapshot_file(a.bundle/'model/robot.xml',source/'robot.xml')
interface=json.loads((a.bundle/'model/interfaces.json').read_text())
scene=export_scene(a.bundle/'model/robot.xml',a.output/'scene.json')
# Native SceneCfg uses a 20 m gym ground. The portable MJCF viewer's 4 m
# display patch does not cover the offset vector-environment origins.
for geom in scene['geoms']:
 if geom['body']=='world' and geom['name']=='floor':geom['size_m'][:2]=[10.,10.]
(a.output/'scene.json').write_text(json.dumps(scene))
model=mujoco.MjModel.from_xml_path(str(a.bundle/'model/robot.xml'));state=mujoco.MjData(model)
d=json.loads(a.evaluation.read_text());rows=[]
if a.skills:
 report=json.loads(a.evaluation.with_name('report.json').read_text())
 snapshot_file(a.evaluation.with_name('report.json'),source/'report.json')
 scope='Native Isaac skill evaluation: body poses are recorded simulator state; camera and joint samples at 5 Hz. Missing images remain missing. No physical qualification.'
 for row in d[report['training_records']:]:
  bodies={link['name']:{'position_m':pose[:3],'quaternion_xyzw':pose[3:]} for link,pose in zip(interface['links'],row['body_poses_world_xyz_xyzw'],strict=True)}
  command=row['commands'][-1]
  item=dict(row,bodies=bodies,root_position_world_m=row['body_poses_world_xyz_xyzw'][0][:3],command=[command['forward_m_s'],0,command['yaw_rad_s']])
  if row.get('image'):snapshot_file(a.evaluation.parent/row['image'],a.output/row['image'])
  rows.append(item)
 outcome='success' if report['evaluation_success'] else 'failed task; inspect retained camera and motion'
else:
 scope='Isaac motor evaluation. Display link poses are forward kinematics from recorded root/joints through the sealed MJCF; not independently measured native body poses. Camera unavailable in this motor-only recording.'
 outcome=d['status']
for row in ([] if a.skills else d['trace']):
 state.qpos[:3]=row['root_position_world_m']
 xyzw=row['root_quaternion_xyzw'];state.qpos[3:7]=[xyzw[3],*xyzw[:3]]
 for name,value in zip(interface['joint_order'],row['q'],strict=True):
  j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name);state.qpos[model.jnt_qposadr[j]]=value
 mujoco.mj_forward(model,state)
 bodies={}
 for i in range(model.nbody):
  name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) or 'world'
  q=state.xquat[i];bodies[name]={'position_m':state.xpos[i].tolist(),'quaternion_xyzw':[q[1],q[2],q[3],q[0]]}
 rows.append(dict(row,bodies=bodies))
scope += (' Shared battery RC, delayed PD and voltage-dependent DC motor proxy; electrical traces are estimates, not measurements.' if rows[0].get('electromechanical') else ' Ideal actuators in this historical rollout.')
if (a.bundle/'cad-manifest.json').exists():
 snapshot_file(a.bundle/'cad-manifest.json',source/'cad-manifest.json')
 scope += ' Visual geometry is the source-bound detailed CAD; collisions remain the sealed simplified model.'
(a.output/'replay.json').write_text(json.dumps({'rows':rows,'interface':interface,'outcome':outcome,'scope':scope}))
subprocess.run(['.venv-observe/bin/python','tools/write_control_rrd.py',str(a.output)],check=True)
subprocess.run(['.venv-observe/bin/rerun','rrd','verify',str(a.output/'episode.rrd')],check=True)
