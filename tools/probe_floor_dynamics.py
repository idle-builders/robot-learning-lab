"""Retain a bounded dynamic floor squat/grasp reference trial, including failures."""
import argparse,json,sys,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from lab.floor_pickup import FloorProbe,build_scene
from lab.artifact_store import snapshot_file,plan_migration,apply_migration,verify_manifest


def run(cap=.7,render=False):
 out=ROOT/'artifacts'/('floor-dynamics-'+uuid.uuid4().hex[:12]);out.mkdir();print(out,flush=True)
 build_scene(out/'scene.xml',source=(.015,.140,.015),block_size=.030,robot_model=ROOT/'artifacts/pip-floor-reach-model-v1/robot.xml')
 env=FloorProbe(out/'scene.xml',cap)
 goal=json.loads((ROOT/'artifacts/floor-cleanup-preflight-18686e4733a4/report.json').read_text())['models']['wheels']['joints']
 upright=env.rest.copy()
 for n in ['left_shoulder_roll_joint','left_upper_arm_joint','left_forearm_joint','left_gripper_joint','right_shoulder_roll_joint']:
  upright[env.order.index(n)]=goal[n]
 crouch=np.array([goal[n] for n in env.order])
 rows=[];frames=[]
 for step in range(1900):
  t=step*.02
  s=np.clip((t-2)/8,0,1);s=s*s*(3-2*s)
  if t>16:s=1-np.clip((t-16)/4,0,1);s=s*s*(3-2*s)
  target=upright+s*(crouch-upright)
  # Park the hand forward while the knee/wheel passes underneath it.
  arm_s=np.clip((t-10)/3,0,1) if t<16 else 1.
  target[env.order.index('left_forearm_joint')]=-.7+arm_s*(goal['left_forearm_joint']+.7)
  if t>14:target[env.order.index('left_gripper_joint')]=.45
  command=.06 if 20<t<26 else 0.
  if t>26.5:
   down=np.clip((t-26.5)/4,0,1);down=down*down*(3-2*down)
   target=upright+down*(crouch-upright)
   target[env.order.index('left_gripper_joint')]=.45 if t<31 else -.45
  row=env.advance(target,command);rows.append(row)
  if render and step%5==0:frames.append(env.render())
  if step%100==0:print(round(t,2),'pitch',round(row['pitch_rad'],3),'hand',np.round(row['hand_m'],3),'obj',np.round(row['block_m'],3),'contact',row['block_contacts_n'],flush=True)
  if abs(row['pitch_rad'])>.45:break
 if render:
  import imageio.v2 as iio
  iio.mimwrite(out/'episode.mp4',frames,fps=10)
 (out/'trajectory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 report={'status':'probe','leg_cap_nm':cap,'duration_s':rows[-1]['time_s'],'min_hand_z_m':min(r['hand_m'][2] for r in rows),'max_block_z_m':max(r['block_m'][2] for r in rows),'final':rows[-1],'scope':'Authored whole-body reference, not learned cleanup; current sealed geometry; unmeasured actuator model.'}
 (out/'report.json').write_text(json.dumps(report,indent=2));env.close()
 for rel in ['lab/floor_pickup.py','tools/probe_floor_dynamics.py','lab/mujoco_learning.py','lab/electromechanical.py','learning/pip-electromechanical-v1.json']:
  snapshot_file(ROOT/rel,out/'source'/rel)
 p=plan_migration(ROOT,[out]);apply_migration(ROOT,p);verify_manifest(ROOT,p)
 print({k:v for k,v in report.items() if k!='final'},flush=True)
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--leg-cap',type=float,default=.7);p.add_argument('--render',action='store_true');a=p.parse_args();run(a.leg_cap,a.render)
