"""Execute the feedback cleanup candidate and retain sensors, truth and failures."""
import argparse,json,sys,uuid,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import mujoco
from lab.floor_pickup import FloorProbe,CleanupReference,build_scene,evaluate_cleanup
from lab.artifact_store import snapshot_file,retain_dependency,plan_migration,apply_migration,verify_manifest


def run(render=False,friction=.8):
 out=ROOT/'artifacts'/('floor-cleanup-'+uuid.uuid4().hex[:12]);out.mkdir();print(out,flush=True)
 build_scene(out/'scene.xml',source=(.015,.14,.015),block_size=.03,friction=friction,robot_model=ROOT/'artifacts/pip-floor-reach-model-v1/robot.xml')
 m=FloorProbe(out/'scene.xml',1.);m.data.qpos[0]=-.25;mujoco.mj_forward(m.model,m.data)
 c=CleanupReference(m);rows=[];frames=[];oldphase='';report={'status':'running'}
 try:
  for step in range(3500):
   if step%10==0:c.capture()
   target,speed,obs=c.command();m.heading_error=c.yaw
   row=m.advance(target,speed)
   row.update(phase=c.phase,estimate=c.estimate,frame_estimate=c.frame_estimate,estimated_world_x_m=obs['world_x_m'],estimated_hand_m=obs['hand'].tolist(),heading_estimate_rad=c.yaw)
   row['block_velocity_m_s']=m.data.qvel[m.model.jnt_dofadr[m.model.joint('task_block_free').id]:][:3].tolist()
   rows.append(row)
   if c.phase!=oldphase:print(c.phase,round(row['time_s'],2),'pose',np.round(row['hand_m'],3),'block',np.round(row['block_m'],3),'estimate',c.estimate,flush=True);oldphase=c.phase
   if render and step%10==0:
    from PIL import Image,ImageDraw
    canvas=Image.new('RGB',(960,440),'#102027');canvas.paste(Image.fromarray(m.render()),(0,0));canvas.paste(Image.fromarray(c.rgb),(640,0));draw=ImageDraw.Draw(canvas)
    draw.text((650,195),'RGB known-orange-cube detection',fill='white');draw.text((650,222),'STATE: '+c.phase,fill='white');draw.text((650,249),f"Estimated x: {obs['world_x_m']:.3f} m",fill='white');draw.text((15,380),'Free-body floor pickup | Feedback reference controller | Provisional longer upper arms',fill='white');draw.text((15,404),'1.0 Nm leg capacity is unmeasured. Object truth is evaluator-only. No welds.',fill='white')
    frames.append(np.asarray(canvas))
   if c.phase=='done' or abs(row['pitch_rad'])>.45:break
  final=rows[-1];lift=max(r['block_m'][2]-.015 for r in rows)
  placed=abs(final['block_m'][0]-.365)<.06 and abs(final['block_m'][1]-.14)<.06 and abs(final['block_m'][2]-.015)<.004
  assessment=evaluate_cleanup(rows)
  report={'evaluation':assessment,'status':'pass' if assessment['success'] else 'fail','scope':'Feedback reference controller; RGB known-cube floor-plane perception and remembered pose; no learned cleanup policy yet. Revised upper-arm envelope and assumed 1.0 Nm leg cap.','final_phase':c.phase,'duration_s':final['time_s'],'max_lift_m':lift,'placed':placed,'final':final,'finger_friction':friction}
  (out/'trajectory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
  if frames:
   import imageio.v2 as iio
   iio.mimwrite(out/'replay.mp4',frames,fps=5,macro_block_size=1)
 except Exception as exc:
  import traceback
  report={'status':'fail','error':str(exc)};(out/'error.log').write_text(traceback.format_exc());raise
 finally:
  c.close();m.close();(out/'report.json').write_text(json.dumps(report,indent=2))
  for rel in ['lab/floor_pickup.py','tools/run_floor_cleanup.py','lab/mujoco_learning.py','lab/electromechanical.py','variants/pip-floor-reach-v1.yaml']:
   snapshot_file(ROOT/rel,out/'source'/rel)
  (out/'dependencies.json').write_text(json.dumps(retain_dependency(ROOT/'artifacts/pip-floor-reach-model-v1'),indent=2))
  p=plan_migration(ROOT,[out]);apply_migration(ROOT,p);verify_manifest(ROOT,p)
 print({k:v for k,v in report.items() if k!='final'},flush=True)
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--render',action='store_true');p.add_argument('--friction',type=float,default=.8);a=p.parse_args();run(a.render,a.friction)
