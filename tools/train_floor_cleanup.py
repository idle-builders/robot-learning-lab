"""Short native PPO grasp adaptation, saved/reloaded, fixed independent evaluation."""
import argparse,json,sys,time,uuid,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from lab.floor_pickup import build_scene
from lab.floor_learning import FloorCleanupEnv
from lab.mujoco_learning import digest
from lab.artifact_store import snapshot_file,retain_dependency,plan_migration,apply_migration,verify_manifest

CASES={'nominal':{'object_x':.015,'object_y':.140},'shift_x':{'object_x':.019,'object_y':.140},'shift_y':{'object_x':.015,'object_y':.143}}
SCOPE='Camera-driven feedback cleanup with native SB3 PPO grasp residual. Authored task order, joint references, balance and floor-plane orange-cube vision; PPO changes gripper effort allowance and forearm offset. 15 mm longer upper arms are provisional envelopes; 1.0 Nm legs, motor dynamics and contacts are unmeasured assumptions. No hardware qualification.'

class Progress(BaseCallback):
 def _on_step(self):
  if self.num_timesteps%128==0:print('PPO transitions',self.num_timesteps,flush=True)
  return True

def evaluate(policy,scene,out,case,options,render=False,reference=False):
 from PIL import Image,ImageDraw
 env=FloorCleanupEnv(scene,capture=render);obs,_=env.reset(seed=8100,options=options);frames=[];dest=out/case;dest.mkdir();decisions=[]
 try:
  for step in range(300):
   tensor=torch.as_tensor(obs[None],dtype=torch.float32)
   with torch.no_grad():value=float(policy.policy.predict_values(tensor).item()) if policy is not None else None
   action=np.array([1.,0.]) if reference else policy.predict(obs,deterministic=True)[0]
   decision={'time_s':float(env.motor.data.time),'observation':obs.tolist(),'action':action.tolist(),'predicted_return':value}
   obs,reward,t,tr,info=env.step(action);decisions.append({**decision,'reward':reward})
   row=env.rows[-1];row.update(predicted_return=value)
   if render:
    camera=Image.fromarray(env.controller.rgb);camera.save(dest/f'camera-{step:03}.png');row['image']=f'{case}/camera-{step:03}.png'
    canvas=Image.new('RGB',(960,440),'#102027');canvas.paste(Image.fromarray(env.motor.render()),(0,0));canvas.paste(camera,(640,0));draw=ImageDraw.Draw(canvas)
    draw.text((650,195),'PHASE: '+row['phase'],fill='white');draw.text((650,223),f"PPO grip allowance: {row['gripper_effort_limit_nm']:.3f} Nm",fill='white');draw.text((650,249),f"PPO forearm residual: {row['forearm_residual_rad']:+.3f} rad",fill='white');draw.text((650,275),f"Voltage proxy: {row['electromechanical']['bus_voltage_v']:.2f} V",fill='white')
    draw.text((15,378),'FLOOR CLEANUP | '+case+' | '+('feedback reference' if reference else 'PPO grasp residual + feedback skill controller'),fill='white');draw.text((15,404),'Free object; contact grasp. Purple upper arms are new envelopes; 1.0 Nm leg cap unmeasured.',fill='white');frames.append(np.asarray(canvas))
   if t or tr:break
  (dest/'trajectory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in env.rows));(dest/'decisions.json').write_text(json.dumps(decisions))
  result=info.get('evaluation',{'success':False,'reason':'runner_horizon'});result.update(duration_s=float(env.motor.data.time),final_phase=env.controller.phase,initial=env.initial.tolist(),reference=reference)
  (dest/'report.json').write_text(json.dumps(result,indent=2))
  if frames:
   import imageio.v2 as iio
   iio.mimwrite(dest/'replay.mp4',frames,fps=5,macro_block_size=1)
  print(case,result,flush=True)
  return result
 finally:env.close()

def run(steps=1024,parent=None,render=True):
 torch.set_num_threads(2);out=ROOT/'artifacts'/('floor-learning-'+uuid.uuid4().hex[:12]);out.mkdir();print(out,flush=True)
 report={'status':'running','scope':SCOPE,'training_steps':steps,'cases':{}};env=None;start=time.monotonic()
 try:
  for rel in ['lab/floor_pickup.py','lab/floor_learning.py','tools/train_floor_cleanup.py','lab/mujoco_learning.py','lab/electromechanical.py','learning/pip-control-v1.json','learning/pip-electromechanical-v1.json','variants/pip-floor-reach-v1.yaml']:
   snapshot_file(ROOT/rel,out/'source'/rel)
  build_scene(out/'scene.xml',source=(.015,.14,.015),block_size=.03,robot_model=ROOT/'artifacts/pip-floor-reach-model-v1/robot.xml')
  env=FloorCleanupEnv(out/'scene.xml')
  policy=PPO.load(parent,env=env,device='cpu') if parent else PPO('MlpPolicy',env,n_steps=256,batch_size=64,n_epochs=6,learning_rate=.0003,gamma=.98,ent_coef=.002,policy_kwargs={'net_arch':[32,32],'log_std_init':-1.2},seed=91,device='cpu',verbose=0)
  if parent:report['parent']={'path':str(parent),'sha256':digest(parent)}
  else:
   # Initial mean is the executed reference (full allowed grip, zero offset).
   # This is declared reference initialization, not foundation-model pretraining.
   with torch.no_grad():policy.policy.action_net.bias[0]=1.
  report['baseline']=evaluate(None,out/'scene.xml',out,'reference',CASES['nominal'],reference=True)
  train_start=time.monotonic();policy.learn(total_timesteps=steps,callback=Progress());report['training_wall_s']=time.monotonic()-train_start
  policy.save(out/'policy.zip');probe=env.reset(seed=889)[0];expected=policy.predict(probe,deterministic=True)[0]
  restored=PPO.load(out/'policy.zip',device='cpu');report['reload_action_match']=bool(np.array_equal(expected,restored.predict(probe,deterministic=True)[0]));report['checkpoint_sha256']=digest(out/'policy.zip')
  env.close();env=None
  for case,options in CASES.items():report['cases'][case]=evaluate(restored,out/'scene.xml',out,case,options,render)
  build_scene(out/'frictionless.xml',source=(.015,.14,.015),block_size=.03,friction=0.,robot_model=ROOT/'artifacts/pip-floor-reach-model-v1/robot.xml')
  report['negative_control']=evaluate(restored,out/'frictionless.xml',out,'frictionless',CASES['nominal'],False)
  report['execution_status']='pass';report['status']='pass' if all(c['success'] for c in report['cases'].values()) and not report['negative_control']['success'] else 'fail'
 except Exception as exc:
  import traceback
  report.update(status='fail',error=str(exc));(out/'error.log').write_text(traceback.format_exc());raise
 finally:
  if env:env.close()
  report['wall_s']=time.monotonic()-start;(out/'report.json').write_text(json.dumps(report,indent=2))
  (out/'dependencies.json').write_text(json.dumps(retain_dependency(ROOT/'artifacts/pip-floor-reach-model-v1'),indent=2))
  (out/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
  p=plan_migration(ROOT,[out]);apply_migration(ROOT,p);verify_manifest(ROOT,p)
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=1024);p.add_argument('--parent',type=Path);p.add_argument('--no-render',action='store_true');a=p.parse_args()
 if not 0<=a.steps<=4096 or a.steps%256:p.error('0–4096 steps, multiples of 256')
 run(a.steps,a.parent,not a.no_render)
