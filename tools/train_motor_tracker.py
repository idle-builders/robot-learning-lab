"""Bounded native PPO training; fixed motor tests, checkpoint and failures retained."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from lab.learned_motor import MotorTrackingEnv
from lab.mujoco_learning import digest
from lab.artifact_store import snapshot_file,retain_dependency,plan_migration,apply_migration,verify_manifest

class Budget(BaseCallback):
    def __init__(self, seconds, path):
        super().__init__()
        self.deadline = time.monotonic()+seconds
        self.path = path

    def _on_step(self):
        if self.num_timesteps % 1024 == 0:
            with self.path.open('a') as f:
                row = {'transitions':self.num_timesteps,'optimizer_epochs':self.model._n_updates,
                       'mean_recent_return':float(np.mean([x['r'] for x in self.model.ep_info_buffer]))
                         if self.model.ep_info_buffer else None}
                f.write(json.dumps(row)+'\n')
            print(row,flush=True)
        return time.monotonic() < self.deadline

def fixed_cases(interface):
    hold = interface.goal.copy()
    forward = hold.copy(); forward[14] = .02
    bend = hold.copy()
    for side in ['left','right']:
        bend[interface.contract()['goal_names'].index(side+'_thigh_joint.position_rad')] = -.15
        bend[interface.contract()['goal_names'].index(side+'_shin_joint.position_rad')] = .3
    return {'hold':hold,'forward':forward,'bend':bend}

def evaluate(policy,scene,out,cases,render=False):
    results = {}
    for name,goal in cases.items():
        env = MotorTrackingEnv(scene)
        dest = out/name; dest.mkdir()
        obs,_ = env.reset(seed=8201,options={'goal':goal})
        rows=[]; decisions=[]; errors=[]; frames=[]
        try:
            while True:
                start=time.perf_counter()
                action = policy.predict(obs,deterministic=True)[0]
                elapsed=time.perf_counter()-start
                decisions.append({'time_s':float(env.motor.data.time),'observation':obs.tolist(),
                                  'action':action.tolist(),'inference_s':elapsed})
                obs,reward,terminal,truncated,info=env.step(action)
                rows.append(env.row);errors.append(info)
                if render and len(rows)%5==0:
                    frames.append(env.motor.render())
                if terminal or truncated:break
            full=len(rows)==env.horizon
            checks={'full_duration':full,'no_fall':not any(x['fallen'] for x in errors),
                    'no_overspeed':not any(x['overspeed'] for x in errors),
                    'posture_rmse_below_0_10rad':bool(np.sqrt(np.mean([x['posture_rmse_rad']**2 for x in errors]))<.1),
                    'velocity_rmse_below_0_02m_s':bool(np.sqrt(np.mean([x['velocity_error_m_s']**2 for x in errors]))<.02)}
            result={'success':all(checks.values()),'checks':checks,'duration_s':rows[-1]['time_s'],
                    'posture_rmse_rad':float(np.sqrt(np.mean([x['posture_rmse_rad']**2 for x in errors]))),
                    'velocity_rmse_m_s':float(np.sqrt(np.mean([x['velocity_error_m_s']**2 for x in errors]))),
                    'mean_inference_s':float(np.mean([x['inference_s'] for x in decisions]))}
            (dest/'report.json').write_text(json.dumps(result,indent=2))
            for filename,values in [('trajectory.jsonl',rows),('decisions.jsonl',decisions)]:
                (dest/filename).write_text(''.join(json.dumps(x)+'\n' for x in values))
            if frames:
                import imageio.v2 as iio
                iio.mimwrite(dest/'replay.mp4',frames,fps=10,macro_block_size=1)
            results[name]=result
            print(name,result,flush=True)
        finally:env.close()
    return results

def run(steps,seconds,render,parent_checkpoint=None):
    torch.set_num_threads(2)
    parent=ROOT/'artifacts/vla-demonstrations-d6140c7e55aa'
    out=ROOT/'artifacts'/('motor-tracker-'+uuid.uuid4().hex[:12]);out.mkdir()
    print(out,flush=True)
    report={'status':'running','trainer':'native Stable Baselines3 PPO',
            'scope':'All-actuator reference tracker; no analytic balance/feedforward. Not task reasoning, VLA RL or qualified WBC.',
            'training_budget_s':seconds,'requested_transitions':steps,'task_success':'not_run'}
    env=None;start=time.monotonic()
    try:
        for name in ['lab/learned_motor.py','lab/floor_pickup.py','lab/mujoco_learning.py',
                     'lab/electromechanical.py','tools/train_motor_tracker.py','lab/vla_control.py',
                     'learning/pip-control-v1.json','learning/pip-electromechanical-v1.json']:
            snapshot_file(ROOT/name,out/'source'/name)
        snapshot_file(parent/'scene.xml',out/'scene.xml')
        env=MotorTrackingEnv(out/'scene.xml')
        obs,_=env.reset(seed=811)
        contract=env.interface.contract()
        (out/'contract.json').write_text(json.dumps(contract,indent=2))
        env.interface.reset(); cases=fixed_cases(env.interface)
        (out/'evaluation-cases.json').write_text(json.dumps({k:v.tolist() for k,v in cases.items()},indent=2))
        if parent_checkpoint is not None:
            if json.loads((parent_checkpoint.parent/'contract.json').read_text()) != contract:
                raise ValueError('Parent motor contract differs from current plant/interface')
            policy=PPO.load(parent_checkpoint,env=env,device='cpu')
            report['parent_checkpoint']={'path':str(parent_checkpoint),'sha256':digest(parent_checkpoint)}
        else:
            policy=PPO('MlpPolicy',env,n_steps=512,batch_size=128,n_epochs=5,learning_rate=3e-4,
                   gamma=.99,gae_lambda=.95,ent_coef=.001,
                   policy_kwargs={'net_arch':[64,64],'log_std_init':-1.},seed=811,device='cpu',verbose=0)
        baseline=out/'before';baseline.mkdir()
        report['before']=evaluate(policy,out/'scene.xml',baseline,cases)
        train_start=time.monotonic()
        initial_transitions=policy.num_timesteps
        policy.learn(total_timesteps=steps,callback=Budget(seconds,out/'learning.jsonl'),reset_num_timesteps=False)
        report.update(training_wall_s=time.monotonic()-train_start,actual_transitions=policy.num_timesteps-initial_transitions,
                      cumulative_transitions=policy.num_timesteps,
                      optimizer_epochs=policy._n_updates)
        policy.save(out/'policy.zip')
        probe=env.reset(seed=899)[0]
        restored=PPO.load(out/'policy.zip',device='cpu')
        report['reload_action_match']=bool(np.array_equal(policy.predict(probe,deterministic=True)[0],restored.predict(probe,deterministic=True)[0]))
        report['checkpoint_sha256']=digest(out/'policy.zip')
        after=out/'after';after.mkdir()
        report['after']=evaluate(restored,out/'scene.xml',after,cases,render)
        report['status']='training_and_evaluation_executed'
    except Exception as error:
        import traceback
        report.update(status='fail',error=str(error))
        (out/'error.log').write_text(traceback.format_exc())
        raise
    finally:
        if env:env.close()
        report['wall_s']=time.monotonic()-start
        (out/'report.json').write_text(json.dumps(report,indent=2))
        dependencies={'demonstrations':retain_dependency(parent),
            'plant':retain_dependency(ROOT/'artifacts/pip-cad-gym-bundle-v4')}
        if parent_checkpoint is not None:
            dependencies['parent_motor']=retain_dependency(parent_checkpoint.parent)
        (out/'dependencies.json').write_text(json.dumps(dependencies,indent=2))
        (out/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        plan=plan_migration(ROOT,[out]);apply_migration(ROOT,plan);verify_manifest(ROOT,plan)
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=32768)
    parser.add_argument('--seconds',type=int,default=180)
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--parent',type=Path)
    args=parser.parse_args()
    if not 512<=args.steps<=131072 or not 10<=args.seconds<=600:
        parser.error('Bounded run: 512–131072 transitions, 10–600 training seconds')
    run(args.steps,args.seconds,args.render,args.parent)
