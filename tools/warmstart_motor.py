"""Bounded native supervised warm start; saved PPO weights, physical evaluation."""
import argparse,json,subprocess,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import mujoco
from stable_baselines3 import PPO
from lab.learned_motor import MotorTrackingEnv
from lab.motor_demonstrations import teacher_action,copy_regressor_to_actor
from lab.artifact_store import snapshot_file,retain_dependency,plan_migration,apply_migration,verify_manifest
from tools.train_motor_tracker import evaluate,fixed_cases

SCENE=ROOT/'artifacts/vla-demonstrations-d6140c7e55aa/scene.xml'

def finalize(out):
    p=plan_migration(ROOT,[out]);apply_migration(ROOT,p);verify_manifest(ROOT,p)

def collect(episodes):
    out=ROOT/'artifacts'/('motor-demonstrations-'+uuid.uuid4().hex[:12]);out.mkdir();print(out,flush=True)
    observations=[];actions=[];episode_ids=[];episodes_report=[]
    env=MotorTrackingEnv(SCENE,horizon=400)
    try:
        for episode in range(episodes):
            obs,_=env.reset(seed=9100+episode,options={'goal':np.r_[env.motor.rest[env.interface.positions],0.,0.,0.]})
            rng=np.random.default_rng(9100+episode)
            base=env.interface.goal.copy();goal=base.copy()
            if episode%4:
                for j,name in enumerate(env.interface.contract()['goal_names'][:14]):
                    if 'shin' in name:goal[j]=.15*(episode%5)
                    elif 'thigh' in name:goal[j]=-.075*(episode%5)
                    elif 'gripper' not in name:goal[j]+=rng.uniform(-.35,.35)
                goal[14:16]=rng.uniform([-.04,-.08],[.04,.08])
            pitch=rng.uniform(-.02,.02)
            env.motor.data.qpos[3:7]=[np.cos(pitch/2),0,np.sin(pitch/2),0]
            mujoco.mj_forward(env.motor.model,env.motor.data)
            episode_start=len(actions)
            for step in range(400):
                if step%5==0:
                    command=base+(goal-base)*min(1.,step/100)
                    command[16]=np.clip(env.motor.reference_x-env.motor.motion()[3],-.12,.12)
                    env.interface.set_goal(command)
                obs=env.interface.observe()
                action=teacher_action(env.interface)
                observations.append(obs);actions.append(action);episode_ids.append(episode)
                _,_,term,trunc,info=env.step(action)
                if term or trunc:break
            result={'episode':episode,'seed':9100+episode,'samples':len(actions)-episode_start,
                    'duration_s':float(env.motor.data.time),'goal':goal.tolist(),
                    'fallen':info['fallen'],'overspeed':info['overspeed'],
                    'posture_rmse_final_rad':info['posture_rmse_rad'],'velocity_error_final_m_s':info['velocity_error_m_s']}
            episodes_report.append(result);print(result,flush=True)
        with (out/'demonstrations.npz').open('xb') as f:
            np.savez_compressed(f,obs=np.asarray(observations),acts=np.asarray(actions),episodes=np.asarray(episode_ids))
        (out/'contract.json').write_text(json.dumps(env.interface.contract(),indent=2))
        (out/'report.json').write_text(json.dumps({'method':'Authored balance teacher mapped into learned motor command interface; training labels only',
            'episodes':episodes_report,'samples':len(actions),'held_out_episode_ids':list(range(episodes-4,episodes)),
            'sample_includes_failures':True,'rl_updates':0},indent=2))
    finally:
        env.close()
        for name in ['tools/warmstart_motor.py','lab/motor_demonstrations.py','lab/learned_motor.py','lab/floor_pickup.py']:
            snapshot_file(ROOT/name,out/'source'/name)
        (out/'dependencies.json').write_text(json.dumps({'scene':retain_dependency(SCENE.parent)},indent=2))
        finalize(out)
    return out

def fit(data,iterations):
    from sklearn.neural_network import MLPRegressor
    from threadpoolctl import threadpool_limits
    import joblib
    out=ROOT/'artifacts'/('motor-warmstart-'+uuid.uuid4().hex[:12]);out.mkdir();print(out,flush=True)
    env=MotorTrackingEnv(SCENE)
    report={'status':'running','method':'scikit-learn MLP supervised warm start → native PPO actor','actual_transitions':0,'rl_updates':0}
    started=time.monotonic()
    try:
        env.reset(seed=1)
        contract=json.loads((data/'contract.json').read_text())
        if env.interface.contract()!=contract:raise ValueError('Demonstration motor contract mismatch')
        (out/'contract.json').write_text(json.dumps(contract,indent=2));snapshot_file(SCENE,out/'scene.xml')
        env.interface.reset();cases=fixed_cases(env.interface)
        (out/'evaluation-cases.json').write_text(json.dumps({k:v.tolist() for k,v in cases.items()},indent=2))
        dataset=np.load(data/'demonstrations.npz',allow_pickle=False)
        meta=json.loads((data/'report.json').read_text())
        test=np.isin(dataset['episodes'],meta['held_out_episode_ids'])
        x,y=dataset['obs'],dataset['acts']
        model=MLPRegressor(hidden_layer_sizes=(64,64),activation='tanh',solver='adam',batch_size=256,
            learning_rate_init=.001,max_iter=iterations,tol=1e-7,n_iter_no_change=25,random_state=914)
        with threadpool_limits(limits=2):model.fit(x[~test],y[~test])
        joblib.dump(model,out/'regressor.joblib')
        report.update(supervised_epochs=model.n_iter_,supervised_loss_curve=model.loss_curve_,
            held_out_action_rmse=np.sqrt(np.mean((model.predict(x[test])-y[test])**2,axis=0)).tolist(),
            training_samples=int((~test).sum()),held_out_samples=int(test.sum()))
        policy=PPO('MlpPolicy',env,n_steps=512,batch_size=128,n_epochs=5,learning_rate=1e-4,
            gamma=.99,gae_lambda=.95,ent_coef=0.,policy_kwargs={'net_arch':[64,64],'log_std_init':-3.},seed=914,device='cpu')
        copy_regressor_to_actor(model,policy.policy)
        probe=x[test][:256]
        expected=np.clip(model.predict(probe),-1,1)
        actual=policy.predict(probe,deterministic=True)[0]
        report['conversion_max_action_error']=float(np.max(abs(actual-expected)))
        if report['conversion_max_action_error']>1e-5:raise ValueError('Regressor → PPO action mismatch')
        policy.save(out/'policy.zip')
        restored=PPO.load(out/'policy.zip',device='cpu')
        report['reload_action_match']=bool(np.array_equal(actual,restored.predict(probe,deterministic=True)[0]))
        after=out/'after';after.mkdir()
        report['after']=evaluate(restored,SCENE,after,cases,render=True)
        report['status']='supervised_warmstart_and_evaluation_executed'
    except Exception as exc:
        import traceback
        report.update(status='fail',error=str(exc));(out/'error.log').write_text(traceback.format_exc());raise
    finally:
        env.close();report['wall_s']=time.monotonic()-started
        (out/'report.json').write_text(json.dumps(report,indent=2))
        for name in ['tools/warmstart_motor.py','lab/motor_demonstrations.py','lab/learned_motor.py','tools/train_motor_tracker.py']:
            snapshot_file(ROOT/name,out/'source'/name)
        (out/'dependencies.json').write_text(json.dumps({'demonstrations':retain_dependency(data),'scene':retain_dependency(SCENE.parent)},indent=2))
        (out/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        finalize(out)
    return out

if __name__=='__main__':
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['collect','fit'])
    parser.add_argument('--data',type=Path)
    parser.add_argument('--episodes',type=int,default=24)
    parser.add_argument('--iterations',type=int,default=200)
    args=parser.parse_args()
    if not 8<=args.episodes<=48 or not 1<=args.iterations<=300:parser.error('Bounded collection/fit required')
    if args.mode=='collect':collect(args.episodes)
    elif args.data is None:parser.error('fit requires --data')
    else:fit(args.data,args.iterations)
