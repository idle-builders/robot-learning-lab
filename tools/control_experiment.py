"""Train, reload, evaluate and inspect Pip with native MuJoCo and SB3.

Example: .venv-learning/bin/python tools/control_experiment.py --mode adapt --steps 2048
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from lab.mujoco_learning import PipMotorEnv, BUNDLE, ISAAC_CHECKPOINT, isaac_actor, digest
from lab.artifact_store import snapshot_file, plan_migration, apply_migration, verify_manifest
from lab.control_assets import verify_control_bundle

CASES={'hold':[0.,0.,0.],'forward':[.035,0.,0.], 'reverse':[-.025,0.,0.], 'turn':[0.,0.,.1]}


def warm_start(model):
    """Copy the existing trained motor actor/critic into SB3 without changing outputs."""
    state=torch.load(ISAAC_CHECKPOINT,map_location='cpu',weights_only=True)
    for kind, destination, head in [('actor',model.policy.mlp_extractor.policy_net,model.policy.action_net),
                                    ('critic',model.policy.mlp_extractor.value_net,model.policy.value_net)]:
        weights=state[kind+'_state_dict']
        for layer in (0,2):
            destination[layer].load_state_dict({k:weights[f'mlp.{layer}.{k}'] for k in ('weight','bias')})
        head.load_state_dict({k:weights[f'mlp.4.{k}'] for k in ('weight','bias')})
    with torch.no_grad(): model.policy.log_std.copy_(state['actor_state_dict']['distribution.std_param'].clamp_min(1e-6).log())
    # Probe away from zero as well: a shape match alone does not establish parity.
    observations=np.random.default_rng(17).normal(0,.1,(7,47)).astype(np.float32)
    expected=np.clip(isaac_actor()(observations),-1,1)
    actual=model.predict(observations,deterministic=True)[0]
    error=float(abs(expected-actual).max())
    if error>1e-6: raise ValueError('Warm-start actor changed output')
    return error


def evaluate(policy, output, *, case_commands=None, render=False):
    cases={}; trace=[]; frames=[]
    for index,(name,command) in enumerate((case_commands or CASES).items()):
        env=PipMotorEnv(command=command);obs,_=env.reset(seed=1000+index)
        if hasattr(policy,'reset'): policy.reset()
        rows=[]; rewards=[]; errors=[]; yaw_errors=[]
        for step in range(400):
            obs,reward,terminated,truncated,info=env.step(policy(obs))
            rewards.append(reward);errors.append(info['tracking_error_m_s'])
            yaw_errors.append(env.motion()[2][2]-command[2])
            if step%5==0 or terminated or truncated:
                rows.append(env.record())
                if render and index==0: frames.append(env.render())
            if terminated or truncated: break
        cases[name]={'seed':1000+index,'command':command,'steps':step+1,'terminated':terminated,
            'truncated':truncated,'fall':info['fallen'],'overspeed':info['overspeed'],
            'return':float(sum(rewards)),'tracking_rmse_m_s':float(np.sqrt(np.mean(np.square(errors)))),
            'yaw_rmse_rad_s':float(np.sqrt(np.mean(np.square(yaw_errors)))),
            'energy_j':env.pack.energy_j,'elapsed_sim_s':float(env.data.time),
            'success':bool(not terminated and step==399 and np.sqrt(np.mean(np.square(errors)))<.035 and np.sqrt(np.mean(np.square(yaw_errors)))<.08)}
        (output/(name+'-trace.json')).write_text(json.dumps(rows))
        if index==0: trace=rows
        env.close()
    if frames:
        import imageio.v2 as imageio
        imageio.mimwrite(output/'replay.mp4',frames,fps=10,codec='libx264',quality=7,macro_block_size=1)
    return cases,trace


def export_rerun(rows, output, scope):
    from lab.rerun_scene import export_scene
    env=PipMotorEnv()
    names=[env.model.body(i).name for i in range(1,env.model.nbody)]
    env.close()
    for row in rows:
        row['bodies']={n:{'position_m':p[:3],'quaternion_xyzw':p[3:]} for n,p in zip(names,row['body_poses_world_xyz_xyzw'],strict=True)}
        row['velocity']=[row['forward_m_s'],0.,0.]
    export_scene(BUNDLE/'model/robot.xml',output/'scene.json')
    (output/'replay.json').write_text(json.dumps({'rows':rows,'interface':env.interface,'scope':scope,'outcome':'See report.json; failures are retained.'}))
    with (output/'rerun-export.log').open('w') as log:
        subprocess.run([str(ROOT/'.venv-observe/bin/python'),'tools/write_control_rrd.py',str(output)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        subprocess.run([str(ROOT/'.venv-observe/bin/rerun'),'rrd','verify',str(output/'episode.rrd')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=30)


def run(mode,steps=2048,checkpoint=None,render=True,command=None):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    torch.set_num_threads(2)
    if mode not in {'reference','isaac','adapt','resume','evaluate'}:raise ValueError('Unsupported experiment')
    if not 128<=steps<=8192 or steps%128:raise ValueError('Use 128–8192 steps in multiples of 128')
    if mode in {'resume','evaluate'}:
        # Artifact files resolve into CAS; validate their logical retained path.
        checkpoint=Path(os.path.abspath(checkpoint))
        if not checkpoint.is_file() or ROOT/'artifacts' not in checkpoint.parents: raise ValueError('Choose a retained artifact checkpoint')
    if command is not None: PipMotorEnv.validate_command(command)
    output=ROOT/'artifacts'/('control-stack-'+mode+'-'+uuid.uuid4().hex[:12]);output.mkdir()
    report={'schema_version':1,'status':'running','backend':'MuJoCo CPU + Stable-Baselines3','mode':mode,'training_steps':0,
        'seed':23,'physics_hz':500,'control_hz':50,'quaternion':'xyzw in recordings; wxyz in MuJoCo',
        'observation':{'q_rad':[0,16],'dq_rad_s':[16,32],'gravity_body':[32,35],'gyro_body_rad_s':[35,38],
            'motion_command':[38,41],'odometry_error_m':[41,42],'odometry_speed_m_s':[42,43],'previous_residual':[43,47]},
        'action':'4 normalized residuals; left/right wheel ±0.5 rad/s, symmetric hip/knee ±0.025 rad; final mixed commands bounded',
        'scope':'Small existing-motor adaptation pilot. Same sealed CAD/mass/collision input; native solver differences unqualified. No manipulation or physical-readiness claim.',
        'robot_bundle':str(BUNDLE.relative_to(ROOT)),'bundle_manifest_sha256':digest(BUNDLE/'manifest.json'),
        'parent_checkpoint':str(checkpoint or ISAAC_CHECKPOINT) if mode!='reference' else None,
        'parent_checkpoint_sha256':digest(checkpoint or ISAAC_CHECKPOINT) if mode!='reference' else None}
    started=time.monotonic(); env=None
    try:
        report['source_verification']=verify_control_bundle(BUNDLE)
        source=output/'source';source.mkdir()
        for path in [Path(__file__),ROOT/'lab/mujoco_learning.py',ROOT/'lab/electromechanical.py',ROOT/'lab/control_assets.py',ROOT/'learning/pip-control-v1.json',ROOT/'learning/pip-electromechanical-v1.json']:
            snapshot_file(path,source/path.name)
        # A manifest reference keeps large meshes in their existing CAS objects.
        snapshot_file(BUNDLE/'manifest.json',source/'bundle-manifest.json')
        (output/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        env=PipMotorEnv();check_env(env)
        if mode=='reference': policy=lambda observation:np.zeros(4,dtype=np.float32)
        elif mode=='isaac':policy=isaac_actor()
        else:
            if mode in {'resume','evaluate'}:model=PPO.load(checkpoint,env=env,device='cpu')
            else:
                model=PPO('MlpPolicy',env,n_steps=128,batch_size=64,n_epochs=3,learning_rate=1e-4,
                    policy_kwargs={'net_arch':{'pi':[64,64],'vf':[64,64]},'activation_fn':torch.nn.ELU},seed=23,device='cpu',verbose=0)
                report['warm_start_max_action_error']=warm_start(model)
            if mode!='evaluate':
                begin=time.monotonic();model.learn(total_timesteps=steps,reset_num_timesteps=(mode!='resume'))
                report['training_seconds']=time.monotonic()-begin;report['training_steps']=steps
            model.save(output/'policy.zip')
            before=model.predict(env.reset(seed=991)[0],deterministic=True)[0]
            model=PPO.load(output/'policy.zip',device='cpu')
            after=model.predict(env.reset(seed=991)[0],deterministic=True)[0]
            report['reload_max_action_error']=float(abs(before-after).max())
            if report['reload_max_action_error']>1e-7:raise ValueError('Checkpoint reload changed action')
            policy=lambda obs:model.predict(obs,deterministic=True)[0]
            report['checkpoint_sha256']=digest(output/'policy.zip')
        report['cases'],rows=evaluate(policy,output,render=render,case_commands={'custom':command} if command is not None else None)
        report['status']='pass' if all(c['success'] for c in report['cases'].values()) else 'fail'
        report['execution_status']='pass'
        if render:export_rerun(rows,output,report['scope'])
    except Exception as error:
        report['status']='fail';report['execution_status']='fail';report['error']=type(error).__name__+': '+str(error)
        raise
    finally:
        if env is not None:env.close()
        report['elapsed_seconds']=time.monotonic()-started
        (output/'report.json').write_text(json.dumps(report,indent=2))
        manifest=plan_migration(ROOT,[output]);apply_migration(ROOT,manifest);verify_manifest(ROOT,manifest)
        print(output,flush=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['reference','isaac','adapt','resume','evaluate'],default='isaac')
    parser.add_argument('--steps',type=int,default=2048)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--no-render',action='store_true')
    parser.add_argument('--forward',type=float)
    parser.add_argument('--yaw',type=float,default=0.)
    args=parser.parse_args()
    run(args.mode,args.steps,args.checkpoint,not args.no_render,None if args.forward is None else [args.forward,0.,args.yaw])
