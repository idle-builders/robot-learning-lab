"""Bounded LeRobot ACT distillation of the existing Pip motor policy.

This is a small state-only imitation exercise, not a pretrained visual policy or
base-model pretraining. Native LeRobot datasets, chunks, loss and checkpoints.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from lab.mujoco_learning import PipMotorEnv, isaac_actor, ISAAC_CHECKPOINT, BUNDLE, digest
from lab.artifact_store import snapshot_file, plan_migration, apply_migration, verify_manifest
from tools.control_experiment import evaluate, export_rerun
from lab.control_assets import verify_control_bundle


def collect(root):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset=LeRobotDataset.create('local/pip-motor',fps=50,root=root,robot_type='pip-wheeled-v4',use_videos=False,
        features={'observation.state':{'dtype':'float32','shape':(47,),'names':None},
            'observation.environment_state':{'dtype':'float32','shape':(3,),'names':['forward_m_s','lateral_m_s','yaw_rad_s']},
            'action':{'dtype':'float32','shape':(4,),'names':['left_wheel_residual','right_wheel_residual','hip_residual','knee_residual']}})
    teacher=isaac_actor();env=PipMotorEnv();episodes=[]
    try:
        for episode in range(10):
            obs,_=env.reset(seed=120+episode)
            for step in range(400):
                action=np.clip(teacher(obs),-1,1).astype(np.float32)
                dataset.add_frame({'observation.state':obs,'observation.environment_state':obs[38:41].copy(),
                    'action':action,'task':'Follow the bounded velocity command while balancing.'})
                obs,reward,terminated,truncated,info=env.step(action)
                if terminated or truncated: break
            dataset.save_episode()
            episodes.append({'episode':episode,'seed':120+episode,'frames':step+1,'terminated':terminated,
                'split':'train' if episode<8 else 'validation','outcome':info})
    finally:dataset.finalize();env.close()
    return episodes


class ACTMotor:
    def __init__(self,policy):self.policy=policy;self.reset()
    def reset(self):self.policy.reset()
    def __call__(self,obs):
        with torch.inference_mode():
            return self.policy.select_action({'observation.state':torch.as_tensor(obs).unsqueeze(0),
                'observation.environment_state':torch.as_tensor(obs[38:41]).unsqueeze(0)}).squeeze(0).cpu().numpy()


def run(steps=100, checkpoint=None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.configs.types import FeatureType,PolicyFeature,NormalizationMode
    if not 1<=steps<=500:raise ValueError('Pilot supports 1–500 optimizer steps')
    if checkpoint is not None:
        checkpoint=Path(checkpoint).resolve()
        if ROOT/'artifacts' not in checkpoint.parents or not (checkpoint/'model.safetensors').is_file():
            raise ValueError('Choose a retained LeRobot pretrained_model folder')
    torch.set_num_threads(2);torch.manual_seed(31);np.random.seed(31)
    output=ROOT/'artifacts'/('control-stack-lerobot-'+uuid.uuid4().hex[:12]);output.mkdir()
    report={'schema_version':1,'status':'running','backend':'LeRobot ACT + native MuJoCo',
        'mode':'act-resume' if checkpoint else 'act-distill','training_steps':steps,'seed':31,
        'robot_bundle':str(BUNDLE.relative_to(ROOT)),'bundle_manifest_sha256':digest(BUNDLE/'manifest.json'),
        'teacher_checkpoint_sha256':digest(ISAAC_CHECKPOINT),
        'scope':'Small state-only motor-policy distillation. No visual foundation-model fine-tuning, manipulation or imagined RL. 10-action prediction chunks; replan every 20 ms.',
        'normalization':'identity; same SI 47-vector as MuJoCo/RSL. environment_state duplicates the 3 commanded velocities, not privileged simulator truth.',
        'optimizer_resume':'fresh AdamW; policy weights restored' if checkpoint else None,
        'parent_checkpoint':str(checkpoint) if checkpoint else str(ISAAC_CHECKPOINT)}
    started=time.monotonic()
    try:
        report['source_verification']=verify_control_bundle(BUNDLE)
        source=output/'source';source.mkdir()
        for path in [Path(__file__),ROOT/'lab/mujoco_learning.py',ROOT/'lab/electromechanical.py',ROOT/'lab/control_assets.py',ROOT/'tools/control_experiment.py',ROOT/'learning/pip-control-v1.json',ROOT/'learning/pip-electromechanical-v1.json']:
            snapshot_file(path,source/path.name)
        snapshot_file(BUNDLE/'manifest.json',source/'bundle-manifest.json')
        (output/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        report['episodes']=collect(output/'dataset')
        delta={'action':[i/50 for i in range(10)]}
        train=LeRobotDataset('local/pip-motor',root=output/'dataset',episodes=list(range(8)),delta_timestamps=delta)
        valid=LeRobotDataset('local/pip-motor',root=output/'dataset',episodes=[8,9],delta_timestamps=delta)
        config=ACTConfig(input_features={'observation.state':PolicyFeature(FeatureType.STATE,(47,)),
                'observation.environment_state':PolicyFeature(FeatureType.ENV,(3,))},
            output_features={'action':PolicyFeature(FeatureType.ACTION,(4,))},device='cpu',
            normalization_mapping={k:NormalizationMode.IDENTITY for k in ['STATE','ENV','ACTION']},
            chunk_size=10,n_action_steps=1,dim_model=64,n_heads=4,dim_feedforward=128,n_encoder_layers=1,
            n_decoder_layers=1,use_vae=False,dropout=0.,pretrained_backbone_weights=None)
        policy=ACTPolicy.from_pretrained(checkpoint) if checkpoint else ACTPolicy(config)
        report['parameter_count']=sum(p.numel() for p in policy.parameters())
        optimizer=torch.optim.AdamW(policy.parameters(),lr=1e-4)
        loader=torch.utils.data.DataLoader(train,batch_size=32,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(31))
        def validation_loss():
            policy.eval();losses=[]
            with torch.no_grad():
                for batch in torch.utils.data.DataLoader(valid,batch_size=64,num_workers=0):losses.append(float(policy(batch)[0]))
            return float(np.mean(losses))
        report['validation_l1_before']=validation_loss()
        history=[];iterator=iter(loader);begin=time.monotonic()
        for step in range(steps):
            try:batch=next(iterator)
            except StopIteration:iterator=iter(loader);batch=next(iterator)
            policy.train();loss,_=policy(batch);optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(),1.);optimizer.step()
            history.append({'step':step+1,'loss':float(loss.detach())})
        report['training_seconds']=time.monotonic()-begin
        report['validation_l1_after']=validation_loss()
        (output/'training.json').write_text(json.dumps(history))
        policy.save_pretrained(output/'pretrained_model')
        sample=train[0];obs=sample['observation.state'].numpy()
        before=ACTMotor(policy)(obs)
        restored=ACTPolicy.from_pretrained(output/'pretrained_model')
        after=ACTMotor(restored)(obs)
        report['reload_max_action_error']=float(abs(before-after).max())
        if report['reload_max_action_error']>1e-7:raise ValueError('Reload changed ACT action')
        report['checkpoint_sha256']=digest(output/'pretrained_model/model.safetensors')
        report['cases'],rows=evaluate(ACTMotor(restored),output,render=True)
        report['status']='pass' if all(c['success'] for c in report['cases'].values()) else 'fail'
        report['execution_status']='pass'
        export_rerun(rows,output,report['scope'])
    except Exception as error:
        report['status']='fail';report['execution_status']='fail';report['error']=type(error).__name__+': '+str(error)
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        (output/'report.json').write_text(json.dumps(report,indent=2))
        manifest=plan_migration(ROOT,[output]);apply_migration(ROOT,manifest);verify_manifest(ROOT,manifest)
        print(output,flush=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--checkpoint',type=Path)
    args=parser.parse_args();run(args.steps,args.checkpoint)
