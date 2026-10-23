"""Gymnasium/SB3 discrete skill selection over a frozen native RSL-RL motor policy."""
import json,math,hashlib,os,shutil
from dataclasses import asdict
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
import cv2
from perception_control import detect_target,approach_skill


class SkillGym(gym.Env):
    def __init__(self,raw,wrapped,motor,output):
        self.raw,self.wrapped,self.motor,self.output=raw,wrapped,motor,Path(output)
        self.action_space=gym.spaces.Discrete(3)
        self.observation_space=gym.spaces.Box(-10,10,shape=(4,),dtype=np.float32)
        self.episode=-1;self.records=[];self.camera=raw.scene['camera'];self.forced_loss=False

    def capture(self):
        self.raw.sim.render();self.camera.update(self.raw.step_dt,force_recompute=True)
        self.rgb=self.camera.data.output['rgb'].torch[0].cpu().numpy()[:,:,:3]
        if self.forced_loss and 3.0<=self.time<4.0:self.rgb=np.zeros_like(self.rgb)
        k=self.camera.data.intrinsic_matrices[0].cpu().numpy()
        self.estimate=detect_target(self.rgb,k,.2,self.time)
        from robotlab.config import INTERFACE
        from isaaclab.utils.math import quat_apply
        robot=self.raw.scene['robot'];head=robot.body_names.index('head')
        pose=robot.data.body_link_pose_w.torch[:,head]
        offset=torch.tensor([INTERFACE['camera_frame']['xyz_m']],device=self.raw.device)
        expected=pose[:,:3]+quat_apply(pose[:,3:],offset)
        self.camera_position=self.camera.data.pos_w.torch[0].tolist()
        self.camera_pose_error_m=float(torch.max(torch.abs(expected-self.camera.data.pos_w.torch)))
        if self.camera_pose_error_m>1e-4:raise RuntimeError('Head-camera pose disagrees with robot kinematics')

    def observation(self):
        e=self.estimate
        return np.array([float(e is not None),min(e.forward_m,5) if e else 0,np.clip(e.left_m,-5,5) if e else 0,self.time/20],np.float32)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);self.episode+=1;self.time=0
        self.raw.reset(seed=seed);self.capture()
        return self.observation(),{}

    def step(self,action):
        before=self.estimate.forward_m if self.estimate else None
        fell=False;commands=[]
        for i in range(10):
            if i%5==0:self.capture()
            cmd=approach_skill(self.estimate,self.time)
            vx,yaw=cmd.forward_m_s,cmd.yaw_rad_s
            if int(action)==0:vx,yaw=0.,0.
            elif int(action)==2:vx=0.
            commands.append({'time_s':self.time,'skill_id':int(action),'forward_m_s':vx,'yaw_rad_s':yaw,'reason':cmd.reason})
            self.raw.command_manager.get_command('motion')[:]=torch.tensor([[vx,0.,yaw]],device=self.raw.device)
            # SB3 resets outside this context; do not leave Isaac state as
            # inference-only tensors that cannot be updated on reset.
            with torch.no_grad():
                obs=self.wrapped.get_observations()
                _,_,done,_=self.wrapped.step(self.motor(obs))
                fell=bool(self.raw.termination_manager.terminated.any())
            self.time+=self.raw.step_dt
            if fell:break
        self.capture();e=self.estimate
        success=e is not None and e.forward_m<=.45 and not fell
        reward=(20*(before-e.forward_m) if before is not None and e else -.02)-.005
        reward+=5 if success else -5 if fell else 0
        asset=self.raw.scene['robot']
        row={'episode':self.episode,'time_s':self.time,'skill_id':int(action),'estimate':asdict(e) if e else None,'commands':commands,'reward':reward,'success':success,'fell':fell,'q':asset.data.joint_pos.torch[0].tolist(),'dq':asset.data.joint_vel.torch[0].tolist(),'body_poses_world_xyz_xyzw':asset.data.body_link_pose_w.torch[0].tolist(),'targets':self.raw.action_manager.get_term('motion').targets[0].tolist()}
        if len(self.records)%10==0 or self.forced_loss:
            name=f'frame-{len(self.records):05}.png';cv2.imwrite(str(self.output/name),cv2.cvtColor(self.rgb,cv2.COLOR_RGB2BGR));row['image']=name
        row['camera_position_world_m']=self.camera_position
        row['camera_pose_error_m']=self.camera_pose_error_m
        if 'shared_pack' in asset.actuators:row['electromechanical']=asset.actuators['shared_pack'].telemetry()
        self.records.append(row)
        return self.observation(),float(reward),success or fell,self.time>=20,{'success':success,'fell':fell}


def run_skills(raw,checkpoint,output):
    import importlib.metadata
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper,handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner
    from robotlab.config import PipRunnerCfg
    from stable_baselines3 import PPO
    output=Path(output);output.mkdir(exist_ok=False)
    wrapped=RslRlVecEnvWrapper(raw,clip_actions=1.)
    cfg=handle_deprecated_rsl_rl_cfg(PipRunnerCfg(),importlib.metadata.version('rsl-rl-lib'))
    runner=OnPolicyRunner(wrapped,cfg.to_dict(),log_dir=None,device=raw.device);runner.load(str(checkpoint))
    env=SkillGym(raw,wrapped,runner.get_inference_policy(device=raw.device),output)
    reused=os.environ.get('ROBOTLAB_SELECTOR_CHECKPOINT')
    if reused:
        shutil.copyfile(reused,output/'selector.zip')
    else:
        model=PPO('MlpPolicy',env,n_steps=64,batch_size=32,n_epochs=4,policy_kwargs={'net_arch':[32,32]},device='cpu',seed=23,verbose=1)
        try:
            model.learn(total_timesteps=512);model.save(output/'selector')
        finally:
            (output/'training-trace.json').write_text(json.dumps(env.records))
    training_records=len(env.records)
    restored=PPO.load(output/'selector.zip',env=env,device='cpu')
    env.forced_loss=True
    observation,_=env.reset(seed=107)
    for _ in range(100):
        action,_=restored.predict(observation,deterministic=True)
        observation,_,terminated,truncated,_=env.step(action)
        if terminated or truncated:break
    evaluation=env.records[training_records:]
    held=[c for r in evaluation for c in r['commands'] if c['reason']=='target_missing']
    loss_stop=bool(held) and all(c['forward_m_s']==0 and c['yaw_rad_s']==0 for c in held)
    report={'status':'partial','training_steps':512,'checkpoint_reloaded':True,'motor_checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),'selector_checkpoint_sha256':hashlib.sha256((output/'selector.zip').read_bytes()).hexdigest(),'training_records':training_records,'evaluation_records':len(evaluation),'evaluation_success':any(r['success'] for r in evaluation),'evaluation_falls':sum(r['fell'] for r in evaluation),'lost_vision_hold':loss_stop,'skills':{'0':'hold','1':'approach visible marker','2':'align to visible marker'},'scope':'bounded discrete PPO integration; fiducial perception and safety override are authored, no general semantic vision or language model'}
    report['training_steps']=0 if reused else 512
    report['selector_reused']=bool(reused)
    (output/'trace.json').write_text(json.dumps(env.records))
    (output/'report.json').write_text(json.dumps(report,indent=2))
    return report
