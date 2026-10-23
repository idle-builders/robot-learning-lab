"""Bounded native reference/checkpoint evaluation. Never treats resets as success."""
import argparse
import json
from pathlib import Path
import torch
from isaaclab.app import add_launcher_args, launch_simulation

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--checkpoint',type=Path)
p.add_argument('--steps',type=int,default=400)
add_launcher_args(p)
a=p.parse_args()
from robotlab.config import PipEnvCfg, PipRunnerCfg, ORDER, INTERFACE
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
import importlib.metadata
from rsl_rl.runners import OnPolicyRunner
cfg=PipEnvCfg();cfg.scene.num_envs=8;cfg.seed=103
with launch_simulation(cfg,a):
    raw=ManagerBasedRLEnv(cfg=cfg)
    raw.reset()
    asset=raw.scene['robot']
    report={'status':'partial','joint_names':asset.joint_names,'body_names':asset.body_names,'checkpoint':str(a.checkpoint) if a.checkpoint else None}
    assert asset.joint_names==ORDER,(asset.joint_names,ORDER)
    assert asset.body_names==[v['name'] for v in INTERFACE['links']]
    env=RslRlVecEnvWrapper(raw,clip_actions=1.0)
    if a.checkpoint:
        agent_cfg=handle_deprecated_rsl_rl_cfg(PipRunnerCfg(),importlib.metadata.version("rsl-rl-lib"))
        runner=OnPolicyRunner(env,agent_cfg.to_dict(),log_dir=None,device=raw.device)
        runner.load(str(a.checkpoint));policy=runner.get_inference_policy(device=raw.device)
    obs=env.get_observations()
    import mujoco, numpy as np, os
    from robotlab.config import REST
    oracle=mujoco.MjModel.from_xml_path(str(Path(os.environ['ROBOTLAB_BUNDLE'])/'model/robot.xml'))
    state=mujoco.MjData(oracle)
    for name,value in REST.items():
        joint=mujoco.mj_name2id(oracle,mujoco.mjtObj.mjOBJ_JOINT,name)
        state.qpos[oracle.jnt_qposadr[joint]]=value
    mujoco.mj_forward(oracle,state)
    ids=[mujoco.mj_name2id(oracle,mujoco.mjtObj.mjOBJ_BODY,n) for n in asset.body_names]
    native_mass=asset.data.body_mass.torch[0].cpu().numpy()
    native_pose=asset.data.body_link_pose_w.torch[0].cpu().numpy()
    mass_error=float(np.max(np.abs(native_mass-oracle.body_mass[ids])))
    pose_error=float(np.max(np.abs(native_pose[:,:3]-raw.scene.env_origins[0].cpu().numpy()-state.xpos[ids])))
    parity={'status':'pass' if mass_error<1e-5 and pose_error<1e-4 else 'fail','mass_max_error_kg':mass_error,'position_max_error_m':pose_error,'mass_kg':native_mass.tolist(),'body_poses_world_xyz_xyzw':native_pose.tolist(),'inertia_world_kg_m2':asset.data.body_inertia.torch[0].cpu().tolist(),'inertia_comparison':'not_run'}
    a.output.with_name(a.output.stem+'-loaded.json').write_text(json.dumps(parity,indent=2))
    if parity['status']!='pass':raise RuntimeError('Loaded mass/pose parity failed')
    rows=[];failures=0
    with torch.inference_mode():
        for i in range(a.steps):
            action=policy(obs) if a.checkpoint else torch.zeros((8,4),device=raw.device)
            obs,reward,done,info=env.step(action)
            failures+=int(raw.termination_manager.terminated.sum().item())
            if i%5==0:
                rows.append({'step':i,'time_s':i*raw.step_dt,'root_position_world_m':asset.data.root_pos_w.torch[0].tolist(),'root_quaternion_xyzw':asset.data.root_quat_w.torch[0].tolist(),'q':asset.data.joint_pos.torch[0].tolist(),'dq':asset.data.joint_vel.torch[0].tolist(),'gravity':asset.data.projected_gravity_b.torch[0].tolist(),'velocity':asset.data.root_lin_vel_b.torch[0].tolist(),'command':raw.command_manager.get_command('motion')[0].tolist(),'reward':float(reward.mean()),'terminated':raw.termination_manager.terminated.tolist(),'targets':raw.action_manager.get_term('motion').targets[0].tolist()})
                if 'shared_pack' in asset.actuators:rows[-1]['electromechanical']=asset.actuators['shared_pack'].telemetry()
    report.update(status='pass' if failures==0 else 'fail',failure_resets=failures,steps=a.steps,num_envs=8,trace=rows)
    a.output.write_text(json.dumps(report,indent=2));env.close()
