"""Fresh shallow feedback recipe, privileged value/rewards labelled explicitly.

No construction, inference or physics at import. Values are proposed experimental
scales, not hardware qualifications or capability thresholds.
"""
FIT=dict(worlds=4,rounds=4,steps_per_round=500,controls=8000,physics_substeps=640000,Adam_calls=80)
IMPULSE=dict(start_control=250,stop_control=255,force_N=1.,direction='world Y',
             duration_s=.1,impulse_N_s=.1,body='torso',reference_feet_load_N=37.56)
REWARDS={
    'pose_tracking':dict(weight=2.,scale_rad=.03,formula='exp(-mean22(((q-poseGoal)/.03)^2))'),
    'upright':dict(weight=1.,scale_rad=.05,formula='exp(-sum(estimatedGravityXY^2)/.05^2)'),
    'gyro':dict(weight=-.05,scale_rad_s=.5,formula='mean3((delayedGyro/.5)^2)'),
    'residual':dict(weight=-.02,scale_rad=.03,formula='mean22((physicalDelta/.03)^2)'),
    'residual_rate':dict(weight=-.005,scale_rad_s=1.5,formula='mean22((physicalDeltaRate/1.5)^2)'),
    'nonfoot':dict(weight=-.5,scale_N=37.56,formula='(sumBodyNorm(nonfootContactForce)/37.56)^2'),
    'foot_load_imbalance':dict(weight=-.1,scale_N=37.56,formula='((rightFootFz-leftFootFz)/37.56)^2'),
}


def agent_config():
    """Fresh upstream MLP/PPO configuration; no saved legacy agent YAML."""
    from mjlab.rl.config import RslRlOnPolicyRunnerCfg
    cfg=RslRlOnPolicyRunnerCfg()
    cfg.num_steps_per_env=500;cfg.max_iterations=4;cfg.save_interval=4
    cfg.logger='tensorboard';cfg.upload_model=False;cfg.clip_actions=None
    for model in (cfg.actor,cfg.critic):
        model.hidden_dims=(64,64);model.activation='tanh';model.obs_normalization=True
    cfg.actor.distribution_cfg=dict(class_name='GaussianDistribution',init_std=.2,std_type='scalar')
    cfg.algorithm.learning_rate=3e-4;cfg.algorithm.schedule='fixed'
    cfg.algorithm.gamma=.995;cfg.algorithm.lam=.95;cfg.algorithm.entropy_coef=.001
    cfg.algorithm.num_learning_epochs=5;cfg.algorithm.num_mini_batches=4
    cfg.algorithm.optimizer='adam'
    return cfg


def zero_fresh_mean(runner):
    """After fresh runner build, before first act; reject accidental continuation."""
    import torch
    actor,critic=runner.alg.actor,runner.alg.critic
    if actor.obs_dim!=120 or critic.obs_dim!=174 or runner.alg.optimizer.state:
        raise ValueError('Expected fresh120/174 actor/value and empty Adam')
    for model in (actor,critic):
        if int(model.obs_normalizer.count)!=0:raise ValueError('Fresh normalizer count must be0')
    output=[m for m in actor.mlp.modules()if isinstance(m,torch.nn.Linear)][-1]
    if output.in_features!=64 or output.out_features!=22:raise ValueError('Fresh raw22 mean layout differs')
    torch.nn.init.zeros_(output.weight);torch.nn.init.zeros_(output.bias)


def value_observation(env, actor120, joint_ids):
    """Training-only174: deployed120 + nativeq22/qd22/v3/w3/z1/feetFz2/nonfootN1."""
    import torch
    robot=env.scene['robot'].data
    feet=env.scene['feet_ground_contact'].data.force.reshape(env.num_envs,2,-1,3).sum(dim=2)
    nonfoot=env.scene['nonfoot_ground_contact'].data.force
    nonfoot_N=nonfoot.norm(dim=-1).reshape(env.num_envs,-1).sum(dim=1,keepdim=True)
    return torch.cat((actor120,robot.joint_pos[:,joint_ids],robot.joint_vel[:,joint_ids],
        robot.root_link_lin_vel_b,robot.root_link_ang_vel_b,env.sim.data.qpos[:,2:3],
        feet[:,:,2],nonfoot_N),dim=1)


def reward_terms(env):
    """RewardManager scales each rate by actual .02s; no legacy reward bus."""
    import torch
    action=env.action_manager.get_term('joint_pos');s=action.feedback
    q=env.scene['robot'].data.joint_pos[:,action.ids]
    feet=env.scene['feet_ground_contact'].data.force.reshape(env.num_envs,2,-1,3).sum(dim=2)
    nonfoot=env.scene['nonfoot_ground_contact'].data.force
    return dict(pose_tracking=torch.exp(-((q-s.pose_goal)/.03).square().mean(dim=1)),
        upright=torch.exp(-s.obs[:,47:49].square().sum(dim=1)/.05**2),
        gyro=(s.obs[:,44:47]/.5).square().mean(dim=1),
        residual=(s.delta/.03).square().mean(dim=1),
        residual_rate=(action.physical_delta_rate/1.5).square().mean(dim=1),
        nonfoot=(nonfoot.norm(dim=-1).reshape(env.num_envs,-1).sum(dim=1)/37.56).square(),
        foot_load_imbalance=((feet[:,0,2]-feet[:,1,2])/37.56).square())


def lateral_force(local_control, signs, *, torch):
    """Real world-frame force, not a velocity/state overwrite or actor feature."""
    active=(local_control>=250)&(local_control<255)
    force=torch.zeros((len(local_control),1,3),device=signs.device,dtype=signs.dtype)
    force[:,0,1]=active*signs*IMPULSE['force_N']
    return force


def operational_json(value, path=()):
    """NEW worker can encode only known structural softlimit infinities safely.

    Native source/report bytes stay unchanged. Other NaN/Infinity, including
    measured motion or reward numbers, remain an error requiring original capture.
    """
    import math
    if isinstance(value,dict):return{k:operational_json(v,path+(str(k),))for k,v in value.items()}
    if isinstance(value,list):return[operational_json(v,path+(str(i),))for i,v in enumerate(value)]
    if isinstance(value,float)and not math.isfinite(value):
        if not math.isnan(value)and'soft_limits22_rad'in path:
            return dict(structural_bound='unbounded',sign='negative'if value<0 else'positive')
        raise ValueError('Nonfinite non-structural operational metadata at '+'.'.join(path))
    return value
