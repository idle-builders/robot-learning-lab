"""Finite CPU fresh-runner/tensor contract exercise, no donor/native evidence."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent


def ref(p):
    b=p.read_bytes();return dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))


def main():
    import torch
    from tensordict import TensorDict
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjlab.managers import ObservationGroupCfg,ObservationTermCfg
    from mjlab.managers.observation_manager import ObservationManager
    from interface import Interface
    from task import observation_terms
    from method import agent_config,zero_fresh_mean,value_observation,reward_terms,lateral_force,operational_json
    def forbidden(*a,**kw):raise RuntimeError('PT loading outside CPU fixture')
    torch.load=forbidden;torch.manual_seed(242100)
    obs=TensorDict({'actor':torch.zeros(4,120),'critic':torch.zeros(4,174)},batch_size=[4])
    env=SimpleNamespace(num_envs=4,num_actions=22,device='cpu',cfg={},get_observations=lambda:obs)
    runner=MjlabOnPolicyRunner(env,asdict(agent_config()),log_dir=None,device='cpu')
    zero_fresh_mean(runner)
    calls=dict(actor=0,critic=0)
    def count(name):
        def before(*a):calls[name]+=1
        return before
    runner.alg.actor.register_forward_pre_hook(count('actor'))
    runner.alg.critic.register_forward_pre_hook(count('critic'))
    with torch.inference_mode():mean=runner.alg.actor(obs)
    assert mean.shape==(4,22)and torch.equal(mean,torch.zeros_like(mean))
    assert torch.allclose(runner.alg.actor.distribution.std_param,torch.full((22,),.2))
    assert len(runner.alg.optimizer.state)==0
    norm_before={n:int(getattr(runner.alg,n).obs_normalizer.count)for n in ('actor','critic')}
    runner.alg.train_mode()
    with torch.inference_mode():runner.alg.act(obs)
    nextobs=obs.clone();nextobs['actor'][:,50]=.5;nextobs['critic'][:,:120]=nextobs['actor']
    runner.alg.process_env_step(nextobs,torch.zeros(4),torch.zeros(4,dtype=torch.bool),{})
    norm_after={n:int(getattr(runner.alg,n).obs_normalizer.count)for n in ('actor','critic')}
    assert norm_before==dict(actor=0,critic=0)and norm_after==dict(actor=4,critic=4)
    assert float(runner.alg.actor.obs_normalizer.mean[50])==.5
    runner.alg.eval_mode();runner.alg.actor.update_normalization(obs);runner.alg.critic.update_normalization(obs)
    assert all(int(getattr(runner.alg,n).obs_normalizer.count)==4 for n in ('actor','critic'))
    q=torch.zeros(4,22);actor=torch.zeros(4,120);actor[:,49]=-1.
    feet=torch.zeros(4,2,1,3);feet[:,:,:,2]=37.56/2
    scene={'robot':SimpleNamespace(data=SimpleNamespace(joint_pos=q,joint_vel=q.clone(),
            root_link_lin_vel_b=torch.zeros(4,3),root_link_ang_vel_b=torch.zeros(4,3))),
        'feet_ground_contact':SimpleNamespace(data=SimpleNamespace(force=feet)),
        'nonfoot_ground_contact':SimpleNamespace(data=SimpleNamespace(force=torch.zeros(4,1,1,3)))}
    s=SimpleNamespace(pose_goal=q.clone(),delta=q.clone(),obs=actor)
    action=SimpleNamespace(ids=list(range(22)),feedback=s,physical_delta_rate=q.clone())
    stub=SimpleNamespace(scene=scene,num_envs=4,sim=SimpleNamespace(data=SimpleNamespace(qpos=torch.zeros(4,29))),
        action_manager=SimpleNamespace(get_term=lambda name:action))
    critic=value_observation(stub,actor,action.ids)
    assert critic.shape==(4,174)and torch.equal(critic[:,:120],actor)
    terms=reward_terms(stub)
    assert torch.equal(terms['pose_tracking'],torch.ones(4))and torch.equal(terms['upright'],torch.ones(4))
    assert all(torch.equal(v,torch.zeros(4))for k,v in terms.items()if k not in ('pose_tracking','upright'))
    class Donor:
        queries=0
        def motor_reference(self,beta):self.queries+=1;return torch.zeros(22)
    class Goals:
        queries=0
        def at(self,beta):self.queries+=1;return dict(q_pose_goal22_rad=[0.]*22)
    donor,goals=Donor(),Goals();names=[f'axis{i}'for i in range(22)]
    interface=Interface(4,names,SimpleNamespace(command=lambda k:(0.,0.)),donor,goals,device='cpu')
    action.feedback=interface;stub.device='cpu';stub.episode_length_buf=torch.zeros(4,dtype=torch.long)
    runtime=SimpleNamespace(names=names,last_values=torch.zeros(4,50),tick=torch.zeros(4,dtype=torch.long),
        valid=torch.zeros(4,dtype=torch.bool),capture_time_s=torch.zeros(4,dtype=torch.float64),physics_dt_s=.00025)
    native=SimpleNamespace(runtime_package=lambda:SimpleNamespace(get_runtime=lambda env:runtime))
    actor_term,critic_term=observation_terms(native)
    obs_cfg={'actor':ObservationGroupCfg(terms={'feedback':ObservationTermCfg(func=actor_term)}),
        'critic':ObservationGroupCfg(terms={'training_only':ObservationTermCfg(func=critic_term)})}
    # Exact pinned manager constructor invokes term.func to infer dimensions.
    manager=ObservationManager(obs_cfg,stub)
    assert manager.group_obs_dim=={'actor':(120,),'critic':(174,)}
    assert donor.queries==goals.queries==0 and not bool(interface.initialized.any())
    assert interface.generation.tolist()==[0,0,0,0]and interface.prepared_generation.tolist()==[-1,-1,-1,-1]
    stub.observation_manager=manager
    unreset_rejected=False
    try:manager.compute(update_history=True)
    except RuntimeError:unreset_rejected=True
    assert unreset_rejected
    interface.reset(torch.arange(4),q)
    actual=manager.compute(update_history=True)
    assert actual['actor'].shape==(4,120)and actual['critic'].shape==(4,174)
    assert donor.queries==goals.queries==4 and interface.generation.tolist()==[1,1,1,1]
    signs=torch.ones(500);force=lateral_force(torch.arange(500),signs,torch=torch)
    active=force[:,0,1].nonzero().flatten().tolist()
    assert active==[250,251,252,253,254]
    impulse=float(force[:,0,1].sum()*.02)
    encoded=operational_json(dict(reset=dict(soft_limits22_rad=[[-float('inf'),float('inf')]])))
    rejected=False
    try:operational_json(dict(measured_q22_rad=[float('nan')]))
    except ValueError:rejected=True
    assert rejected
    report=dict(schema='cove50.coordinated-arm-feedback-cpu-method-proof.v1',
        source=[ref(HERE/name)for name in ('interface.py','method.py','task.py','record.py','run.py','prove_method.py')],
        runtime_source=[ref(Path(__import__(n,fromlist=['x']).__file__))for n in ('rsl_rl.algorithms.ppo','mjlab.rl.runner')],
        observations=dict(actor_width=runner.alg.actor.obs_dim,critic_width=runner.alg.critic.obs_dim,
            deterministic_fresh_mean_max=float(mean.abs().max()),raw_initial_std=.2,empty_Adam=len(runner.alg.optimizer.state)==0,
            normalizer_before=norm_before,normalizer_after_one_synthetic_nextbatch=norm_after,
            actual_returned_command_feature_mean=.5,evaluation_norms_frozen=True,
            critic_prefix_exact_actor=True,balanced_nominal_reward_rates={k:v.tolist()for k,v in terms.items()},
            actual_ObservationManager_shape_probe=dict(widths=manager.group_obs_dim,donor_queries_before_reset=0,
                generation_before_reset=[0,0,0,0],cache_generation_before_reset=[-1,-1,-1,-1],
                unreset_real_observation_rejected=unreset_rejected,queries_after_actual_reset=donor.queries,
                actual_post_reset_generation=interface.generation.tolist()),
            force_controls=active,impulse_N_s=impulse,structural_bounds=encoded,nonstructural_NaN_rejected=rejected),
        counts=dict(fresh_models=2,NN_forward_calls=calls,synthetic_rows_added_to_PPO_storage=4,
            optimizer_updates=0,PT_loads=0,native_envs=0,physics_steps=0,provider_requests=0,new_candidates=0),
        limitations='CPU stub data only; no robot/donor inference, physical reset/recorder/80-substep execution, complete native nextobs ordering or native motion. Exact ObservationManager constructor/reset-cache path exercised, not native environment construction. Synthetic rows are not simulator transitions or campaign counts.')
    out=Path(os.environ['LAB_EVIDENCE_DIR'])/'records/cpu-method-proof.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(report=ref(out),observed_counts=report['counts'])))


if __name__=='__main__':main()
