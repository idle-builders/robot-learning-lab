"""One plan-bound attained-state gait visit through the original native task."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import traceback
from profiles import Profile
from replay import ROOT,CHECKPOINT,CHECKPOINT_SHA,ref,verify_inputs,load_native,validate_command_columns,command_observations
from gait_evidence import TransitionCapture,state_attestation,live_rewards


def planned_case(plan_path,case_name,method_path):
    plan=json.loads(Path(plan_path).read_text());method=verify_inputs(method_path)
    if ref(plan_path)!=method['execution_plan']:raise ValueError('Execution plan differs from frozen method')
    cases=[x for x in plan['cases'] if x['name']==case_name]
    if len(cases)!=1:raise ValueError('Visit is outside the frozen plan')
    case=cases[0];profiles=[x for x in plan['profiles'] if x['name']==case['profile']]
    if len(profiles)!=1:raise ValueError('Profile missing/ambiguous')
    p=profiles[0];profile=Profile(p['name'],p['duration_s'],tuple(p['initial']),tuple(p['successor']) if p.get('successor') is not None else None,p.get('switch_s'))
    vectors=[profile.initial]+([profile.successor] if profile.successor is not None else [])
    if any(len(v)!=3 or any(not math.isfinite(x) for x in v) for v in vectors):raise ValueError('Finite three-axis commands required')
    if profile.switch_s is not None and (profile.successor is None or not 0<profile.switch_s<profile.duration_s or p['switch_control']!=round(profile.switch_s/.02)):raise ValueError('Invalid pre-action switch schedule')
    if case['seed'] not in plan['seed_cohorts'] or case['controls']!=round(profile.duration_s/.02) or case['physics_steps']!=80*case['controls']:raise ValueError('Native visit accounting differs from plan')
    if plan['checkpoint']!=ref(CHECKPOINT) or plan['checkpoint']['sha256']!=CHECKPOINT_SHA:raise ValueError('Only exact candidate200 is frozen')
    if plan['optimizer_updates']!=0 or plan['new_policies']!=0:raise ValueError('This collector never fits')
    return plan,method,case,profile


def collect(plan_path,case_name,method_path,out,device='cuda:0'):
    out=Path(out)
    if out.exists():raise FileExistsError('Every original attempt needs a fresh directory')
    out.mkdir(parents=True)
    report=dict(schema='cove50.plan-bound-gait-replay.v1',requested_case=case_name,device=device,policy='candidate200',status='starting',optimizer_updates=0,ppo_rounds=0,distinct_new_policies=0,native_physics_steps=0,simulator_transitions=0,reset_at_command_change=False)
    native=env=recorder=capture=runner=saved=None;start=time.monotonic()
    try:
        plan,frozen,case,profile=planned_case(plan_path,case_name,method_path);seed=case['seed']
        report.update(case=case,profile=asdict(profile),seed=seed,method=ref(method_path),execution_plan=ref(plan_path),checkpoint=ref(CHECKPOINT))
        native=load_native();torch=native.torch;torch.manual_seed(seed)
        cfg=native.bind(native.morphology_cfg(native.runtime_package().robot_cfg(seed=seed),play=True))
        cfg.scene.num_envs=1;cfg.seed=seed;profile.configure(cfg);validate_command_columns(cfg)
        report['configured_rewards']=native.reward_recipe(cfg)
        cfg.recorders={'full_episodes':native.RecorderTermCfg(func=native.Episodes,params={'directory':str(out/'episodes')})}
        env=native.ManagerBasedRlEnv(cfg=cfg,device=device);original_step=env.sim.step
        def counted_step(*args,**kwargs):
            result=original_step(*args,**kwargs);report['native_physics_steps']+=1;return result
        env.sim.step=counted_step;recorder=env.recorder_manager._terms[0]
        vec=native.RslRlVecEnvWrapper(env);agent=native.agent_config()
        runner=native.MjlabOnPolicyRunner(vec,asdict(agent),log_dir=None,device=device)
        report['transfer']=native.initialize(runner,env,checkpoint=CHECKPOINT)
        saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        report['post_initialize_fullstate']=state_attestation(runner,saved)
        if not all(x['equal'] for x in report['post_initialize_fullstate'].values()):raise RuntimeError('Exact saved fullstate not restored')
        report['initial_counter']=env.common_step_counter;report['post_initialize_live_rewards']=live_rewards(env,native)
        if len(report['post_initialize_live_rewards'])!=12 or env.reward_manager.get_term_cfg('action_rate_l2').weight!=-.6:raise RuntimeError('Unexpected live reward contract')
        report['runtime_versions']=native.runtime_versions();report['numerical_options']=native.serializable(cfg.sim)
        model=env.sim.mj_model;action_term=env.action_manager.get_term('joint_pos');layout=[]
        for name in action_term.interface.current_names:
            matches=[i for i in range(model.njnt) if model.joint(i).name.split('/')[-1]==name]
            if len(matches)!=1:raise RuntimeError('Ambiguous named native joint layout: '+name)
            i=matches[0];layout.append(dict(name=name,native_name=model.joint(i).name,qpos_address=int(model.jnt_qposadr[i]),qvel_address=int(model.jnt_dofadr[i])))
        report['physical_joint_layout']=layout;report['feet_contact_order']=['R_foot','L_foot']
        policy=runner.get_inference_policy(device=device);term=env.command_manager.get_term('twist');capture=TransitionCapture(env,native,profile)
        with torch.inference_mode():
            for control in range(case['controls']):
                # Installed get_observations returns the existing noisy/delayed
                # cache. Fetch once; clone and patch only direct command cols.
                cached=vec.get_observations();before=capture.snapshot(control,'before_command',cached)
                requested=profile.apply(term,control);observations=command_observations(cached,requested)
                command=observations['actor'].new_tensor(requested)[None,:]
                if not torch.equal(observations['actor'][:,69:72],command) or not torch.equal(observations['critic'][:,72:75],command):raise RuntimeError('Pre-action command differs from schedule')
                env.obs_buf=observations;after=capture.snapshot(control,'after_command',observations);capture.compare(control,before,after)
                recorder.cache_inputs(runner.alg.critic(observations));env.step(policy(observations,stochastic_output=False))
                env._manual_reset_pending[:]=False
        recorder.finish(0,'requested_horizon');report['status']='completed'
    except BaseException as exc:
        report.update(status='interrupted',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc()[-5000:])
        if recorder is not None:recorder.finish(0,'collector_interruption_censoring')
        raise
    finally:
        if env is not None:
            try:
                report['final_counter']=env.common_step_counter;report['final_live_rewards']=live_rewards(env,native)
                if runner is not None and saved is not None:
                    report['final_fullstate']=state_attestation(runner,saved)
                    if not all(x['equal'] for x in report['final_fullstate'].values()):report.update(status='interrupted',state_error='Frozen replay actor/value/normalizers/Adam changed')
            except Exception as exc:report.update(status='interrupted',state_attestation_error=str(exc))
        if capture is not None:
            try:report['command_switch_state']=capture.save(out,ref)
            except Exception as exc:report.update(status='interrupted',command_snapshot_save_error=str(exc))
        if env is not None:
            try:env.close()
            except Exception as exc:report['close_error']=str(exc)
        report.update(wall_s=time.monotonic()-start,world_slots_constructed=int(env is not None),seed_cohorts=1 if env is not None else 0,independent_start_note='Actual saved initial qpos/qvel/bias vectors are compared offline; seed labels and original recorder presence boolean are not distinct physical start counts.',simulator_transitions=recorder.transitions if recorder is not None else 0,complete_or_terminated_episodes=recorder.completed if recorder is not None else 0,censored_episodes=recorder.censored if recorder is not None else 0)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--case',required=True);p.add_argument('--method',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();collect(a.plan,a.case,a.method,a.out,a.device)


if __name__=='__main__':main()
