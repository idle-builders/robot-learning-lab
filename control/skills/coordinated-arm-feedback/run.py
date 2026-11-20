"""Fresh upstream PPO or exact candidate replay for shallow whole22 feedback.

Source description is inert. Native execution requires a separate lead release.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from interface import ACTOR_WIDTH

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
DATA=ROOT/'data/coordinated-arm-feedback-20261030'


def ref(path):
    with path.open('rb')as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha,bytes=path.stat().st_size)


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def execute(args,method):
    import numpy as np
    import torch
    from method import agent_config,operational_json,zero_fresh_mean
    from mjlab.managers.recorder_manager import RecorderTermCfg
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner
    for expected in method['inputs']+method['source']:
        if ref(ROOT/expected['path'])!=expected:raise ValueError('Exact feedback source/input changed: '+expected['path'])
    # Native imports its own generic record/config/boundary first. Our recorder
    # loads under a unique name, never shadowing those retained startup modules.
    teacher_dir=ROOT/'control/skills/coordinated-arm-reference'
    sys.path.insert(0,str(teacher_dir))
    teacher=load(teacher_dir/'run.py','feedback_frozen_teacher')
    cfg=teacher.verify(teacher.DATA/'config.json',teacher.DATA/'source-seal.json',payload=True)
    native=teacher.load_native()
    task=load(HERE/'task.py','feedback_task_source')
    recorder_source=load(HERE/'record.py','feedback_recorder_source')
    training=args.mode=='train'
    if training:
        cases=method['training_cases'];signs=method['training_impulse_signs']
    else:
        cases=[method['comparisons']['cases'][args.case_index]]
        signs=[method['comparisons']['impulse_signs'][args.case_index]]
    torch.manual_seed(args.seed if training else cases[0]['seed'])
    configuration=task.build_config(native,teacher,cfg,cases,signs,training=training,seed=args.seed if training else cases[0]['seed'])
    Episodes=recorder_source.make(native,dict(cases=cases,joint_order=cfg['joint_order']))
    configuration.recorders={'full_episodes':RecorderTermCfg(func=Episodes,params={'directory':str(args.out/'episodes')})}
    counts=dict(simulator_controls=0,physics_substeps=0,PPO_rounds=0,optimizer_updates=0,new_weight_sets=0)
    report=dict(schema='cove50.coordinated-arm-feedback.native.v1',status='starting',mode=args.mode,
        method=ref(args.method),source=ref(Path(__file__)),counts=counts,cases=cases,impulse_signs=signs,
        start_family_ids=[('training_condition:'if training else'evaluation:')+str(c.get('condition_label',c.get('seed')))for c in cases],
        actual_batch_seed=args.seed if training else cases[0]['seed'],
        training_rng_semantics='One shared four-row Runtime generator and Torch stream seeded242100; per-row condition labels identify fixed entry shapes, not independently reseeded streams.'if training else'One world reset with this comparison case seed; paired policies use the same case seed.',
        counting='Completed env.step returns and actual sim.step calls weighted by worldslots; recorder transitions retained independently, including terminal rows. On failure these counters are observed quantities, not planned counts.',
        fresh=training,checkpoint=ref(args.checkpoint)if args.checkpoint else None,zero_residual=args.zero_residual,
        reset_records='Each actual reset saved in initialization NPZ and per-episode manifest, including empty post-timeout initializations.',
        reward_rates=method['reward_rates'],normalizer_policy=method['state'])
    def write_report():
        (args.out/'report.json').write_text(json.dumps(operational_json(report),indent=2,allow_nan=False)+'\n')
    env=runner=recorder=None;started=time.monotonic();journal=[]
    try:
        env=native.ManagerBasedRlEnv(cfg=configuration,device=args.device)
        action=env.action_manager.get_term('joint_pos');rt=native.runtime_package().get_runtime(env)
        original_step=env.sim.step
        def physics_step(*a,**kw):
            value=original_step(*a,**kw);counts['physics_substeps']+=env.num_envs
            action.sampled_substep();return value
        env.sim.step=physics_step
        recorder=env.recorder_manager._terms[0]
        vec=RslRlVecEnvWrapper(env)
        if not args.zero_residual:
            agent=agent_config();runner=MjlabOnPolicyRunner(vec,asdict(agent),log_dir=str(args.work/'logger'),device=args.device)
            if training:
                zero_fresh_mean(runner)
                def optimizer_step(optimizer,a,kw):
                    counts['optimizer_updates']+=1
                    journal.append(dict(call=counts['optimizer_updates'],common_counter=env.common_step_counter,
                        Adam_parameter_steps=sorted({int(s['step'])for s in optimizer.state.values()if'step'in s})))
                runner.alg.optimizer.register_step_post_hook(optimizer_step)
                update=runner.alg.update
                def counted_update(*a,**kw):
                    value=update(*a,**kw);counts['PPO_rounds']+=1;return value
                runner.alg.update=counted_update;runner.save=lambda *a,**kw:None
            else:
                saved=torch.load(args.checkpoint,map_location=args.device,weights_only=False)
                if saved['method']!=ref(args.method)or saved['layout']!=dict(actor=120,critic=174,raw=22):
                    raise ValueError('Exact candidate method/layout differs')
                runner.alg.actor.load_state_dict(saved['actor_state_dict'],strict=True)
                runner.alg.critic.load_state_dict(saved['critic_state_dict'],strict=True)
                runner.alg.eval_mode()
        original_env_step=env.step
        def recorded_step(raw):
            with torch.inference_mode():
                value=(runner.alg.transition.values if training else runner.alg.critic(vec.get_observations()))if runner else torch.zeros((env.num_envs,1),device=env.device)
                recorder.cache_inputs(value)
            result=original_env_step(raw);counts['simulator_controls']+=env.num_envs
            return result
        env.step=recorded_step
        report['initial_common_counter']=env.common_step_counter
        report['live_reward_weights_initial']={n:env.reward_manager.get_term_cfg(n).weight for n in env.reward_manager.active_terms}
        if training:
            report['initial_normalizer_counts']={n:int(getattr(runner.alg,n).obs_normalizer.count)for n in ('actor','critic')}
            runner.learn(num_learning_iterations=4,init_at_random_ep_len=False)
            counts['new_weight_sets']=1
            path=args.out/'candidate.pt'
            torch.save(dict(actor_state_dict=runner.alg.actor.state_dict(),critic_state_dict=runner.alg.critic.state_dict(),
                optimizer_state_dict=runner.alg.optimizer.state_dict(),method=ref(args.method),layout=dict(actor=120,critic=174,raw=22),
                frozen_donor=cfg['checkpoint'],goal_table=cfg['goal_table'],counts=dict(counts),common_counter=env.common_step_counter,
                normalizer_counts={n:int(getattr(runner.alg,n).obs_normalizer.count)for n in ('actor','critic')}),path)
            report['candidate']=ref(path)
        else:
            with torch.inference_mode():
                for _ in range(500):
                    raw=torch.zeros((env.num_envs,22),device=env.device)if args.zero_residual else runner.alg.actor(vec.get_observations())
                    env.step(raw);env._manual_reset_pending[:]=False
            recorder.finish(0,'requested_horizon')
        report['status']='completed'
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()[-8000:])
        if env is not None:
            try:
                rt=native.runtime_package().get_runtime(env)
                action=env.action_manager.get_term('joint_pos')
                values=dict(qpos=env.sim.data.qpos,qvel=env.sim.data.qvel,runtime_tick=rt.tick,
                    native_time_s=env.sim.data.time,requested_target22_rad=action.requested_target,
                    accepted_target22_rad=rt.previous_command,applied_target22_rad=rt.target,
                    delayed_values50=rt.last_values,raw_residual22=env.action_manager.action,
                    sensor_valid=rt.valid,sensor_capture_s=rt.capture_time_s,
                    own_previous_raw22=action.feedback.previous_raw,physical_delta22_rad=action.feedback.delta)
                path=args.out/'interrupted-native-state.npz'
                np.savez_compressed(path,**{k:v.detach().cpu().numpy().copy()for k,v in values.items()})
                report['interrupted_native_state']=ref(path)
            except BaseException as capture_error:
                report['interrupted_native_state_capture_error']=str(capture_error)
        raise
    finally:
        if recorder is not None:recorder.close()
        if env is not None:
            action=env.action_manager.get_term('joint_pos')
            report.update(final_common_counter=env.common_step_counter,donor_queries=action.feedback.donor.queries,
                world_slots=env.num_envs,actual_world_slots=env.num_envs,recorded_transitions=recorder.transitions,
                live_reward_weights_final={n:env.reward_manager.get_term_cfg(n).weight for n in env.reward_manager.active_terms})
            identity=cfg['checkpoint']['sha256']if args.zero_residual else report.get('candidate',report.get('checkpoint'))
            report['controller_identity']=('frozen_donor:'+identity)if isinstance(identity,str)else('feedback:'+identity['sha256'])if identity else None
            if runner is not None:
                report['final_normalizer_counts']={n:int(getattr(runner.alg,n).obs_normalizer.count)for n in ('actor','critic')}
            env.close()
        (args.out/'optimizer-journal.json').write_text(json.dumps(journal,indent=2)+'\n')
        report['wall_s']=time.monotonic()-started;write_report()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--mode',choices=('train','replay'),default='train')
    p.add_argument('--method',type=Path,default=DATA/'config.json');p.add_argument('--checkpoint',type=Path)
    p.add_argument('--zero-residual',action='store_true');p.add_argument('--case-index',type=int,choices=range(4),default=0)
    p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=242100)
    p.add_argument('--out',type=Path);p.add_argument('--work',type=Path)
    args=p.parse_args();method=json.loads(args.method.read_text())
    if not args.execute:
        print(json.dumps(dict(method=ref(args.method),source=ref(Path(__file__)),activity='source description only; no scientific imports')));return
    if not args.out or not args.work or not os.environ.get('LAB_EVIDENCE_DIR')or not os.environ.get('LAB_WORK_DIR'):
        p.error('Managed evidence/work paths required')
    if not args.out.resolve().is_relative_to(Path(os.environ['LAB_EVIDENCE_DIR']).resolve()):p.error('Native evidence must be managed')
    if args.mode=='train'and(args.checkpoint or args.zero_residual):p.error('Fresh training cannot load old/candidate state')
    if args.mode=='train'and args.seed!=method['actual_training_batch_seed']:p.error('Use exact frozen shared training batch seed')
    if args.mode=='replay'and bool(args.checkpoint)==args.zero_residual:p.error('Replay exact candidate OR zero-residual baseline')
    args.out.mkdir(parents=True,exist_ok=False);args.work.mkdir(parents=True,exist_ok=True)
    execute(args,method)


if __name__=='__main__':main()
