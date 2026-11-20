"""Native R3 old-forward lesson: full replay or bounded upstream PPO.

Provisioning and compute health are owned by the campaign lead. This launcher
does not allocate remote machines or change the selected robot model.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import traceback
import importlib.metadata
import torch
import yaml
import numpy as np

from config import ROOT, SAVED_ENV, morphology_cfg, reward_recipe, serializable, typed
from boundary import bind, runtime_package
from record import Episodes
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.recorder_manager import RecorderTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.rl.config import RslRlOnPolicyRunnerCfg
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'policy'))
from adapter import CHECKPOINT, Interface, expand_checkpoint, ref
from checkpoint import initialize, save as save_checkpoint, LAYOUT, CRITIC


def agent_config():
    saved = typed(yaml.load((SAVED_ENV.parent / 'agent.yaml').read_text(), Loader=yaml.BaseLoader))
    cfg = RslRlOnPolicyRunnerCfg()
    for name in ('seed', 'num_steps_per_env', 'save_interval', 'clip_actions', 'logger', 'upload_model'):
        setattr(cfg, name, saved[name])
    for group in ('actor', 'critic', 'algorithm'):
        for name, value in saved[group].items(): setattr(getattr(cfg, group), name, value)
    cfg.upload_model = False
    return cfg


def runtime_versions():
    result = {}
    for name in ('mjlab', 'mujoco', 'mujoco-warp', 'torch', 'rsl-rl-lib'):
        try: result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result[name] = 'distribution name unavailable; inspect retained worker environment'
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('smoke', 'replay', 'capacity', 'train'), required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--worlds', type=int, default=1)
    p.add_argument('--updates', type=int, default=1, help='PPO rounds; each uses24 simulator steps/world and20 optimizer steps')
    p.add_argument('--steps', type=int, default=2, help='Smoke control steps only')
    p.add_argument('--seed', type=int, default=135726)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--source-checkpoint', type=Path, help='Original72/87 actor/value source, expanded with fresh Adam moments')
    p.add_argument('--fresh', action='store_true', help='New random network; same task/reward and R3 physics')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--velocity-x', type=float, help='Replay fixed x command; omit to use original sampler')
    args = p.parse_args()
    if args.out.exists(): raise FileExistsError('Use a fresh evidence directory')
    if args.worlds < 1 or args.updates < 1 or args.steps < 1: p.error('Positive world/update/step counts required')
    if sum(bool(x) for x in (args.fresh,args.checkpoint,args.source_checkpoint)) > 1: p.error('Choose fresh, R3 checkpoint, or original source checkpoint')
    if args.mode in ('train','capacity') and not args.device.startswith('cuda'): p.error('Training/capacity requires GPU')
    if args.mode in ('train','capacity') and args.velocity_x is not None: p.error('Training/capacity keeps the saved command sampler')
    args.out.mkdir(parents=True)
    start = time.monotonic()
    env = runner = cfg = recorder = None
    counts = dict(native_physics_steps=0, optimizer_steps=0, ppo_rounds=0)
    report = dict(schema='cove50.walk-resume-native-run.v1', mode=args.mode, device=args.device,
        seed=args.seed, worlds=args.worlds, independent_start_families=0, status='starting',
        planned=dict(ppo_rounds=args.updates if args.mode == 'train' else 0,
            simulator_transitions=args.worlds * (args.updates * 24 if args.mode == 'train' else args.steps if args.mode in ('smoke','capacity') else 400),
            optimizer_steps=args.updates * 20 if args.mode == 'train' else 0),
        accounting_method='Completed native sim.step calls, optimizer post-step hook, completed alg.update calls and recorder capture counter; planned denominators remain separate.')
    receipt = args.out / 'report.json'
    receipt.write_text(json.dumps(report, indent=2) + '\n')
    try:
        torch.manual_seed(args.seed)
        robot = runtime_package().robot_cfg(seed=args.seed)
        cfg = bind(morphology_cfg(robot, play=args.mode not in ('train','capacity')))
        cfg.scene.num_envs = args.worlds
        cfg.seed = args.seed
        cfg.auto_reset = args.mode in ('train','capacity')
        cfg.recorders = {'full_episodes': RecorderTermCfg(func=Episodes, params={'directory': str(args.out / 'episodes')})}
        if args.velocity_x is not None:
            c = cfg.commands['twist']; c.ranges.lin_vel_x = (args.velocity_x, args.velocity_x)
            c.ranges.lin_vel_y = (0., 0.); c.ranges.ang_vel_z = (0., 0.)
            c.rel_standing_envs = 0.; c.resampling_time_range = (9., 9.)
        report.update(selected_interface=Interface().contract(),
            implementation_sources=[ref(path) for path in sorted(Path(__file__).resolve().parents[1].rglob('*.py'))],
            runtime_versions=runtime_versions(),
            source_environment=ref(SAVED_ENV), source_agent=ref(SAVED_ENV.parent / 'agent.yaml'),
            rewards=reward_recipe(cfg), command_recipe=serializable(cfg.commands), numerical_options=serializable(cfg.sim),
            terrain_collision_config=serializable(cfg.scene.terrain.collisions),
            actor81='Original72 prefix plus newq3,newqd3,previousrawnew3',
            critic96=CRITIC, actor_value_layout=LAYOUT,
            first_period='First20ms after every reset holds mapped home while sensor_valid=False; raw output/history and physics stay inside episode/transition counts.',
            action_rate='Original21 raw channels only; no unrequested new-head penalty.',
            physical_variation='Nominal selected R3 with saved reset variation. Old-body inertia/friction/effort/PD randomizers are not ported silently.',
            trace_resolution='Every control transition and native qpos/qvel at50Hz; boundary runs4kHz, not a4kHz full-state trace.',
            source_curriculum_counter=0 if args.fresh else 20400,
            source_curriculum_note='Warm-start retains20400 source common steps, initial action-rate−.4. New PPO iteration0; source BC800updates/inheritediter844 remain provenance.')
        (args.out / 'contract.json').write_text(json.dumps(report, indent=2) + '\n')
        env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
        native_step = env.sim.step
        def counted_native_step(*a, **kw):
            result = native_step(*a, **kw)
            counts['native_physics_steps'] += args.worlds
            return result
        env.sim.step = counted_native_step
        recorder = env.recorder_manager._terms[0]
        vec = RslRlVecEnvWrapper(env)
        agent = agent_config(); agent.logger = 'tensorboard'; agent.save_interval = max(1, args.updates)
        runner = MjlabOnPolicyRunner(vec, asdict(agent), log_dir=str(args.out / 'learner'), device=args.device)
        def optimizer_step_completed(optimizer, a, kw): counts['optimizer_steps'] += 1
        runner.alg.optimizer.register_step_post_hook(optimizer_step_completed)
        update = runner.alg.update
        def counted_update(*a, **kw):
            result = update(*a, **kw)
            counts['ppo_rounds'] += 1
            return result
        runner.alg.update = counted_update
        transfer = {} if args.fresh else initialize(runner,env,source_checkpoint=args.source_checkpoint,checkpoint=args.checkpoint)
        report['transfer'] = transfer
        report['source_curriculum_counter'] = env.common_step_counter
        report['algorithm'] = serializable(agent.algorithm)
        report['saved_agent_yaml_learning_rate'] = report['algorithm']['learning_rate']
        report['algorithm']['learning_rate'] = runner.alg.learning_rate
        # Save exact completed counts/Adam through our checkpoint writer. The
        # upstream writer uses a zero-based iteration label and otherwise
        # duplicates large source-identical checkpoints in learner/.
        runner.save = lambda path, infos=None: save_checkpoint(runner,Path(path),transfer,counts) if Path(path).parent == args.out else None
        native_env_step = env.step
        def recorded_step(action):
            with torch.inference_mode():
                value = runner.alg.transition.values if args.mode == 'train' else runner.alg.critic(vec.get_observations())
                recorder.cache_inputs(value)
            return native_env_step(action)
        env.step = recorded_step
        if args.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
        collection_start = time.monotonic()
        if args.mode == 'train':
            runner.learn(num_learning_iterations=args.updates, init_at_random_ep_len=False)
            report['candidate'] = save_checkpoint(runner,args.out/'candidate.pt',transfer,counts)
        else:
            policy = runner.get_inference_policy(device=args.device)
            steps = args.steps if args.mode in ('smoke','capacity') else env.max_episode_length
            if args.device.startswith('cuda'): torch.cuda.synchronize()
            measured_start = time.monotonic()
            with torch.inference_mode():
                for index in range(steps):
                    if args.mode == 'capacity' and index == steps//2:
                        torch.cuda.synchronize(); measured_start = time.monotonic()
                    env.step(policy(vec.get_observations(), stochastic_output=args.mode == 'capacity'))
                    if not cfg.auto_reset: env._manual_reset_pending[:] = False
            if args.device.startswith('cuda'): torch.cuda.synchronize()
            report['measured_collection_s'] = time.monotonic()-measured_start
            report['measured_controls_per_world'] = steps-steps//2 if args.mode == 'capacity' else steps
            report['collection_wall_s'] = time.monotonic()-collection_start
            reason = 'requested_horizon' if args.mode == 'replay' else 'capacity_budget_censoring' if args.mode == 'capacity' else 'interface_smoke_censoring'
            for i in range(args.worlds): recorder.finish(i, reason)
        report['status'] = 'completed'
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
            error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc()[-8000:])
        if env is not None:
            try:
                path = args.out / 'interrupted-native-state.npz'
                np.savez_compressed(path, qpos=env.sim.data.qpos.cpu().numpy(),
                    qvel=env.sim.data.qvel.cpu().numpy(), native_time_s=env.sim.data.time.cpu().numpy())
                report['interrupted_native_state'] = ref(path)
            except Exception as state_exc: report['interrupted_native_state_error'] = str(state_exc)
        if runner is not None and counts['optimizer_steps']:
            try:
                report['partial_candidate'] = save_checkpoint(runner,args.out/'partial-candidate.pt',report.get('transfer',{}),counts)
            except Exception as save_exc: report['partial_candidate_error'] = str(save_exc)
        raise
    finally:
        if runner is not None:
            report['adam_parameter_step_counts'] = sorted({int(state['step'])
                for state in runner.alg.optimizer.state.values() if 'step' in state})
            report['final_effective_learning_rate'] = runner.alg.learning_rate
        if args.device.startswith('cuda') and torch.cuda.is_available():
            report['cuda_memory'] = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(), free_total_bytes=list(torch.cuda.mem_get_info()))
        if env is not None:
            try: env.close()
            except Exception as close_exc: report['close_error'] = str(close_exc)
        report.update(wall_s=time.monotonic()-start,
            world_slots_constructed=env.num_envs if env is not None else 0,
            independent_start_families=int(recorder is not None and bool(recorder.initial)),
            simulator_transitions=recorder.transitions if recorder is not None else 0,
            **counts,
            complete_or_terminated_episodes=recorder.completed if recorder is not None else 0,
            censored_episodes=recorder.censored if recorder is not None else 0,
            distinct_new_trained_policies=int(counts['optimizer_steps'] > 0),
            claim_limit='Native evidence on one assumed model; no physical qualification or general robustness.')
        report['retained_episode_bytes'] = sum(p.stat().st_size for p in (args.out/'episodes').glob('*.npz'))
        receipt.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('mode','worlds','simulator_transitions','optimizer_steps','wall_s')}))


if __name__ == '__main__': main()
