"""GPU stages of the WBC teacher pilots: benchmark, train, evaluate.

Runs inside the worker's isolated venv from the extracted capsule root with
plain upstream mjlab (no task plugins). Each stage writes
<evidence>/<stage>/report.json and its native outputs. --variant selects the
native BeyondMimic actor (T-e7vm69) or the GEAR-SONIC actor (sonic.py), warm
started from the released G1 weights or randomly initialized.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sonic  # noqa: E402
import tracking  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.tracking import mdp  # noqa: E402
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner  # noqa: E402

DEVICE = 'cuda:0'  # --device cpu only for local smoke runs


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def hardware():
    if not torch.cuda.is_available():
        return dict(gpu=None, torch=torch.__version__, device=DEVICE)
    return dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
        memory_GB=torch.cuda.get_device_properties(0).total_memory / 1e9)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def task(args):
    return sonic if args.variant.startswith('sonic') else tracking


def plant(args):
    """Reward widths and servo options shared by every stage's env_cfg."""
    out = dict(reward_widths=args.reward_widths, peak=args.peak_torque, actuator=args.actuator, dynamics=args.dynamics)
    if args.stop_weight:
        out['stop_weight'] = args.stop_weight
    if args.joint_weight:
        out['joint_weight'] = args.joint_weight
    if args.knee_weight:
        out['knee_weight'] = args.knee_weight
    if args.reward_set != 'v1':
        out['reward_set'] = args.reward_set
    if args.command_delay is not None:
        out['command_delay'] = args.command_delay
    if args.feedback_delay is not None:
        out['feedback_delay'] = args.feedback_delay
    if args.feet_scale:
        out['feet_scale'] = tuple(args.feet_scale)
    if args.arm_scale:
        out['arm_scale'] = args.arm_scale[0] if len(args.arm_scale) == 1 else tuple(args.arm_scale)
    if args.fist:
        out['fist'] = tuple(args.fist)
    if getattr(args, 'terrain', None):
        out['terrain'] = args.terrain
        if getattr(args, 'flat_references', False):
            out['slope_references'] = False
    if args.reference_lead:
        out['reference_lead'] = args.reference_lead
    if args.preview_steps:
        out['preview_steps'] = tuple(args.preview_steps)
    if args.residual_action:
        out['residual_action'] = True
    if args.anchor_tolerance:
        out['anchor_tolerance'] = args.anchor_tolerance
    if args.error_sampling:
        out['error_sampling'] = args.error_sampling
    if args.physics_dt:
        out['physics_dt'] = args.physics_dt
    return out


def dynamics_record(args):
    """The resolved plant dynamics for the reports: None for R3, the r4 sidecar parameters otherwise."""
    return tracking.runtime_dynamics().resolve(args.actuator) if args.dynamics == 'r4' else None


def agent(args, env, **overrides):
    """(PPO cfg, runner dict); SONIC arms swap in the SONIC actor on R3 actuator rows."""
    if args.variant.startswith('sonic'):
        overrides['recipe'] = args.recipe
    cfg = task(args).ppo_cfg(**overrides)
    if args.minibatch_samples:
        # More worlds per iteration: keep each gradient step's sample count, take more steps.
        cfg.algorithm.num_mini_batches = max(cfg.algorithm.num_mini_batches,
            round(env.num_envs * cfg.num_steps_per_env / args.minibatch_samples))
    if not args.variant.startswith('sonic'):
        return cfg, asdict(cfg)
    names = env.unwrapped.action_manager.get_term('joint_pos').target_names
    # sonic-random keeps the release FSQ levels with random layers.
    return cfg, dict(asdict(cfg), actor=dict(class_name=sonic.model_class(), rows=sonic.rows_for(names),
        weights=str(args.sonic), pretrained=args.variant == 'sonic-warm', init_std=sonic.INIT_STD))


def benchmark(args):
    """Batched control-step throughput by world count (random actions, no learning)."""
    rows = []
    for worlds in args.worlds:
        row = dict(worlds=worlds)
        try:
            start = time.perf_counter()
            env = ManagerBasedRlEnv(cfg=task(args).env_cfg(args.motion, num_envs=worlds, **plant(args)), device=DEVICE)
            env.reset(seed=args.seed)
            row['build_s'] = time.perf_counter() - start
            dim = env.action_manager.total_action_dim
            for _ in range(3):
                env.step(torch.randn(worlds, dim, device=DEVICE))
            sync()
            start = time.perf_counter()
            for _ in range(args.steps):
                env.step(torch.randn(worlds, dim, device=DEVICE))
            sync()
            elapsed = time.perf_counter() - start
            row.update(control_steps=args.steps, step_s=elapsed / args.steps, transitions_per_s=worlds * args.steps / elapsed,
                physics_steps_per_s=worlds * args.steps * env.cfg.decimation / elapsed,
                peak_memory_GB=torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None)
            env.close()
            del env
        except Exception as exc:  # e.g. out of memory: retain as evidence, try the next size
            row.update(error_type=type(exc).__name__, error=str(exc)[:2000])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        rows.append(row)
        print(json.dumps(row), flush=True)
    good = [r for r in rows if 'transitions_per_s' in r]
    # The most worlds within 5% of the best throughput: more samples per PPO iteration at no throughput cost.
    best = max((r['transitions_per_s'] for r in good), default=None)
    chosen = max((r for r in good if r['transitions_per_s'] >= .95 * best), key=lambda r: r['worlds']) if good else None
    write(args.out / 'report.json', dict(schema='cove50.wbc.benchmark.v1', hardware=hardware(), motion=str(args.motion),
        rows=rows, chosen_worlds=chosen and chosen['worlds'], native=dict(simulator_transitions=sum(r.get('control_steps', 0) * r['worlds'] for r in rows),
        optimizer_steps=0, world_slots_constructed=sum(r['worlds'] for r in rows))))


def flattened(chunks, window, gain):
    """A compute stop, not a grade: the mean return of the last `window` full
    chunks (~5 min each) against the `window` before them. The two resume
    warm-up chunks (random episode lengths, then a refilling buffer) and short
    tail chunks are left out. Records the gain on the newest chunk."""
    full = [c['mean_reward'] for c in chunks[2:] if c['iterations'] >= 10 and c['mean_reward'] is not None]
    if len(full) < 2 * window:
        return None
    recent, before = float(np.mean(full[-window:])), float(np.mean(full[-2 * window:-window]))
    chunks[-1]['window_gain'] = (recent - before) / abs(before)
    if chunks[-1]['window_gain'] < gain:
        return dict(window_chunks=window, gain_below=gain, recent_mean=recent, before_mean=before,
            gain=chunks[-1]['window_gain'], chunk=len(chunks) - 1)
    return None


def widened(checkpoint, runner, stats=None):
    """checkpoint, or a copy whose actor and critic take the env's extra trailing observations (reference_preview)
    through zero input columns, so the retained teacher's behaviour is unchanged. stats = (mean, var) of the extra
    columns seed their normaliser (otherwise 0 and 1)."""
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    grown = {}
    for key, net in (('actor_state_dict', runner.alg.actor), ('critic_state_dict', runner.alg.critic)):
        sd, target = ck[key], net.mlp[0].in_features
        old = sd['mlp.0.weight'].shape[1]
        if old == target:
            continue
        if old > target:
            raise ValueError(f'{key}: checkpoint takes {old} observations, the env gives {target}')
        extra = target - old
        grown[tuple(sd['mlp.0.weight'].shape)] = extra
        sd['mlp.0.weight'] = torch.cat([sd['mlp.0.weight'], sd['mlp.0.weight'].new_zeros(sd['mlp.0.weight'].shape[0], extra)], 1)
        mean, var = (torch.zeros(extra), torch.ones(extra)) if stats is None else (stats[0].cpu(), stats[1].cpu().clamp(min=1e-6))
        if len(mean) != extra:
            raise ValueError(f'{key}: {extra} extra observations, statistics for {len(mean)}')
        for name, value in (('_mean', mean), ('_var', var), ('_std', var.sqrt())):
            t = sd[f'obs_normalizer.{name}']
            sd[f'obs_normalizer.{name}'] = torch.cat([t, value.to(t.dtype)[None]], 1)
    if not grown:
        return checkpoint
    for state in ck['optimizer_state_dict'].get('state', {}).values():  # Adam moments of the widened input layers
        for name in ('exp_avg', 'exp_avg_sq'):
            t = state.get(name)
            if t is not None and tuple(t.shape) in grown:
                state[name] = torch.cat([t, t.new_zeros(t.shape[0], grown[tuple(t.shape)])], 1)
    path = Path(tempfile.mkdtemp()) / 'init-widened.pt'  # derived from the checkpoint; kept out of the evidence
    torch.save(ck, path)
    return path


def train(args):
    worlds = args.worlds[0] if args.worlds else json.loads((args.benchmark / 'report.json').read_text())['chosen_worlds']
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=task(args).env_cfg(args.motion, num_envs=worlds, **plant(args)), device=DEVICE))
    agent_cfg, runner_cfg = agent(args, env, logger='tensorboard', seed=args.seed, save_interval=10**9)
    log = args.out / 'log'
    runner = MotionTrackingOnPolicyRunner(env, runner_cfg, str(log), DEVICE)
    init = None
    if args.init_checkpoint:  # continue a retained teacher: actor, critic, noise, normalizers, optimizer
        stats = None
        if args.preview_steps:
            stats = tracking.preview_statistics(env.unwrapped.command_manager.get_term('motion').motion.joint_pos, args.preview_steps)
        runner.load(str(widened(args.init_checkpoint, runner, stats)), map_location=DEVICE)
        # Resume the adaptive rate where the teacher left it, not at the 1e-3 default.
        runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]['lr']
        init = dict(path=str(args.init_checkpoint), sha256=hashlib.sha256(args.init_checkpoint.read_bytes()).hexdigest(),
            iteration=runner.current_learning_iteration, std=runner.alg.actor.distribution.std_param.mean().item(),
            learning_rate=runner.alg.learning_rate)
        if args.init_widths:  # the teacher's last reward widths, from its train report beside the checkpoint
            last = json.loads((args.init_checkpoint.parent / 'report.json').read_text())['chunks'][-1]['adaptive_std']
            init['widths'] = tracking.set_adaptive_widths(env, last)
        if args.reset_actor_output:
            # Keep the hidden features, critic, noise and normalizers; zero the action head so every
            # joint starts at its default target (a plant change made the old commands meaningless).
            head = [m for m in runner.alg.actor.modules() if isinstance(m, torch.nn.Linear)][-1]
            assert head.out_features == env.num_actions, head
            with torch.no_grad():
                head.weight.zero_()
                head.bias.zero_()
            init['reset_actor_output'] = True
    warmup = 0
    if args.variant.startswith('sonic'):
        sonic.split_optimizer(runner.alg, args.recipe)  # the critic learns alone first
        warmup = sonic.CRITIC_WARMUP
    # Per-iteration mean of the last 100 episode returns, the value rsl-rl
    # writes as Train/mean_reward, for the plateau stop below.
    rewards, log_iteration = [], runner.logger.log

    def logged(*a, **k):
        log_iteration(*a, **k)
        if runner.logger.rewbuffer:
            rewards.append(float(np.mean(runner.logger.rewbuffer)))
    runner.logger.log = logged
    plateau = None
    began = time.perf_counter()
    deadline = began + args.seconds
    cap = args.iterations or 10**9
    chunks, chunk, first = [], min(warmup or 5, cap), True
    while True:
        start = time.perf_counter()
        seen = len(rewards)
        runner.learn(num_learning_iterations=chunk, init_at_random_ep_len=first)
        first = False
        elapsed = time.perf_counter() - start
        chunks.append(dict(iterations=chunk, wall_s=elapsed, actor_lr=runner.alg.optimizer.param_groups[0]['lr'], adaptive_std=tracking.adaptive_widths(env),
            mean_reward=float(np.mean(rewards[seen:])) if len(rewards) > seen else None))
        per_iteration = elapsed / chunk
        if args.plateau_window:
            plateau = flattened(chunks, args.plateau_window, args.plateau_gain)
        print(json.dumps(chunks[-1]), flush=True)
        # learn() saves model_<it>.pt when it ends; keep only the newest, as
        # latest.pt (atomic replace), for the worker's mid-run upload.
        saved = sorted(log.glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[1]))
        if saved:
            os.replace(saved[-1], args.out / 'latest.pt')
            for path in saved[:-1]:
                path.unlink()
        if warmup and len(chunks) == 1:
            sonic.release_actor(runner.alg, args.recipe)
        done = sum(c['iterations'] for c in chunks)
        remaining = deadline - time.perf_counter()
        # Chunks of roughly five minutes; stop while a full chunk still fits.
        chunk = max(1, min(int(300 / per_iteration), int(remaining / per_iteration / 1.2), cap - done))
        if remaining < 1.2 * per_iteration or done >= cap or plateau:
            break
    final = args.out / 'teacher.pt'
    runner.save(str(final))
    # rsl-rl's current_learning_iteration is the last index, not a count.
    iterations = sum(c['iterations'] for c in chunks)
    transitions = iterations * worlds * agent_cfg.num_steps_per_env
    write(args.out / 'report.json', dict(schema='cove50.wbc.train.v1', variant=args.variant, hardware=hardware(),
        motion=str(args.motion), worlds=worlds, seed=args.seed, wall_s=time.perf_counter() - began, chunks=chunks,
        stop=dict(reason='plateau' if plateau else 'iterations' if sum(c['iterations'] for c in chunks) >= cap else 'deadline',
            plateau=plateau, rule=dict(window_chunks=args.plateau_window, gain_below=args.plateau_gain) if args.plateau_window else None,
            iteration_rewards=rewards),
        checkpoint=final.name, init_checkpoint=init, ppo=asdict(agent_cfg), reward_std=tracking.REWARD_STD[args.reward_widths], ranges=tracking.RANGES, action_scale_rad=tracking.ACTION_SCALE,
        peak_torque=args.peak_torque, actuator=dict(name=args.actuator, spec=tracking.ACTUATORS[args.actuator]),
        dynamics=dict(name=args.dynamics, resolved=dynamics_record(args)),
        sonic=dict(sha256={k: v[0] for k, v in sonic.FILES.items()}, recipe=args.recipe, actor_lr=sonic.ACTOR_LR,
            critic_lr=sonic.CRITIC_LR, init_std=sonic.INIT_STD, desired_kl=sonic.DESIRED_KL, trust_actor_lr=sonic.TRUST_ACTOR_LR,
            critic_warmup_iterations=warmup, map=sonic.MAP) if warmup else None,
        native=dict(simulator_transitions=transitions, native_physics_steps=transitions * env.unwrapped.cfg.decimation,
            ppo_rounds=iterations, optimizer_steps=iterations * agent_cfg.algorithm.num_learning_epochs * agent_cfg.algorithm.num_mini_batches,
            world_slots_constructed=worlds)))
    # Intermediate end-of-chunk checkpoints are inferior candidates of the
    # same run; keep the final checkpoint, its ONNX export and the logs.
    for path in [*log.glob('model_*.pt'), args.out / 'latest.pt']:
        path.unlink(missing_ok=True)


def pushes(worlds, device):
    """Eight horizontal directions x two speeds, one per world (G1 push range x HEIGHT_RATIO)."""
    speed = torch.tensor([.1 if i < worlds // 2 else .2 for i in range(worlds)], device=device)
    angle = torch.tensor([2 * math.pi * (i % 8) / 8 for i in range(worlds)], device=device)
    return torch.stack((speed * torch.cos(angle), speed * torch.sin(angle)), dim=1)


def episode(env, mode, policy, push_step, kick, max_steps=None):
    command = env.unwrapped.command_manager.get_term('motion')
    action = env.unwrapped.action_manager.get_term('joint_pos')
    robot = env.unwrapped.scene['robot']
    rt = tracking.runtime().get_runtime(env.unwrapped)
    ends = tuple(command.cfg.body_names.index(n) for n in tracking.END_EFFECTORS)
    obs, _ = env.reset()
    log = {k: [] for k in ('qpos', 'qvel', 'ref_joint_pos', 'ref_anchor_pos', 'ref_anchor_quat', 'ref_end_z', 'end_z',
        'target', 'applied_target', 'torque', 'current', 'saturated', 'flag_anchor_pos', 'flag_anchor_ori', 'flag_end_pos')}
    if getattr(command.cfg, 'slope_references', False):
        log['ref_joint_pos_tape'] = []  # the flat tape's joint angles, next to the slope-aware reference
    steps = min(command.motion.time_step_total - 1, max_steps or 10**9)
    for t in range(steps):
        if push_step is not None and t == push_step:
            velocity = robot.data.root_link_vel_w.clone()
            velocity[:, :2] += kick
            robot.write_root_link_velocity_to_sim(velocity)
        if mode == 'teacher':
            with torch.inference_mode():
                act = policy(obs)
        else:
            act = (command.joint_pos[:, action._target_ids] - action._offset) / action._scale
        obs, _, _, _ = env.step(act)
        d = robot.data
        log['qpos'].append(torch.cat((d.root_link_pos_w, d.root_link_quat_w, d.joint_pos), dim=1))
        log['qvel'].append(torch.cat((d.root_link_lin_vel_w, d.root_link_ang_vel_w, d.joint_vel), dim=1))
        # The reference at the robot state's time, also when the command leads it (tracking.ErrorAwareMotionCommand).
        now = getattr(command, 'reference_now', None)
        if now is None:
            log['ref_joint_pos'].append(command.joint_pos)
            log['ref_anchor_pos'].append(command.anchor_pos_w)
            log['ref_anchor_quat'].append(command.anchor_quat_w)
            log['ref_end_z'].append(command.body_pos_relative_w[:, ends, 2])
        else:
            anchor = command.motion_anchor_body_index
            log['ref_joint_pos'].append(now('joint_pos'))
            if 'ref_joint_pos_tape' in log:
                log['ref_joint_pos_tape'].append(command.motion.joint_pos[command.time_steps])
            log['ref_anchor_pos'].append(now('body_pos_w', anchor) + env.unwrapped.scene.env_origins)
            log['ref_anchor_quat'].append(now('body_quat_w', anchor))
            log['ref_end_z'].append(now('body_pos_w', list(ends))[..., 2])
        log['end_z'].append(command.robot_body_pos_w[:, ends, 2])
        log['target'].append(action._processed_actions)
        log['applied_target'].append(rt.target)
        log['torque'].append(rt.torque)
        log['current'].append(rt.current)
        log['saturated'].append(rt.saturated)
        u = env.unwrapped
        log['flag_anchor_pos'].append(mdp.bad_anchor_pos_z_only(u, 'motion', tracking.TERMINATION_HEIGHT_M))
        log['flag_anchor_ori'].append(mdp.bad_anchor_ori(u, SceneEntityCfg('robot'), 'motion', .8))
        log['flag_end_pos'].append(mdp.bad_motion_body_pos_z_only(u, 'motion', tracking.TERMINATION_HEIGHT_M, tracking.END_EFFECTORS))
    return {k: torch.stack(v).cpu().numpy() for k, v in log.items()}


def summarize(trace, push_step):
    q, ref = trace['qpos'][..., 7:], trace['ref_joint_pos']
    error = np.abs(q - ref).mean(axis=2)  # [T, W] rad
    height_error = trace['qpos'][..., 2] - trace['ref_anchor_pos'][..., 2]
    flags = trace['flag_anchor_pos'] | trace['flag_anchor_ori'] | trace['flag_end_pos']
    first = np.where(flags.any(0), flags.argmax(0), -1)
    pct = lambda x: {p: float(np.percentile(x, p)) for p in (10, 50, 90)}
    # The motion command wraps to the clip start after its last step, so the
    # final row's reference is the start; compare against the row before it.
    root, path = trace['qpos'][-2, :, :2], trace['ref_anchor_pos'][-2, :, :2]
    return dict(worlds=int(q.shape[1]), control_steps=int(q.shape[0]), push_step=push_step,
        joint_abs_error_rad=dict(per_world_mean=pct(error.mean(0)), per_world_max=pct(error.max(0))),
        anchor_height_error_m=dict(min=float(height_error.min()), per_world_min=pct(height_error.min(0))),
        native_termination_condition=dict(worlds_flagged=int((first >= 0).sum()),
            first_flag_s=[float(s / 50) if s >= 0 else None for s in first]),
        peak_abs_torque_Nm=float(np.abs(trace['torque']).max()),
        saturated_actuator_fraction=float(trace['saturated'].mean()),
        final_root_xy_error_m=pct(np.linalg.norm(root - path, axis=1)),
        root_travel_m=pct(np.linalg.norm(root - trace['qpos'][0, :, :2], axis=1)),
        reference_travel_m=pct(np.linalg.norm(path - trace['ref_anchor_pos'][0, :, :2], axis=1)))


def world_record(u):
    """Per-world draws that stay fixed through an evaluation: spawn origin, lowest sole friction (the floor's 0.3 base
    is lower, so the sole's value decides on the obstacle world) and forearm mass with any carried toy."""
    try:
        m, model = u.sim.mj_model, u.sim.model
        tensor = lambda a: (a if isinstance(a, torch.Tensor) else a[:]).detach()  # mjlab TorchArray: indexing gives the tensor
        feet = [i for i in range(m.ngeom) if (m.geom(i).name or '').split('/')[-1].startswith(('collision_R_foot_', 'collision_L_foot_'))]
        friction, mass = tensor(model.geom_friction).cpu(), tensor(model.body_mass).cpu()
        sole = friction[:, feet, 0] if friction.ndim == 3 else friction[None, feet, 0].expand(u.num_envs, -1)
        arms = [m.body(f'robot/{s}_forearm').id for s in 'RL']
        forearms = mass[:, arms] if mass.ndim == 2 else mass[None, arms].expand(u.num_envs, -1)
        origins = u.scene.env_origins.cpu()
        return [dict(origin_m=[round(float(v), 3) for v in origins[w]], sole_friction_min=round(float(sole[w].min()), 3),
            sole_friction_max=round(float(sole[w].max()), 3), forearm_mass_kg=[round(float(v), 4) for v in forearms[w]])
            for w in range(u.num_envs)]
    except Exception as exc:  # a record, never a reason to lose the evaluation
        return dict(unavailable=f'{type(exc).__name__}: {exc}')


def evaluate(args):
    cfg = task(args).env_cfg(args.motion, play=True, num_envs=args.eval_worlds, **plant(args))
    for name in ('anchor_pos', 'anchor_ori', 'ee_body_pos'):
        cfg.terminations.pop(name)  # full episodes; native conditions become per-step flags
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=DEVICE))
    runner = MotionTrackingOnPolicyRunner(env, agent(args, env, logger='tensorboard')[1], None, DEVICE)
    if args.checkpoint:  # without one, sonic-warm evaluates the released weights zero-shot
        runner.load(str(widened(args.checkpoint, runner)), load_cfg=dict(actor=True), map_location=DEVICE)
    policy = runner.get_inference_policy(device=DEVICE)
    kick = pushes(args.eval_worlds, DEVICE)
    tiles = None
    if args.terrain:  # fixed tiles, the same in every evaluation, so runs compare world by world
        import parcour
        tiles = parcour.place_eval_worlds(env.unwrapped)
    worlds = world_record(env.unwrapped)
    if tiles and isinstance(worlds, list):
        worlds = [dict(w, **t) for w, t in zip(worlds, tiles)]
    elif tiles:
        worlds = dict(worlds, tiles=tiles)
    rows = []
    command = env.unwrapped.command_manager.get_term('motion')
    segments = [dict(name=None, start=0, frames=None)]
    if args.segments:  # tape segments: each episode starts at the segment's first frame (with the usual start draws)
        listed = json.loads(args.segments.read_text())
        listed = listed['segments'] if 'segments' in listed else listed['motions'][0]['segments']
        segments = [g for g in listed if not args.segment_names or g['name'] in args.segment_names]
        command.cfg.sampling_mode = 'uniform'
    for segment in segments:
        if segment['name']:
            command._uniform_sampling = lambda ids, start=segment['start']: command.time_steps.__setitem__(ids, start)
        steps = segment['frames'] - 1 if segment['frames'] else None
        steps = min(steps, args.max_steps) if steps and args.max_steps else steps or args.max_steps
        for condition, push_step in [c for c in (('nominal', None), ('pushed', args.push_step)) if c[0] in args.conditions]:
            for mode in args.modes:
                torch.manual_seed(args.seed)
                trace = episode(env, mode, policy, push_step, kick if push_step is not None else None, steps)
                name = f"{segment['name'] + '-' if segment['name'] else ''}{condition}-{mode}.npz"
                np.savez_compressed(args.out / name, push_velocity_xy=kick.cpu().numpy(), **trace)
                rows.append(dict(segment=segment['name'], condition=condition, mode=mode, trace=name, **summarize(trace, push_step)))
                print(json.dumps({k: rows[-1][k] for k in ('segment', 'condition', 'mode', 'joint_abs_error_rad')}), flush=True)
    steps = sum(r['control_steps'] * r['worlds'] for r in rows)
    write(args.out / 'report.json', dict(schema='cove50.wbc.evaluate.v1', variant=args.variant, hardware=hardware(), motion=str(args.motion),
        checkpoint=str(args.checkpoint), peak_torque=args.peak_torque, actuator=args.actuator,
        dynamics=dict(name=args.dynamics, resolved=dynamics_record(args)), push_velocity_xy_m_s=kick.cpu().tolist(), worlds=worlds, rows=rows,
        native=dict(simulator_transitions=steps, optimizer_steps=0, world_slots_constructed=args.eval_worlds)))


def main():
    global DEVICE
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('stage', choices=('benchmark', 'train', 'evaluate'))
    parser.add_argument('--motion', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worlds', type=int, nargs='*', default=[])
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--seconds', type=float, default=2700)
    parser.add_argument('--benchmark', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--eval-worlds', type=int, default=16)
    parser.add_argument('--push-step', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--variant', choices=('native', 'sonic-warm', 'sonic-random'), default='native')
    parser.add_argument('--sonic', type=Path, help='directory with the pinned SONIC ONNX files (sonic arms)')
    parser.add_argument('--recipe', choices=sonic.RECIPES, default='release', help='SONIC fine-tune recipe')
    parser.add_argument('--iterations', type=int, help='stop after this many PPO iterations (matched arms)')
    parser.add_argument('--init-checkpoint', type=Path, help='continue training from this retained teacher.pt')
    parser.add_argument('--init-widths', action='store_true', help='with --init-checkpoint: start the adaptive reward widths at the teacher\'s last ones (its report.json)')
    parser.add_argument('--reward-widths', choices=sorted(tracking.REWARD_STD), default='g1', help='tracking reward widths (tracking.REWARD_STD)')
    parser.add_argument('--minibatch-samples', type=int, help='scale PPO mini-batches to keep this many samples per gradient step')
    parser.add_argument('--peak-torque', action='store_true', help="servo stall-torque curve instead of the rated-torque cap")
    parser.add_argument('--actuator', choices=sorted(tracking.ACTUATORS), default='c047', help='servo law on the C047 joints (tracking.ACTUATORS)')
    parser.add_argument('--segments', type=Path, help='tape motions.json (or segments.json): evaluate each segment from its first frame')
    parser.add_argument('--segment-names', nargs='*', help='evaluate only these segments')
    parser.add_argument('--modes', nargs='*', choices=('teacher', 'reference'), default=['teacher', 'reference'])
    parser.add_argument('--dynamics', choices=tracking.DYNAMICS, default='r3', help='plant actuator dynamics: R3, or the r4 best-estimate sidecar')
    parser.add_argument('--plateau-window', type=int, help='stop training when the mean return of the last N full chunks...')
    parser.add_argument('--plateau-gain', type=float, default=0.01, help='...is less than this fraction above the N chunks before them')
    parser.add_argument('--stop-weight', type=float, help='weight of the target_beyond_stop penalty (rad past the joint stops)')
    parser.add_argument('--joint-weight', type=float, help='weight of the joint_position_tracking reward (reference joint angles)')
    parser.add_argument('--knee-weight', type=float, help='weight of the knee_tracking reward (reference knee angles, 0.1 rad width)')
    parser.add_argument('--reward-set', choices=('v1', 'v2', 'v3', 'v3k', 'v3r', 'v3rk', 'v4', 'v4k', 'v5', 'v5k', 'v6', 'v6k'), default='v1', help='v2: adaptive tracking widths plus joint-velocity and foot tracking')
    parser.add_argument('--command-delay', type=float, help='actuation delay in s (whole physics ticks; R3 timing 0.02)')
    parser.add_argument('--feedback-delay', type=float, help='joint feedback delay in s (R3 timing 0.01)')
    parser.add_argument('--feet-scale', type=float, nargs=2, help='foot collision length and width scale (tracking.scale_feet)')
    parser.add_argument('--reference-lead', type=int, help='control frames the command and rewards lead the reference by (tracking.ErrorAwareMotionCommand)')
    parser.add_argument('--fist', type=float, nargs=2, help='knuckle fist length (m) and mass (kg) on each forearm (tracking.add_fists)')
    parser.add_argument('--arm-scale', type=float, nargs='+', help='upper-arm and forearm length scale, one or two values (tracking.scale_arms)')
    parser.add_argument('--preview-steps', type=int, nargs='+', help='control frames of reference preview the actor and critic see (tracking.reference_preview)')
    parser.add_argument('--residual-action', action='store_true', help='actions are offsets on the reference joint angles')
    parser.add_argument('--anchor-tolerance', type=float, help='torso-height termination threshold in m (end effectors keep theirs)')
    parser.add_argument('--error-sampling', type=float, help='share of adaptive starts drawn by per-bin joint-angle error')
    parser.add_argument('--physics-dt', type=float, help='r4 physics step in s (default R3 .00025)')
    parser.add_argument('--reset-actor-output', action='store_true', help='with --init-checkpoint: zero the action head, keep features')
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--max-steps', type=int, help='cap evaluation episodes (local smoke only)')
    parser.add_argument('--terrain', choices=('parcour',), help='obstacle world (parcour.py) instead of the plane')
    parser.add_argument('--flat-references', action='store_true', help='terrain: keep the flat tape on inclined ground (no slope references)')
    parser.add_argument('--conditions', nargs='+', choices=('nominal', 'pushed'), default=['nominal', 'pushed'], help='evaluation conditions')
    args = parser.parse_args()
    DEVICE = args.device
    args.out.mkdir(parents=True, exist_ok=True)
    dict(benchmark=benchmark, train=train, evaluate=evaluate)[args.stage](args)


if __name__ == '__main__':
    main()
