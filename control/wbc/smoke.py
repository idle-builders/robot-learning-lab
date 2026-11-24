"""Local CPU smoke of the APE0 R3 port of the native MJLab tracking task.

    infra/lab storage run cove50-wbc-port-smoke -- .venv-control/bin/python \
      control/wbc/smoke.py --scratch '{work}' --output '{evidence}'

Evidence receives the compact report: sources, ranges, per-mode reward terms,
terminations, tracking error and motor demand, PPO wall time and export sizes.
Motion files, the replay sheet and the throwaway PPO checkpoint are cheap and
go to scratch. --inspect copies the replay sheet somewhere that outlives scratch.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
from PIL import Image
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

import motions
import tracking


def write_motions(folder):
    cfg = tracking.env_cfg(motion_file='')
    scene = Scene(cfg.scene, device='cpu')
    sim = Simulation(num_envs=1, cfg=cfg.sim, model=scene.compile(), device='cpu')
    scene.initialize(sim.mj_model, sim.model, sim.data)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in motions.references():
        motion, summary = motions.convert(scene, sim, source)
        path = folder / f"{source['name']}.npz"
        np.savez(path, **motion)
        rows.append(dict(**summary, motion_sha256=motions.sha(path)))
    return rows


def replay_sheet(folder, rows, path, frames=5):
    """Kinematic replay of each converted motion on the native R3 MJCF."""
    model = mujoco.MjModel.from_xml_path(str(tracking.MODEL / 'cove50-rigid.xml'))
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = 1.1, -12, 135
    tiles = []
    with mujoco.Renderer(model, 240, 240) as renderer:
        for row in rows:
            motion = np.load(folder / f"{row['name']}.npz")
            strip = []
            for i in np.linspace(0, len(motion['joint_pos']) - 1, frames).astype(int):
                data.qpos[:3], data.qpos[3:7] = motion['body_pos_w'][i, 0], motion['body_quat_w'][i, 0]
                data.qpos[7:] = motion['joint_pos'][i]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = data.qpos[:3] - [0, 0, .08]
                renderer.update_scene(data, camera)
                strip.append(renderer.render())
            tiles.append(np.concatenate(strip, axis=1))
    Image.fromarray(np.concatenate(tiles, axis=0)).save(path)


def rollout(env, mode, steps, seed):
    """Untrained action baselines: zero target, open-loop reference, random."""
    torch.manual_seed(seed)
    env.reset(seed=seed)
    command = env.command_manager.get_term('motion')
    action = env.action_manager.get_term('joint_pos')
    rt = tracking.runtime().get_runtime(env)
    names = env.reward_manager.active_terms
    rewards, done_terms = [], {n: 0 for n in env.termination_manager.active_terms}
    joint_error, saturated, torque, torso = [], [], [], []
    for _ in range(steps):
        if mode == 'zero':
            act = torch.zeros(env.num_envs, action.action_dim)
        elif mode == 'reference':
            act = (command.joint_pos[:, action._target_ids] - action._offset) / tracking.ACTION_SCALE
        else:
            act = torch.randn(env.num_envs, action.action_dim)
        env.step(act)
        rewards.append(env.reward_manager._step_reward.clone())
        for n in done_terms:
            done_terms[n] += int(env.termination_manager.get_term(n).sum())
        joint_error.append((command.robot_joint_pos - command.joint_pos).abs().mean(dim=1))
        saturated.append(rt.saturated.float().mean(dim=1))
        torque.append(rt.torque.abs().max(dim=1).values)
        torso.append(command.robot_anchor_pos_w[:, 2])
    rewards = torch.stack(rewards)
    stat = lambda t: dict(mean=float(t.mean()), min=float(t.min()), max=float(t.max()))
    return dict(mode=mode, control_steps=steps, worlds=env.num_envs,
        reward_terms={n: stat(rewards[..., i]) for i, n in enumerate(names)},
        all_rewards_finite=bool(torch.isfinite(rewards).all()),
        termination_events=done_terms,
        joint_abs_error_rad=stat(torch.stack(joint_error)),
        saturated_actuator_fraction=stat(torch.stack(saturated)),
        peak_abs_torque_Nm=stat(torch.stack(torque)),
        torso_height_m=stat(torch.stack(torso)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inspect', type=Path)
    parser.add_argument('--rollout-motions', nargs='*', default=['forward-walk', 'deep-squat', 'bilateral-arms'])
    parser.add_argument('--worlds', type=int, default=4)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--ppo-worlds', type=int, default=8)
    parser.add_argument('--ppo-iterations', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    folder = args.scratch / 'motions'
    report = dict(schema='cove50.wbc.port-smoke.v1', idle_task='T-ge3q2k',
        model=tracking.runtime().source_contract()['version'], ranges=tracking.RANGES,
        tracked_bodies=tracking.TRACKED_BODIES, end_effectors=tracking.END_EFFECTORS,
        action_scale_rad=tracking.ACTION_SCALE, termination_height_m=tracking.TERMINATION_HEIGHT_M,
        soft_joint_limit_factor=tracking.SOFT_LIMIT_FACTOR,
        device='cpu', gpu_hours=0)

    start = time.perf_counter()
    report['motions'] = write_motions(folder)
    report['conversion_s'] = time.perf_counter() - start
    sheet = args.scratch / 'replay-sheet.png'
    replay_sheet(folder, report['motions'], sheet)
    if args.inspect:
        args.inspect.mkdir(parents=True, exist_ok=True)
        shutil.copy(sheet, args.inspect / sheet.name)

    report['rollouts'] = []
    for name in args.rollout_motions:
        start = time.perf_counter()
        env = ManagerBasedRlEnv(cfg=tracking.env_cfg(folder / f'{name}.npz', num_envs=args.worlds), device='cpu')
        build_s = time.perf_counter() - start
        for mode in ('zero', 'reference', 'random'):
            start = time.perf_counter()
            row = rollout(env, mode, args.steps, args.seed)
            row.update(motion=name, env_build_s=build_s, wall_s=time.perf_counter() - start,
                actor_obs_dim=int(env.observation_manager.group_obs_dim['actor'][0]),
                critic_obs_dim=int(env.observation_manager.group_obs_dim['critic'][0]))
            report['rollouts'].append(row)
            print(f"{name} {mode}: joint error {row['joint_abs_error_rad']['mean']:.3f} rad, "
                  f"terminations {sum(row['termination_events'].values()) - row['termination_events']['time_out']}, {row['wall_s']:.0f} s")
        env.close()

    start = time.perf_counter()
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=tracking.env_cfg(folder / 'forward-walk.npz', num_envs=args.ppo_worlds), device='cpu'))
    agent = tracking.ppo_cfg(logger='tensorboard', max_iterations=args.ppo_iterations, seed=args.seed)
    runner = MotionTrackingOnPolicyRunner(env, asdict(agent), str(args.scratch / 'ppo'), 'cpu')
    runner.learn(num_learning_iterations=args.ppo_iterations, init_at_random_ep_len=True)
    checkpoint = args.scratch / 'ppo' / 'smoke.pt'
    runner.save(str(checkpoint))
    onnx = sorted((args.scratch / 'ppo').rglob('*.onnx'))
    actor = runner.alg.get_policy()
    report['ppo'] = dict(motion='forward-walk', worlds=args.ppo_worlds, iterations=args.ppo_iterations,
        transitions=args.ppo_worlds * agent.num_steps_per_env * args.ppo_iterations,
        wall_s=time.perf_counter() - start,
        actor_parameters=sum(p.numel() for p in actor.parameters()),
        actor_parameters_finite=all(bool(torch.isfinite(p).all()) for p in actor.parameters()),
        onnx_bytes=onnx[0].stat().st_size if onnx else None,
        note='Throwaway checkpoint in scratch; proves the native runner, update and export path only.')
    env.close()
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['ppo'], indent=2))


if __name__ == '__main__':
    main()
