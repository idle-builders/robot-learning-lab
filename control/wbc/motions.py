"""Convert retained R3 skill episodes into native MJLab tracking motion files.

Each source is an exact native episode (qpos29/qvel28 sampled at the 50 Hz
policy rate). Like the native csv_to_npz, body link poses and velocities are
read back from the same MJLab entity the task uses, so array order matches
MotionCommand. Outputs are cheap derived files: references.json plus

    .venv-control/bin/python control/wbc/motions.py --output DIR

regenerate them exactly. Set APE0_FEET_SCALE, APE0_ARM_SCALE and APE0_FIST as for agility/library.py when the references
were authored on a scaled body.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from mjlab.scene import Scene
from mjlab.sim.sim import Simulation
from mjlab.utils.lab_api.math import quat_apply

import tracking

HERE = Path(__file__).resolve().parent
FPS = 50


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def references():
    return json.loads((HERE / 'references.json').read_text())['references']


def stretch(qpos, qvel, speed):
    """The episode replayed at speed x its rate, resampled at 50 Hz; velocities scale with it.

    Positions interpolate linearly and the root quaternion by normalized lerp
    (source frames are 20 ms apart). speed 1 returns the episode unchanged.
    """
    if speed == 1:
        return qpos, qvel
    t = np.arange(0, len(qpos) - 1 + 1e-9, speed)
    i = np.minimum(t.astype(int), len(qpos) - 2)
    w = (t - i)[:, None]
    q = qpos[i] * (1 - w) + qpos[i + 1] * w
    a, b = qpos[i, 3:7], qpos[i + 1, 3:7]
    b = np.where((a * b).sum(1, keepdims=True) < 0, -b, b)
    quat = a * (1 - w) + b * w
    q[:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    return q, (qvel[i] * (1 - w) + qvel[i + 1] * w) * speed


def convert(scene, sim, source, speed=1.):
    if sha(tracking.ROOT / source['trace']) != source['sha256']:
        raise ValueError(f"{source['name']}: retained trace changed")
    episode = np.load(tracking.ROOT / source['trace'])
    source_qpos, source_qvel = stretch(episode['qpos'], episode['qvel'], speed)
    motion = convert_arrays(scene, sim, source_qpos, source_qvel, source['name'])
    robot = scene['robot']
    feet = [robot.body_names.index(n) for n in ('R_foot', 'L_foot')]
    summary = dict(name=source['name'], skill=source['skill'], result_id=source['result_id'], clip=source['clip'],
        trace=source['trace'], trace_sha256=source['sha256'], speed=speed, frames=len(source_qpos), fps=FPS,
        duration_s=len(source_qpos) / FPS, torso_height_m=[float(motion['body_pos_w'][:, 0, 2].min()), float(motion['body_pos_w'][:, 0, 2].max())],
        lowest_foot_origin_m=float(motion['body_pos_w'][:, feet, 2].min()),
        source_terminated=int(np.asarray(episode['source_terminated']).sum()) if 'source_terminated' in episode.files else None)
    return dict(fps=np.array([FPS]), **motion), summary


def convert_arrays(scene, sim, source_qpos, source_qvel, name='motion'):
    """Native tracking arrays (joint and body link states) for qpos29/qvel28 rows at 50 Hz."""
    qpos = torch.as_tensor(source_qpos, dtype=torch.float32, device=sim.device)
    qvel = torch.as_tensor(source_qvel, dtype=torch.float32, device=sim.device)
    if qpos.shape[1:] != (29,) or qvel.shape[1:] != (28,) or len(qpos) != len(qvel):
        raise ValueError(f"{name}: expected native qpos29/qvel28")
    robot = scene['robot']
    log = {k: [] for k in ('joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w')}
    for q, v in zip(qpos, qvel):
        # Free-joint qvel: world linear velocity, body-frame angular velocity.
        # MJLab root state takes both in world frame.
        ang_w = quat_apply(q[None, 3:7], v[None, 3:6])
        robot.write_root_state_to_sim(torch.cat((q[None, :7], v[None, :3], ang_w), dim=1))
        robot.write_joint_state_to_sim(q[None, 7:], v[None, 6:])
        sim.forward()
        scene.update(sim.mj_model.opt.timestep)
        data = robot.data
        for key, value in (('joint_pos', data.joint_pos), ('joint_vel', data.joint_vel),
                ('body_pos_w', data.body_link_pos_w), ('body_quat_w', data.body_link_quat_w),
                ('body_lin_vel_w', data.body_link_lin_vel_w), ('body_ang_vel_w', data.body_link_ang_vel_w)):
            log[key].append(value[0].cpu().numpy().copy())
        torch.testing.assert_close(data.root_link_ang_vel_w[0], ang_w[0])
    motion = {k: np.stack(v) for k, v in log.items()}
    np.testing.assert_allclose(motion['joint_pos'], np.asarray(source_qpos)[:, 7:], atol=1e-5)
    return motion


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--only', nargs='*', help='reference names to convert (default: all)')
    parser.add_argument('--speed', type=float, default=1., help='replay rate; other than 1 writes NAME-speedPCT.npz')
    parser.add_argument('--tape', type=Path, help='agility/tape.py output folder: convert its tape.npz to agility-tape.npz instead')
    args = parser.parse_args()
    # The body the references were authored on (agility/body.py reads the same variables): longer arms move the
    # forearm bodies, so their tracked positions must come from the same body. Foot scaling moves no body frame.
    scales = {}
    if os.environ.get('APE0_FEET_SCALE'):
        scales['feet_scale'] = tuple(float(x) for x in os.environ['APE0_FEET_SCALE'].split(','))
    if os.environ.get('APE0_ARM_SCALE'):
        a = [float(x) for x in os.environ['APE0_ARM_SCALE'].split(',')]
        scales['arm_scale'] = a[0] if len(a) == 1 else tuple(a)
    if os.environ.get('APE0_FIST'):
        scales['fist'] = tuple(float(x) for x in os.environ['APE0_FIST'].split(','))
    cfg = tracking.env_cfg(motion_file='', **scales)
    scene = Scene(cfg.scene, device='cpu')
    sim = Simulation(num_envs=1, cfg=cfg.sim, model=scene.compile(), device='cpu')
    scene.initialize(sim.mj_model, sim.model, sim.data)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.tape:
        tape = np.load(args.tape / 'tape.npz')
        motion = convert_arrays(scene, sim, tape['qpos'], tape['qvel'], 'agility-tape')
        path = args.output / 'agility-tape.npz'
        np.savez(path, fps=np.array([FPS]), **motion)
        segments = json.loads((args.tape / 'segments.json').read_text())
        (args.output / 'motions.json').write_text(json.dumps(dict(schema='cove50.wbc.motions.v1',
            body_names=list(scene['robot'].body_names), joint_names=list(scene['robot'].joint_names),
            motions=[dict(name='agility-tape', frames=int(len(tape['qpos'])), fps=FPS, motion=str(path), motion_sha256=sha(path),
                tape_sha256=segments['tape_sha256'], segments=segments['segments'])]), indent=2) + '\n')
        print(f'agility-tape: {len(tape["qpos"])} frames, {len(segments["segments"])} segments')
        return
    summaries = []
    for source in references():
        if args.only and source['name'] not in args.only:
            continue
        motion, summary = convert(scene, sim, source, args.speed)
        suffix = '' if args.speed == 1 else f'-speed{round(args.speed * 100)}'
        path = args.output / f"{source['name']}{suffix}.npz"
        np.savez(path, **motion)
        summaries.append(dict(**summary, motion=str(path), motion_sha256=sha(path)))
        print(f"{source['name']}: {summary['frames']} frames, torso {summary['torso_height_m'][0]:.3f}-{summary['torso_height_m'][1]:.3f} m")
    (args.output / 'motions.json').write_text(json.dumps(dict(schema='cove50.wbc.motions.v1',
        body_names=list(scene['robot'].body_names), joint_names=list(scene['robot'].joint_names),
        motions=summaries), indent=2) + '\n')


if __name__ == '__main__':
    main()
