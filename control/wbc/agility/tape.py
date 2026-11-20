"""Chain the agility motion set into one continuous 50 Hz reference tape for multi-motion WBC training.

Each motion is moved so it starts where the previous one ended (root position
and heading), and a 1 s min-jerk blend joins them (joints, torso height and
tilt; feet stay roughly planted because every motion starts and ends standing).
Velocities are recomputed from the whole tape, so the joins are smooth. The
tracker's adaptive sampling then picks hard segments across all motions.
segments.json gives each motion's first frame and length for evaluation.

    .venv-control/bin/python control/wbc/agility/tape.py --motions DIR [DIR ...] --output DIR [--exclude NAME ...]

DIR from library.py (several are chained in the order given, e.g. the agility and ape catalogues). Writes tape.npz (qpos, qvel at 50 Hz) and segments.json;
motions.py --tape converts it to the native tracking motion file.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

import library

FPS = 50
BLEND_S = 1.


def yaw_of(quat):
    w, x, y, z = quat
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def place(qpos, end):
    """qpos moved so its first root pose has `end`'s position (x, y) and heading."""
    q = qpos.copy()
    dyaw = yaw_of(end[3:7]) - yaw_of(q[0, 3:7])
    c, s = np.cos(dyaw), np.sin(dyaw)
    R = np.array([[c, -s], [s, c]])
    q[:, :2] = (q[:, :2] - q[0, :2]) @ R.T + end[:2]
    turn = np.array([np.cos(dyaw / 2), 0, 0, np.sin(dyaw / 2)])
    for t in range(len(q)):
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, turn, q[t, 3:7])
        q[t, 3:7] = out
    return q


def blend(a, b, frames):
    """Min-jerk blend from pose a to pose b (b already placed at a's position and heading)."""
    out = []
    for i in range(1, frames + 1):
        s = library.minjerk(i / (frames + 1))
        q = a + (b - a) * s
        qb = b[3:7] if a[3:7] @ b[3:7] >= 0 else -b[3:7]
        quat = a[3:7] * (1 - s) + qb * s  # normalized lerp: the two torso poses differ by a few degrees
        q[3:7] = quat / np.linalg.norm(quat)
        out.append(q)
    return np.array(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--motions', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exclude', nargs='*', default=[])
    parser.add_argument('--blend-s', type=float, default=BLEND_S, help='blend between motions (training tapes: 1 s)')
    parser.add_argument('--blend-range', type=float, nargs=2, help='draw each blend uniformly from this range (s) instead')
    parser.add_argument('--clip-stops', type=float, help='clip every joint angle to its stop less this margin (rad): '
        'references past a stop ask for what the body cannot do (chest beats: shoulder yaw 0.21 rad past its 60 deg stop)')
    parser.add_argument('--shuffle', type=int, help='shuffle the motions of each folder with this seed (an order never trained)')
    parser.add_argument('--windows', type=int, help='also list windows of this many consecutive motions of the first '
        'folder, blends included, as segments named window-N (chained motions in the tape order)')
    args = parser.parse_args()
    entries = []
    for i, d in enumerate(args.motions):
        rows = [(d, m, i) for m in json.loads((d / 'motions.json').read_text())['motions']]
        if args.shuffle is not None:
            rows = [rows[k] for k in np.random.default_rng(args.shuffle + i).permutation(len(rows))]
        entries += rows
    B = library.Body()
    blend_rng = np.random.default_rng(0 if args.shuffle is None else args.shuffle)
    pieces, segments, end = [], [], None
    frames = 0
    for folder, m, source in entries:
        if m['name'] in args.exclude or m['family'] in args.exclude:
            continue
        mo = np.load(folder / m['motion'])
        q = mo['qpos'][::int(mo['fps'][0]) // FPS]
        join_start = frames
        if end is not None:
            q = place(q, end)
            blend_s = blend_rng.uniform(*args.blend_range) if args.blend_range else args.blend_s
            join = blend(end, q[0], int(blend_s * FPS))
            pieces.append(join)
            frames += len(join)
        segments.append(dict(name=m['name'], family=m['family'], tempo=m['tempo'], tempo_unit=m['tempo_unit'],
            start=frames, frames=len(q), duration_s=round(len(q) / FPS, 2), source=source, join_start=join_start))
        pieces.append(q)
        frames += len(q)
        end = q[-1]
    qpos = np.concatenate(pieces)
    clipped = {}
    if args.clip_stops is not None:
        for j in range(B.m.njnt):
            if B.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE or not B.m.jnt_limited[j]:
                continue
            a, (lo, hi) = B.m.jnt_qposadr[j], B.m.jnt_range[j]
            before = qpos[:, a].copy()
            qpos[:, a] = np.clip(qpos[:, a], lo + args.clip_stops, hi - args.clip_stops)
            if np.abs(before - qpos[:, a]).max() > 1e-4:
                clipped[B.m.joint(j).name] = round(float(np.abs(before - qpos[:, a]).max()), 4)
    qvel, _ = library.derivatives(B.m, qpos, FPS)
    windows = []
    if args.windows:
        first = [s for s in segments if s['source'] == 0]
        for k in range(0, len(first) - args.windows + 1, args.windows):
            group = first[k:k + args.windows]
            windows.append(dict(name=f'window-{k // args.windows + 1}', family='chained motions', tempo=None, tempo_unit=None,
                start=group[0]['start'], frames=group[-1]['start'] + group[-1]['frames'] - group[0]['start'],
                duration_s=round((group[-1]['start'] + group[-1]['frames'] - group[0]['start']) / FPS, 2),
                motions=[s['name'] for s in group]))
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / 'tape.npz', qpos=qpos, qvel=qvel)
    digest = hashlib.sha256((args.output / 'tape.npz').read_bytes()).hexdigest()
    (args.output / 'segments.json').write_text(json.dumps(dict(schema='cove50.agility.tape.v1', fps=FPS, blend_s=args.blend_s,
        frames=len(qpos), duration_s=round(len(qpos) / FPS, 2), tape_sha256=digest, excluded=args.exclude,
        shuffle=args.shuffle, blend_range=args.blend_range, clip_stops=args.clip_stops, clipped_max_rad=clipped,
        segments=segments + windows), indent=1) + '\n')
    print(f'{len(segments)} motions, {len(windows)} windows, {len(qpos)} frames ({len(qpos) / FPS:.0f} s)')


if __name__ == '__main__':
    main()
