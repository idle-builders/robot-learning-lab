"""Per-world and per-tile-type measures of an obstacle-world evaluation (stage.py evaluate --terrain parcour): falls, leg/arm/ankle RMS, drift.
usage: world_tiles.py EVAL_DIR [EVAL_DIR ...]  -> prints a table and writes EVAL_DIR/per_tile.json"""
import json, sys
from pathlib import Path
import numpy as np, mujoco
ROOT = Path(__file__).resolve().parents[3]
m = mujoco.MjModel.from_xml_path(str(ROOT / 'artifacts/robot/model/timestep/cove50-rigid.xml'))
J = [m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] != 0]
LEGS = [i for i, a in enumerate(J) if any(k in a for k in ('hip', 'knee', 'ankle'))]
ARMS = [i for i, a in enumerate(J) if any(k in a for k in ('shoulder', 'elbow'))]
ANK = [i for i, a in enumerate(J) if 'ankle' in a]

for d in map(Path, sys.argv[1:]):
    rep = json.loads((d / 'report.json').read_text())
    worlds = rep['worlds']
    rows = []
    for r in rep['rows']:
        t = np.load(d / r['trace'])
        q, ref = t['qpos'][:-1, :, 7:], t['ref_joint_pos'][:-1]
        tape = t['ref_joint_pos_tape'][:-1] if 'ref_joint_pos_tape' in t else ref
        flags = (t['flag_anchor_pos'] | t['flag_anchor_ori'] | t['flag_end_pos'])[:-1]
        first = np.where(flags.any(0), flags.argmax(0), len(flags))  # measure up to the first flag
        root, path = t['qpos'][:-1, :, :2], t['ref_anchor_pos'][:-1, :, :2]
        for w in range(q.shape[1]):
            n = max(int(first[w]), 1)
            rms = lambda a, b, idx: float(np.sqrt(((a[:n, w, idx] - b[:n, w, idx]) ** 2).mean()))
            rows.append(dict(segment=r['segment'], world=w, tile=worlds[w]['tile'], difficulty=worlds[w]['difficulty'],
                fell=bool(flags[:, w].any()), first_flag_s=round(first[w] / 50, 2) if flags[:, w].any() else None,
                leg=rms(q, ref, LEGS), arm=rms(q, ref, ARMS), ankle=rms(q, ref, ANK), ankle_vs_tape=rms(q, tape, ANK),
                ankle_shift=float(np.abs(ref[:n, w][:, ANK] - tape[:n, w][:, ANK]).mean()),
                drift_cm=100 * float(np.linalg.norm(root[n - 1, w] - path[n - 1, w]))))
    (d / 'per_tile.json').write_text(json.dumps(rows, indent=1))
    print(f'== {d}  ({len(rows)} episodes)')
    print(f"{'tile':15s} {'eps':>4s} {'falls':>5s} {'leg':>7s} {'arm':>7s} {'ankle':>7s} {'ank/tape':>8s} {'shift':>6s} {'drift p50/p90 cm':>17s}")
    for tile in dict.fromkeys(w['tile'] for w in worlds):
        s = [x for x in rows if x['tile'] == tile]
        med = lambda k: np.median([x[k] for x in s])
        dr = [x['drift_cm'] for x in s]
        print(f"{tile:15s} {len(s):4d} {sum(x['fell'] for x in s):5d} {med('leg'):7.4f} {med('arm'):7.4f} {med('ankle'):7.4f} "
              f"{med('ankle_vs_tape'):8.4f} {med('ankle_shift'):6.3f} {np.percentile(dr, 50):8.1f}/{np.percentile(dr, 90):.1f}")
