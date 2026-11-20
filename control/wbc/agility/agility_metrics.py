"""Agility measurements of evaluated WBC segments, robot against its reference, in native units.

Per segment (nominal condition), median over worlds:
  speed ratio   p99 |joint speed| of the robot / of the reference, for knee, hip pitch, ankle pitch, hip roll
                (both sides), over the frames where the reference moves;
  lag_ms        delay of the robot's knee angle behind the reference (cross-correlation peak within +-200 ms,
                refined between 20 ms frames);
  contact match share of frames whose foot contact (foot origin within 5 mm of its 5th-percentile height) agrees
                with the reference's, both feet;
  swing ratio   peak swing height of each foot / the reference's, on segments where the reference lifts it > 8 mm;
  flight        frames with both feet more than 3 mm up, robot and reference;
  torso vz      p99 |vertical torso speed| robot / reference.

    .venv/bin/python control/wbc/agility/agility_metrics.py RUN [RUN ...] \
        [--stage evaluate-native] [--out FILE.json]
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
XML = ROOT / 'artifacts/robot/model/timestep/cove50-rigid.xml'
M = mujoco.MjModel.from_xml_path(str(XML))
D = mujoco.MjData(M)
J = [M.joint(i).name for i in range(M.njnt) if M.jnt_type[i] != 0]
FEET = [M.body('R_foot').id, M.body('L_foot').id]
SPEED_JOINTS = ('knee', 'hip_pitch', 'ankle_pitch', 'hip_roll')
FPS = 50


def feet_z(qpos):
    out = np.zeros((len(qpos), 2))
    for k, q in enumerate(qpos):
        D.qpos[:] = q
        mujoco.mj_kinematics(M, D)
        out[k] = D.xpos[FEET, 2]
    return out


def lag_ms(a, b, max_lag=10):
    """Lag (ms) of a behind b: the cross-correlation peak of their deviations from the mean, refined between frames
    by a parabola through the peak and its two neighbours (one 50 Hz frame is 20 ms)."""
    a, b = a - a.mean(), b - b.mean()
    if a.std() < 1e-4 or b.std() < 1e-4:
        return None
    lags = range(-max_lag, max_lag + 1)
    c = np.array([np.dot(a[max(L, 0):len(a) + min(L, 0)], b[max(-L, 0):len(b) - max(L, 0)]) for L in lags])
    k = int(c.argmax())
    shift = 0.0
    if 0 < k < len(c) - 1 and c[k - 1] - 2 * c[k] + c[k + 1] < 0:
        shift = 0.5 * (c[k - 1] - c[k + 1]) / (c[k - 1] - 2 * c[k] + c[k + 1])
    return round((lags[k] + shift) * 1000 / FPS, 1)


def segment(t, worlds=range(0, 32, 4)):
    n = len(t['qpos']) - 1
    q, ref = t['qpos'][:n, :, 7:], t['ref_joint_pos'][:n]
    qd = t['qvel'][:n, :, 6:]
    rqd = np.gradient(ref, 1 / FPS, axis=0)
    out = {}
    for name in SPEED_JOINTS:
        ids = [J.index(f'{s}_{name}') for s in 'RL']
        moving = np.abs(rqd[:, :, ids]).max(-1) > .5
        r = []
        for w in range(q.shape[1]):
            if moving[:, w].sum() < 10:
                continue
            r.append(np.percentile(np.abs(qd[:, w, ids])[moving[:, w]], 99) / max(np.percentile(np.abs(rqd[:, w, ids])[moving[:, w]], 99), 1e-6))
        out[f'speed_ratio_{name}'] = round(float(np.median(r)), 3) if r else None
    k = J.index('R_knee')
    lags = [lag_ms(q[:, w, k], ref[:, w, k]) for w in range(q.shape[1])]
    lags = [x for x in lags if x is not None]
    out['knee_lag_ms'] = float(np.median(lags)) if lags else None
    match, swing, flight, rflight = [], [], [], []
    for w in worlds:
        rq = t['qpos'][:n, w].copy()
        rq[:, :3], rq[:, 3:7], rq[:, 7:] = t['ref_anchor_pos'][:n, w], t['ref_anchor_quat'][:n, w], ref[:, w]
        z, rz = feet_z(t['qpos'][:n, w]), feet_z(rq)
        h, rh = z - np.percentile(z, 5, axis=0), rz - np.percentile(rz, 5, axis=0)
        c, rc = h < .005, rh < .005
        match.append(float((c == rc).mean()))
        flight.append(int((h > .003).all(1).sum()))
        rflight.append(int((rh > .003).all(1).sum()))
        for f in range(2):
            if rh[:, f].max() > .008:
                swing.append(h[:, f].max() / rh[:, f].max())
    out.update(contact_match=round(float(np.median(match)), 3), swing_ratio=round(float(np.median(swing)), 3) if swing else None,
        flight_frames=float(np.median(flight)), reference_flight_frames=float(np.median(rflight)))
    vz = np.gradient(t['qpos'][:n, :, 2], 1 / FPS, axis=0)
    rvz = np.gradient(t['ref_anchor_pos'][:n, :, 2], 1 / FPS, axis=0)
    rv = np.percentile(np.abs(rvz), 99, axis=0)
    out['torso_vz_ratio'] = round(float(np.median(np.percentile(np.abs(vz), 99, axis=0) / np.maximum(rv, 1e-6))), 3) if np.median(rv) > .02 else None
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--stage', default='evaluate-native')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    res = {}
    for run in a.runs:
        rep = json.loads((run / 'evidence' / a.stage / 'report.json').read_text())
        res[run.name] = {r['segment']: segment(np.load(run / 'evidence' / a.stage / r['trace'])) for r in rep['rows'] if r['condition'] == 'nominal'}
    if a.out:
        a.out.write_text(json.dumps(res, indent=1) + '\n')
    for run, segs in res.items():
        print(f'== {run} ({a.stage})')
        for s, v in segs.items():
            print(f"  {s:24s} knee x{v['speed_ratio_knee']} hipP x{v['speed_ratio_hip_pitch']} ankP x{v['speed_ratio_ankle_pitch']} lag {v['knee_lag_ms']} ms"
                  f" contact {v['contact_match']} swing x{v['swing_ratio']} flight {v['flight_frames']}/{v['reference_flight_frames']} vz x{v['torso_vz_ratio']}")


if __name__ == '__main__':
    main()
