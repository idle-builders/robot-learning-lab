"""Servo demand of the agility motion set on the R3 body, against the C047 and STS3250 envelopes.

For every motion and body condition (C047 or STS3250 mass, armature none or the
BAM figure) body.Dynamics gives each frame's joint torques, the floor forces and
the base residual. Each torque is compared with the servo's torque-speed
envelope at that joint speed: 'use' is the demanded torque over the available
torque in that direction (above 1 the servo cannot supply it). Results are
distributions in native units, not verdicts.

    .venv-control/bin/python control/wbc/agility/demand.py \
      --motions DIR --work DIR --evidence DIR [--only NAME ...]

DIR from library.py. Per-frame demand goes to --work (cheap: this command
regenerates it); report.json and the configuration go to --evidence.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

import body

# (servo body, rotor inertia level): every body with R3's none, and the sidecar's low, nominal and high.
CONDITIONS = [(b, level) for b in body.BODIES for level in ('none', 'low', 'nominal', 'high')]
ENVELOPES = {b: tuple(e for e in body.SERVOS if body.SERVOS[e]['body'] == b) for b in body.BODIES}
SERVO_IDX = [body.JOINTS.index(j) for j in body.SERVO_JOINTS]
LEG_IDX = [body.JOINTS.index(j) for j in body.LEGS]
ARM_IDX = [body.JOINTS.index(j) for j in body.ARMS]
_DYN = {}
FRICTION_MASK = np.array([j in body.SERVO_JOINTS for j in body.JOINTS], float)
OMEGA_DOFS = {}


def dynamics(condition):
    if condition not in _DYN:
        b, level = condition
        _DYN[condition] = body.Dynamics(body.model(b, body.armature(b, level)))
    return _DYN[condition]


def run(path, condition):
    """Per-frame torques (JOINTS order), joint speeds, base residual and foot normal forces for one motion."""
    dyn = dynamics(condition)
    OMEGA_DOFS.setdefault(condition, body.dadr(dyn.m, body.JOINTS))
    mo = np.load(path)
    q, v, a = mo['qpos'], mo['qvel'], mo['qacc']
    tau, res, normal = [], [], []
    for t in range(len(q)):
        active = body.active_contacts(dyn, mo, q[t], t)
        tt, r, n = dyn.frame(q[t], v[t], a[t], active)
        if condition[1] != 'none':  # r4: the servo also overcomes its Coulomb friction (smoothed sign)
            tt = tt + FRICTION_MASK * body.friction(condition[0]) * np.tanh(v[t, OMEGA_DOFS[condition]] / .05)
        tau.append(tt)
        res.append(r)
        normal.append([n['R_foot'], n['L_foot']])
    return dict(tau=np.array(tau, np.float32), omega=v[:, body.dadr(dyn.m, body.JOINTS)].astype(np.float32),
        residual=np.array(res, np.float32), normal=np.array(normal, np.float32), fps=int(mo['fps'][0]), weight=dyn.weight)


def stats(x):
    x = np.abs(x)
    return dict(p50=round(float(np.percentile(x, 50)), 4), p95=round(float(np.percentile(x, 95)), 4), max=round(float(x.max()), 4))


def summarise(out, envelope):
    tau, omega = out['tau'].astype(float), out['omega'].astype(float)
    use, current = body.envelope(envelope, tau[:, SERVO_IDX], omega[:, SERVO_IDX])
    s = body.SERVOS[envelope]
    joints = {}
    for k, j in enumerate(body.SERVO_JOINTS):
        i = SERVO_IDX[k]
        joints[j] = dict(torque_Nm=stats(tau[:, i]), rms_torque_Nm=round(float(np.sqrt((tau[:, i] ** 2).mean())), 4),
            speed_rad_s=stats(omega[:, i]), use=stats(use[:, k]), beyond_share=round(float((use[:, k] > 1).mean()), 4),
            rms_over_rated=round(float(np.sqrt((tau[:, i] ** 2).mean()) / s['rated']), 3))
    legs = [body.SERVO_JOINTS.index(j) for j in body.LEGS]
    arms = [body.SERVO_JOINTS.index(j) for j in body.ARMS]
    worst = sorted(joints, key=lambda j: -joints[j]['use']['max'])[:3]
    total = current.sum(1)
    return dict(joints=joints, worst_joints=worst,
        leg_use_max=round(float(use[:, legs].max()), 3), arm_use_max=round(float(use[:, arms].max()), 3),
        leg_beyond_share=round(float((use[:, legs] > 1).any(1).mean()), 4), arm_beyond_share=round(float((use[:, arms] > 1).any(1).mean()), 4),
        total_current_A=dict(mean=round(float(total.mean()), 2), p95=round(float(np.percentile(total, 95)), 2), max=round(float(total.max()), 2)))


def task(args):
    path, condition, work = args
    out = run(path, condition)
    name = Path(path).stem
    np.savez(Path(work) / f'{name}--{condition[0]}-{condition[1]}.npz', **{k: v for k, v in out.items() if isinstance(v, np.ndarray)})
    res = out['residual'].astype(float)
    row = dict(motion=name, body=condition[0], armature=condition[1], frames=len(res),
        base_residual=dict(force_share_of_weight=stats(np.linalg.norm(res[:, :3], axis=1) / out['weight']),
            moment_Nm=stats(np.linalg.norm(res[:, 3:], axis=1))),
        peak_foot_normal_N=round(float(out['normal'].max()), 2), weight_N=round(out['weight'], 2),
        envelopes={e: summarise(out, e) for e in ENVELOPES[condition[0]]})
    return row


def validation(motions_dir):
    """The retained rollouts that recorded their applied servo torques, against the inverse-dynamics demand."""
    import motions
    rows = []
    for name in ('forward-walk', 'walk-stop', 'bilateral-arms', 'alternating-arms'):
        out = run(Path(motions_dir) / f'{name}.npz', ('c047', 'none'))  # R3: the plant those policies trained on
        src = next(r for r in motions.references() if r['name'] == name)
        rec = np.load(body.tracking.ROOT / src['trace'])['torque22_Nm']
        sl = slice(5, -5)
        n = min(len(rec), len(out['tau']))
        tau = out['tau'][:n].astype(float)
        row = dict(motion=name, joints={})
        for j in body.SERVO_JOINTS:
            i = body.JOINTS.index(j)
            a, b = tau[sl, i], rec[:n][sl, i]
            row['joints'][j] = dict(rms_demand=round(float(np.sqrt((a ** 2).mean())), 4), rms_applied=round(float(np.sqrt((b ** 2).mean())), 4),
                correlation=round(float(np.corrcoef(a, b)[0, 1]), 3) if a.std() > 1e-6 and b.std() > 1e-6 else None)
        for group, names in (('legs', body.LEGS), ('arms', body.ARMS)):
            ra = [row['joints'][j]['rms_demand'] for j in names]
            rb = [row['joints'][j]['rms_applied'] for j in names]
            row[f'{group}_rms_demand_over_applied'] = round(float(np.sum(ra) / np.sum(rb)), 3)
        rows.append(row)
    return rows


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--motions', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--only', nargs='*')
    parser.add_argument('--jobs', type=int, default=6)
    args = parser.parse_args()
    catalogue = json.loads((args.motions / 'motions.json').read_text())
    motions = [m for m in catalogue['motions'] if not args.only or m['name'] in args.only or m['family'] in args.only]
    for m in motions:
        if sha(args.motions / m['motion']) != m['sha256']:
            raise ValueError(f"{m['name']}: motion file changed")
    (args.work / 'demand').mkdir(parents=True, exist_ok=True)
    args.evidence.mkdir(parents=True, exist_ok=True)
    jobs = [(str(args.motions / m['motion']), c, str(args.work / 'demand')) for m in motions for c in CONDITIONS]
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(task, jobs, chunksize=1))
    meta = {m['name']: {k: v for k, v in m.items() if k not in ('motion',)} for m in motions}
    for r in rows:
        r['family'], r['tempo'], r['tempo_unit'] = meta[r['motion']]['family'], meta[r['motion']]['tempo'], meta[r['motion']]['tempo_unit']
    here = Path(__file__).resolve().parent
    report = dict(schema='cove50.agility.demand.v1',
        method=body.__doc__.strip(),
        servos=body.SERVOS, dynamics_sidecar=body.SIDECAR, friction_pyramid_mu=body.MU,
        servo_joints=body.SERVO_JOINTS, legs=body.LEGS, arms=body.ARMS,
        sources={p.name: sha(p) for p in (here / 'body.py', here / 'library.py', here / 'demand.py')},
        model=dict(rigid_mjcf=str(body.tracking.MODEL.relative_to(body.tracking.ROOT) / 'cove50-rigid.xml'),
            rigid_mjcf_sha256=sha(body.tracking.MODEL / 'cove50-rigid.xml'),
            actuators_sha256=sha(body.tracking.MODEL / 'actuators.json')),
        motions=[meta[m['name']] for m in motions], rows=rows,
        validation=validation(args.motions) if not args.only else None)
    (args.evidence / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    for r in rows:
        if r['armature'] == 'nominal':
            e = r['envelopes'][f"{r['body']}-stall"]
            print(f"{r['motion']:24s} {r['body']:8s} beyond stall legs {e['leg_beyond_share']:6.1%} arms {e['arm_beyond_share']:6.1%} worst {e['worst_joints'][0]}")


if __name__ == '__main__':
    main()
