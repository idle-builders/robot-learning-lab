"""Per-joint demand traces of a motion set, for scoring actuator candidates without re-running inverse dynamics.

Same inverse dynamics and floor forces as allocation.py, with the rotor inertia and friction left out. A candidate
adds its own reflected rotor inertia x joint acceleration and Coulomb friction per joint, and is scored against its
torque-speed line (categories.py). Servo mass changes link inertia, so traces are made for one mass allocation;
re-run allocation.py on the chosen category sets to check that coupling.

    .venv-control/bin/python control/wbc/agility/requirements.py \
        --motions DIR [DIR ...] --allocation NAME --evidence DIR [--jobs N]

Writes traces.npz (per motion: tau_rigid, qvel, qacc for the 20 body joints, residual share) and report.json
(motion list with sources, allocation, joint order, per-joint percentiles of torque and speed).
"""
import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import allocation as alloc
import body

_DYN = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _weight(name):
    m = alloc.model(name)
    return float(m.body_subtreemass[1]) * body.G


def task(args):
    path, name = args
    if name not in _DYN:
        m = alloc.model(name)
        m.dof_armature[:] = 0.  # the candidate adds its own rotor inertia
        _DYN[name] = body.Dynamics(m)
    dyn = _DYN[name]
    dofs = body.dadr(dyn.m, body.JOINTS)
    mo = np.load(path)
    q, v, a = mo['qpos'], mo['qvel'], mo['qacc']
    tau, res = [], []
    for t in range(len(q)):
        active = body.active_contacts(dyn, mo, q[t], t)
        tt, r, _ = dyn.frame(q[t], v[t], a[t], active)
        tau.append(tt)
        res.append(r)
    idx = [body.JOINTS.index(j) for j in body.SERVO_JOINTS]
    return Path(path).stem, dict(tau_rigid=np.array(tau)[:, idx].astype(np.float32), qvel=v[:, dofs][:, idx].astype(np.float32),
        qacc=a[:, dofs][:, idx].astype(np.float32), residual=np.array(res, dtype=np.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--motions', type=Path, nargs='+', required=True)
    parser.add_argument('--allocation', default='st3025-pitch-roll-sts3250', choices=sorted(alloc.ALLOCATIONS))
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    catalogue = []
    for d in args.motions:
        meta = json.loads((d / 'motions.json').read_text())
        catalogue += [dict(m, path=str(d / m['motion'])) for m in meta['motions']]
    jobs = [(m['path'], args.allocation) for m in catalogue]
    with Pool(args.jobs) as pool:
        rows = dict(pool.map(task, jobs, chunksize=1))
    args.evidence.mkdir(parents=True, exist_ok=True)
    flat = {f'{k}/{f}': v for k, d in rows.items() for f, v in d.items()}
    np.savez_compressed(args.evidence / 'traces.npz', **flat)
    pct = lambda x: {p: round(float(np.percentile(np.abs(x), p)), 4) for p in (50, 95, 99, 100)}
    joints = {j: dict(torque_Nm=pct(np.concatenate([r['tau_rigid'][:, i] for r in rows.values()])),
        speed_rad_s=pct(np.concatenate([r['qvel'][:, i] for r in rows.values()])))
        for i, j in enumerate(body.SERVO_JOINTS)}
    here = Path(__file__).resolve().parent
    report = dict(schema='cove50.agility.requirements.v1', method=__doc__.strip(), allocation=args.allocation,
        allocation_groups=alloc.ALLOCATIONS[args.allocation], joints=body.SERVO_JOINTS, weight_N=round(float(_weight(args.allocation)), 3),
        sources={p.name: sha(p) for p in (here / 'body.py', here / 'allocation.py', here / 'requirements.py', here / 'library.py')},
        sidecar_sha256=sha(body.tracking.ROOT / 'robot/model/dynamics/actuator-dynamics.json'),
        motions=[{k: v for k, v in m.items() if k != 'path'} | dict(source=m['path'], sha256=sha(m['path'])) for m in catalogue],
        rigid_demand=joints)
    (args.evidence / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    print(f'{len(rows)} motions, {sum(len(r["qvel"]) for r in rows.values())} frames')


if __name__ == '__main__':
    main()
