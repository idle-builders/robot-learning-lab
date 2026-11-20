"""Servo demand of the agility set with a different servo on each joint group (mixed sizing).

Same inverse dynamics and floor forces as demand.py (body.Dynamics), but each of
the 20 body joints carries its own servo: its mass at the joint (the point-mass
rule of tracking.heavier_servos), its nominal r4 rotor inertia and Coulomb
friction, and its own torque-speed envelope and current line. The neck keeps its
XM335 values. Results are distributions in native units, not verdicts.

    .venv-control/bin/python control/wbc/agility/allocation.py \
      --motions DIR --evidence DIR [--allocation NAME ...] [--jobs N]

DIR from library.py. Writes report.json (per motion and allocation: per-joint
torque, speed, envelope use, share of frames beyond the stall curve, current).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

import body

GROUPS = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'hip_yaw', 'hip_pitch', 'hip_roll', 'knee', 'ankle_pitch', 'ankle_roll')
ARM_G, PITCH = GROUPS[:4], ('hip_pitch', 'knee', 'ankle_pitch')
LEG_G = GROUPS[4:]


def alloc(default, **groups):
    return {g: groups.get(g, default) for g in GROUPS}


ALLOCATIONS = {
    'c047': alloc('c047'),
    'st3025': alloc('st3025'),
    'sts3250': alloc('sts3250'),  # same 45 x 25 x 35 mm case as the C047 the current CAD is built around
    # The locked three-tier set (robot/model/motor-map/motor-map.json).
    'tier3-st3025-sts3250-c047': alloc('c047', **{g: 'st3025' for g in ('hip_pitch', 'knee', 'ankle_pitch')},
        **{g: 'sts3250' for g in ('hip_roll', 'shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow')}),
    'st3025-pitch': alloc('c047', **{g: 'st3025' for g in PITCH}),
    'st3025-pitch-roll': alloc('c047', **{g: 'st3025' for g in PITCH + ('hip_roll',)}),
    'st3025-legs': alloc('c047', **{g: 'st3025' for g in LEG_G}),
    'st3025-pitch-roll-sts3250': alloc('sts3250', **{g: 'st3025' for g in PITCH + ('hip_roll',)}),
    # Stronger knees on the mixed set: 24 V brushless knee actuators (sidecar revision 3), everything else as the mixed set.
    'st3025-pitch-roll-sts3250-ak45knee': alloc('sts3250', **{**{g: 'st3025' for g in PITCH + ('hip_roll',)}, 'knee': 'ak45-10'}),
    'st3025-pitch-roll-sts3250-ak60knee': alloc('sts3250', **{**{g: 'st3025' for g in PITCH + ('hip_roll',)}, 'knee': 'ak60-6'}),
    # 12 V knee candidates on the existing rail (sidecar revision 3.1).
    'st3025-pitch-roll-sts3250-st3215hsknee': alloc('sts3250', **{**{g: 'st3025' for g in PITCH + ('hip_roll',)}, 'knee': 'st3215-hs'}),
    'st3025-pitch-roll-sts3250-rsbl85knee': alloc('sts3250', **{**{g: 'st3025' for g in PITCH + ('hip_roll',)}, 'knee': 'rsbl85-12'}),
    'st3025-pitch-roll-sts3250-geared-knee': alloc('sts3250', **{**{g: 'st3025' for g in PITCH + ('hip_roll',)}, 'knee': 'st3025-x1.5'}),
    # Whole-body 24 V brushless sets (sidecar revision 4; category round, torque margin 1.5-2x).
    'qdd-edulite-gim': alloc('gim3505-8', **{g: 'edulite-05' for g in PITCH + ('hip_roll', 'ankle_roll')}),
    'qdd-damiao-gim': alloc('gim3505-8', **{**{g: 'dm-j3507' for g in PITCH + ('ankle_roll',)}, 'hip_roll': 'dm-j4310'}),
}
SIDE = body.SIDECAR['models']


def group(joint):
    return joint[2:]


def servo_of(allocation, joint):
    return ALLOCATIONS[allocation][group(joint)]


def model(allocation):
    """The selected R3 body with each body joint's servo mass, rotor inertia and the neck's nominal r4 rotor."""
    spec = body.robot_spec()
    skew = lambda r: r @ r * np.eye(3) - np.outer(r, r)
    for axis in body.CONTRACT['actuators']:
        if axis['model'] != 'c047':
            continue
        extra = SIDE[servo_of(allocation, axis['joint'])]['mass_kg'] - body.C047['mass_kg']
        if not extra:
            continue
        box = extra / 12 * 2 * sum(d * d for d in body.tracking.SERVO_CASE_M) / 3
        joint, source = spec.joint(axis['joint']), axis['source_axis']
        child = spec.body(source['child'])
        anchor = np.array(joint.pos)
        if axis['joint'] in body.tracking._CASE_IN_CHILD:
            b, point = child, anchor
        else:
            b, point = spec.body(source['parent']), np.zeros(3)
            mujoco.mju_rotVecQuat(point, anchor, np.array(child.quat))
            point += np.array(child.pos)
        m, c, f = b.mass, np.array(b.ipos), b.fullinertia
        inertia = np.array([[f[0], f[3], f[4]], [f[3], f[1], f[5]], [f[4], f[5], f[2]]])
        mass = m + extra
        com = (m * c + extra * point) / mass
        inertia = inertia + m * skew(c - com) + extra * skew(point - com) + box * np.eye(3)
        b.mass, b.ipos = mass, com
        b.fullinertia = [inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]
    m = spec.compile()
    for name in body.SERVO_JOINTS:
        m.dof_armature[m.jnt_dofadr[m.joint(name).id]] = SIDE[servo_of(allocation, name)]['armature_kg_m2']['nominal']
    for name in set(body.JOINTS) - set(body.SERVO_JOINTS):
        m.dof_armature[m.jnt_dofadr[m.joint(name).id]] = SIDE['xm335']['armature_kg_m2']['nominal']
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    return m


_DYN = {}


def stats(x):
    x = np.abs(x)
    return dict(p50=round(float(np.percentile(x, 50)), 4), p95=round(float(np.percentile(x, 95)), 4), max=round(float(x.max()), 4))


def task(args):
    path, allocation = args
    if allocation not in _DYN:
        _DYN[allocation] = body.Dynamics(model(allocation))
    dyn = _DYN[allocation]
    dofs = body.dadr(dyn.m, body.JOINTS)
    fric = np.array([SIDE[servo_of(allocation, j)]['frictionloss_Nm']['nominal'] if j in body.SERVO_JOINTS else 0. for j in body.JOINTS])
    mo = np.load(path)
    q, v, a = mo['qpos'], mo['qvel'], mo['qacc']
    tau, res = [], []
    for t in range(len(q)):
        active = body.active_contacts(dyn, mo, q[t], t)
        tt, r, _ = dyn.frame(q[t], v[t], a[t], active)
        tau.append(tt + fric * np.tanh(v[t, dofs] / .05))
        res.append(r)
    tau, omega, res = np.array(tau), v[:, dofs], np.array(res)
    joints, uses, currents = {}, {}, []
    for j in body.SERVO_JOINTS:
        i, s = body.JOINTS.index(j), servo_of(allocation, j)
        use, cur = body.envelope(f'{s}-stall', tau[:, i], omega[:, i])
        uses[j] = use
        currents.append(cur)
        joints[j] = dict(servo=s, torque_Nm=stats(tau[:, i]), rms_torque_Nm=round(float(np.sqrt((tau[:, i] ** 2).mean())), 4),
            speed_rad_s=stats(omega[:, i]), use=stats(use), beyond_share=round(float((use > 1).mean()), 4),
            rms_over_rated=round(float(np.sqrt((tau[:, i] ** 2).mean()) / SIDE[s]['rated_torque_Nm']), 3))
    total = np.sum(currents, 0)
    share = lambda names: round(float(np.any([uses[j] > 1 for j in names], 0).mean()), 4)
    return dict(motion=Path(path).stem, allocation=allocation, frames=len(q), robot_mass_kg=round(float(dyn.m.body_subtreemass[1]), 3),
        servo_mass_kg=round(sum(SIDE[servo_of(allocation, j)]['mass_kg'] for j in body.SERVO_JOINTS), 3),
        leg_beyond_share=share(body.LEGS), arm_beyond_share=share(body.ARMS),
        total_current_A=dict(mean=round(float(total.mean()), 2), p95=round(float(np.percentile(total, 95)), 2), max=round(float(total.max()), 2)),
        base_residual_force_share_of_weight=stats(np.linalg.norm(res[:, :3], axis=1) / dyn.weight), joints=joints)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--motions', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--allocation', nargs='*', default=list(ALLOCATIONS))
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    catalogue = json.loads((args.motions / 'motions.json').read_text())
    for m in catalogue['motions']:
        if sha(args.motions / m['motion']) != m['sha256']:
            raise ValueError(f"{m['name']}: motion file changed")
    jobs = [(str(args.motions / m['motion']), a) for m in catalogue['motions'] for a in args.allocation]
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(task, jobs, chunksize=1))
    meta = {m['name']: {k: v for k, v in m.items() if k != 'motion'} for m in catalogue['motions']}
    for r in rows:
        r['family'], r['tempo'], r['tempo_unit'] = meta[r['motion']]['family'], meta[r['motion']]['tempo'], meta[r['motion']]['tempo_unit']
    here = Path(__file__).resolve().parent
    args.evidence.mkdir(parents=True, exist_ok=True)
    report = dict(schema='cove50.agility.allocation.v1', method=__doc__.strip(), allocations={a: ALLOCATIONS[a] for a in args.allocation},
        servos={k: SIDE[k] for k in sorted({m for a in args.allocation for m in ALLOCATIONS[a].values()})}, legs=body.LEGS, arms=body.ARMS,
        sources={p.name: sha(p) for p in (here / 'body.py', here / 'allocation.py', here / 'library.py')},
        sidecar_sha256=sha(body.tracking.ROOT / 'robot/model/dynamics/actuator-dynamics.json'),
        model=dict(rigid_mjcf_sha256=sha(body.tracking.MODEL / 'cove50-rigid.xml'), actuators_sha256=sha(body.tracking.MODEL / 'actuators.json')),
        motions=[meta[m['name']] for m in catalogue['motions']], rows=rows)
    (args.evidence / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    for a in args.allocation:
        rs = [r for r in rows if r['allocation'] == a]
        walk = next((r for r in rs if r['motion'] == 'forward-walk'), None)
        larger = np.mean([max(r['leg_beyond_share'], r['arm_beyond_share']) for r in rs])
        walked = f"  walk legs {walk['leg_beyond_share']:6.1%}  walk current {walk['total_current_A']['mean']:5.2f} A" if walk else ''
        print(f"{a:28s} robot {rs[0]['robot_mass_kg']:.2f} kg  mean larger-group share beyond stall {larger:6.1%}{walked}")


if __name__ == '__main__':
    main()
