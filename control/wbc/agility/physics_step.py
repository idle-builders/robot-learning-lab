"""How the r4 plant changes with the physics step: servo step responses and an open-loop squat on the floor.

Plain MuJoCo (CPU) with the walk-resume BatchedRuntime as the servo boundary, so
the same firmware duty loop, rotor inertia, friction, backlash, delays and
sensor cadence run at each step size. Two probes per servo set:
- step: base pinned in the air; at 0.1 s the knee, hip pitch, shoulder pitch and
  elbow targets jump; joint angles for 0.6 s;
- squat: robot standing on the floor; the deep-squat-x2 reference joint angles
  are commanded open loop (no policy) from the tape; torso height and feet.
Differences are reported against the 0.25 ms run in native units. CPU wall time
per simulated second is recorded as a speed proxy (GPU throughput is measured
by the training job's benchmark stage).

    .venv-control/bin/python control/wbc/agility/physics_step.py \
      --tape DIR --evidence DIR [--steps 0.00025 0.0005 0.001 0.002] [--bodies c047 st3025 st3025-pitch-roll-sts3250]

DIR: motions.py --tape output (agility-tape.npz, motions.json). Writes report.json.
"""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import tracking  # noqa: E402

RT = tracking.runtime()
from cove50_r3_runtime import boundary as B, dynamics as D, mjlab_adapter as A  # noqa: E402

STEP_JOINTS = {'R_knee': .6, 'R_hip_pitch': -.6, 'R_shoulder_pitch': .8, 'R_elbow': .8}


def build(body, dt, floor):
    dyn = D.resolve(body)
    spec_fn = partial(A.robot_spec, dyn)
    c047_kg = json.loads((tracking.MODEL / 'actuators.json').read_text())['models']['c047']['mass_kg']
    if 'joints' in dyn:
        spec_fn = partial(tracking.heavier_servos, spec_fn, {j: v['mass_kg'] - c047_kg for j, v in dyn['joints'].items()})
    elif body != 'c047':
        spec_fn = partial(tracking.heavier_servos, spec_fn, dyn['mass_kg'] - c047_kg)
    spec = spec_fn()
    if floor:
        native = mujoco.MjModel.from_xml_path(str(tracking.MODEL / 'cove50-rigid.xml'))
        f = mujoco.mj_name2id(native, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        g = spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, .1])
        g.friction, g.solref, g.solimp = native.geom_friction[f], native.geom_solref[f], native.geom_solimp[f]
        g.condim, g.priority = int(native.geom_condim[f]), int(native.geom_priority[f])
    m = spec.compile()
    m.opt.timestep = dt
    rt = B.BatchedRuntime(1, rail_V=dyn['rail_V'], peak=True, c047=dyn['law'], duty=dyn['duty'], ranges_deg=dyn['ranges_deg'],
        joint_models=dyn.get('joints') and {j: dict(law=v['law'], duty=v['duty']) for j, v in dyn['joints'].items()},
        physics_dt_s=dt)
    rt.contract['models']['c047'].update(dyn['law'])
    return m, rt


def sensor(m, suffix):
    i = [k for k in range(m.nsensor) if m.sensor(k).name.endswith(suffix)][0]
    return m.sensor_adr[i]


def run(m, rt, targets_fn, seconds, pin=None):
    d = mujoco.MjData(m)
    d.qpos[:3] = 0, 0, .30201 if pin is None else .6
    d.qpos[3] = 1
    mujoco.mj_forward(m, d)
    qa = np.array([m.jnt_qposadr[m.joint(n).id] for n in rt.names])
    va = np.array([m.jnt_dofadr[m.joint(n).id] for n in rt.names])
    acc, gyr = sensor(m, 'imu_accel'), sensor(m, 'imu_gyro')
    ticks = round(seconds / m.opt.timestep)
    per = round(rt.policy_dt_s / m.opt.timestep)
    log = dict(t=[], q=[], torso_z=[], foot_z=[], torque=[], current=[])
    rt.reset()
    target = torch.zeros(1, 22, dtype=torch.float64 if False else torch.float32)
    start = time.perf_counter()
    for k in range(ticks):
        if k % per == 0:
            target = torch.as_tensor(targets_fn(k * m.opt.timestep), dtype=torch.float32)[None]
        if pin is not None:
            d.qpos[:7], d.qvel[:6] = pin, 0
        q = torch.as_tensor(d.qpos[qa], dtype=torch.float32)[None]
        qd = torch.as_tensor(d.qvel[va], dtype=torch.float32)[None]
        tau = rt.step(target, q, qd, torch.as_tensor(d.sensordata[acc:acc + 3], dtype=torch.float32)[None],
            torch.as_tensor(d.sensordata[gyr:gyr + 3], dtype=torch.float32)[None])
        d.qfrc_applied[:] = 0
        d.qfrc_applied[va] = tau[0].numpy()
        mujoco.mj_step(m, d)
        if k % max(1, round(.002 / m.opt.timestep)) == 0:  # log every 2 ms of simulated time
            log['t'].append(k * m.opt.timestep)
            log['q'].append(d.qpos[qa].copy())
            log['torso_z'].append(float(d.xpos[m.body('torso').id][2]))
            log['foot_z'].append([float(d.xpos[m.body(f).id][2]) for f in ('R_foot', 'L_foot')])
            log['torque'].append(rt.torque[0].numpy().copy())
            log['current'].append(rt.current[0].numpy().copy())
    wall = time.perf_counter() - start
    return {k: np.array(v) for k, v in log.items()}, wall / seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--tape', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--steps', type=float, nargs='*', default=[.00025, .0005, .001, .002])
    parser.add_argument('--bodies', nargs='*', default=['c047', 'st3025', 'st3025-pitch-roll-sts3250'])
    args = parser.parse_args()
    catalogue = json.loads((args.tape / 'motions.json').read_text())['motions'][0]
    tape = np.load(args.tape / 'agility-tape.npz')
    seg = next(s for s in catalogue['segments'] if s['name'] == 'deep-squat-x2')
    joints_tape = tape['joint_pos']  # [T, 22] in robot joint order (motion file)
    rows = []
    for body in args.bodies:
        results = {}
        for dt in args.steps:
            m, rt = build(body, dt, floor=False)
            idx = {n: i for i, n in enumerate(rt.names)}
            def step_targets(t):
                q = np.zeros(22)
                if t >= .1:
                    for j, v in STEP_JOINTS.items():
                        q[idx[j]] = v
                return q
            step_log, step_wall = run(m, rt, step_targets, .7, pin=np.array([0, 0, .6, 1, 0, 0, 0]))
            m, rt = build(body, dt, floor=True)
            jn = [m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] != 0]
            order = [jn.index(n) for n in rt.names]  # motion (robot joint) order -> actuator order
            def squat_targets(t):
                f = seg['start'] + min(int(t * 50), seg['frames'] - 1)
                return joints_tape[f][order]
            squat_log, squat_wall = run(m, rt, squat_targets, seg['frames'] / 50)
            results[dt] = dict(step=step_log, squat=squat_log, wall=dict(step=step_wall, squat=squat_wall))
            print(f'{body:28s} dt {dt * 1e3:5.2f} ms  CPU wall per simulated s: step {step_wall:6.2f} s, squat {squat_wall:6.2f} s  '
                  f'squat torso z {squat_log["torso_z"].min():.3f}-{squat_log["torso_z"].max():.3f} m', flush=True)
        base = results[min(args.steps)]
        names = list(STEP_JOINTS)
        for dt, r in results.items():
            row = dict(body=body, physics_dt_s=dt, cpu_wall_per_sim_s=r['wall'])
            for probe in ('step', 'squat'):
                a, b = r[probe], base[probe]
                n = min(len(a['t']), len(b['t']))
                ii = [list(rt.names).index(j) for j in names] if probe == 'step' else list(range(22))
                diff = np.abs(a['q'][:n][:, ii] - b['q'][:n][:, ii])
                row[probe] = dict(joint_rms_diff_rad=round(float(np.sqrt((diff ** 2).mean())), 4), joint_max_diff_rad=round(float(diff.max()), 4),
                    torque_rms_diff_Nm=round(float(np.sqrt(((a['torque'][:n] - b['torque'][:n]) ** 2).mean())), 4),
                    current_mean_A=round(float(np.abs(a['current'][:n]).sum(1).mean()), 3))
                if probe == 'step':
                    row[probe]['per_joint'] = {j: dict(final_rad=round(float(a['q'][-1][list(rt.names).index(j)]), 4),
                        overshoot_rad=round(float(np.max(np.sign(STEP_JOINTS[j]) * a['q'][:, list(rt.names).index(j)]) - abs(STEP_JOINTS[j])), 4),
                        t90_s=round(float(a['t'][np.argmax(np.sign(STEP_JOINTS[j]) * a['q'][:, list(rt.names).index(j)] >= .9 * abs(STEP_JOINTS[j]))] - .1), 4))
                        for j in names}
                else:
                    row[probe].update(torso_z_rms_diff_m=round(float(np.sqrt(((a['torso_z'][:n] - b['torso_z'][:n]) ** 2).mean())), 4),
                        torso_z_range_m=[round(float(a['torso_z'].min()), 4), round(float(a['torso_z'].max()), 4)],
                        foot_z_max_m=round(float(a['foot_z'].max()), 4), fell=bool(a['torso_z'].min() < .15))
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != 'cpu_wall_per_sim_s'})[:400], flush=True)
    args.evidence.mkdir(parents=True, exist_ok=True)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    (args.evidence / 'report.json').write_text(json.dumps(dict(schema='cove50.r4.physics-step.v1', method=__doc__.strip(),
        steps_s=args.steps, bodies=args.bodies, step_joints=STEP_JOINTS, squat_segment=seg,
        sources={p.name: sha(p) for p in (HERE / 'physics_step.py', tracking.ROOT / 'control/common/plant/boundary.py',
            tracking.ROOT / 'control/common/plant/dynamics.py')},
        tape_sha256=sha(args.tape / 'agility-tape.npz'), rows=rows), indent=1) + '\n')


if __name__ == '__main__':
    main()
