"""Export the locked three-tier motor map (motor-map.json) onto the R3 rigid MJCF and URDF.

    .venv-control/bin/python -B robot/model/motor-map/build.py

For each of the 22 joints, the servo of its tier (values from the r4 sidecar models) sets:
- rotor inertia (MJCF armature; a comment in the URDF, which has no armature field) and Coulomb friction;
- the URDF effort and velocity limits: catalogue stall torque and no-load speed at 12 V. The runtime scales them to
  the 11.1 V pack rail;
- the servo mass change against the R3 CAD servo at that joint (C047 55 g, XM335 27 g at the neck). It is placed as a
  point at the joint anchor in the body holding the case, plus a box inertia of the tier's case. This is the
  tracking.heavier_servos rule, which the training plant uses too.

The best-estimate hip ranges of r4 are carried over. Link geometry, collisions and sensors are R3's. The long-arm
ape body (arm and foot scaling, fists) is applied at training time and is not exported here.

Writes artifacts/robot/model/motor-map/{cove50-rigid.xml, cove50.urdf, report.json}.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
R3 = ROOT / 'artifacts/robot/model/timestep'
URDF = ROOT / 'artifacts/robot/model/urdf/cove50.urdf'
SIDECAR = ROOT / 'robot/model/dynamics/actuator-dynamics.json'
OUT = ROOT / 'artifacts/robot/model/motor-map'
CASE_IN_CHILD = ('R_elbow', 'L_elbow', 'R_ankle_roll', 'L_ankle_roll')  # as tracking._CASE_IN_CHILD


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def skew(r):
    return r @ r * np.eye(3) - np.outer(r, r)


def add_servo_mass(spec, axis, extra_kg, case_m):
    """tracking.heavier_servos for one axis: extra_kg as a point at the joint anchor plus a box inertia."""
    if not extra_kg:
        return None
    box = extra_kg / 12 * 2 * sum(d * d for d in case_m) / 3
    joint, source = spec.joint(axis['joint']), axis['source_axis']
    child = spec.body(source['child'])
    anchor = np.array(joint.pos)
    if axis['joint'] in CASE_IN_CHILD:
        body, point = child, anchor
    else:
        body, point = spec.body(source['parent']), np.zeros(3)
        mujoco.mju_rotVecQuat(point, anchor, np.array(child.quat))
        point += np.array(child.pos)
    m, c, f = body.mass, np.array(body.ipos), body.fullinertia
    inertia = np.array([[f[0], f[3], f[4]], [f[3], f[1], f[5]], [f[4], f[5], f[2]]])
    mass = m + extra_kg
    com = (m * c + extra_kg * point) / mass
    inertia = inertia + m * skew(c - com) + extra_kg * skew(point - com) + box * np.eye(3)
    body.mass, body.ipos = mass, com
    body.fullinertia = [inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]
    return body.name


def main():
    motor_map = json.loads((HERE / 'motor-map.json').read_text())
    sidecar = json.loads(SIDECAR.read_text())
    r3 = json.loads((R3 / 'actuators.json').read_text())
    r3_mass = {k: v['mass_kg'] for k, v in r3['models'].items()}
    OUT.mkdir(parents=True, exist_ok=True)

    spec = mujoco.MjSpec.from_file(str(R3 / 'cove50-rigid.xml'))
    spec.meshdir = os.path.relpath(R3, OUT)
    written, changed_bodies = {}, set()
    for axis in r3['actuators']:
        tier = motor_map['joints'][axis['joint']]
        t = motor_map['tiers'][tier]
        m = sidecar['models'][t['sidecar_model']]
        joint = spec.joint(axis['joint'])
        joint.armature, joint.frictionloss = m['armature_kg_m2']['nominal'], m['frictionloss_Nm']['nominal']
        extra = t['mass_kg'] - r3_mass[axis['model']]
        body = add_servo_mass(spec, axis, extra, [d / 1000 for d in t['case_mm']])
        if body:
            changed_bodies.add(body)
        written[axis['joint']] = dict(tier=tier, model=t['sidecar_model'], r3_model=axis['model'], armature_kg_m2=joint.armature,
            frictionloss_Nm=joint.frictionloss, mass_change_kg=round(extra, 4), mass_body=body,
            effort_Nm=m['stall_torque_Nm'], velocity_rad_s=m['no_load_speed_rad_s'])
    ranges = {f'{side}_{axis}': r['nominal'] for axis, r in sidecar.get('joint_ranges_deg', {}).get('joints', {}).items() for side in 'RL'}
    for name, (lo, hi) in ranges.items():
        joint = spec.joint(name)
        joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        joint.range = [lo, hi] if spec.compiler.degree else [math.radians(lo), math.radians(hi)]
        written[name]['range_deg'] = [lo, hi]
    mjcf = OUT / 'cove50-rigid.xml'
    mjcf.write_text(spec.to_xml())
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    for name, values in written.items():
        dof = model.jnt_dofadr[model.joint(name).id]
        assert abs(model.dof_armature[dof] - values['armature_kg_m2']) < 1e-12
        assert abs(model.dof_frictionloss[dof] - values['frictionloss_Nm']) < 1e-12

    tree = ET.parse(URDF)
    for mesh in tree.getroot().iter('mesh'):
        mesh.set('filename', os.path.relpath(URDF.parent / mesh.get('filename'), OUT))
    for link in tree.getroot().iter('link'):  # link and body frames coincide; copy the recomputed inertials
        if link.get('name') not in changed_bodies:
            continue
        b = model.body(link.get('name')).id
        inertial = link.find('inertial')
        inertial.find('mass').set('value', f'{model.body_mass[b]:.9g}')
        inertial.find('origin').set('xyz', ' '.join(f'{v:.9g}' for v in model.body_ipos[b]))
        inertial.find('origin').set('rpy', '0 0 0')
        i = spec.body(link.get('name')).fullinertia
        inertial.find('inertia').attrib.update({k: f'{v:.9g}' for k, v in zip(('ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz'), i)})
        link.insert(0, ET.Comment(' r5-tier3: inertial includes the servo mass change at this joint (motor-map.json) '))
    for joint in tree.getroot().iter('joint'):
        name = joint.get('name')
        if name not in written:
            continue
        v = written[name]
        dyn = joint.find('dynamics')
        if dyn is None:
            dyn = ET.SubElement(joint, 'dynamics')
        dyn.set('friction', f"{v['frictionloss_Nm']:g}")
        limit = joint.find('limit')
        limit.set('effort', f"{v['effort_Nm']:g}")
        limit.set('velocity', f"{v['velocity_rad_s']:g}")
        if 'range_deg' in v:
            lo, hi = v['range_deg']
            joint.set('type', 'revolute')
            limit.set('lower', f'{math.radians(lo):.6f}')
            limit.set('upper', f'{math.radians(hi):.6f}')
        joint.insert(0, ET.Comment(f" r5-tier3: {v['tier']} tier, {motor_map['tiers'][v['tier']]['part']}; rotor inertia "
            f"{v['armature_kg_m2']:g} kg m2 at the output; effort and velocity are the 12 V catalogue stall torque and no-load speed "))
    urdf = OUT / 'cove50.urdf'
    tree.write(urdf, encoding='unicode', xml_declaration=True)
    from_urdf = mujoco.MjSpec.from_file(str(urdf)).compile()
    urdf_mass, mjcf_mass = float(from_urdf.body_subtreemass[0]), float(model.body_subtreemass[0])

    report = dict(schema='cove50.model-motor-map-export.v1', version=motor_map['version'],
        motor_map=dict(path=str((HERE / 'motor-map.json').relative_to(ROOT)), sha256=sha(HERE / 'motor-map.json')),
        sidecar=dict(path=str(SIDECAR.relative_to(ROOT)), sha256=sha(SIDECAR)),
        sources=dict(rigid_MJCF=dict(path=str((R3 / 'cove50-rigid.xml').relative_to(ROOT)), sha256=sha(R3 / 'cove50-rigid.xml')),
            rigid_URDF=dict(path=str(URDF.relative_to(ROOT)), sha256=sha(URDF))),
        outputs={p.name: dict(path=str(p.relative_to(ROOT)), sha256=sha(p), bytes=p.stat().st_size) for p in (mjcf, urdf)},
        joints=written, mass_kg_mjcf=round(mjcf_mass, 4), mass_kg_urdf=round(urdf_mass, 4),
        mass_kg_r3=round(float(mujoco.MjModel.from_xml_path(str(R3 / 'cove50-rigid.xml')).body_subtreemass[0]), 4))
    (OUT / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ('version', 'mass_kg_r3', 'mass_kg_mjcf', 'mass_kg_urdf')}, indent=1))
    for tier in motor_map['tiers']:
        print(f"{tier:9s}", ', '.join(j for j, v in written.items() if v['tier'] == tier))


if __name__ == '__main__':
    main()
