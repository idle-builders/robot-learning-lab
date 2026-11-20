"""Export the r4 best-estimate dynamics onto the unchanged R3 rigid MJCF and URDF.

    .venv-control/bin/python -B robot/model/dynamics/build.py

Reads actuator-dynamics.json (nominal values, C047 on the 20 body joints and
XM335 on the neck) and writes, next to this file's artifact folder:

- cove50-rigid.xml: the R3 rigid MJCF with joint armature and frictionloss;
  meshes stay in artifacts/robot/model/timestep/meshes;
- cove50.urdf: the R3 URDF with <dynamics friction> per joint, the effort and
  velocity limits of the C047 stall curve, and the rotor inertia (URDF has no
  armature field) as a comment on each joint; meshes stay in the URDF
  inspection package;
- report.json: hashes and the values written.

Both exports also carry the best-estimate hip ranges (joint_ranges_deg, nominal)
for the six hip axes that R3 left unlimited/continuous.

Link masses, inertias, geometry, collisions and sensors are R3's. The
Feetech firmware duty loop, rate limit and backlash are runtime behaviour, not
model attributes: control/common/plant applies them when
given dynamics.resolve(servo).
"""
import hashlib
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
R3 = ROOT / 'artifacts/robot/model/timestep'
URDF = ROOT / 'artifacts/robot/model/urdf/cove50.urdf'
OUT = ROOT / 'artifacts/robot/model/dynamics'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    sidecar = json.loads((HERE / 'actuator-dynamics.json').read_text())
    axes = json.loads((R3 / 'actuators.json').read_text())['actuators']
    nominal = {m: {k: v['nominal'] for k, v in sidecar['models'][m].items() if isinstance(v, dict) and 'nominal' in v}
        for m in ('c047', 'xm335')}
    OUT.mkdir(parents=True, exist_ok=True)

    spec = mujoco.MjSpec.from_file(str(R3 / 'cove50-rigid.xml'))
    spec.meshdir = os.path.relpath(R3, OUT)  # mesh files are named meshes/...
    written = {}
    for axis in axes:
        joint, values = spec.joint(axis['joint']), nominal[axis['model']]
        joint.armature, joint.frictionloss = values['armature_kg_m2'], values['frictionloss_Nm']
        written[axis['joint']] = dict(model=axis['model'], armature_kg_m2=joint.armature, frictionloss_Nm=joint.frictionloss)
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
        if 'range_deg' in values:
            j = model.joint(name).id
            assert model.jnt_limited[j] and all(abs(a - math.radians(b)) < 1e-5 for a, b in zip(model.jnt_range[j], values['range_deg']))

    tree = ET.parse(URDF)
    for mesh in tree.getroot().iter('mesh'):
        mesh.set('filename', os.path.relpath(URDF.parent / mesh.get('filename'), OUT))
    c047 = sidecar['models']['c047']
    for joint in tree.getroot().iter('joint'):
        name = joint.get('name')
        if name not in written:
            continue
        values = written[name]
        dynamics = joint.find('dynamics')
        dynamics.set('friction', f"{values['frictionloss_Nm']:g}")
        if values['model'] == 'c047':
            joint.find('limit').set('effort', f"{c047['stall_torque_Nm']:g}")  # the full curve; rated stays in the sidecar
        if 'range_deg' in values:  # R3 exported these continuous: no limit was specified
            lo, hi = values['range_deg']
            joint.set('type', 'revolute')
            limit = joint.find('limit')
            limit.set('lower', f'{math.radians(lo):.6f}')
            limit.set('upper', f'{math.radians(hi):.6f}')
            joint.insert(0, ET.Comment(f' r4 best-estimate range {lo}..{hi} deg (assumption, see actuator-dynamics.json joint_ranges_deg) '))
        joint.insert(0, ET.Comment(f" r4 rotor inertia (armature) {values['armature_kg_m2']:g} kg m2 at the output; "
            f"{values['model']} in robot/model/dynamics/actuator-dynamics.json "))
    urdf = OUT / 'cove50.urdf'
    tree.write(urdf, encoding='unicode', xml_declaration=True)
    mujoco.MjSpec.from_file(str(urdf))  # parses and resolves meshes

    report = dict(schema='cove50.model-dynamics-export.v1', version=sidecar['version'], revision=sidecar.get('revision', 1), parent=sidecar['parent_model'],
        sidecar=dict(path=str((HERE / 'actuator-dynamics.json').relative_to(ROOT)), sha256=sha(HERE / 'actuator-dynamics.json')),
        sources=dict(rigid_MJCF=dict(path=str((R3 / 'cove50-rigid.xml').relative_to(ROOT)), sha256=sha(R3 / 'cove50-rigid.xml')),
            rigid_URDF=dict(path=str(URDF.relative_to(ROOT)), sha256=sha(URDF))),
        outputs={p.name: dict(path=str(p.relative_to(ROOT)), sha256=sha(p), bytes=p.stat().st_size) for p in (mjcf, urdf)},
        joints=written, mass_kg=float(model.body_subtreemass[1]))
    (OUT / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ('version', 'outputs', 'mass_kg')}, indent=1))


if __name__ == '__main__':
    main()
