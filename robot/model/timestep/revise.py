"""Release a numerical-resolution derivative of sealed R2; physical inputs unchanged."""
import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
import mujoco
from common import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume this prepared numerical derivative after a packaging/source repair')
    args = parser.parse_args()
    parent = ROOT / 'artifacts/robot/model/collisions'
    if OUT.exists() and not args.resume:
        raise FileExistsError('Retain the released model; regeneration needs a fresh COVE50_MODEL_DIR')
    if args.resume and read(OUT / 'revision-inputs.json')['parent_seal'] != ref(parent / 'evidence-seal.json'):
        raise ValueError('Resume requires this exact prepared parent')
    seal = read(parent / 'evidence-seal.json')
    srdf = next(row for row in read(parent / 'package-inventory.json')['files'] if row['path'] == 'cove50.srdf')
    verified = seal['models'] + seal['meshes'] + seal['contracts'] + [dict(srdf, path=rel(parent / srdf['path']))]
    for row in verified:
        p = ROOT / row['path']
        if sha(p) != row['sha256'] or p.stat().st_size != row['bytes']:
            raise ValueError('Parent identity drift: ' + str(p))
    OUT.mkdir(parents=True, exist_ok=args.resume)
    for row in seal['models'] + seal['meshes']:
        p = ROOT / row['path']; target = OUT / p.relative_to(parent)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, target)
    contract_names = ['frames.json', 'actuators.json', 'sensors.json', 'timing.json',
        'part-mass-ledger.json', 'link-inertias.json', 'mesh-transfer.json', 'collision-model.json',
        'collision-decomposition.json', 'compliance.json', 'compliance-mass-splits.json', 'FEM-geometry-identity.json', 'cove50.srdf']
    for name in contract_names:
        shutil.copyfile(parent / name, OUT / name)
    rigid = OUT / 'cove50-rigid.xml'
    text = rigid.read_text()
    if text.count('timestep="0.001"') != 1:
        raise ValueError('Unexpected parent integration declaration')
    rigid.write_text(text.replace('timestep="0.001"', 'timestep="0.00025"'))
    timing = read(OUT / 'timing.json')
    timing.update(physics_timestep_s=.00025,
        inner_servo_evaluation=dict(rate_Hz=4000, basis='Numerical evaluation of the assumed continuous Runtime PD/torque-speed law at each rigid physics step; not measured digital-servo rate.',
            previous_rate_Hz=1000, unchanged='Gains, caps, physical inertia/armature, target envelopes,20ms policy period and command delay,10ms feedback delay and nominal sensor rates.',
            sensor_resolution='Accumulated nominal sensor clocks are quantized to physics steps. Actual capture/receive timestamps must be retained; nominal rates alone do not imply identical event times.'),
        numerical_scope='Matched slow-reference comparison removed unloaded knee/ankle1ms chatter but did not complete a step. This resolves an observed numerical mode, not general convergence or physical servo calibration.')
    write(OUT / 'timing.json', timing)
    evidence_paths = [ROOT / 'data/cove50-forward-walk-r1-20261026/review-diagnosis/servo-mode-audit.json',
        ROOT / 'artifacts/cove50-forward-walk/r2-20261026/rate-aware-replay/summary.json',
        ROOT / 'artifacts/cove50-forward-walk/r2-20261026/rate-aware-replay-substep/summary.json',
        ROOT / 'artifacts/cove50-forward-walk/r2-20261026/rate-reference-native-observations.json']
    evidence = []
    for p in evidence_paths:
        target = OUT / 'numerical-actuation-evidence' / (p.parent.name + '-' + p.name)
        target.parent.mkdir(exist_ok=True); shutil.copyfile(p, target)
        evidence.append(dict(source=ref(p), snapshot=ref(target)))
    write(OUT / 'revision-inputs.json', dict(schema='cove50.numerical-resolution-revision.v1', version=VERSION,
        parent_seal=ref(parent / 'evidence-seal.json'), parent_verified_files=len(verified), source=ref(Path(__file__)),
        parent_package_inventory=ref(parent / 'package-inventory.json'),
        change='Only rigid MJCF integration timestep and its timing/adapter documentation. URDF, all physical geometry and model arrays, compliant timestep and motor force law retained.',
        evidence=evidence, limitations='Continuous-law sensitivity; source motor impedance, reflected inertia and digital inner cadence remain unmeasured. No useful walking claim.'))
    arrays = ['body_mass', 'body_ipos', 'body_inertia', 'body_iquat', 'body_parentid', 'body_pos', 'body_quat',
        'jnt_type', 'jnt_axis', 'jnt_pos', 'jnt_range', 'jnt_limited', 'dof_damping', 'dof_armature',
        'actuator_trnid', 'actuator_gear', 'actuator_ctrlrange', 'site_pos', 'site_quat',
        'geom_type', 'geom_bodyid', 'geom_contype', 'geom_conaffinity', 'geom_friction', 'geom_pos', 'geom_quat',
        'geom_size', 'mesh_vert', 'mesh_face', 'exclude_signature']
    records = []
    for name in ['cove50.urdf', 'cove50-rigid.xml', 'cove50-compliant.xml']:
        old = mujoco.MjModel.from_xml_path(str(parent / name)); new = mujoco.MjModel.from_xml_path(str(OUT / name))
        for attr in ['nq', 'nv', 'nu', 'nbody', 'njnt', 'nsensor']:
            if getattr(old, attr) != getattr(new, attr): raise ValueError(name + '/' + attr)
        for attr in arrays:
            if not np.array_equal(getattr(old, attr), getattr(new, attr)): raise ValueError(name + '/' + attr)
        od, nd = mujoco.MjData(old), mujoco.MjData(new)
        mujoco.mj_forward(old, od); mujoco.mj_forward(new, nd)
        for attr in ['xpos', 'xmat', 'xipos', 'site_xpos', 'site_xmat']:
            if not np.array_equal(getattr(od, attr), getattr(nd, attr)): raise ValueError(name + '/' + attr)
        if name != 'cove50-rigid.xml' and old.opt.timestep != new.opt.timestep:
            raise ValueError('Only rigid resolution may change')
        records.append(dict(model=ref(OUT / name), nq=new.nq, nv=new.nv, nu=new.nu,
            timestep_s=new.opt.timestep, parent_timestep_s=old.opt.timestep, unchanged_arrays=arrays,
            mass_kg=float(new.body_mass.sum()), neutral_FK_max_error_m=float(np.max(abs(nd.xpos-od.xpos)))))
    write(OUT / 'native-revision-validation.json', dict(schema='cove50.numerical-resolution-validation.v1', version=VERSION,
        models=records, parent_seal=ref(parent / 'evidence-seal.json'), numerical_evidence=evidence,
        observation='Native physical arrays, collision meshes/exclusions and FK identical. R3 rigid default matches the R2 numerical override used for the retained0.25ms comparison.',
        limitations=['Matched comparison resolves the observed alternating mode; it is not a general integration-convergence study.',
            'The reference still falls after unloading. Floating-base tracking, useful steps, installed actuator dynamics and physical design remain separate questions.']))
    model_names = ['cove50.urdf', 'cove50.ros.urdf', 'cove50-rigid.xml', 'cove50-compliant.xml']
    models = [ref(OUT / n) for n in model_names]
    build = read(parent / 'model-build.json')
    build.update(version=VERSION, parent_build=ref(parent / 'model-build.json'),
        numerical_revision=ref(OUT / 'revision-inputs.json'), authored_inputs=[ref(Path(__file__)), ref(OUT / 'timing.json')])
    write(OUT / 'model-build.json', build)
    sources = [ref(p) for p in sorted(Path(__file__).parent.glob('*.py'))]
    write(OUT / 'source-lock.json', dict(schema='cove50.numerical-revision-source-lock.v1', version=VERSION,
        parent_source_lock=ref(parent / 'source-lock.json'), inputs=ref(OUT / 'revision-inputs.json'),
        authored_sources=sources, models=models, validation=ref(OUT / 'native-revision-validation.json')))
    meshes = set()
    for n in model_names:
        for node in ET.parse(OUT / n).iter('mesh'):
            f = node.get('file', node.get('filename', '')).replace('package://cove50_description/', '')
            if f: meshes.add(OUT / f)
    contracts = [ref(OUT / n) for n in contract_names + ['revision-inputs.json', 'model-build.json',
        'source-lock.json', 'native-revision-validation.json']]
    contracts += [e['snapshot'] for e in evidence]
    write(OUT / 'evidence-seal.json', dict(schema='cove50.learning-model.seal.v1', version=VERSION,
        parent_seal=ref(parent / 'evidence-seal.json'), authored_source=sources, models=models,
        meshes=[ref(p) for p in sorted(meshes)], contracts=contracts,
        environment=dict(python=sys.version, mujoco=mujoco.__version__, numpy=np.__version__),
        qualification='Numerical-resolution derivative; physical/learned behavior confidence remains bounded by the retained native evidence.'))
    pointer = read(parent / 'current-model.json')
    pointer.update(version=VERSION, status='provisional_continuous_servo_resolution_corrected_model',
        rigid_URDF=ref(OUT / 'cove50.urdf'), rigid_MJCF=ref(rigid), compliant_MJCF=ref(OUT / 'cove50-compliant.xml'),
        actuators=ref(OUT / 'actuators.json'), sensors=ref(OUT / 'sensors.json'), timing=ref(OUT / 'timing.json'),
        report=ref(OUT / 'native-revision-validation.json'), parent_model=ref(parent / 'current-model.json'),
        evidence_seal=ref(OUT / 'evidence-seal.json'),
        scope='Same selected CAD/physical contracts with0.25ms rigid continuous-servo evaluation. No completed step, installed-servo calibration or tuning-readiness promotion implied.')
    write(OUT / 'current-model.json', pointer)
    print(dict(version=VERSION, native_models=len(records), mesh_dependencies=len(meshes), rigid_timestep_s=.00025))


if __name__ == '__main__': main()
