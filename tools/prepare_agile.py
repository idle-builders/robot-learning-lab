"""Seal the current floor-reach robot for native AGILE asset/actuator preflight."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import uuid
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare(train=False, resume=None, export_checkpoint=None):
    import mujoco
    from lab.floor_pickup import FloorProbe
    from lab.artifact_store import snapshot_file, retain_dependency

    parent = ROOT / 'artifacts/vla-demonstrations-d6140c7e55aa'
    out = ROOT / 'artifacts' / ('agile-inputs-' + uuid.uuid4().hex[:12])
    model_dir = out / 'model'
    model_dir.mkdir(parents=True)
    motor = FloorProbe(parent / 'scene.xml', 1.0)
    tree = ET.parse(parent / 'scene.xml')
    root = tree.getroot()
    # Keep the exact robot, contacts and CAD. Remove only task props: this asset
    # becomes an articulation; the native task owns its ground and object scene.
    world = root.find('worldbody')
    for child in list(world):
        if child.tag == 'body' and child.get('name') != 'torso':
            world.remove(child)
        elif child.tag == 'geom' and child.get('name') == 'destination':
            world.remove(child)
    for pair in list(root.find('contact')):
        if 'task_block_geom' in (pair.get('geom1'), pair.get('geom2')):
            root.find('contact').remove(pair)
    for mesh in root.findall('./asset/mesh'):
        source = Path(mesh.get('file'))
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        relative = Path('meshes') / (digest + source.suffix)
        if not (model_dir / relative).exists():
            snapshot_file(source, model_dir / relative)
        mesh.set('file', str(relative))
    for actuator in root.findall('./actuator/*'):
        if actuator.get('joint') in motor.order:
            cap = float(motor.pack.effort_limit[motor.order.index(actuator.get('joint'))])
            actuator.set('ctrlrange', f'{-cap} {cap}')
    tree.write(model_dir / 'robot.xml')
    compiled = mujoco.MjModel.from_xml_path(str(model_dir / 'robot.xml'))
    # Compare every retained named robot body's mass/inertia, not just totals.
    import numpy as np
    for i in range(1, compiled.nbody):
        name = mujoco.mj_id2name(compiled, mujoco.mjtObj.mjOBJ_BODY, i)
        old = motor.model.body(name).id
        for field in ['body_mass', 'body_inertia', 'body_ipos', 'body_iquat']:
            np.testing.assert_allclose(getattr(compiled, field)[i], getattr(motor.model, field)[old], rtol=1e-12, atol=1e-12)
    contract = {
        'scope': 'Current provisional floor-reach humanoid; parameters unmeasured; 1 Nm leg cap is an experimental assumption.',
        'root_body': 'torso', 'root_position_m': motor.data.qpos[:3].tolist(),
        'joint_order': motor.order, 'rest_rad': dict(zip(motor.order, motor.rest.tolist())),
        'joints': [dict(j, kp=float(motor.pack.kp[i]), kd=float(motor.pack.kd[i]),
                        effort_limit_nm=float(motor.pack.effort_limit[i])) for i, j in enumerate(motor.joints)],
        'bodies': [{'name': mujoco.mj_id2name(compiled, mujoco.mjtObj.mjOBJ_BODY, i),
                    'mass_kg': float(compiled.body_mass[i]), 'inertia_kg_m2': compiled.body_inertia[i].tolist()}
                   for i in range(1, compiled.nbody)],
        'robot_mass_kg': float(compiled.body_mass.sum()),
        'physics_hz': 500, 'controller_hz': 50,
        'mechanical_parameter_copy': 'pass',
        'electrical_model': 'learning/pip-electromechanical-v1.json; native actuator integration not yet verified',
    }
    (model_dir / 'contract.json').write_text(json.dumps(contract, indent=2))
    for name in ['workers/agile/preflight.py', 'workers/agile/pip_cfg.py', 'workers/agile/entry.py', 'workers/agile/zero_baseline.py']:
        snapshot_file(ROOT / name, out / 'code' / Path(name).name)
    snapshot_file(ROOT / 'workers/agile/evaluation.yaml', out / 'evaluation.yaml')
    experiment = {
        'operation': 'export' if export_checkpoint else ('train_evaluate' if train or resume else 'preflight'),
        'seed': 42, 'num_envs': 512 if resume else 64, 'max_iterations': 100,
        'zero_baseline': bool(resume),
        'scope': 'Bounded native motor RL pilot; no task-learning qualification.'}
    checkpoint = export_checkpoint or resume
    if checkpoint:
        import torch
        checkpoint = Path(checkpoint).absolute()
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        snapshot_file(checkpoint, out / 'resume/model.pt')
        experiment['resume'] = {'path': 'resume/model.pt', 'iteration': int(saved['iter']),
            'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(), 'origin': str(checkpoint)}
    (out / 'experiment.json').write_text(json.dumps(experiment, indent=2))
    snapshot_file(ROOT / 'learning/pip-electromechanical-v1.json', model_dir / 'electromechanical.json')
    snapshot_file(ROOT / 'workers/control_loop/robotlab/electromechanical.py', out / 'code/electromechanical.py')
    files = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in out.rglob('*') if p.is_file()}
    (out / 'manifest.json').write_text(json.dumps({'files': files}, indent=2))
    dependencies = {'plant': retain_dependency(parent)}
    if checkpoint:
        run_name = checkpoint.relative_to(ROOT / 'artifacts').parts[0]
        dependencies['resume_run'] = retain_dependency(ROOT / 'artifacts' / run_name)
    (out / 'dependencies.json').write_text(json.dumps(dependencies, indent=2))
    print(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument('--train', action='store_true', help='Include bounded native PPO and fixed evaluation after asset preflight')
    operation.add_argument('--resume', type=Path, help='Continue a native checkpoint with 512 parallel environments and compare a zero-action baseline')
    operation.add_argument('--export-checkpoint', type=Path, help='Run native LEAPP export/validation of an existing checkpoint without training')
    args = parser.parse_args()
    prepare(args.train, args.resume, args.export_checkpoint)
