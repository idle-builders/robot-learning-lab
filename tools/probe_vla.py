"""Run bounded native policy probes and LeRobot reference tutorials.

Prepare with .venv-learning; run with .venv-experiments. Probe mode does not apply
unadapted outputs. Reference modes execute upstream training/evaluation; their
retained results determine what actually passed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare():
    import mujoco
    import numpy as np
    from PIL import Image
    from lab.floor_pickup import FloorProbe
    from lab.artifact_store import snapshot_file, retain_dependency

    parent = ROOT / 'artifacts/floor-learning-2f86cd18caaa'
    output = ROOT / 'artifacts' / ('vla-inputs-' + uuid.uuid4().hex[:12])
    output.mkdir()
    motor = FloorProbe(parent / 'scene.xml', 1.)
    renderer = None
    try:
        motor.data.qpos[0] = -.25
        for name, value in [('neck_yaw_joint', .65), ('head_joint', .5)]:
            motor.data.qpos[motor.qidx[motor.order.index(name)]] = value
        mujoco.mj_forward(motor.model, motor.data)
        _, gravity, gyro, *_ = motor.motion()
        # Wheel encoders supply speed; continuous accumulated wheel angle is not
        # equivalent to the wheel-speed command. All other joints supply radians.
        joint_state = np.where(motor.wheels, motor.data.qvel[motor.vidx], motor.data.qpos[motor.qidx])
        low = [(-j['velocity_limit_rad_s'] if j['continuous'] else j['limits'][0]) for j in motor.joints]
        high = [(j['velocity_limit_rad_s'] if j['continuous'] else j['limits'][1]) for j in motor.joints]
        names = [n + ('.velocity_rad_s' if wheel else '.position_rad')
                 for n, wheel in zip(motor.order, motor.wheels, strict=True)]
        request = {
            'schema_version': 1, 'robot_model': str(parent / 'scene.xml'),
            'robot_model_sha256': hashlib.sha256((parent / 'scene.xml').read_bytes()).hexdigest(),
            'acquisition_time_s': float(motor.data.time),
            'scope': 'Synchronized simulated sensors at a declared reset pose; no controller or task rollout executed.',
            'state': np.r_[joint_state, gravity, gyro].tolist(),
            'state_names': names + ['gravity_body.' + a for a in 'xyz'] + ['gyro_body_rad_s.' + a for a in 'xyz'],
            'action_names': names,
            'engineering_bounds': {
                'observation.state': {'min': low + [-1.] * 3 + [-10.] * 3, 'max': high + [1.] * 3 + [10.] * 3},
                'action': {'min': low, 'max': high},
            },
            'normalization_scope': 'Provisional engineering limits for IO probing only; not demonstration statistics.',
            'action_application': 'disabled; NEW_EMBODIMENT has not been adapted',
        }
        renderer = mujoco.Renderer(motor.model, height=180, width=320)
        renderer.update_scene(motor.data, camera='head_camera')
        Image.fromarray(renderer.render()).save(output / 'head.png')
        (output / 'observation.json').write_text(json.dumps(request, indent=2))
        (output / 'dependencies.json').write_text(json.dumps(retain_dependency(parent), indent=2))
        for source in ['tools/probe_vla.py', 'lab/floor_pickup.py', 'lab/mujoco_learning.py']:
            snapshot_file(ROOT / source, output / 'source' / source)
    finally:
        if renderer is not None:
            renderer.close()
        motor.close()
    finalize(output)
    return output


def finalize(output):
    from lab.artifact_store import plan_migration, apply_migration, verify_manifest
    plan = plan_migration(ROOT, [output])
    apply_migration(ROOT, plan)
    verify_manifest(ROOT, plan)


def endpoint_ids():
    result = subprocess.run(['runpodctl', 'serverless', 'list'], capture_output=True,
                            text=True, check=True, timeout=20)
    return {x['id'] for x in json.loads(result.stdout) if 'robot-lab-vla' in x.get('name', '')}


def adaptation_files(inputs):
    from xml.etree import ElementTree as ET
    from workers.s3_artifacts import INPUT_BYTES, INPUT_FILES
    files = {}
    for path in (inputs / 'dataset').rglob('*'):
        if path.is_file():
            files['dataset/' + str(path.relative_to(inputs / 'dataset'))] = path.read_bytes()
    for src, dst in [('report.json', 'dataset-report.json'), ('contract.json', 'contract.json'),
                     ('evaluation-cases.json', 'evaluation-cases.json')]:
        files[dst] = (inputs / src).read_bytes()
    files['native_adapt.py'] = (ROOT / 'workers/vla/native_adapt.py').read_bytes()
    for name in ['lab/vla_control.py', 'lab/learned_motor.py', 'lab/floor_pickup.py', 'lab/mujoco_learning.py', 'lab/electromechanical.py',
                 'learning/pip-control-v1.json', 'learning/pip-electromechanical-v1.json']:
        files[name] = (ROOT / name).read_bytes()
    files['lab/__init__.py'] = b''
    bundle = ROOT / 'artifacts/pip-cad-gym-bundle-v4/model'
    for name in ['interfaces.json', 'resolved-config.json']:
        files['bundle/model/' + name] = (bundle / name).read_bytes()
    scene = ET.parse(inputs / 'scene.xml')
    for mesh in scene.findall('./asset/mesh'):
        source = Path(mesh.get('file'))
        if not source.is_absolute():
            source = inputs / source
        # Preserve native mesh suffix and relative XML directory semantics.
        name = 'bundle/model/meshes/' + source.name
        data = source.read_bytes()
        if name in files and files[name] != data:
            raise ValueError('Ambiguous portable mesh filename')
        files[name] = data
        mesh.set('file', name)
    files['scene.xml'] = ET.tostring(scene.getroot())
    if len(files) > INPUT_FILES or sum(map(len, files.values())) > INPUT_BYTES:
        raise ValueError('VLA experiment exceeds transport limits')
    return files


def checkpoint_files(checkpoint, contract):
    """Bind reused weights to the exact named robot interface they were trained on."""
    from workers.vla.native_adapt import PINS
    required = {'config.json', 'adapter_config.json', 'adapter_model.safetensors',
                'policy_preprocessor.json', 'policy_postprocessor.json'}
    paths = [p for p in checkpoint.iterdir() if p.is_file()]
    if not required <= {p.name for p in paths}:
        raise ValueError('Incomplete native PEFT checkpoint')
    origin = next((p/'submitted-source/contract.json' for p in checkpoint.parents
                   if (p/'submitted-source/contract.json').is_file()), None)
    if origin is None or json.loads(origin.read_text()) != contract:
        raise ValueError('Checkpoint lacks a matching retained robot interface contract')
    files = {'checkpoint/' + p.name:p.read_bytes() for p in paths}
    cfg = json.loads(files['checkpoint/config.json'])
    if (cfg['input_features']['observation.state']['shape'] != [len(contract['state_names'])]
        or cfg['output_features']['action']['shape'] != [len(contract['action_names'])]):
        raise ValueError('Checkpoint and robot state/action dimensions differ')
    adapter = json.loads(files['checkpoint/adapter_config.json'])
    if adapter['base_model_name_or_path'] != 'lerobot/smolvla_base' or adapter['revision'] != PINS['lerobot/smolvla_base']:
        raise ValueError('Checkpoint base model does not match pinned runtime')
    return files


def run(inputs, mode="probe", checkpoint=None, action_steps=1, motor_checkpoint=None):
    from huggingface_hub import get_token, get_hf_file_metadata, hf_hub_url
    if mode == "reference-rl":
        PINS = {}  # Public tutorial assets are pinned and downloaded by the worker.
    elif mode in {"adapt", "evaluate"}:
        from workers.vla.native_adapt import PINS
    elif mode in {'reference', 'reference-train'}:
        from workers.vla.lerobot_reference import PINS
    else:
        from workers.vla.native_probe import PINS
    from workers.s3_artifacts import Artifacts, CONFIG, profile_auth, client_from_auth, redact
    from lab.artifact_store import snapshot_file, retain_dependency
    token = get_token()
    if not token:
        raise RuntimeError('Hugging Face authentication missing')
    for repo, revision in PINS.items():
        if repo == 'lerobot/libero-assets':
            from huggingface_hub import HfApi
            HfApi(token=token).dataset_info(repo, revision=revision)
            continue
        name = 'model-00001-of-00002.safetensors' if 'GR00T' in repo else 'model.safetensors'
        get_hf_file_metadata(hf_hub_url(repo, name, revision=revision), token=token, timeout=15)
    if mode == 'evaluate' and checkpoint is None:
        raise ValueError('Evaluation requires a retained pretrained_model checkpoint directory')
    if action_steps not in (1, 5):
        raise ValueError('This bounded comparison supports action_steps 1 or 5')
    if checkpoint is not None and mode != 'evaluate':
        raise ValueError('Checkpoint reuse is currently evaluation-only; adaptation resume is not implemented')
    if motor_checkpoint is not None and mode != 'evaluate':
        raise ValueError('Learned motor comparison is evaluation-only')
    output = ROOT / 'artifacts' / ({'adapt':'vla-adaptation-', 'evaluate':'vla-evaluation-', 'probe':'vla-probe-', 'reference':'lerobot-reference-', 'reference-train':'lerobot-reference-train-', 'reference-rl':'lerobot-reference-rl-'}[mode] + uuid.uuid4().hex[:12])
    output.mkdir()
    print(output, flush=True)
    auth = profile_auth('runpod')
    secrets = dict(auth, hf_token=token)
    storage = Artifacts(client_from_auth(CONFIG, auth), CONFIG)
    if mode in {'adapt', 'evaluate'}:
        files = adaptation_files(inputs)
        if checkpoint is not None:
            # Only native adapter/processor files; no prior source snapshots.
            files.update(checkpoint_files(checkpoint, json.loads(files['contract.json'])))
        if motor_checkpoint is not None:
            motor_contract=json.loads((motor_checkpoint.parent/'contract.json').read_text())
            vla_contract=json.loads(files['contract.json'])
            if any(motor_contract[a] != vla_contract[b] for a,b in
                   [('goal_names','action_names'),('goal_min','action_min'),('goal_max','action_max')]):
                raise ValueError('Motor goals do not match the named VLA action interface')
            files['motor/policy.zip']=motor_checkpoint.read_bytes()
            files['motor/contract.json']=json.dumps(motor_contract,indent=2).encode()
            reference=output/'motor-plant-reference.npz'
            subprocess.run([str(ROOT/'.venv-learning/bin/python'),'-m','lab.learned_motor',
                '--scene',str(inputs/'scene.xml'),'--contract',str(motor_checkpoint.parent/'contract.json'),
                '--output',str(reference)],cwd=ROOT,check=True,timeout=30)
            files['motor/plant-reference.npz']=reference.read_bytes()
    elif mode == 'reference-rl':
        files = {'lerobot_hilserl.py': (ROOT / 'workers/vla/lerobot_hilserl.py').read_bytes()}
    elif mode in {'reference', 'reference-train'}:
        files = {'lerobot_reference.py': (ROOT / 'workers/vla/lerobot_reference.py').read_bytes()}
    else:
        files = {name: (inputs / name).read_bytes() for name in ['head.png', 'observation.json']}
        files['native_probe.py'] = (ROOT / 'workers/vla/native_probe.py').read_bytes()
    from workers.s3_artifacts import INPUT_BYTES, INPUT_FILES
    if len(files) > INPUT_FILES or sum(map(len, files.values())) > INPUT_BYTES:
        raise ValueError('VLA inputs and checkpoint exceed transport limits')
    for name, data in files.items():
        dest = output / 'submitted-source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    for name in ['tools/probe_vla.py', 'workers/serverless/vla.py', 'workers/serverless/s3_artifacts.py']:
        snapshot_file(ROOT / name, output / 'source' / name)
    dependencies = {} if mode in {'reference', 'reference-rl'} else {'demonstrations': retain_dependency(inputs)}
    if checkpoint is not None:
        dependencies['checkpoint'] = retain_dependency(checkpoint)
    if motor_checkpoint is not None:
        dependencies['motor_checkpoint'] = retain_dependency(motor_checkpoint.parent)
    (output / 'dependencies.json').write_text(json.dumps(dependencies, indent=2))
    payload = {'mode': mode, 'action_steps': action_steps, 'run_id': output.name, 'files': {name: storage.put_bytes(data) for name, data in files.items()}, 'storage': dict(CONFIG)}
    if mode == 'reference-train':
        dataset = inputs / 'dataset'
        report = json.loads((inputs/'report.json').read_text())
        verification = json.loads((inputs/'conversion-verification.json').read_text())
        if report.get('status') != 'converted_and_loaded' or verification.get('status') != 'pass':
            raise ValueError('Reference training requires the verified native dataset conversion')
        paths = {str(p.relative_to(dataset)): p for p in dataset.rglob('*') if p.is_file()}
        if len(paths) > INPUT_FILES or sum(p.stat().st_size for p in paths.values()) > 800_000_000:
            raise ValueError('Reference dataset exceeds its explicit 800 MB bound')
        payload['reference_dataset_files'] = {name: storage.put_file(p) for name,p in paths.items()}
    (output / 'transfer.json').write_text(json.dumps(payload, indent=2))
    before = endpoint_ids()
    process = None
    completed = False
    try:
        import socket
        with socket.socket() as check:
            check.bind(('127.0.0.1', 8903))
        with (output / 'flash.log').open('x') as log:
            process = subprocess.Popen([str(ROOT / 'tools/runpod-flash'), 'dev', '--port', '8903', '--no-reload'],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            base = 'http://127.0.0.1:8903'
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError('Flash exited before readiness')
                try:
                    with urllib.request.urlopen(base + '/openapi.json', timeout=1) as response:
                        json.load(response)
                    break
                except Exception:
                    time.sleep(.5)
            else:
                raise TimeoutError('Flash readiness deadline')
            envelope = {'input': {'payload': payload, 'storage_auth': auth, 'model_auth': token}}
            request = urllib.request.Request(base + '/vla/runsync', data=json.dumps(envelope).encode(),
                                             headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=1380) as response:
                result = redact(json.load(response), secrets)
            (output / 'response.json').write_text(json.dumps(result, indent=2))
            worker = result.get('output', {})
            if 's3_evidence' in worker:
                verification = storage.download_run(worker['s3_evidence'], output / 'evidence')
                (output / 'download-verification.json').write_text(json.dumps(verification, indent=2))
            print('Worker status:', worker.get('status', 'missing'), flush=True)
            completed = worker.get('status') == {'adapt':'adaptation_executed', 'evaluate':'evaluation_executed', 'probe':'io_probe_pass', 'reference':'reference_evaluation_executed', 'reference-train':'reference_training_evaluated', 'reference-rl':'reference_rl_executed'}[mode]
    except Exception as exc:
        (output / 'client-error.json').write_text(json.dumps({'status': 'fail', 'error_type': type(exc).__name__}))
        print('Probe failed:', type(exc).__name__, flush=True)
    finally:
        if process is not None:
            try:
                process.send_signal(signal.SIGINT)
                process.wait(timeout=25)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
        cleanup = []
        try:
            for endpoint in endpoint_ids() - before:
                result = subprocess.run(['runpodctl', 'serverless', 'delete', endpoint], capture_output=True, timeout=30)
                cleanup.append({'endpoint': endpoint, 'delete_exit_code': result.returncode})
            (output / 'cleanup.json').write_text(json.dumps({'deletions': cleanup, 'remaining_owned_endpoints': sorted(endpoint_ids() - before)}))
        except Exception as exc:
            (output / 'cleanup.json').write_text(json.dumps({'status': 'needs_check', 'error_type': type(exc).__name__}))
        log = output / 'flash.log'
        if log.exists():
            log.write_text(redact(log.read_text(errors='replace'), secrets))
        finalize(output)
    if not completed:
        raise RuntimeError('Native VLA probe failed; evidence retained at ' + str(output))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--mode', choices=['probe', 'adapt', 'evaluate', 'reference', 'reference-train', 'reference-rl'], default='probe')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--action-steps', type=int, choices=[1,5], default=1)
    parser.add_argument('--motor-checkpoint', type=Path)
    args = parser.parse_args()
    if args.stage == 'run' and args.inputs is None and args.mode not in {'reference', 'reference-rl'}:
        parser.error('run requires --inputs')
    print(prepare() if args.stage == 'prepare' else run(args.inputs, args.mode, args.checkpoint, args.action_steps,args.motor_checkpoint))
