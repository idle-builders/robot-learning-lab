"""Bounded pinned upstream AGILE installation and Pip native preflight."""
from runpod_flash import Endpoint, GpuGroup, PodTemplate


@Endpoint(name='robot-lab-agile', gpu=[GpuGroup.ADA_48_PRO, GpuGroup.ADA_24], min_cuda_version='13.0',
    workers=(0, 1), gpu_count=1, idle_timeout=5, execution_timeout_ms=1_800_000,
    template=PodTemplate(containerDiskInGb=100),
    dependencies=['boto3==1.43.98', 'uv==0.12.17'],
    system_dependencies=['git', 'libegl1', 'libgl1', 'libopengl0', 'libvulkan1', 'libxt6'],
    env={'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility,graphics', 'RUNPOD_API_KEY': '',
         'ACCEPT_EULA': 'Y', 'PRIVACY_CONSENT': 'N'})
async def agile(payload: dict, storage_auth: dict) -> dict:
    import hashlib
    import json
    import os
    from pathlib import Path
    import signal
    import subprocess
    import sys
    import tempfile
    import time
    import traceback
    from s3_artifacts import Artifacts, client_from_auth, redact

    root = Path(tempfile.mkdtemp(prefix='robot-agile-'))
    output = root / 'evidence'
    output.mkdir()
    storage = Artifacts(client_from_auth(payload['storage'], storage_auth), payload['storage'])
    started = time.monotonic()
    report = {'status': 'fail', 'robot_rl': 'not_run', 'bootstrap_version': 9, 'stages': []}
    env = os.environ.copy()
    # Check out the whole small Python package, but do not download unrelated
    # released robot checkpoints through Git LFS.
    env['GIT_LFS_SKIP_SMUDGE'] = '1'

    def run(name, command, cwd, timeout):
        remaining = min(timeout, 1500 - (time.monotonic() - started))
        if remaining <= 0:
            raise TimeoutError('AGILE preflight total deadline')
        stage_started = time.monotonic()
        # Flash captures Python stdout until the function returns. Stage-only
        # progress on the worker's native stdout remains visible in CLI logs.
        os.write(1, (json.dumps({'agile_stage': name, 'state': 'starting'}) + '\n').encode())
        log_path = output / (name + '.log')
        log_path.touch(exist_ok=False)
        # Standard tee retains exact native output and exposes it to Runpod's
        # worker logs. Command arguments stay separate from shell source.
        process = subprocess.Popen(['bash', '-o', 'pipefail', '-c',
            '"$@" 2>&1 | tee -a "$ROBOTLAB_STAGE_LOG"', 'agile-stage', *command],
            cwd=cwd, env={**env, 'ROBOTLAB_STAGE_LOG': str(log_path)},
            stdout=1, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            report['stages'].append({'stage': name, 'status': 'timeout',
                'elapsed_s': time.monotonic() - stage_started, 'command': command})
            raise TimeoutError(name + ' deadline')
        elapsed = time.monotonic() - stage_started
        report['stages'].append({'stage': name, 'exit_code': code, 'command': command, 'elapsed_s': elapsed})
        os.write(1, (json.dumps({'agile_stage': name, 'exit_code': code, 'elapsed_s': elapsed}) + '\n').encode())
        if code:
            raise RuntimeError(name + ' failed; retained log has details')

    try:
        run('hardware', ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], root, 20)
        hardware = (output / 'hardware.log').read_text().strip()
        version = tuple(map(int, hardware.split(',')[1].strip().split('.')))
        if version < (580, 95, 5):
            raise RuntimeError('Worker driver below Isaac Sim 6 tested Linux baseline 580.95.05: ' + hardware)
        report['input_transfer'] = storage.materialize(payload['files'], root / 'transfer')
        bundle = root / 'transfer/bundles/pip'
        if hashlib.sha256((bundle / 'manifest.json').read_bytes()).hexdigest() != payload['bundle_sha256']:
            raise ValueError('Input manifest mismatch')
        checkout = root / 'WBC-AGILE'
        run('clone', ['git', 'clone', '--filter=blob:none', '--sparse',
            'https://github.com/nvidia-isaac/WBC-AGILE.git', str(checkout)], root, 90)
        run('sparse', ['git', 'sparse-checkout', 'set', '--cone',
            'agile', 'scripts', 'third_party'], checkout, 30)
        run('checkout', ['git', 'checkout', '6830cf995714e81c91ce63247e8e016d36e28f14'], checkout, 90)
        run('root-files', ['git', 'checkout', '--ignore-skip-worktree-bits', 'HEAD', '--',
            'uv.lock', 'pyproject.toml', 'README.md'], checkout, 30)
        report['upstream_revision'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip()
        report['upstream_lock_sha256'] = hashlib.sha256((checkout / 'uv.lock').read_bytes()).hexdigest()
        if report['upstream_lock_sha256'] != '08fa2e1d941c9e59a0247220af1cc8bf1ca72e692350166b7742fb464a328ff6':
            raise ValueError('Pinned upstream dependency lock mismatch')
        run('install', [sys.executable, '-m', 'uv', 'sync', '--frozen'], checkout, 1000)
        python = str(checkout / '.venv/bin/python')
        run('inventory', [sys.executable, '-m', 'uv', 'pip', 'freeze', '--python', python], checkout, 30)
        env.update(PYTHONPATH=str(bundle / 'code') + ':' + str(checkout), AGILE_CHECKOUT=str(checkout))
        run('native-preflight', [python, str(bundle / 'code/preflight.py'),
            '--inputs', str(bundle), '--output', str(output / 'native'), '--headless'], root, 360)
        report['native_preflight'] = json.loads((output / 'native/report.json').read_text())
        if report['native_preflight']['status'] != 'pass':
            raise RuntimeError('Native preflight report failed despite process exit zero')
        plan_path = bundle / 'experiment.json'
        plan = json.loads(plan_path.read_text()) if plan_path.exists() else {'operation': 'preflight'}
        if plan['operation'] not in {'preflight', 'train_evaluate', 'export'}:
            raise ValueError('Unknown native experiment operation')
        if plan['operation'] == 'export':
            checkpoint = bundle / 'resume/model.pt'
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != plan['resume']['sha256']:
                raise ValueError('Export checkpoint hash mismatch')
            env.update(ROBOTLAB_USD=str(output / 'native/articulation.usda'),
                ROBOTLAB_CONTRACT=str(bundle / 'model/contract.json'),
                ROBOTLAB_ELECTRICAL=str(bundle / 'model/electromechanical.json'))
            report['checkpoint'] = plan['resume']
            run('native-leapp-export', [python, str(bundle / 'code/entry.py'), 'export',
                '--task', 'RobotLab-Pip-Agile-v0', '--checkpoint', str(checkpoint),
                '--export_save_path', str(output / 'leapp'), '--validation_steps', '5',
                '--disable_graph_visualization', '--headless'], output, 360)
            graph = output / 'leapp/RobotLab-Pip-Agile-v0/RobotLab-Pip-Agile-v0.yaml'
            if not graph.is_file():
                raise RuntimeError('Native export returned without LEAPP graph')
            report['leapp_export'] = {'status': 'executed', 'graph': str(graph.relative_to(output)),
                'sha256': hashlib.sha256(graph.read_bytes()).hexdigest(), 'validation_steps': 5}
            report['task_success'] = 'not_run'
        if plan['operation'] == 'train_evaluate':
            # Fixed economical pilot. Use the unchanged native entry points;
            # this worker only manages compute, inputs and retained evidence.
            if (plan['num_envs'], plan['max_iterations'], plan['seed']) not in {(64, 100, 42), (512, 100, 42)}:
                raise ValueError('Pilot must use 64 or 512 environments, 100 iterations and seed 42')
            env.update(ROBOTLAB_USD=str(output / 'native/articulation.usda'),
                ROBOTLAB_CONTRACT=str(bundle / 'model/contract.json'),
                ROBOTLAB_ELECTRICAL=str(bundle / 'model/electromechanical.json'))
            entry = [python, str(bundle / 'code/entry.py')]
            common = ['--task', 'RobotLab-Pip-Agile-v0', '--headless']
            def evaluate(name, policy_path):
                run(name, entry + ['eval'] + common + [
                    '--num_envs', '4', '--checkpoint', str(policy_path), '--num_steps', '450',
                    '--eval_config', str(bundle / 'evaluation.yaml'), '--run_evaluation',
                    '--save_trajectories', '--generate_report',
                    '--metrics_file', str(output / name / 'metrics.json')], output, 180)
                metrics = output / name / 'metrics.json'
                if not metrics.is_file():
                    raise RuntimeError(name + ' returned without native metrics')
                return json.loads(metrics.read_text())

            if plan.get('zero_baseline'):
                baseline = output / 'zero_baseline.pt'
                run('make-zero-baseline', [python, str(bundle / 'code/zero_baseline.py'), str(baseline)], root, 30)
                report['unlearned_zero_baseline'] = evaluate('zero-baseline', baseline)
            resume_args = []
            expected_iteration = 99
            if plan.get('resume'):
                source = bundle / 'resume/model.pt'
                if hashlib.sha256(source.read_bytes()).hexdigest() != plan['resume']['sha256']:
                    raise ValueError('Resume checkpoint hash mismatch')
                resume_args = ['--resume', 'True', '--checkpoint', str(source)]
                expected_iteration += plan['resume']['iteration']
                report['resume'] = plan['resume']
            report['robot_rl'] = 'started_updates_unverified'
            run('native-train', entry + ['train'] + common + [
                '--num_envs', str(plan['num_envs']), '--max_iterations', '100', '--seed', '42',
                '--logger', 'tensorboard'] + resume_args, output, 600)
            checkpoints = list((output / 'logs').rglob('model_*.pt'))
            if not checkpoints:
                raise RuntimeError('Native trainer returned without any checkpoint')
            checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
            report['checkpoint'] = {'path': str(checkpoint.relative_to(output)),
                'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
            # Persist training before starting a separate evaluation process.
            report['training_evidence'] = storage.retain_tree(output / 'logs', root.name + '-training')
            if int(checkpoint.stem.split('_')[-1]) < expected_iteration:
                raise RuntimeError('Native training stopped before the final requested iteration; checkpoints retained')
            report['robot_rl'] = 'executed_not_qualified'
            report['new_rollout_transitions'] = plan['num_envs'] * 24 * 100
            report['evaluation_metrics'] = evaluate('evaluation', checkpoint)
            report['task_success'] = 'not_run'
        report['status'] = 'partial'
    except Exception:
        report['error'] = traceback.format_exc()
    finally:
        report['elapsed_s'] = time.monotonic() - started
        for path in output.rglob('*'):
            if path.is_file() and path.suffix in {'.log', '.json', '.txt'}:
                path.write_text(redact(path.read_text(errors='replace'), storage_auth))
        (output / 'report.json').write_text(json.dumps(report, indent=2))
    return {'status': report['status'], 'report': report,
            's3_evidence': storage.retain_tree(output, root.name)}
