def run_trial(data):
    """One bounded native R3 smoke, replay and PPO pilot; no provider polling."""
    import hashlib
    import json
    import os
    from pathlib import Path
    import signal
    import shutil
    import subprocess
    import tarfile
    import tempfile
    import time
    import traceback
    import urllib.request
    import boto3
    from botocore.config import Config

    started = time.monotonic()
    root = Path(tempfile.mkdtemp(prefix='r3-walk-'))
    evidence = root / 'evidence'
    evidence.mkdir()
    report = {'schema': 'cove50.runpod-walking-trial.v1', 'stages': [],
              'bundle_sha256': data['bundle_sha256'], 'status': 'starting'}
    storage = data['storage']
    client = boto3.client('s3', endpoint_url=storage['endpoint'], region_name=storage['region'],
        config=Config(signature_version='s3v4', s3={'addressing_style': 'path'},
                      connect_timeout=10, read_timeout=90, retries={'total_max_attempts': 1},
                      request_checksum_calculation='when_required', response_checksum_validation='when_required'),
        **data['storage_auth'])

    def stage(name, command, timeout):
        remaining = min(timeout, 1650 - (time.monotonic() - started))
        if remaining <= 0:
            raise RuntimeError('Worker wall budget exhausted')
        with (evidence / (name + '.log')).open('wb') as log:
            proc = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                code = proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                report['stages'].append({'name': name, 'timeout_s': remaining})
                raise RuntimeError(name + ' exceeded bounded stage time')
        report['stages'].append({'name': name, 'exit_code': code})
        if code:
            raise RuntimeError(name + ' exited ' + str(code))

    try:
        response = client.get_object(Bucket=storage['bucket'], Key=data['input_key'])
        bundle = response['Body'].read(200_000_001)
        response['Body'].close()
        if len(bundle) > 200_000_000 or hashlib.sha256(bundle).hexdigest() != data['bundle_sha256']:
            raise ValueError('Bundle hash/size mismatch')
        archive = root / 'source.tar.gz'
        archive.write_bytes(bundle)
        with tarfile.open(archive) as tf:
            # The input archive is created locally; still reject links and traversal.
            for member in tf.getmembers():
                if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                    raise ValueError('Non-file bundle member')
                if not (root / member.name).resolve().is_relative_to(root):
                    raise ValueError('Unsafe bundle member path')
            tf.extractall(root, filter='data')
        stage('hardware', ['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                           '--format=csv,noheader'], 20)
        stage('bootstrap', ['bash', data['bootstrap']], 700)
        shutil.copy2(root / 'worker-attestation.json', evidence / 'worker-attestation.json')
        python = str(root / '.venv-worker/bin/python')
        stage('packages', [python, '-m', 'pip', 'freeze'], 30)
        task = 'control/skills/walker/task/run.py'
        common = [python, task, '--device', 'cuda:0']
        stage('smoke', common + ['--mode', 'smoke', '--worlds', '1', '--steps', '4',
                                '--out', str(evidence / 'smoke')], 300)
        stage('baseline', common + ['--mode', 'replay', '--worlds', '1', '--seed', '135731',
                                   '--velocity-x', '.30', '--out', str(evidence / 'baseline')], 240)
        stage('pilot', common + ['--mode', 'train', '--worlds', '128', '--updates', '2',
                                '--seed', '135726', '--out', str(evidence / 'pilot')], 300)
        stage('adapted', common + ['--mode', 'replay', '--worlds', '1', '--seed', '135731',
                                  '--velocity-x', '.30', '--checkpoint', str(evidence / 'pilot/candidate.pt'),
                                  '--out', str(evidence / 'adapted')], 240)
        report['status'] = 'completed'
    except BaseException as exc:
        report.update(status='interrupted', error_type=type(exc).__name__,
                      traceback=traceback.format_exc()[-8000:])
        for secret in data['storage_auth'].values():
            report['traceback'] = report['traceback'].replace(secret, '[REDACTED]')
    report['wall_s'] = time.monotonic() - started
    (evidence / 'worker-report.json').write_text(json.dumps(report, indent=2) + '\n')
    output = root / 'evidence.tar.gz'
    with tarfile.open(output, 'w:gz') as tf:
        tf.add(evidence, arcname='evidence')
    content = output.read_bytes()
    response = client.put_object(Bucket=storage['bucket'], Key=data['output_key'], Body=content)
    report['upload_status'] = response['ResponseMetadata']['HTTPStatusCode']
    return {'status': report['status'], 'wall_s': report['wall_s'],
            'stages': report['stages'], 'output_sha256': hashlib.sha256(content).hexdigest(),
            'output_bytes': len(content)}
