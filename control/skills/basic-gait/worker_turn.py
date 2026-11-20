def run_turns(data):
    """One bounded sequential native batch; no provider calls or learning."""
    import hashlib
    import importlib.util
    import json
    import os
    from pathlib import Path
    import shutil
    import signal
    import subprocess
    import sys
    import tarfile
    import tempfile
    import time
    import traceback
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config

    start = time.monotonic()
    root = Path(tempfile.mkdtemp(prefix='r3-turn-')).resolve()
    evidence = root / 'evidence'; evidence.mkdir()
    plan = data['plan']
    if len(plan['cases']) != 16 or plan['hard_deadline_s'] != 1650 or plan['planned_controls'] != 3000 or plan['optimizer_updates'] != 0:
        raise ValueError('Not the authorized frozen batch')
    storage = data['storage']; prefix = data['output_prefix']
    client = boto3.client('s3', endpoint_url=storage['endpoint'], region_name=storage['region'],
        config=Config(signature_version='s3v4', s3={'addressing_style':'path'}, connect_timeout=10,
            read_timeout=90, retries={'total_max_attempts':1}, request_checksum_calculation='when_required',
            response_checksum_validation='when_required'), **data['storage_auth'])
    report = dict(schema='cove50.native-turn-batch.v1', status='starting', plan=plan,
        bundle_sha256=data['bundle_sha256'], stages=[], packs=[], progress_key=prefix+'/progress.json')
    transport = None

    def digest(path):
        with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()

    def progress():
        report['wall_s'] = time.monotonic() - start
        body = (json.dumps(report, indent=2)+'\n').encode()
        (evidence/'progress.json').write_bytes(body)
        client.put_object(Bucket=storage['bucket'], Key=report['progress_key'], Body=body, ContentType='application/json')

    def publish(name, paths):
        archive = root/(name+'.tar.gz')
        item = transport.pack(root, paths, archive)
        key = prefix+'/packs/'+name+'.tar.gz'
        client.upload_file(str(archive), storage['bucket'], key, Config=TransferConfig(max_concurrency=1,use_threads=False))
        response = client.get_object(Bucket=storage['bucket'], Key=key)
        try: sha = hashlib.file_digest(response['Body'],'sha256').hexdigest()
        finally: response['Body'].close()
        if sha != item['sha256']: raise RuntimeError('Stored pack hash differs: '+name)
        item.update(key=key, stage=name)
        report['packs'].append(item); progress(); archive.unlink()

    def stage(name, command, timeout):
        remaining = min(timeout, plan['hard_deadline_s']-(time.monotonic()-start)-120)
        if remaining <= 0: raise RuntimeError('Worker deadline before '+name)
        report['active_stage'] = name; progress()
        began = time.monotonic(); timed_out = False
        with (evidence/(name+'.log')).open('wb') as log:
            proc = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try: code = proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out = True; os.killpg(proc.pid,signal.SIGINT)
                try: code = proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL); code = proc.wait()
        row = dict(name=name, exit_code=code, wall_s=time.monotonic()-began, operator_truncated=timed_out)
        native = evidence/name/'report.json'
        if native.exists(): row['native'] = json.loads(native.read_text())
        report['stages'].append(row)
        publish(name, [evidence/name, evidence/(name+'.log')])
        if code or timed_out: raise RuntimeError('Stage incomplete; no retry: '+name)
        return row

    try:
        progress()
        response = client.get_object(Bucket=storage['bucket'], Key=data['input_key'])
        archive = root/'inputs.tar.gz'
        try:
            with archive.open('wb') as output: shutil.copyfileobj(response['Body'],output)
        finally: response['Body'].close()
        if archive.stat().st_size > 200_000_000 or digest(archive) != data['bundle_sha256']:
            raise ValueError('Bundle identity/size differs')
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                if not member.isfile() or not (root/member.name).resolve().is_relative_to(root):
                    raise ValueError('Unsafe bundle member')
            tf.extractall(root, filter='data')
        here = root/'control/skills/basic-gait'
        manifest = json.loads((root/'data/basic-skills-retry-20261027/turn/compute/bundle-manifest.json').read_text())
        for item in manifest['files']:
            path = root/item['path']
            if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
                raise ValueError('Input closure differs: '+item['path'])
        if json.loads((here/'cuda-batch.json').read_text()) != plan:
            raise ValueError('Dispatch plan differs from bundled plan')
        spec = importlib.util.spec_from_file_location('turn_transport',here/'turn_transport.py')
        transport = importlib.util.module_from_spec(spec); spec.loader.exec_module(transport)
        stage('hardware',['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],20)
        stage('bootstrap',['bash',data['bootstrap']],700)
        for name in ('worker-attestation.json','worker-pip-freeze.txt'):
            shutil.copy2(root/name,evidence/name)
        python = str(root/'.venv-worker/bin/python')
        for case in plan['cases']:
            row = stage(case['name'],[python,'-B','-u',str(here/'replay.py'),'--device','cuda:0',
                '--profile',case['profile'],'--seed',str(case['seed']),'--policy',case['policy'],
                '--method',case['method'],'--out',str(evidence/case['name'])],plan['stage_timeout_s'])
            native = row.get('native',{})
            if native.get('status') != 'completed' or native.get('simulator_transitions') != case['controls'] or native.get('native_physics_steps') != 80*case['controls']:
                raise RuntimeError('Native attempt accounting differs; retain without retry: '+case['name'])
        report['status'] = 'completed'
    except BaseException as exc:
        detail = traceback.format_exc()[-8000:]
        for secret in data['storage_auth'].values():
            if secret: detail = detail.replace(secret,'[REDACTED]')
        report.update(status='incomplete',error_type=type(exc).__name__,traceback=detail)
    finally:
        report.pop('active_stage',None)
        report['wall_s'] = time.monotonic()-start
        native_rows = [row for row in report['stages'] if row['name'] not in ('hardware','bootstrap')]
        report['actual_attempts'] = len(native_rows)
        report['totals'] = {}
        report['unknown_counter_stages'] = {}
        for key in ('simulator_transitions','native_physics_steps'):
            unknown = [row['name'] for row in native_rows if key not in row.get('native',{})]
            report['unknown_counter_stages'][key] = unknown
            report['totals'][key] = None if unknown else sum(row['native'][key] for row in native_rows)
        report['totals'].update(optimizer_updates=0,ppo_rounds=0,distinct_new_policies=0)
        (evidence/'worker-report.json').write_text(json.dumps(report,indent=2)+'\n')
        progress()
        final = root/'evidence.tar.gz'
        paths = [evidence/name for name in ('worker-report.json','progress.json','worker-attestation.json','worker-pip-freeze.txt') if (evidence/name).exists()]
        if transport is None:
            # Setup failures precede extraction; preserve their compact receipt.
            with tarfile.open(final,'w:gz') as tf:
                for path in paths: tf.add(path,arcname='evidence/'+path.name)
        else: transport.pack(root,paths,final)
        client.upload_file(str(final),storage['bucket'],data['output_key'],Config=TransferConfig(max_concurrency=1,use_threads=False))
    return dict(status=report['status'],wall_s=report['wall_s'],packs=report['packs'],totals=report['totals'],
        output_sha256=digest(final),output_bytes=final.stat().st_size)
