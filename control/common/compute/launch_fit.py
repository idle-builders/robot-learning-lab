"""Explicit submit/status/restore for one bounded actor-critic continuation.

No background monitoring or automatic retries. Status and progress reads share
a five-minute minimum interval; evidence is streamed rather than read into RAM.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
import urllib.error

from launch import ROOT, call, s3


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def inspect_allowed(receipt, record):
    previous = record.get('last_inspection_at', record['submitted_at_epoch'])
    if time.time() - previous < 300:
        raise SystemExit('Give the job time: wait five minutes between explicit inspections.')
    record['last_inspection_at'] = time.time()
    write(receipt, record)


def restore_pack(client, bucket, item, run_dir):
    packs = run_dir / 'packs'
    packs.mkdir(exist_ok=True)
    target = packs / (item['sha256'] + '.tar.gz')
    def digest(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    if target.exists() and digest(target) != item['sha256']:
        raise ValueError('Existing evidence archive differs: ' + str(target))
    if not target.exists():
        if shutil.disk_usage(run_dir).free < item['bytes'] * 3 + 1_000_000_000:
            raise ValueError('Insufficient local space for native evidence and extraction')
        response = client.get_object(Bucket=bucket, Key=item['key'])
        temporary = target.with_suffix('.partial')
        try:
            with temporary.open('wb') as output:
                shutil.copyfileobj(response['Body'], output, length=4 * 1024 * 1024)
        finally:
            response['Body'].close()
        if temporary.stat().st_size != item['bytes'] or digest(temporary) != item['sha256']:
            raise ValueError('Transferred evidence identity differs: ' + item['stage'])
        temporary.rename(target)
    marker = target.with_suffix('.restored.json')
    if not marker.exists():
        with tarfile.open(target) as archive:
            for member in archive.getmembers():
                if not (member.isfile() or member.isdir()) or member.issym() or member.islnk():
                    raise ValueError('Invalid native evidence member')
                if not (run_dir / member.name).resolve().is_relative_to(run_dir.resolve()):
                    raise ValueError('Unsafe native evidence member')
            archive.extractall(run_dir, filter='data')
        write(marker, item)
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('submit', 'status', 'fetch'))
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--bootstrap')
    parser.add_argument('--recipe', type=Path)
    args = parser.parse_args()
    receipt = args.run_dir / 'submission.json'
    if args.action == 'submit':
        if not args.bundle or not args.bootstrap or not args.recipe:
            parser.error('submit needs --bundle, --bootstrap and --recipe')
        recipe = json.loads(args.recipe.read_text())
        args.run_dir.mkdir(parents=True, exist_ok=False)
        source = args.run_dir / 'submitted-source'
        source.mkdir()
        for name in ('launch_fit.py', 'worker_fit.py'):
            shutil.copy2(Path(__file__).with_name(name), source / name)
        shutil.copy2(args.recipe, source / 'recipe.json')
        client, bucket, endpoint = s3()
        if not endpoint:
            raise SystemExit('Enable the Terraform walking_trial endpoint before submitting.')
        with args.bundle.open('rb') as stream:
            sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        prefix = 'robot-learning-lab/v1/walking-fit/' + args.run_dir.name
        input_key, output_key = prefix + '/input.tar.gz', prefix + '/evidence.tar.gz'
        with args.bundle.open('rb') as stream:
            client.put_object(Bucket=bucket, Key=input_key, Body=stream)
        from infra.compute.serverless.s3_artifacts import CONFIG, profile_auth
        data = {'storage': dict(CONFIG), 'storage_auth': profile_auth(),
                'input_key': input_key, 'output_key': output_key, 'output_prefix': prefix,
                'bundle_sha256': sha, 'bootstrap': args.bootstrap, 'recipe': recipe}
        record = {'endpoint': endpoint, 'bucket': bucket, 'input_key': input_key,
                  'output_key': output_key, 'output_prefix': prefix, 'bundle_sha256': sha,
                  'submitted_at_epoch': time.time(),
                  'submitted_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'submission_state': 'request_pending', 'recipe': recipe}
        write(receipt, record)
        from runpod_flash.runtime.serialization import serialize_kwargs
        payload = {'function_name': 'run_fit', 'function_code': (source / 'worker_fit.py').read_text(),
                   'kwargs': serialize_kwargs({'data': data}), 'serialization_format': 'cloudpickle',
                   'dependencies': ['boto3==1.43.98'], 'system_dependencies': [],
                   'accelerate_downloads': False}
        response = call(endpoint, 'run', {'input': payload,
                        'policy': {'executionTimeout': 3600000, 'ttl': 4200000}})
        record.update(job_id=response['id'], submission_state='accepted')
        write(receipt, record)
        print(json.dumps(response))
        return
    record = json.loads(receipt.read_text())
    status_path = args.run_dir / 'status.json'
    previous_status = json.loads(status_path.read_text()) if status_path.exists() else {}
    terminal = previous_status.get('status') in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMED_OUT')
    if args.action == 'status':
        if terminal:
            raise SystemExit('Job is already terminal; use saved status and fetch evidence.')
        inspect_allowed(receipt, record)
        result = call(record['endpoint'], 'status/' + record['job_id'])
        write(status_path, result)
        print(json.dumps({k: v for k, v in result.items() if k != 'output'}))
        return
    # Progress is worker-pushed to storage. This explicit read is throttled too.
    if (args.run_dir / 'transport-cleanup.json').exists():
        raise SystemExit('Closed run: disposable transport containers were removed after verified regeneration. '
                         'Use retained native evidence and transport-cleanup.json restore references.')
    if not terminal:
        inspect_allowed(receipt, record)
    client, _, _ = s3()
    response = client.get_object(Bucket=record['bucket'], Key=record['output_prefix'] + '/progress.json')
    try:
        progress = json.loads(response['Body'].read(2_000_000))
    finally:
        response['Body'].close()
    write(args.run_dir / 'progress.json', progress)
    restored = [restore_pack(client, record['bucket'], item, args.run_dir) for item in progress.get('packs', [])]
    if terminal:
        inner = previous_status.get('output') or {}
        info = inner.get('json_result')
        if info is None and inner.get('result'):
            import base64
            import cloudpickle
            info = cloudpickle.loads(base64.b64decode(inner['result']))
        if info and info.get('output_sha256'):
            restored.append(restore_pack(client, record['bucket'], {
                'key': record['output_key'], 'sha256': info['output_sha256'],
                'bytes': info['output_bytes'], 'stage': 'final-receipt'}, args.run_dir))
        write(args.run_dir / 'terminal-info.json', info)
    write(args.run_dir / 'restored.json', {'packs': restored, 'terminal': terminal})
    print(json.dumps({'worker_status': progress.get('status'), 'packs_restored': len(restored),
                      'stages': [item['stage'] for item in restored], 'terminal': terminal}))


if __name__ == '__main__':
    try:
        main()
    except urllib.error.HTTPError as exc:
        raise SystemExit(f'HTTP {exc.code}; stop and inspect. No automatic retry.')
