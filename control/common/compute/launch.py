"""Submit once, inspect once, or fetch a bounded Terraform-owned walking trial."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(ROOT))


def s3():
    import boto3
    from botocore.config import Config
    cfg = json.loads(subprocess.check_output(['terraform', 'output', '-json'], cwd=ROOT / 'infra/compute/runpod'))
    volume = cfg['transport']['value']
    client = boto3.Session(profile_name='runpod-robot-lab-new').client(
        's3', endpoint_url=volume['s3_endpoint'], region_name=volume['data_center'],
        config=Config(signature_version='s3v4', s3={'addressing_style': 'path'},
                      connect_timeout=10, read_timeout=90, retries={'total_max_attempts': 1},
                      request_checksum_calculation='when_required', response_checksum_validation='when_required'))
    return client, volume['id'], cfg.get('walking_trial_id', {}).get('value')


def call(endpoint, path, payload=None):
    from runpod_flash.core.credentials import get_api_key
    if path.split('/')[0] in ('run', 'runsync'):  # every GPU job is matched to its written spend reflection
        from infra.storage import spend_guard
        timeout = ((payload or {}).get('policy') or {}).get('executionTimeout')
        if not timeout:  # the provider timeout is what stops a job billing after the work ends
            raise ValueError('a GPU job needs a provider execution timeout')
        spend_guard.consume(timeout)
    request = urllib.request.Request('https://api.runpod.ai/v2/' + endpoint + '/' + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Authorization': 'Bearer ' + get_api_key(), 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=40) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('submit', 'status', 'fetch'))
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--bootstrap')
    args = parser.parse_args()
    receipt = args.run_dir / 'submission.json'
    if args.action == 'submit':
        if not args.bundle or not args.bootstrap:
            parser.error('submit needs --bundle and --bootstrap')
        args.run_dir.mkdir(parents=True, exist_ok=False)
        client, bucket, endpoint = s3()
        if not endpoint:
            raise SystemExit('Enable Terraform walking_trial before submitting.')
        body = args.bundle.read_bytes()
        sha = hashlib.sha256(body).hexdigest()
        prefix = 'robot-learning-lab/v1/walking-trial/' + args.run_dir.name
        input_key, output_key = prefix + '/input.tar.gz', prefix + '/evidence.tar.gz'
        client.put_object(Bucket=bucket, Key=input_key, Body=body)
        from infra.compute.serverless.s3_artifacts import CONFIG, profile_auth
        # S3 keys travel only as transient function arguments to our Runpod worker.
        # Never persist payloads, credentials or authentication URLs in receipts.
        data = {'storage': dict(CONFIG), 'storage_auth': profile_auth(),
                'input_key': input_key, 'output_key': output_key,
                'bundle_sha256': sha, 'bootstrap': args.bootstrap}
        record = {'endpoint': endpoint, 'bucket': bucket, 'input_key': input_key,
                  'output_key': output_key, 'bundle_sha256': sha,
                  'submitted_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'submission_state': 'request_pending'}
        receipt.write_text(json.dumps(record, indent=2) + '\n')
        # The pinned Flash worker accepts its native FunctionRequest protocol.
        from runpod_flash.runtime.serialization import serialize_kwargs
        payload = {'function_name': 'run_trial',
                   'function_code': Path(__file__).with_name('worker.py').read_text(),
                   'kwargs': serialize_kwargs({'data': data}), 'serialization_format': 'cloudpickle',
                   'dependencies': ['boto3==1.43.98'], 'system_dependencies': [], 'accelerate_downloads': False}
        response = call(endpoint, 'run', {'input': payload,
                                         'policy': {'executionTimeout': 1800000, 'ttl': 2400000}})
        record.update(job_id=response['id'], submission_state='accepted')
        receipt.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(response))
        return
    record = json.loads(receipt.read_text())
    if args.action == 'status':
        if time.time() - record.get('last_status_at', 0) < 300:
            raise SystemExit('Wait five minutes between explicit status requests.')
        record['last_status_at'] = time.time()
        receipt.write_text(json.dumps(record, indent=2) + '\n')
        result = call(record['endpoint'], 'status/' + record['job_id'])
        (args.run_dir / 'status.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result))
        return
    result = json.loads((args.run_dir / 'status.json').read_text())
    inner = result.get('output') or {}
    info = inner.get('json_result')
    if info is None and inner.get('result'):
        import base64, cloudpickle
        info = cloudpickle.loads(base64.b64decode(inner['result']))
    if not info or not info.get('output_sha256'):
        raise SystemExit('No returned evidence checksum. Inspect the saved terminal response.')
    client, _, _ = s3()
    response = client.get_object(Bucket=record['bucket'], Key=record['output_key'])
    content = response['Body'].read(); response['Body'].close()
    if hashlib.sha256(content).hexdigest() != info['output_sha256']:
        raise SystemExit('Evidence archive hash mismatch')
    archive = args.run_dir / 'evidence.tar.gz'
    archive.write_bytes(content)
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise ValueError('Invalid evidence member')
            if not (args.run_dir / member.name).resolve().is_relative_to(args.run_dir.resolve()):
                raise ValueError('Unsafe evidence member')
        tf.extractall(args.run_dir, filter='data')
    (args.run_dir / 'restored.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info))


if __name__ == '__main__':
    try:
        main()
    except urllib.error.HTTPError as exc:
        raise SystemExit(f'HTTP {exc.code}; stop and inspect before any further request. No automatic retry.')
