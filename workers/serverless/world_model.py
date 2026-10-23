"""Bounded, frozen pretrained world-model exploration using native Cosmos."""
from runpod_flash import Endpoint, GpuGroup, PodTemplate

@Endpoint(name='robot-lab-world-model', gpu=GpuGroup.AMPERE_80, workers=(0, 1), gpu_count=1,
          idle_timeout=5, execution_timeout_ms=1_800_000, template=PodTemplate(containerDiskInGb=100),
          dependencies=['boto3==1.43.98'], system_dependencies=['git', 'git-lfs', 'ffmpeg', 'libx11-dev'],
          env={'RUNPOD_API_KEY': ''})
async def world_model(payload: dict, storage_auth: dict, model_auth: str) -> dict:
    import hashlib
    import json
    import os
    from pathlib import Path
    import signal
    import subprocess
    import sys
    import tempfile
    import time
    from s3_artifacts import Artifacts, CONFIG, client_from_auth, redact

    root = Path(tempfile.mkdtemp(prefix='world-model-'))
    output = root/'evidence';output.mkdir()
    storage = Artifacts(client_from_auth(CONFIG, storage_auth), CONFIG)
    started = time.monotonic()
    report = {'status': 'partial', 'stages': [], 'training_steps': 0}
    env = dict(os.environ, HF_TOKEN=model_auth, GIT_LFS_SKIP_SMUDGE='1',
               HF_HOME='/tmp/robot-lab-cosmos-cache', UV_HTTP_TIMEOUT='30', UV_HTTP_RETRIES='1')

    def run(stage, command, cwd, seconds):
        budget = min(seconds, 1500-(time.monotonic()-started))
        if budget <= 0: raise TimeoutError('Job deadline reached')
        print('world-model stage: '+stage, flush=True)
        with (output/(stage+'.log')).open('x') as log:
            proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try: code = proc.wait(timeout=budget)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL);proc.wait();raise
        report['stages'].append({'stage': stage, 'exit_code': code})
        print('world-model stage finished: '+stage+' exit='+str(code), flush=True)
        if code: raise RuntimeError(stage+' failed')

    try:
        storage.materialize(payload['files'], root/'inputs')
        config = json.loads((root/'inputs/config.json').read_text())
        request = json.loads((root/'inputs/request.json').read_text())
        if config['source_commit'] != 'a2c298b0a3df3778b973fe65e9e58877b292d8a7': raise ValueError('Unrecognized source')
        if request['mode'] not in ['hold_arm','reverse_translation']: raise ValueError('Invalid action mode')
        report['source_commit'] = config['source_commit']
        run('hardware', ['nvidia-smi'], root, 20)
        run('install-uv', [sys.executable, '-m', 'pip', 'install', 'uv==0.12.17'], root, 90)
        checkout = root/'cosmos'
        run('clone', ['git','clone','--filter=blob:none','--no-checkout',config['source_repo'],str(checkout)], root, 180)
        run('checkout', ['git','checkout',config['source_commit']], checkout, 90)
        report['upstream_lock_sha256'] = hashlib.sha256((checkout/'uv.lock').read_bytes()).hexdigest()
        # Upstream flash-attn wheel is cp310; the generic worker currently uses 3.13.
        run('install', [sys.executable,'-m','uv','sync','--python','3.10','--frozen','--extra','cu128'], checkout, 600)
        python = str(checkout/'.venv/bin/python')
        run('versions', [sys.executable,'-m','uv','pip','freeze','--python',python], checkout, 20)
        env['PYTHONPATH'] = str(root/'inputs')
        # Record whole-worker allocation, including model subprocesses, while inference runs.
        with (output/'gpu-memory.csv').open('x') as memory_log:
            monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,memory.used,memory.total,utilization.gpu',
                '--format=csv,nounits','--loop=2'],stdout=memory_log,stderr=subprocess.STDOUT)
            try:
                run('predict', [python,str(root/'inputs/probe.py'),'--inputs',str(root/'inputs'),
                    '--output',str(output/'prediction'),'--mode',request['mode']], checkout, 1000)
            finally:
                monitor.terminate();monitor.wait(timeout=10)
        report['status'] = 'needs_image_review'
    except Exception as error:
        report['status'] = 'fail';report['error_type'] = type(error).__name__
    finally:
        report['elapsed_s'] = time.monotonic()-started
        # Neither HF nor S3 secrets enter retained artifacts, including native failure logs.
        secrets = dict(storage_auth, hf_token=model_auth)
        for path in output.rglob('*'):
            if path.is_file() and path.suffix in ['.json','.log','.txt']:
                path.write_text(redact(path.read_text(errors='replace'), secrets))
        report['files'] = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file()}
        (output/'report.json').write_text(json.dumps(report,indent=2))
    reference = storage.retain_tree(output, root.name)
    return {'status':report['status'],'report':report,'s3_evidence':reference}
