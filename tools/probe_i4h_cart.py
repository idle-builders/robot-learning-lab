"""Run the cheap NVIDIA Vulkan gate on one disposable Serverless GPU worker."""
import json
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.probe_vla import finalize


def endpoint_ids():
    result = subprocess.run(['runpodctl','serverless','list'], capture_output=True,
                           text=True, check=True, timeout=20)
    return {x['id'] for x in json.loads(result.stdout)
            if 'robot-lab-i4h-cart' in x.get('name','')}


def main():
    from huggingface_hub import get_token
    from workers.s3_artifacts import Artifacts, CONFIG, profile_auth, client_from_auth, redact
    output = ROOT/'artifacts'/('i4h-cart-vulkan-'+uuid.uuid4().hex[:12])
    output.mkdir()
    print(output, flush=True)
    source = ROOT/'workers/serverless/i4h_cart.py'
    (output/'source').mkdir()
    run_id = output.name
    full_run = '--evaluate' in sys.argv
    worker_source = ROOT/('workers/serverless/i4h_cart_eval.py' if full_run else 'workers/serverless/i4h_cart.py')
    (output/'source'/worker_source.name).write_bytes(worker_source.read_bytes())
    for name in ['tools/probe_i4h_cart.py','tools/export_i4h_cart_recording.py','workers/serverless/s3_artifacts.py',
                 'workers/serverless/i4h_cart.py','workers/serverless/i4h_cart_eval.py']:
        target=output/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    payload = {'run_id':run_id}
    auth = profile_auth('runpod') if full_run else None
    token = get_token() if full_run else None
    if full_run and not token:
        raise RuntimeError('Hugging Face token is not available in the existing login')
    if full_run:
        payload['recording_exporter']=(ROOT/'tools/export_i4h_cart_recording.py').read_text()
    (output/'request.json').write_text(json.dumps({'purpose':'pinned i4h cart native evaluation' if full_run else 'NVIDIA Vulkan preflight only',
        'upstream':'isaac-for-healthcare/i4h-workflows v0.7.1',
        'policy_requested':full_run,
        'weights_expected_after_preflight':full_run},indent=2))
    before = endpoint_ids()
    process = None
    passed = False
    try:
        port = '8910' if full_run else '8908'
        with socket.socket() as check:
            check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            check.bind(('127.0.0.1',int(port)))
        with (output/'flash.log').open('x') as log:
            process = subprocess.Popen([str(ROOT/'tools/runpod-flash'),'dev','--port',port,'--no-reload'],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            base = None
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError('Flash exited before readiness')
                flash_text = (output/'flash.log').read_text(errors='replace')
                match = re.search(r'flash dev\s+localhost:(\d+)', flash_text)
                if match:
                    base = 'http://127.0.0.1:' + match.group(1)
                try:
                    if base is None:
                        time.sleep(.5)
                        continue
                    with urllib.request.urlopen(base+'/openapi.json',timeout=1) as response:
                        schema = json.load(response)
                    route = '/i4h_cart_eval/runsync' if full_run else '/i4h_cart/runsync'
                    if route not in schema['paths']:
                        raise RuntimeError('Missing i4h preflight route')
                    break
                except (OSError,TimeoutError):
                    time.sleep(.5)
            else:
                raise TimeoutError('Flash readiness deadline')
            request = urllib.request.Request(base+route,
                data=json.dumps({'input':{'payload':payload, **({'storage_auth':auth,'model_auth':token} if full_run else {})}}).encode(),
                headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=3660 if full_run else 240) as response:
                result = json.load(response)
            result = redact(result, dict(auth or {}, hf_token=token or ''))
            (output/'response.json').write_text(json.dumps(result,indent=2))
            worker = result.get('output',{})
            if full_run and 's3_evidence' in worker:
                storage = Artifacts(client_from_auth(CONFIG,auth),CONFIG)
                verification = storage.download_run(worker['s3_evidence'],output/'evidence')
                (output/'download-verification.json').write_text(json.dumps(verification,indent=2))
            passed = worker.get('status') == ('native_rollout_executed' if full_run else 'vulkan_preflight_passed')
            print('i4h cart:', worker.get('status','missing'), flush=True)
    except Exception as exc:
        (output/'client-error.json').write_text(json.dumps({'status':'fail',
            'error':type(exc).__name__+': '+str(exc)},indent=2))
        print('Preflight invocation failed:',type(exc).__name__,flush=True)
    finally:
        if process is not None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=25)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
        cleanup = []
        try:
            for endpoint in endpoint_ids()-before:
                result = subprocess.run(['runpodctl','serverless','delete',endpoint],
                                        capture_output=True,text=True,timeout=30)
                cleanup.append({'endpoint':endpoint,'delete_exit_code':result.returncode})
            remaining = sorted(endpoint_ids()-before)
            (output/'cleanup.json').write_text(json.dumps({'deletions':cleanup,
                'remaining_owned_endpoints':remaining},indent=2))
        except Exception as exc:
            (output/'cleanup.json').write_text(json.dumps({'status':'needs_check',
                'error_type':type(exc).__name__},indent=2))
        log = output/'flash.log'
        if log.exists():
            log.write_text(log.read_text(errors='replace'))
        finalize(output)
    if not passed:
        raise SystemExit(('i4h cart evaluation failed' if full_run else 'Vulkan gate failed')+'; retained evidence: '+str(output))


if __name__ == '__main__':
    main()
