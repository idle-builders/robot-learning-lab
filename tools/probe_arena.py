"""Execute the pinned upstream G1 tutorial; retain evidence and remove our endpoint."""
import json
from pathlib import Path
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
    result = subprocess.run(['runpodctl','serverless','list'],capture_output=True,text=True,check=True,timeout=20)
    return {x['id'] for x in json.loads(result.stdout) if 'robot-lab-arena' in x.get('name','')}


def main():
    from huggingface_hub import get_token, get_hf_file_metadata, hf_hub_url
    from workers.s3_artifacts import Artifacts, CONFIG, profile_auth, client_from_auth, redact
    from lab.artifact_store import snapshot_file
    pins = json.loads((ROOT/'workers/arena/pins.json').read_text())
    token = get_token()
    for name in ['model-00001-of-00002.safetensors','model-00002-of-00002.safetensors']:
        get_hf_file_metadata(hf_hub_url(pins['checkpoint_repo'],name,revision=pins['checkpoint_revision']),token=token,timeout=15)
    auth = profile_auth('runpod')
    secrets = dict(auth,hf_token=token)
    storage = Artifacts(client_from_auth(CONFIG,auth),CONFIG)
    output = ROOT/'artifacts'/('arena-reference-'+uuid.uuid4().hex[:12])
    output.mkdir()
    print(output,flush=True)
    for name in ['tools/probe_arena.py','tools/probe_vla.py','workers/serverless/arena.py','workers/serverless/s3_artifacts.py','workers/arena/pins.json']:
        snapshot_file(ROOT/name,output/'source'/name)
    payload = {'pins':pins,'run_id':output.name}
    (output/'request.json').write_text(json.dumps(payload,indent=2))
    before = endpoint_ids()
    process = None
    completed = False
    try:
        with socket.socket() as check:
            check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            check.bind(('127.0.0.1',8906))
        with (output/'flash.log').open('x') as log:
            process = subprocess.Popen([str(ROOT/'tools/runpod-flash'),'dev','--port','8906','--no-reload'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            base = 'http://127.0.0.1:8906'
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError('Flash exited before readiness')
                try:
                    with urllib.request.urlopen(base+'/openapi.json',timeout=1) as response:
                        schema = json.load(response)
                    if '/arena/runsync' not in schema['paths']:
                        raise RuntimeError('Missing Arena route')
                    break
                except (OSError,TimeoutError):
                    time.sleep(.5)
            else:
                raise TimeoutError('Flash readiness deadline')
            envelope = {'input':{'payload':payload,'storage_auth':auth,'model_auth':token}}
            request = urllib.request.Request(base+'/arena/runsync',data=json.dumps(envelope).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=1980) as response:
                result = redact(json.load(response),secrets)
            (output/'response.json').write_text(json.dumps(result,indent=2))
            worker = result.get('output',{})
            if 's3_evidence' in worker:
                verification = storage.download_run(worker['s3_evidence'],output/'evidence')
                (output/'download-verification.json').write_text(json.dumps(verification,indent=2))
            print('Worker status:',worker.get('status','missing'),flush=True)
            completed = worker.get('status') == 'reference_evaluation_executed'
    except Exception as exc:
        (output/'client-error.json').write_text(json.dumps({'status':'fail','error':redact(str(exc),secrets)}))
        print('Arena attempt failed:',type(exc).__name__,flush=True)
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
                result = subprocess.run(['runpodctl','serverless','delete',endpoint],capture_output=True,timeout=30)
                cleanup.append({'endpoint':endpoint,'delete_exit_code':result.returncode})
            (output/'cleanup.json').write_text(json.dumps({'deletions':cleanup,'remaining_owned_endpoints':sorted(endpoint_ids()-before)},indent=2))
        except Exception as exc:
            (output/'cleanup.json').write_text(json.dumps({'status':'needs_check','error_type':type(exc).__name__}))
        log = output/'flash.log'
        if log.exists():
            log.write_text(redact(log.read_text(errors='replace'),secrets))
        finalize(output)
    if not completed:
        raise SystemExit('Failed; retained evidence: '+str(output))


if __name__ == '__main__':
    main()
