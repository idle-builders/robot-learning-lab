"""Local access checks and one bounded Flash request; credentials never enter manifests."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT/'artifacts/world-model-inputs-v1'
MODES = {'hold_arm': 'Hold the arm still', 'reverse_translation': 'Reverse the arm translation'}
SOURCES = {'bridge': INPUTS, 'pip': ROOT/'artifacts/world-model-pip-inputs-v1'}


def access_status():
    from huggingface_hub import get_token, get_hf_file_metadata, hf_hub_url
    config = json.loads((ROOT/'workers/world_model/config.json').read_text())
    token = get_token()
    if not token:
        return {'status':'blocked', 'reason':'Hugging Face login and model access are required. No GPU job was started.'}
    checks = []
    revision = None
    for repo, filename in [
        (config['model_repo'], config['model_path']),
        ('nvidia/Cosmos-Guardrail1', 'video_content_safety_filter/models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3/config.json'),
    ]:
        try:
            meta = get_hf_file_metadata(hf_hub_url(repo, filename), token=token, timeout=15)
            checks.append({'repo': repo, 'status': 'ready'})
            if repo == config['model_repo']: revision = meta.commit_hash
        except Exception as exc:
            # Never serialize credential-bearing request objects or exception text.
            checks.append({'repo': repo, 'status': 'blocked', 'error_type': type(exc).__name__})
    failed = [c for c in checks if c['status'] != 'ready']
    if failed:
        return {'status': 'blocked', 'checks': checks, 'reason': 'Local login detected. Access failed for '+', '.join(c['repo']+' ('+c['error_type']+')' for c in failed)+'. Accept the listed model agreements and enable gated-repository access on the token. No GPU job started.'}
    return {'status': 'ready', 'checks': checks, 'model_revision': revision, 'reason': 'Checkpoint and guardrail access verified.'}


def public_inputs(mode, revision, source='bridge'):
    if mode not in MODES: raise ValueError('Unsupported intervention')
    if not isinstance(revision,str) or len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('A pinned model revision is required')
    if source not in SOURCES: raise ValueError('Choose a known source clip')
    inputs = SOURCES[source]
    manifest = json.loads((inputs/'manifest.json').read_text())
    files = {}
    for item in manifest['files']:
        data = (inputs/item['file']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=item['sha256']: raise ValueError('Example input integrity failure')
        files[item['file']] = data
    for name in ['probe.py','action_loader.py','config.json','upstream-example.json']:
        files[name]=(ROOT/'workers/world_model'/name).read_bytes()
    files['request.json']=json.dumps({'mode':mode,'model_revision':revision,'training_steps':0,
        'source':source,'input_manifest':manifest}).encode()
    return files


def endpoint_ids():
    result=subprocess.run(['runpodctl','serverless','list'],capture_output=True,text=True,check=True,timeout=20)
    return {x['id'] for x in json.loads(result.stdout) if 'robot-lab-world-model' in x.get('name','')}


def run_prediction(mode, notify=lambda message: None, source='bridge'):
    from huggingface_hub import get_token
    from workers.s3_artifacts import Artifacts, CONFIG, profile_auth, client_from_auth, redact
    from lab.artifact_store import plan_migration, apply_migration, verify_manifest
    access=access_status()
    if access['status']!='ready': raise RuntimeError(access['reason'])
    files=public_inputs(mode,access['model_revision'],source)
    output=ROOT/'artifacts'/('world-model-probe-'+uuid.uuid4().hex[:12]);output.mkdir()
    source=output/'submitted-source';source.mkdir()
    for name,data in files.items():(source/name).write_bytes(data)
    for path in [Path(__file__),ROOT/'workers/serverless/world_model.py',ROOT/'workers/serverless/s3_artifacts.py']:
        (source/('worker-'+path.name)).write_bytes(path.read_bytes())
    token=get_token();auth=profile_auth('runpod');secrets=dict(auth,hf_token=token)
    storage=Artifacts(client_from_auth(CONFIG,auth),CONFIG)
    payload={'files':{name:storage.put_bytes(data) for name,data in files.items()},'storage':dict(CONFIG)}
    (output/'transfer.json').write_text(json.dumps(payload,indent=2))
    before=endpoint_ids();process=None
    notify('Inputs verified. Starting one bounded GPU experiment; cold setup may take several minutes.')
    try:
        import socket
        with socket.socket() as port_check:
            port_check.bind(('127.0.0.1',8902))
        with (output/'flash.log').open('x') as log:
            process=subprocess.Popen([str(ROOT/'tools/runpod-flash'),'dev','--port','8902','--no-reload'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            url='http://127.0.0.1:8902'
            for _ in range(120):
                if process.poll() is not None: raise RuntimeError('Flash exited before readiness')
                try:
                    with urllib.request.urlopen(url+'/openapi.json',timeout=1) as response: json.load(response)
                    break
                except Exception: time.sleep(.5)
            else: raise RuntimeError('Flash readiness deadline exceeded')
            # Never retain this envelope: it holds the separate private credentials.
            envelope={'input':{'payload':payload,'storage_auth':auth,'model_auth':token}}
            req=urllib.request.Request(url+'/world_model/runsync',data=json.dumps(envelope).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=2100) as response: result=redact(json.load(response),secrets)
            (output/'response.json').write_text(json.dumps(result,indent=2))
            worker=result.get('output',{})
            if 's3_evidence' in worker:
                verified=storage.download_run(worker['s3_evidence'],output/'evidence')
                (output/'download-verification.json').write_text(json.dumps(verified,indent=2))
            if worker.get('status')!='needs_image_review': raise RuntimeError('Native prediction failed; the original evidence has been retained.')
    except Exception as exc:
        (output/'client-error.json').write_text(json.dumps({'status':'fail','error_type':type(exc).__name__,'scope':'No prediction success claimed.'}))
        raise RuntimeError('Experiment did not complete. Retained run: '+output.name) from None
    finally:
        if process is not None:
            try: process.send_signal(signal.SIGINT);process.wait(timeout=25)
            except subprocess.TimeoutExpired: process.terminate();process.wait(timeout=10)
        cleanup=[]
        try:
            for endpoint in endpoint_ids()-before:
                r=subprocess.run(['runpodctl','serverless','delete',endpoint],capture_output=True,timeout=30)
                cleanup.append({'endpoint':endpoint,'delete_exit_code':r.returncode})
            remaining=sorted(endpoint_ids()-before)
            (output/'cleanup.json').write_text(json.dumps({'remaining_owned_endpoints':remaining,'deletions':cleanup}))
        except Exception as exc:(output/'cleanup.json').write_text(json.dumps({'status':'needs_check','error_type':type(exc).__name__}))
        logpath=output/'flash.log'
        if logpath.exists():logpath.write_text(redact(logpath.read_text(errors='replace'),secrets))
        manifest=plan_migration(ROOT,[output]);apply_migration(ROOT,manifest);verify_manifest(ROOT,manifest)
    return output
