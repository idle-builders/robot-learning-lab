"""Run the unchanged upstream Arena G1/GR00T locomanipulation tutorial."""
from runpod_flash import Endpoint, GpuGroup, PodTemplate


@Endpoint(name='robot-lab-arena', gpu=GpuGroup.ADA_24,
          workers=(0, 1), gpu_count=1, idle_timeout=5, min_cuda_version='13.0',
          execution_timeout_ms=1_800_000, template=PodTemplate(containerDiskInGb=150),
          dependencies=[],
          system_dependencies=['git', 'ffmpeg', 'libegl1', 'libgl1', 'libopengl0', 'libvulkan1', 'build-essential', 'ninja-build', 'libxt6', 'vulkan-tools'],
          env={'RUNPOD_API_KEY': '', 'NVIDIA_DRIVER_CAPABILITIES': 'all',
               'OMNI_KIT_ACCEPT_EULA': 'YES', 'ACCEPT_EULA': 'Y', 'PRIVACY_CONSENT': 'N'})
async def arena(payload: dict, storage_auth: dict, model_auth: str) -> dict:
    import hashlib
    import json
    import os
    from pathlib import Path
    import shutil
    import signal
    import subprocess
    import sys
    import tempfile
    import time
    import traceback
    from s3_artifacts import Artifacts, CONFIG, client_from_auth, redact

    root = Path(tempfile.mkdtemp(prefix='arena-'))
    output = root / 'evidence'
    output.mkdir()
    started = time.monotonic()
    storage = Artifacts(client_from_auth(CONFIG, storage_auth), CONFIG)
    secrets = dict(storage_auth, hf_token=model_auth)
    env = dict(os.environ, HF_TOKEN=model_auth, HF_HOME=str(root/'hf-cache'),
               UV_HTTP_TIMEOUT='60', UV_HTTP_RETRIES='1', UV_LINK_MODE='copy',
               HF_HUB_ENABLE_HF_TRANSFER='0',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false', MAX_JOBS='4', PYTHONUNBUFFERED='1')
    report = {'status': 'running', 'stages': [], 'task_success': 'not_run',
              'training': 'not_run; released checkpoint evaluation', 'pins': payload['pins']}
    server = None
    server_log = None

    def run(stage, command, cwd, seconds):
        budget = min(seconds, 1640 - (time.monotonic()-started))
        if budget <= 0:
            raise TimeoutError('Arena worker deadline')
        os.write(1, ('Arena stage: '+stage+'\n').encode())
        item = {'stage': stage, 'command': command, 'status': 'running'}
        report['stages'].append(item)
        begin = time.monotonic()
        log_path = output/(stage+'.log')
        log_path.touch(exist_ok=False)
        # Reuse the existing AGILE worker's native tee pattern. argv stays separate
        # from shell source; no credential values are included in command arguments.
        process = subprocess.Popen(['bash','-o','pipefail','-c',
            '"$@" 2>&1 | tee -a "$ROBOTLAB_STAGE_LOG"','arena-stage',*command],
            cwd=cwd, env={**env,'ROBOTLAB_STAGE_LOG':str(log_path)},
            stdout=1, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=budget)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            item.update(status='timeout', elapsed_s=time.monotonic()-begin)
            raise
        item.update(status='pass' if code == 0 else 'fail', exit_code=code, elapsed_s=time.monotonic()-begin)
        # Stream a bounded tail; complete native logs are retained below.
        os.write(1, (redact((output/(stage+'.log')).read_text(errors='replace')[-5000:], secrets)+'\n').encode())
        if code:
            raise RuntimeError(stage+' failed')

    try:
        pins = payload['pins']
        run('hardware', ['nvidia-smi'], root, 20)
        run('cuda', [sys.executable, '-c', "import torch; print(torch.cuda.get_device_name()); assert torch.cuda.device_count()==1; assert 'MIG' not in torch.cuda.get_device_name(); print(torch.ones(1,device='cuda').cpu())"], root, 30)
        # Fail cheaply before installing models if the worker cannot render.
        run('graphics-libraries', ['ldconfig','-p'], root, 15)
        try:
            run('vulkan-default', ['vulkaninfo','--summary'], root, 30)
        except RuntimeError:
            # NVIDIA documents libEGL_nvidia as the headless Vulkan ICD.
            # Some compute images mount the driver but omit its loader manifest.
            import ctypes
            ctypes.CDLL('libEGL_nvidia.so.0')
            manifest = root/'nvidia-headless-icd.json'
            manifest.write_text(json.dumps({'file_format_version':'1.0.0',
                'ICD':{'library_path':'libEGL_nvidia.so.0','api_version':'1.3.0'}}))
            shutil.copyfile(manifest,output/manifest.name)
            env['VK_DRIVER_FILES'] = str(manifest)
            env['VK_ICD_FILENAMES'] = str(manifest)
            run('vulkan-headless', ['vulkaninfo','--summary'], root, 30)
        vk_log = output/('vulkan-headless.log' if (output/'vulkan-headless.log').exists() else 'vulkan-default.log')
        if 'NVIDIA' not in vk_log.read_text():
            raise RuntimeError('Vulkan did not enumerate NVIDIA graphics; refusing software-rendered evaluation')
        run('install-uv', [sys.executable,'-m','pip','install','--disable-pip-version-check',
            '--retries','1','--timeout','30','uv==0.12.17'], root, 90)
        run('bootstrap-versions', [sys.executable,'-c',
            "from importlib.metadata import version; print({n:version(n) for n in ['boto3','runpod','uv']})"], root, 15)
        checkout = root/'IsaacLab-Arena'
        run('clone', ['git','clone','--filter=blob:none','https://github.com/isaac-sim/IsaacLab-Arena.git',str(checkout)], root, 90)
        run('checkout', ['git','checkout',pins['arena']], checkout, 20)
        run('submodules', ['git','-c','url.https://github.com/.insteadOf=git@github.com:', 'submodule','update','--init','--depth','1'], checkout, 180)
        for name, revision in [('IsaacLab',pins['isaaclab']),('Isaac-GR00T',pins['gr00t'])]:
            actual = subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout/'submodules'/name,text=True).strip()
            if actual != revision:
                raise ValueError(name+' revision mismatch')
        if hashlib.sha256((checkout/'uv.lock').read_bytes()).hexdigest() != pins['arena_lock_sha256']:
            raise ValueError('Arena lock mismatch')
        uv = [sys.executable,'-m','uv']
        run('arena-install', [*uv,'sync','--frozen','--group','gr00t-client'], checkout, 600)
        sim_python = str(checkout/'.venv/bin/python')
        run('arena-versions', [*uv,'pip','freeze','--python',sim_python], checkout, 30)
        groot = checkout/'submodules/Isaac-GR00T'
        run('groot-install', [*uv,'sync','--python','3.10'], groot, 600)
        run('groot-editable', [*uv,'pip','install','--no-deps','-e','.'], groot, 30)
        policy_python = str(groot/'.venv/bin/python')
        run('groot-versions', [*uv,'pip','freeze','--python',policy_python], groot, 30)
        for src, dst in [(checkout/'uv.lock','arena-uv.lock'),(groot/'uv.lock','groot-uv.lock'),(groot/'pyproject.toml','groot-pyproject.toml')]:
            shutil.copyfile(src,output/dst)
        model = root/'checkpoint'
        download = ('from huggingface_hub import snapshot_download; '
                    'snapshot_download(repo_id='+repr(pins['checkpoint_repo'])+',revision='+repr(pins['checkpoint_revision'])+',local_dir='+repr(str(model))+",allow_patterns=['*.json','*.safetensors','README.md'])")
        run('checkpoint-download', [policy_python,'-c',download], groot, 240)
        # LFS metadata ties each weight to the immutable Hub commit; hash locally for verification.
        manifest = {}
        for path in model.rglob('*'):
            if path.is_file() and '.cache' not in path.parts:
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda:stream.read(8*1024*1024),b''):
                        digest.update(block)
                manifest[str(path.relative_to(model))] = {'sha256':digest.hexdigest(),'bytes':path.stat().st_size}
        (output/'checkpoint-manifest.json').write_text(json.dumps(manifest,indent=2))
        server_command = [policy_python,'gr00t/eval/run_gr00t_server.py',
            '--modality-config-path','../../isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_config.py',
            '--model-path',str(model),'--embodiment-tag','NEW_EMBODIMENT','--device','cuda','--host','127.0.0.1','--port','5555']
        report['server_command'] = server_command
        server_log = (output/'groot-server.log').open('x')
        server = subprocess.Popen(server_command,cwd=groot,env=env,stdout=server_log,stderr=subprocess.STDOUT,start_new_session=True)
        readiness = "from gr00t.policy.server_client import PolicyClient; c=PolicyClient(host='127.0.0.1',port=5555,timeout_ms=150000); print(c.get_modality_config())"
        run('server-ready', [policy_python,'-c',readiness], groot, 180)
        command = [sim_python,'isaaclab_arena/evaluation/policy_runner.py',
            '--policy_type','isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy',
            '--policy_config_yaml_path','isaaclab_arena_gr00t/policy/config/g1_locomanip_gr00t_closedloop_config.yaml',
            '--remote_host','127.0.0.1','--remote_port','5555','--num_steps','1500',
            '--device','cpu','--num_envs','1','--seed','42','--enable_cameras',
            '--record_camera_video','--output_base_dir',str(output/'evaluation'),
            'galileo_g1_locomanip_pick_and_place','--object','brown_box','--embodiment','g1_wbc_joint']
        run('native-evaluation',command,checkout,900)
        report.update(status='reference_evaluation_executed',task_success='inspect native metrics; exit code alone is not success')
    except Exception:
        report.update(status='fail',error=traceback.format_exc())
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid,signal.SIGKILL)
                server.wait()
        if server_log is not None:
            server_log.close()
        report['elapsed_s'] = time.monotonic()-started
        for path in output.rglob('*'):
            if path.is_file() and path.suffix in {'.log','.json','.txt','.html','.yaml'}:
                path.write_text(redact(path.read_text(errors='replace'),secrets))
        report = redact(report,secrets)
        (output/'report.json').write_text(json.dumps(report,indent=2))
    ref = storage.retain_tree(output,payload['run_id'])
    return {'status':report['status'],'report':report,'s3_evidence':ref}
