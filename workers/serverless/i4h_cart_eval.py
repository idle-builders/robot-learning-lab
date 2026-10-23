"""Run the pinned upstream i4h G1 cart-push policy and Arena evaluation."""
from runpod_flash import Endpoint, GpuGroup, PodTemplate


@Endpoint(name='robot-lab-i4h-cart', gpu=GpuGroup.ADA_32_PRO,
          workers=(0, 1), gpu_count=1, idle_timeout=5, min_cuda_version='13.0',
          execution_timeout_ms=3_600_000, template=PodTemplate(containerDiskInGb=150),
          dependencies=[],
          system_dependencies=['git', 'ffmpeg', 'libegl1', 'libgl1', 'libopengl0',
              'libglu1-mesa', 'libvulkan1', 'build-essential', 'ninja-build', 'libxt6', 'vulkan-tools'],
          env={'NVIDIA_DRIVER_CAPABILITIES': 'all', 'OMNI_KIT_ACCEPT_EULA': 'YES',
               'ACCEPT_EULA': 'Y', 'PRIVACY_CONSENT': 'N'})
async def evaluate(payload: dict, storage_auth: dict, model_auth: str) -> dict:
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

    started = time.monotonic()
    root = Path(tempfile.mkdtemp(prefix='i4h-cart-'))
    output = root/'evidence'
    output.mkdir()
    secrets = dict(storage_auth, hf_token=model_auth)
    storage = Artifacts(client_from_auth(CONFIG, storage_auth), CONFIG)
    env = dict(os.environ, HF_TOKEN=model_auth, HF_HOME=str(root/'hf-cache'),
        UV_HTTP_TIMEOUT='60', UV_HTTP_RETRIES='1', UV_LINK_MODE='copy',
        HF_HUB_ENABLE_HF_TRANSFER='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', MAX_JOBS='4',
        PYTHONUNBUFFERED='1', NVIDIA_DRIVER_CAPABILITIES='all',
        OMNI_KIT_ACCEPT_EULA='YES', ACCEPT_EULA='Y', PRIVACY_CONSENT='N')
    report = {'status':'running', 'native_termination_success':'not_run',
              'cart_fallback_success':'not_run', 'runner_episode_completion':'not_run',
              'policy_checkpoint_loaded':False,
              'repo':'https://github.com/isaac-for-healthcare/i4h-workflows.git',
              'tag':'v0.7.1', 'commit':'f81569477a514ff1be2380d2071e04d25f306d38',
              'environment':'locomanip_push_cart',
              'policy_model_repo':'nvidia/GR00T-N1.6-Rheo-Sim-PushCart',
              'policy_model_revision':'0e24b9b225da4a2cff53049eb8b6e57f435814ec',
              'episodes':1, 'stages':[]}
    policy = None
    policy_log = None
    recording = None

    def run(name, argv, cwd, seconds, *, stage_env=None):
        left = 3_500 - (time.monotonic()-started)
        budget = min(seconds, left)
        if budget <= 0:
            raise TimeoutError('i4h worker deadline')
        item = {'name':name, 'command':argv, 'status':'running'}
        report['stages'].append(item)
        path = output/(name+'.log')
        begin = time.monotonic()
        with path.open('x') as log:
            process = subprocess.Popen(argv, cwd=cwd, env=stage_env or env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=budget)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                item.update(status='timeout', elapsed_s=time.monotonic()-begin)
                raise
        item.update(status='pass' if code == 0 else 'fail', exit_code=code,
                    elapsed_s=time.monotonic()-begin)
        tail = path.read_text(errors='replace')[-6000:]
        os.write(1, (redact(tail,secrets)+'\n').encode())
        if code:
            raise RuntimeError(name+' failed')
        return tail

    def check_vulkan():
        import ctypes
        run('hardware',['nvidia-smi'],root,20)
        run('cuda',[sys.executable,'-c',"import torch; assert torch.cuda.device_count()==1; "
            "print(torch.cuda.get_device_name()); print(torch.ones(1,device='cuda').cpu())"],root,30)
        try:
            text = run('vulkan-default',['vulkaninfo','--summary'],root,30)
        except RuntimeError:
            ctypes.CDLL('libEGL_nvidia.so.0')
            manifest=root/'nvidia-headless-icd.json'
            manifest.write_text(json.dumps({'file_format_version':'1.0.0',
                'ICD':{'library_path':'libEGL_nvidia.so.0','api_version':'1.3.0'}}))
            icd_env=dict(env,VK_DRIVER_FILES=str(manifest),VK_ICD_FILENAMES=str(manifest))
            text=run('vulkan-headless',['vulkaninfo','--summary'],root,30,stage_env=icd_env)
        if 'NVIDIA' not in text:
            raise RuntimeError('Vulkan did not enumerate NVIDIA graphics')
        report['graphics_preflight']='pass'

    try:
        # The exact worker that installs and loads the policy must pass graphics first.
        check_vulkan()
        run('install-uv',[sys.executable,'-m','pip','install','--disable-pip-version-check',
            '--retries','1','--timeout','30','uv==0.12.17'],root,90)
        checkout=root/'i4h-workflows'
        run('clone',['git','clone','--branch','v0.7.1','--depth','1',
            'https://github.com/isaac-for-healthcare/i4h-workflows.git',str(checkout)],root,120)
        actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()
        if actual != report['commit']:
            raise ValueError('i4h upstream commit mismatch: '+actual)
        (output/'upstream-head.txt').write_text(actual+'\n')
        # Install only the shared runtime, Arena, and GR00T N1.6 path required by
        # this reference. The upstream all-in-one setup also installs unrelated
        # Mimic, data, annotator, N1.5/N1.7, and PI0 environments.
        agentic=checkout/'workflows/agentic'
        run('common-setup',['env','-u','VIRTUAL_ENV','uv','sync'],agentic/'common',300)
        run('third-party-setup',['workflows/agentic/third_party/setup.sh'],checkout,900)
        setup_env=dict(env,AGENTIC_THIRD_PARTY_SETUP_DONE='1')
        run('arena-setup',['workflows/agentic/arena/setup.sh'],checkout,900,stage_env=setup_env)
        run('policy-gr00t-n16-setup',['workflows/agentic/policy/setup.sh','gr00t_n16'],checkout,1500,
            stage_env=setup_env)
        root_agentic=checkout/'workflows/agentic'
        run('policy-envs',['workflows/agentic/policy/run.sh','--list-envs'],checkout,60)
        run('arena-envs',['workflows/agentic/arena/run.sh','--list-envs'],checkout,60)
        run('policy-dry-run',['workflows/agentic/policy/run.sh','--env','locomanip_push_cart','--dry-run'],checkout,180)
        run('arena-dry-run',['workflows/agentic/arena/run.sh','--env','locomanip_push_cart','--dry-run'],checkout,180)
        run('zero-action-scene',['workflows/agentic/arena/run.sh','--env','locomanip_push_cart',
            '--headless','--enable_cameras','--num-steps','40'],checkout,300)
        exporter=root/'export_i4h_cart_recording.py'
        exporter.write_text(payload['recording_exporter'])
        run('recording-exporter-deps',['uv','run','--project',str(agentic/'arena'),'python','-c',
            'import h5py, numpy, PIL; print("recording exporter dependencies ready")'],checkout,90)
        run_dir=checkout/'workflows/agentic/runs'/('eval_locomanip_push_cart_'+str(int(time.time())))
        (run_dir/'data').mkdir(parents=True)
        (run_dir/'logs').mkdir()
        policy_log=(run_dir/'logs/policy.log').open('x')
        policy_cmd=['workflows/agentic/policy/run.sh','--env','locomanip_push_cart']
        report['policy_command']=policy_cmd
        policy=subprocess.Popen(policy_cmd,cwd=checkout,env=env,stdout=policy_log,
            stderr=subprocess.STDOUT,start_new_session=True)
        ready=False
        deadline=time.monotonic()+900
        while time.monotonic()<deadline:
            if policy.poll() is not None:
                raise RuntimeError('policy daemon exited before readiness')
            log_text=(run_dir/'logs/policy.log').read_text(errors='replace')
            if 'policy ready' in log_text.lower():
                ready=True
                break
            if any(word in log_text.lower() for word in ['traceback','error','failed']):
                raise RuntimeError('policy daemon reported an error before readiness')
            time.sleep(2)
        if not ready:
            raise TimeoutError('policy daemon readiness deadline')
        report['policy_checkpoint_loaded']=True
        (output/'policy-ready.txt').write_text('policy ready\n')
        arena_cmd=['workflows/agentic/arena/run.sh','--env','locomanip_push_cart',
            '--headless','--enable_cameras','--episodes','1','--max-timesteps','1800',
            '--record-to',str(run_dir/'data/cart.hdf5')]
        report['arena_command']=arena_cmd
        native_log_tail=run('native-cart-rollout',arena_cmd,checkout,900)
        native_failure='native success did not fire' in native_log_tail.lower()
        fallback_pass='fallback_success state:' in native_log_tail and 'near_target=True' in native_log_tail
        runner_complete='1/1 episodes succeeded' in native_log_tail
        report.update(status='native_rollout_executed',
            native_termination_success='fail' if native_failure else 'not_confirmed_in_log',
            cart_fallback_success='pass' if fallback_pass else 'not_confirmed_in_log',
            runner_episode_completion='1/1 completed' if runner_complete else 'not_confirmed_in_log')
        # Keep the large upstream HDF5 on-worker; retain a compact visual/trace export instead.
        recording=run_dir/'data/cart.hdf5'
        if recording.exists():
            size=recording.stat().st_size
            report['recording_bytes']=size
            report['recording_retained']=False
            report['recording_retention']='full HDF5 omitted from S3 after prior 413; compact MP4, stills, HDF5 key index, timestamped actions/states, and cart events retained when export succeeds'
            export_dir=output/'compact-recording'
            run('compact-recording-export',['uv','run','--project',str(agentic/'arena'),'python',
                str(exporter),str(recording),str(export_dir),'--fps','30',
                '--video-stride','3',
                '--rollout-log',str(output/'native-cart-rollout.log')],checkout,900)
            report['compact_recording']={'mp4':'compact-recording/cart-camera.mp4',
                'stills':'compact-recording/stills/', 'trace':'compact-recording/state-action-trace.csv',
                'event_trace':'compact-recording/cart-events.json', 'dataset_index':'compact-recording/hdf5-index.json',
                'trace_fps':30,'video_fps':10,'video_stride':3}
        else:
            report['recording_retained']=False
            report['recording_retention']='upstream Arena produced no HDF5 recording'
        shutil.copyfile(run_dir/'logs/policy.log',output/'policy.log')
        for path in sorted(run_dir.rglob('*')):
            if path.is_file() and path.suffix.lower() in {'.json','.jsonl','.csv','.txt'} and path.stat().st_size < 5_000_000:
                target=output/('native-'+path.name)
                if not target.exists():
                    shutil.copyfile(path,target)
    except Exception:
        report.update(status='fail',error=traceback.format_exc())
        if recording is not None and recording.exists():
            report['recording_bytes']=recording.stat().st_size
            report['recording_retained']=False
            report['recording_retention']='full HDF5 intentionally omitted from S3; compact export did not finish'
            report['compact_export_status']='failed_or_timed_out; PNG stills are written before MP4 encoding; any cart-camera.partial.mp4 is explicitly unfinalized'
        if policy is not None:
            try:
                (output/'policy.log').write_text((run_dir/'logs/policy.log').read_text(errors='replace'))
            except Exception:
                pass
    finally:
        if policy is not None and policy.poll() is None:
            try:
                subprocess.run(['workflows/agentic/stop.sh','policy','--env','locomanip_push_cart'],
                    cwd=root/'i4h-workflows',env=env,stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,timeout=45,check=False)
            except Exception:
                pass
            try:
                os.killpg(policy.pid,signal.SIGTERM)
                policy.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(policy.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if policy_log is not None:
            policy_log.close()
        report['elapsed_s']=time.monotonic()-started
        for path in output.rglob('*'):
            if path.is_file() and path.suffix in {'.log','.json','.jsonl','.txt','.yaml','.csv'}:
                try:
                    path.write_text(redact(path.read_text(errors='replace'),secrets))
                except UnicodeDecodeError:
                    pass
        (output/'report.json').write_text(json.dumps(redact(report,secrets),indent=2))
    ref=storage.retain_tree(output,payload['run_id'])
    return {'status':report['status'],'report':redact(report,secrets),'s3_evidence':ref}
