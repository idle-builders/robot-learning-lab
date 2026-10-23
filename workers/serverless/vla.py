"""One bounded native pretrained-policy check on Runpod, with zero idle workers."""
from runpod_flash import Endpoint, GpuGroup, PodTemplate


@Endpoint(name='robot-lab-vla', gpu=GpuGroup.ADA_24, workers=(0, 1), gpu_count=1,
          idle_timeout=5, execution_timeout_ms=1_200_000,
          template=PodTemplate(containerDiskInGb=100), dependencies=['boto3==1.43.98'],
          system_dependencies=['ffmpeg', 'libegl1', 'libgl1', 'libopengl0'], env={'RUNPOD_API_KEY': '', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility,graphics', 'MUJOCO_GL': 'egl'})
async def vla(payload: dict, storage_auth: dict, model_auth: str) -> dict:
    import json
    import os
    from pathlib import Path
    import signal
    import subprocess
    import sys
    import tempfile
    import time
    import traceback
    from s3_artifacts import Artifacts, CONFIG, client_from_auth, redact

    root = Path(tempfile.mkdtemp(prefix='vla-'))
    output = root / 'evidence'
    output.mkdir()
    storage = Artifacts(client_from_auth(CONFIG, storage_auth), CONFIG)
    started = time.monotonic()
    report = {'status': 'running', 'training_steps': 0, 'task_success': 'not_run', 'stages': []}
    env = dict(os.environ, HF_TOKEN=model_auth, HF_HOME=str(root / 'hf-cache'),
               UV_HTTP_TIMEOUT='30', UV_HTTP_RETRIES='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')

    def run(stage, command, seconds):
        budget = min(seconds, 1050 - (time.monotonic() - started))
        if budget <= 0:
            raise TimeoutError('VLA job deadline')
        print('VLA stage: ' + stage, flush=True)
        with (output / (stage + '.log')).open('x') as log:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=budget)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
        report['stages'].append({'stage': stage, 'exit_code': code})
        if code:
            raise RuntimeError(stage + ' failed')

    try:
        storage.materialize(payload['files'], root / 'inputs')
        run('hardware', ['nvidia-smi'], 15)
        run('install-uv', [sys.executable, '-m', 'pip', 'install', 'uv==0.12.17'], 60)
        python = str(root / 'runtime/bin/python')
        adapt = payload.get('mode') in {'adapt', 'evaluate'}
        reference = payload.get('mode') in {'reference', 'reference-train'}
        reference_train = payload.get('mode') == 'reference-train'
        reference_rl = payload.get('mode') == 'reference-rl'
        if adapt:
            run('base-runtime', [sys.executable, '-c', "import torch,torchvision; print(torch.__version__,torchvision.__version__,torch.version.cuda); assert torch.__version__=='2.9.1+cu128' and torchvision.__version__=='0.24.1+cu128'"], 45)
            run('venv', [sys.executable, '-m', 'uv', 'venv', '--python', sys.executable, '--system-site-packages', str(root / 'runtime')], 60)
            packages = ['lerobot[smolvla,peft,training]==0.6.1', 'torch==2.9.1+cu128', 'torchvision==0.24.1+cu128', 'torchcodec==0.8.1']
        else:
            run('venv', [sys.executable, '-m', 'uv', 'venv', '--python', '3.12', str(root / 'runtime')], 90)
            packages = ['lerobot[groot]==0.6.1', 'torch==2.11.0', 'torchvision==0.26.0']
        if reference_rl:
            packages = ['lerobot[hilserl,training]==0.6.1', 'gym-hil==0.1.14', 'torch==2.11.0', 'torchvision==0.26.0']
        elif reference:
            packages += ['lerobot[libero,evaluation]==0.6.1']
            if reference_train:
                packages += ['lerobot[training,peft]==0.6.1']
        else:
            packages += ['mujoco==3.13.0', 'gymnasium==1.2.3', 'imageio[ffmpeg]==2.37.3']
        if 'motor/policy.zip' in payload['files']:
            packages += ['stable-baselines3==2.9.0']
        if adapt:
            # uv's resolver reinstalls inherited CUDA packages. pip recognizes
            # --system-site-packages while installing new dependencies privately.
            run('install', [sys.executable, '-m', 'pip', '--python', python, 'install',
                '--disable-pip-version-check', '--no-input', '--retries', '1', '--timeout', '30',
                '--report', str(output/'install-report.json'), *packages], 240)
            run('torch-reuse-check', [python, '-c',
                "import torch,sys; from pathlib import Path; print(torch.__version__,torch.__file__); assert Path(sys.prefix) not in Path(torch.__file__).parents"], 30)
        else:
            run('install', [sys.executable, '-m', 'uv', 'pip', 'install', '--python', python,
                '--torch-backend', 'cu128', *packages], 240)
        run('training-cuda-check', [python, '-c', "import torch; x=torch.ones(2,device='cuda'); torch.cuda.synchronize(); print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(),x.cpu().tolist())"], 30)
        run('versions', [python, '-m', 'pip', 'freeze'] if adapt else
            [sys.executable, '-m', 'uv', 'pip', 'freeze', '--python', python], 15)
        if adapt:
            run('training-import-check', [python, '-u', '-c', "import faulthandler; faulthandler.dump_traceback_later(30,repeat=True); print('trainer import'); from lerobot.scripts.lerobot_train import train; print('physics import'); import mujoco; print('adapter import'); import peft; print('imports passed'); faulthandler.cancel_dump_traceback_later()"], 150)
        probe = [python, str(root / ('inputs/native_adapt.py' if adapt else 'inputs/native_probe.py'))]
        common = ['--inputs', str(root / 'inputs'), '--output', str(output)]
        if adapt:
            run('plant-preflight', [*probe, 'preflight', *common], 90)
        if reference_rl:
            run('native-rl-tutorial', [python, str(root/'inputs/lerobot_hilserl.py'), str(output)], 540)
        elif reference:
            run('download', [python, str(root/'inputs/lerobot_reference.py'), str(output)], 360)
        else:
            run('download', [*probe, 'download', *common], 360)
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        if reference_rl:
            summary = json.loads((output/'rl-summary.json').read_text())
            report.update(status='reference_rl_executed', training_steps=summary['optimizer_steps'],
                task_success='not_evaluated', scope=summary['scope'])
        elif reference:
            models = json.loads((output/'model-provenance.json').read_text())
            checkpoint = models['nvidia/gr00t17-lerobot-libero_10-640']['path']
            base_model = models['nvidia/GR00T-N1.7-3B']['path']
            if reference_train:
                storage.materialize(payload['reference_dataset_files'], root/'reference-dataset', max_bytes=800_000_000)
                train_command = [python, '-m', 'lerobot.scripts.lerobot_train',
                    '--policy.path=nvidia/gr00t17-lerobot-libero_10-640',
                    '--policy.pretrained_revision=' + models['nvidia/gr00t17-lerobot-libero_10-640']['revision'],
                    '--policy.base_model_path=' + base_model, '--policy.device=cuda',
                    '--policy.push_to_hub=false', '--policy.optimizer_lr=0.00001',
                    '--policy.max_steps=20', '--dataset.repo_id=IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot',
                    '--dataset.root=' + str(root/'reference-dataset'), '--dataset.video_backend=pyav',
                    '--steps=20', '--batch_size=1', '--num_workers=0', '--seed=42',
                    '--save_freq=20', '--log_freq=1', '--env_eval_freq=0', '--eval_steps=0',
                    '--wandb.enable=false', '--peft.method_type=LORA', '--peft.r=8', '--peft.lora_alpha=16',
                    '--peft.target_modules=' + json.dumps(r'.*_groot_model\.action_head\.model\.transformer_blocks\.[0-9]+\.attn1\.(to_q|to_v)'),
                    '--output_dir=' + str(output/'training')]
                (output/'native-train-command.json').write_text(json.dumps(train_command,indent=2))
                run('native-training', train_command, 300)
                run('seal-reference-checkpoint', [python, str(root/'inputs/lerobot_reference.py'), str(output), 'seal'], 30)
                summary = json.loads((output/'training-summary.json').read_text())
                report['training_steps'] = summary['training_steps']
                checkpoint_dir = output/summary['checkpoint']
                checkpoint = str(checkpoint_dir/'pretrained_model')
                ref = storage.retain_tree(checkpoint_dir, payload['run_id']+'-checkpoint')
                report['checkpoint_evidence'] = ref
                (output/'checkpoint-upload.json').write_text(json.dumps(ref,indent=2))
            command = [python, '-m', 'lerobot.scripts.lerobot_eval',
                '--policy.path=' + checkpoint, '--policy.base_model_path=' + base_model,
                '--policy.device=cuda', '--env.type=libero', '--env.task=libero_10',
                '--rename_map={"observation.images.image2":"observation.images.wrist_image"}',
                '--env.task_ids=[0]', '--eval.n_episodes=1', '--eval.batch_size=1',
                '--env.max_parallel_tasks=1', '--eval.recording=false', '--seed=1000',
                '--output_dir=' + str(output/'evaluation')]
            (output/'native-command.json').write_text(json.dumps(command, indent=2))
            run('native-evaluation', command, 420)
            report.update(status='reference_training_evaluated' if reference_train else 'reference_evaluation_executed', task_success='see_native_eval_info',
                scope='One native tutorial episode; not an unseen-task generalization claim.')
        elif payload.get('mode') == 'evaluate':
            evaluation_args = ['--checkpoint', str(root/'inputs/checkpoint'), '--action-steps', str(payload['action_steps'])]
            run('audit', [*probe, 'audit', *common, *evaluation_args], 120)
            run('evaluate', [*probe, 'evaluate', *common, *evaluation_args], 240)
            report['task_success'] = json.loads((output/'evaluation.json').read_text())['task_success']
            report['status'] = 'evaluation_executed'
        elif payload.get('mode') == 'adapt':
            env.update(MUJOCO_GL='egl')
            run('train', [*probe, 'train', *common], 600)
            report['training_steps'] = 300
            # Durably retain native checkpoint before evaluation can fail.
            checkpoint = (output / 'training/checkpoints/last').resolve(strict=True)
            reference = storage.retain_tree(checkpoint, payload['run_id'] + '-checkpoint')
            report['checkpoint_evidence'] = reference
            (output / 'checkpoint-upload.json').write_text(json.dumps(reference, indent=2))
            run('evaluate', [*probe, 'evaluate', *common, '--action-steps', str(payload.get('action_steps',1))], 240)
            report['task_success'] = json.loads((output / 'evaluation.json').read_text())['task_success']
            report['status'] = 'adaptation_executed'
        else:
            run('inference', [*probe, 'inference', *common], 240)
            report['status'] = 'io_probe_pass'
    except Exception as error:
        report.update(status='fail', error_type=type(error).__name__)
        (output / 'error.log').write_text(traceback.format_exc())
    finally:
        if payload.get('mode') == 'reference-train' and (output/'training/checkpoints/last').is_symlink():
            try:
                subprocess.run([python, str(root/'inputs/lerobot_reference.py'), str(output), 'seal'],
                    check=True, timeout=30, env=env, capture_output=True)
            except Exception as error:
                report['reference_seal_error'] = type(error).__name__
        if (root / 'inputs/native_adapt.py').exists():
            try:
                subprocess.run([sys.executable, str(root / 'inputs/native_adapt.py'), 'seal', '--inputs', str(root / 'inputs'), '--output', str(output)], check=True, timeout=15, env=env, capture_output=True)
                report['stages'].append({'stage':'seal-native-checkpoint','exit_code':0})
            except Exception as error:
                report['seal_error'] = type(error).__name__
        report['elapsed_s'] = time.monotonic() - started
        secrets = dict(storage_auth, hf_token=model_auth)
        for path in output.rglob('*'):
            if path.is_file() and path.suffix in {'.json', '.log', '.txt'}:
                path.write_text(redact(path.read_text(errors='replace'), secrets))
        (output / 'report.json').write_text(json.dumps(report, indent=2))
    reference = storage.retain_tree(output, root.name)
    return {'status': report['status'], 'report': report, 's3_evidence': reference}
