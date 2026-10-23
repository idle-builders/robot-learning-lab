"""Cheap graphics preflight for the pinned i4h cart reference."""
from runpod_flash import Endpoint, GpuGroup, PodTemplate


@Endpoint(name='robot-lab-i4h-cart', gpu=GpuGroup.ADA_24,
          workers=(0, 1), gpu_count=1, idle_timeout=5, min_cuda_version='13.0',
          execution_timeout_ms=180_000, template=PodTemplate(containerDiskInGb=150),
          dependencies=[],
          system_dependencies=['libegl1', 'libgl1', 'libopengl0', 'libvulkan1', 'vulkan-tools'],
          env={'NVIDIA_DRIVER_CAPABILITIES': 'all'})
async def preflight(payload: dict) -> dict:
    import ctypes
    import json
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import tempfile
    import traceback

    result = {'status': 'running', 'weights_downloaded': False, 'stages': []}

    def stage(name, argv, timeout=30, env=None):
        item = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                              env=env, check=False)
        result['stages'].append({'name': name, 'exit_code': item.returncode,
                                 'stdout': item.stdout[-12000:],
                                 'stderr': item.stderr[-12000:]})
        if item.returncode:
            raise RuntimeError(name + ' failed')
        return item.stdout + item.stderr

    try:
        stage('nvidia-smi', ['nvidia-smi'], 20)
        stage('cuda', [os.sys.executable, '-c',
            "import torch; assert torch.cuda.device_count()==1; "
            "print(torch.cuda.get_device_name()); print(torch.ones(1,device='cuda').cpu())"], 30)
        summary = ''
        try:
            summary = stage('vulkan-default', ['vulkaninfo', '--summary'], 30)
        except RuntimeError:
            ctypes.CDLL('libEGL_nvidia.so.0')
            manifest = Path(tempfile.gettempdir()) / 'nvidia-headless-icd.json'
            manifest.write_text(json.dumps({'file_format_version':'1.0.0',
                'ICD':{'library_path':'libEGL_nvidia.so.0','api_version':'1.3.0'}}))
            env = dict(os.environ, VK_DRIVER_FILES=str(manifest), VK_ICD_FILENAMES=str(manifest))
            summary = stage('vulkan-headless', ['vulkaninfo', '--summary'], 30, env)
        if 'NVIDIA' not in summary:
            raise RuntimeError('Vulkan did not enumerate NVIDIA graphics')
        result.update(status='vulkan_preflight_passed', nvidia_graphics=True,
                      next_gate='Only now may a later worker request install i4h or download policy weights')
    except Exception:
        result.update(status='fail', error=traceback.format_exc())
    return result
