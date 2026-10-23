"""Bounded Runpod Flash compatibility probe; no policy qualification implied.

Serve with the Flash dev CLI; submit through workers/serverless_submit.py.
Only the sealed robot XML travels to the worker, not the artifact directory tree.
"""
from runpod_flash import Endpoint, GpuGroup


@Endpoint(
    name="robot-lab-serverless-probe",
    gpu=GpuGroup.ADA_48_PRO,
    workers=(0, 1),
    idle_timeout=5,
    execution_timeout_ms=600_000,
    gpu_count=1,
    dependencies=["mujoco==3.11.0", "warp-lang==1.16.0"],
    system_dependencies=["libegl1", "libgl1", "libvulkan1", "vulkan-tools"],
    env={"NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics", "MUJOCO_GL": "egl", "RUNPOD_API_KEY": ""},
)
async def probe(payload: dict) -> dict:
    import base64
    import hashlib
    import importlib.metadata
    import platform
    import subprocess
    import traceback
    import numpy as np
    import torch
    import warp as wp
    import mujoco

    result = {"status": "partial", "python": platform.python_version(), "stages": {}}
    for command in (["nvidia-smi"], ["vulkaninfo", "--summary"]):
        p = subprocess.run(command, capture_output=True, text=True, timeout=30)
        result[command[0]] = {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    result["packages"] = {n: importlib.metadata.version(n) for n in ("torch", "mujoco", "warp-lang")}
    try:
        assert torch.cuda.is_available()
        value = torch.ones(32, device="cuda").sum().item()
        assert value == 32
        wp.init()
        assert wp.is_cuda_available()
        device_array = wp.zeros(32, dtype=wp.float32, device="cuda")
        assert device_array.numpy().sum() == 0
        result["stages"]["cuda"] = {"status": "pass", "gpu": torch.cuda.get_device_name()}
    except Exception:
        result["stages"]["cuda"] = {"status": "fail", "error": traceback.format_exc()}
    try:
        xml = payload["xml"]
        assert hashlib.sha256(xml.encode()).hexdigest() == payload["sha256"]
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for _ in range(20):
            mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all()
        result["stages"]["robot_physics"] = {"status": "pass", "bodies": model.nbody, "joints": model.njnt, "source_sha256": payload["sha256"]}
        with mujoco.Renderer(model, height=120, width=160) as renderer:
            renderer.update_scene(data, camera="head_camera")
            pixels = renderer.render()
            assert pixels.std() > 1
            from OpenGL.GL import glGetString, GL_RENDERER, GL_VENDOR
            vendor = glGetString(GL_VENDOR).decode()
            device = glGetString(GL_RENDERER).decode()
            assert "NVIDIA" in vendor, (vendor, device)
            result["stages"]["camera"] = {"status": "pass", "vendor": vendor, "renderer": device, "shape": list(pixels.shape), "rgb_base64": base64.b64encode(pixels.tobytes()).decode()}
    except Exception:
        result["stages"]["camera_or_physics"] = {"status": "fail", "error": traceback.format_exc()}
    result["scope"] = "GPU/runtime and robot camera compatibility only; Isaac training and complete robot loop not yet verified"
    return result

