"""Run the existing pinned Isaac pilot on a scale-to-zero worker."""
from runpod_flash import Endpoint, GpuGroup, PodTemplate


@Endpoint(
    name="robot-lab-isaac-pilot",
    gpu=GpuGroup.ADA_48_PRO,
    workers=(0, 1),
    gpu_count=1,
    idle_timeout=5,
    execution_timeout_ms=1_800_000,
    dependencies=["uv==0.12.17"],
    system_dependencies=["git", "libegl1", "libgl1", "libopengl0", "libvulkan1"],
    template=PodTemplate(containerDiskInGb=100),
    env={"NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics", "MUJOCO_GL": "egl", "RUNPOD_API_KEY": ""},
)
async def pilot(payload: dict) -> dict:
    import base64
    import hashlib
    import io
    import json
    import os
    from pathlib import Path, PurePosixPath
    import signal
    import subprocess
    import sys
    import tempfile
    import time
    import traceback
    import zipfile

    root = Path(tempfile.mkdtemp(prefix="robot-isaac-"))
    output = root / "evidence"
    output.mkdir()
    start = time.monotonic()
    report = {"status": "partial", "stages": [], "robot_rl": "not_run"}

    def run(name, command, cwd, timeout):
        timeout = min(timeout, 1500 - (time.monotonic() - start))
        if timeout <= 0:
            raise RuntimeError("Pilot wall-clock budget exhausted")
        with (output / (name + ".log")).open("xb") as log:
            p = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                import psutil
                for child in psutil.Process(p.pid).children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                raise RuntimeError(name + " exceeded deadline")
        report["stages"].append({"stage": name, "exit_code": code})
        print(name, "exit", code, flush=True)
        if code:
            raise RuntimeError(name + " failed; inspect retained log")

    try:
        # Only a bounded, explicitly named transfer bundle is materialized.
        if len(payload["files"]) > 150:
            raise ValueError("too many inputs")
        total = 0
        for name, item in payload["files"].items():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or path.parts.count("source") > 1 or "artifacts" in path.parts:
                raise ValueError("unsafe input path")
            data = base64.b64decode(item["base64"], validate=True)
            total += len(data)
            if total > 5_000_000 or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("input size/hash mismatch")
            dest = root / "transfer" / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        commit = "ae37b028ea415c91ea2bc32609efcd759ed2b974"
        checkout = root / "IsaacLab"
        run("clone", ["git", "clone", "--filter=blob:none", "https://github.com/isaac-sim/IsaacLab.git", str(checkout)], root, 180)
        run("checkout", ["git", "checkout", commit], checkout, 180)
        lock = hashlib.sha256((checkout / "uv.lock").read_bytes()).hexdigest()
        if lock != "98570361a2f86abcfdebfda35f853da0945a8ccbe422f51d5fffaf1936d478a4":
            raise ValueError("upstream lock mismatch")
        run("install", [sys.executable, "-m", "uv", "sync", "--frozen", "--extra", "importers"], checkout, 900)
        run("inventory", [sys.executable, "-m", "uv", "pip", "freeze", "--python", str(checkout / ".venv/bin/python")], checkout, 30)
        run("pilot", [str(checkout / ".venv/bin/python"), str(root / "transfer/isaac_pilot.py"), "--checkout", str(checkout), "--bundle", str(root / "transfer/bundles/pip"), "--expected-sha256", payload["bundle_sha256"], "--output", str(output / "pilot"), "--timeout", "180"], root, 600)
    except Exception:
        report["status"] = "fail"
        report["error"] = traceback.format_exc()
    finally:
        report["elapsed_s"] = time.monotonic() - start
        report["files"] = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob("*") if p.is_file()}
        (output / "bootstrap.json").write_text(json.dumps(report, indent=2))
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in output.rglob("*"):
            if path.is_file():
                z.write(path, path.relative_to(output))
    blob = archive.getvalue()
    if len(blob) > 6_000_000:
        return {"status": "fail", "reason": "evidence exceeds inline transport; persistent storage required", "report": report}
    return {"status": report["status"], "report": report, "archive_sha256": hashlib.sha256(blob).hexdigest(), "archive_base64": base64.b64encode(blob).decode()}
