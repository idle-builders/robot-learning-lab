"""Run the existing pinned Isaac pilot on a scale-to-zero worker."""
from runpod_flash import Endpoint, GpuGroup, PodTemplate

@Endpoint(
    name="robot-lab-pip-graphics",
    gpu=GpuGroup.ADA_24,
    workers=(0,1), gpu_count=1, idle_timeout=5,
    execution_timeout_ms=1_800_000,
    template=PodTemplate(containerDiskInGb=100),
    env={"NVIDIA_DRIVER_CAPABILITIES":"compute,utility,graphics", "MUJOCO_GL":"egl", "RUNPOD_API_KEY":""},
    dependencies=["boto3==1.43.98"],
    system_dependencies=["git","libegl1","libgl1","libopengl0","libvulkan1"],
)
async def control(payload: dict, storage_auth: dict = None) -> dict:
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
    from s3_artifacts import Artifacts, CONFIG, INPUT_BYTES, INPUT_FILES, client_from_auth, safe_name, redact

    volume = Path("/tmp/robot-lab-cache")
    volume.mkdir(exist_ok=True)
    runs = volume / "control-runs"
    runs.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="pip-", dir=runs))
    output = root / "evidence"
    output.mkdir()
    start = time.monotonic()
    report = {"status": "partial", "stages": [], "robot_rl": "not_run"}
    storage = None

    # Recovery re-uploads retained evidence from this worker's bounded cache;
    # it neither launches a simulator nor accepts an arbitrary filesystem path.
    if payload.get("recover_run"):
        import re
        run_id = payload["recover_run"]
        if not isinstance(run_id, str) or not re.fullmatch(r"pip-[a-z0-9_]{1,32}", run_id):
            raise ValueError("Invalid recovery run identity")
        previous = runs / run_id / "evidence"
        if not previous.is_dir() or previous.resolve().parent.parent != runs.resolve():
            return {"status": "fail", "reason": "Previous worker evidence is unavailable on this worker"}
        storage = Artifacts(client_from_auth(CONFIG, storage_auth), CONFIG)
        try:
            reference = storage.retain_tree(previous, run_id)
            return {"status": "partial", "s3_evidence": reference, "scope": "Evidence recovery only; original outcome unchanged"}
        except Exception as error:
            logs = {str(p.relative_to(previous)): p.read_text(errors="replace")[-18000:]
                    for p in previous.rglob("*") if p.suffix in (".log", ".json") and p.stat().st_size < 100000}
            return redact({"status": "fail", "error_type": type(error).__name__,
                           "error_code": getattr(error, "response", {}).get("Error", {}).get("Code"),
                           "diagnostic_logs": logs}, storage_auth)

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
        print("Entered bounded control handler", flush=True)
        run("hardware", ["nvidia-smi", "--query-gpu=name,mig.mode.current", "--format=csv,noheader"], root, 20)
        if "Enabled" in (output/"hardware.log").read_text():
            if "storage" in payload:
                storage=Artifacts(client_from_auth(payload["storage"],storage_auth),payload["storage"])
            raise RuntimeError("MIG worker excluded for this verified RTX camera path")
        # Only a bounded, explicitly named transfer bundle is materialized.
        if len(payload["files"]) > INPUT_FILES:
            raise ValueError("too many inputs")
        if "storage" in payload:
            storage = Artifacts(client_from_auth(payload["storage"], storage_auth), payload["storage"])
            report["input_transfer"] = storage.materialize(payload["files"], root / "transfer")
        else:
            total = 0
            for name, item in payload["files"].items():
                path = safe_name(name)
                data = base64.b64decode(item["base64"], validate=True)
                total += len(data)
                if total > 5_000_000 or hashlib.sha256(data).hexdigest() != item["sha256"]:
                    raise ValueError("input size/hash mismatch")
                dest = root / "transfer" / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
        seal = (root / "transfer/bundles/pip/manifest.json").read_bytes()
        if hashlib.sha256(seal).hexdigest() != payload["bundle_sha256"]:
            raise ValueError("materialized bundle seal mismatch")
        for name, expected in json.loads(seal)["files"].items():
            safe_name(name)
            if hashlib.sha256((root / "transfer/bundles/pip" / name).read_bytes()).hexdigest() != expected:
                raise ValueError("materialized bundle file mismatch")
        run("bootstrap-uv", [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--timeout", "20", "--retries", "1", "--index-url", "https://pypi.org/simple", "uv==0.12.17"], root, 90)
        commit = "ae37b028ea415c91ea2bc32609efcd759ed2b974"
        checkout = volume / "IsaacLab"
        if not checkout.exists():
            run("clone", ["git", "clone", "--filter=blob:none", "https://github.com/isaac-sim/IsaacLab.git", str(checkout)], root, 180)
        run("checkout", ["git", "checkout", commit], checkout, 180)
        lock = hashlib.sha256((checkout / "uv.lock").read_bytes()).hexdigest()
        if lock != "98570361a2f86abcfdebfda35f853da0945a8ccbe422f51d5fffaf1936d478a4":
            raise ValueError("upstream lock mismatch")
        run("install", [sys.executable, "-m", "uv", "sync", "--frozen", "--extra", "importers"], checkout, 900)
        run("inventory", [sys.executable, "-m", "uv", "pip", "freeze", "--python", str(checkout / ".venv/bin/python")], checkout, 30)
        run("control", [str(checkout / ".venv/bin/python"), str(root / "transfer/control/run.py"), "--bundle", str(root / "transfer/bundles/pip"), "--output", str(output / "control")], root, 1100)
        report["robot_rl"] = json.loads((output / "control/report.json").read_text()).get("robot_rl", "not_run")
    except Exception as error:
        report["status"] = "fail"
        # S3 errors can carry request details. Keep a type-only error here;
        # engineering subprocess errors remain in their retained stage logs.
        report["error_type"] = type(error).__name__
    finally:
        report["worker_run"] = str(root)
        report["storage"] = "S3 immutable file objects and run manifest" if storage else "ephemeral; returned evidence must be retained locally"
        report["elapsed_s"] = time.monotonic() - start
        report["files"] = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob("*") if p.is_file()}
        (output / "bootstrap.json").write_text(json.dumps(report, indent=2))
    if storage is not None:
        try:
            reference = storage.retain_tree(output, root.name)
            return {"status": report["status"], "report": report, "s3_evidence": reference}
        except Exception as error:
            # Keep the stage outcome and failure code without serializing auth.
            report["status"] = "fail"
            report["storage_error_type"] = type(error).__name__
            (output / "s3-error.json").write_text(json.dumps({"error_type": type(error).__name__}))
            return {"status": "fail", "report": report, "reason": "S3 evidence transfer incomplete; no successful retention claim"}
    if "storage" in payload:
        return {"status": "fail", "report": report, "reason": "S3 initialization failed before durable output retention"}
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in output.rglob("*"):
            if path.is_file():
                z.write(path, path.relative_to(output))
    blob = archive.getvalue()
    if len(blob) > 6_000_000:
        return {"status": "fail", "reason": "evidence exceeds inline transport; persistent storage required", "report": report}
    return {"status": report["status"], "report": report, "archive_sha256": hashlib.sha256(blob).hexdigest(), "archive_base64": base64.b64encode(blob).decode()}
