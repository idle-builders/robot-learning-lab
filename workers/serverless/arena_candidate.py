"""Evaluate an immutable native GR00T candidate on the pinned Arena G1 task."""
from runpod_flash import Endpoint, GpuGroup, PodTemplate


@Endpoint(name="robot-lab-arena-candidate", gpu=GpuGroup.ADA_24,
          workers=(0, 1), gpu_count=1, idle_timeout=5, min_cuda_version="13.0",
          execution_timeout_ms=1_800_000, template=PodTemplate(containerDiskInGb=150),
          dependencies=[],
          system_dependencies=["git", "ffmpeg", "libegl1", "libgl1", "libopengl0",
                               "libvulkan1", "build-essential", "ninja-build", "libxt6",
                               "vulkan-tools"],
          env={"RUNPOD_API_KEY": "", "NVIDIA_DRIVER_CAPABILITIES": "all",
               "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "N"})
async def arena_candidate(payload: dict, storage_auth: dict, model_auth: str) -> dict:
    import hashlib
    import json
    import os
    from pathlib import Path
    import re
    import shutil
    import signal
    import subprocess
    import sys
    import tempfile
    import time
    import traceback
    from native_checkpoint import validate_hf_checkpoint_reference, verify_hf_checkpoint_tree
    from s3_artifacts import Artifacts, CONFIG, client_from_auth, redact

    root = Path(tempfile.mkdtemp(prefix="arena-candidate-"))
    evidence = root / "evidence"
    evidence.mkdir()
    started = time.monotonic()
    storage = Artifacts(client_from_auth(CONFIG, storage_auth), CONFIG)
    secrets = dict(storage_auth, hf_token=model_auth)
    env = dict(os.environ, HF_TOKEN=model_auth, HF_HOME=str(root / "hf-cache"),
               UV_HTTP_TIMEOUT="60", UV_HTTP_RETRIES="1", UV_LINK_MODE="copy",
               HF_HUB_ENABLE_HF_TRANSFER="0", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false", MAX_JOBS="4",
               PYTHONUNBUFFERED="1")
    checkpoint_reference = payload["checkpoint_reference"]
    report = {"status": "running", "stages": [], "task_success": "not_run",
              "training": "not_run in this worker; immutable trained checkpoint input",
              "training_kind": "imitation continuation, not RL", "pins": payload["pins"],
              "checkpoint_reference": checkpoint_reference, "fresh_process_reload": False}
    server = None
    server_log = None

    def run(stage, command, cwd, seconds):
        budget = min(seconds, 1640 - (time.monotonic() - started))
        if budget <= 0:
            raise TimeoutError("Arena candidate worker deadline")
        os.write(1, ("Arena candidate stage: " + stage + "\n").encode())
        item = {"stage": stage, "command": command, "status": "running"}
        report["stages"].append(item)
        begin = time.monotonic()
        log_path = evidence / (stage + ".log")
        log_path.touch(exist_ok=False)
        process = subprocess.Popen(
            ["bash", "-o", "pipefail", "-c",
             '"$@" 2>&1 | tee -a "$ROBOTLAB_STAGE_LOG"', "arena-candidate-stage", *command],
            cwd=cwd, env={**env, "ROBOTLAB_STAGE_LOG": str(log_path)}, stdout=1,
            stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=budget)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            item.update(status="timeout", elapsed_s=time.monotonic() - begin)
            raise
        item.update(status="pass" if code == 0 else "fail", exit_code=code,
                    elapsed_s=time.monotonic() - begin)
        os.write(1, (redact(log_path.read_text(errors="replace")[-5000:], secrets) + "\n").encode())
        if code:
            raise RuntimeError(stage + " failed")

    try:
        pins = payload["pins"]
        run_id = payload["run_id"]
        if not re.fullmatch(r"arena-candidate-[a-f0-9]{12}", run_id):
            raise ValueError("invalid candidate evaluation run id")
        run("hardware", ["nvidia-smi"], root, 20)
        run("cuda", [sys.executable, "-c",
            "import torch; print(torch.cuda.get_device_name()); assert torch.cuda.device_count()==1; "
            "assert torch.cuda.get_device_properties(0).total_memory >= 22*1024**3; "
            "assert 'RTX 4090' in torch.cuda.get_device_name(); print(torch.ones(1,device='cuda').cpu())"],
            root, 30)
        run("driver", ["bash", "-c",
            "test \"$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)\" -ge 580"],
            root, 15)
        run("graphics-libraries", ["ldconfig", "-p"], root, 15)
        try:
            run("vulkan-default", ["vulkaninfo", "--summary"], root, 30)
        except RuntimeError:
            import ctypes
            ctypes.CDLL("libEGL_nvidia.so.0")
            manifest = root / "nvidia-headless-icd.json"
            manifest.write_text(json.dumps({"file_format_version": "1.0.0",
                "ICD": {"library_path": "libEGL_nvidia.so.0", "api_version": "1.3.0"}}))
            shutil.copyfile(manifest, evidence / manifest.name)
            env["VK_DRIVER_FILES"] = str(manifest)
            env["VK_ICD_FILENAMES"] = str(manifest)
            run("vulkan-headless", ["vulkaninfo", "--summary"], root, 30)
        vk_log = evidence / ("vulkan-headless.log" if (evidence / "vulkan-headless.log").exists()
                             else "vulkan-default.log")
        if "NVIDIA" not in vk_log.read_text():
            raise RuntimeError("Vulkan did not enumerate NVIDIA graphics")

        run("install-uv", [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
            "--retries", "1", "--timeout", "30", "uv==0.12.17"], root, 90)
        run("bootstrap-versions", [sys.executable, "-c",
            "from importlib.metadata import version; print({n:version(n) for n in ['boto3','runpod','uv']})"], root, 15)
        checkout = root / "IsaacLab-Arena"
        run("clone", ["git", "clone", "--filter=blob:none",
            "https://github.com/isaac-sim/IsaacLab-Arena.git", str(checkout)], root, 90)
        run("checkout", ["git", "checkout", pins["arena"]], checkout, 20)
        run("submodules", ["git", "-c", "url.https://github.com/.insteadOf=git@github.com:",
            "submodule", "update", "--init", "--depth", "1"], checkout, 180)
        for name, revision in [("IsaacLab", pins["isaaclab"]), ("Isaac-GR00T", pins["gr00t"])]:
            actual = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                             cwd=checkout / "submodules" / name, text=True).strip()
            if actual != revision:
                raise ValueError(name + " revision mismatch")
        if hashlib.sha256((checkout / "uv.lock").read_bytes()).hexdigest() != pins["arena_lock_sha256"]:
            raise ValueError("Arena lock mismatch")
        uv = [sys.executable, "-m", "uv"]
        run("arena-install", [*uv, "sync", "--frozen", "--group", "gr00t-client"], checkout, 600)
        sim_python = str(checkout / ".venv/bin/python")
        run("arena-versions", [*uv, "pip", "freeze", "--python", sim_python], checkout, 30)
        groot = checkout / "submodules/Isaac-GR00T"
        run("groot-install", [*uv, "sync", "--python", "3.10"], groot, 600)
        run("groot-editable", [*uv, "pip", "install", "--no-deps", "-e", "."], groot, 30)
        policy_python = str(groot / ".venv/bin/python")
        run("groot-versions", [*uv, "pip", "freeze", "--python", policy_python], groot, 30)
        for src, dst in [(checkout / "uv.lock", "arena-uv.lock"),
                         (groot / "uv.lock", "groot-uv.lock"),
                         (groot / "pyproject.toml", "groot-pyproject.toml")]:
            shutil.copyfile(src, evidence / dst)

        model = root / "candidate-checkpoint"
        files = validate_hf_checkpoint_reference(checkpoint_reference)
        download = (
            "import json,sys; from huggingface_hub import snapshot_download; "
            "snapshot_download(repo_id=sys.argv[1],revision=sys.argv[2],"
            "local_dir=sys.argv[3],allow_patterns=json.loads(sys.argv[4]))")
        run("checkpoint-download", [policy_python, "-c", download,
            checkpoint_reference["repo_id"], checkpoint_reference["revision"],
            str(model), json.dumps(sorted(files))], groot, 600)
        materialized = verify_hf_checkpoint_tree(checkpoint_reference, model)
        (evidence / "checkpoint-materialization.json").write_text(json.dumps(materialized, indent=2))
        report["checkpoint_materialization"] = {key: materialized[key]
                                                for key in ("status", "files", "bytes",
                                                            "excluded_resume_files")}
        server_command = [policy_python, "gr00t/eval/run_gr00t_server.py",
            "--modality-config-path", "../../isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_config.py",
            "--model-path", str(model), "--embodiment-tag", "NEW_EMBODIMENT",
            "--device", "cuda", "--host", "127.0.0.1", "--port", "5555"]
        report["server_command"] = server_command
        server_log = (evidence / "groot-server.log").open("x")
        server = subprocess.Popen(server_command, cwd=groot, env=env, stdout=server_log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        readiness = ("from gr00t.policy.server_client import PolicyClient; "
            "c=PolicyClient(host='127.0.0.1',port=5555,timeout_ms=150000); "
            "print(c.get_modality_config())")
        run("fresh-checkpoint-server-ready", [policy_python, "-c", readiness], groot, 180)
        report["fresh_process_reload"] = True
        eval_command = [sim_python, "isaaclab_arena/evaluation/policy_runner.py",
            "--policy_type", "isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy",
            "--policy_config_yaml_path", "isaaclab_arena_gr00t/policy/config/g1_locomanip_gr00t_closedloop_config.yaml",
            "--remote_host", "127.0.0.1", "--remote_port", "5555", "--num_steps", "1500",
            "--device", "cpu", "--num_envs", "1", "--seed", "42", "--enable_cameras",
            "--record_camera_video", "--output_base_dir", str(evidence / "evaluation"),
            "galileo_g1_locomanip_pick_and_place", "--object", "brown_box",
            "--embodiment", "g1_wbc_joint"]
        run("native-candidate-evaluation", eval_command, checkout, 900)
        results = list((evidence / "evaluation").rglob("episode_results_rank0.jsonl"))
        if len(results) != 1:
            raise RuntimeError("expected exactly one native episode result")
        lines = [json.loads(line) for line in results[0].read_text().splitlines() if line.strip()]
        if len(lines) != 1 or "success" not in lines[0]:
            raise RuntimeError("native episode result is incomplete")
        report.update(status="candidate_evaluation_executed", task_success=bool(lines[0]["success"]),
                      episode_result=lines[0])
    except Exception:
        report.update(status="fail", error=traceback.format_exc())
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        if server_log is not None:
            server_log.close()
        report["elapsed_s"] = time.monotonic() - started
        for path in evidence.rglob("*"):
            if path.is_file() and path.suffix in {".log", ".json", ".txt", ".html", ".yaml"}:
                path.write_text(redact(path.read_text(errors="replace"), secrets))
        report = redact(report, secrets)
        (evidence / "report.json").write_text(json.dumps(report, indent=2))
    ref = storage.retain_tree(evidence, payload["run_id"])
    return {"status": report["status"], "task_success": report["task_success"],
            "report": report, "s3_evidence": ref}
