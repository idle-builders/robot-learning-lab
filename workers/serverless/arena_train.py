"""Bounded native GR00T continuation training and unchanged Arena evaluation."""
from runpod_flash import Endpoint, GpuType, PodTemplate


@Endpoint(name="robot-lab-arena-train", gpu=GpuType.NVIDIA_A100_80GB_PCIe,
          workers=(0, 1), gpu_count=1, idle_timeout=5,
          min_cuda_version="12.8",
          execution_timeout_ms=3_600_000, template=PodTemplate(containerDiskInGb=180),
          dependencies=[],
          system_dependencies=["git", "ffmpeg", "libegl1", "libgl1", "libopengl0",
                               "libvulkan1", "build-essential", "ninja-build", "libxt6",
                               "vulkan-tools"],
          env={"RUNPOD_API_KEY": "", "NVIDIA_DRIVER_CAPABILITIES": "all",
               "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "N"})
async def arena_train(payload: dict, storage_auth: dict, model_auth: str) -> dict:
    import hashlib
    import json
    import math
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
    from s3_artifacts import Artifacts, CONFIG, client_from_auth, object_key, redact

    root = Path(tempfile.mkdtemp(prefix="arena-train-"))
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
    report = {"status": "running", "stages": [], "task_success": "not_run",
              "training": "running", "training_kind": "imitation continuation, not RL",
              "pins": payload["pins"], "bounded_config": payload["training"]}
    def run(stage, command, cwd, seconds):
        budget = min(seconds, 3560 - (time.monotonic() - started))
        if budget <= 0:
            raise TimeoutError("Arena training worker deadline")
        os.write(1, ("Arena training stage: " + stage + "\n").encode())
        item = {"stage": stage, "command": command, "status": "running"}
        report["stages"].append(item)
        begin = time.monotonic()
        log_path = evidence / (stage + ".log")
        log_path.touch(exist_ok=False)
        process = subprocess.Popen(
            ["bash", "-o", "pipefail", "-c",
             '"$@" 2>&1 | tee -a "$ROBOTLAB_STAGE_LOG"', "arena-train-stage", *command],
            cwd=cwd, env={**env, "ROBOTLAB_STAGE_LOG": str(log_path)},
            stdout=1, stderr=subprocess.STDOUT, start_new_session=True)
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

    def sha256_file(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    try:
        pins = payload["pins"]
        training = payload["training"]
        run_id = payload["run_id"]
        if not re.fullmatch(r"arena-training-[a-f0-9]{12}", run_id):
            raise ValueError("invalid training run id")
        run("hardware", ["nvidia-smi"], root, 20)
        run("cuda", [sys.executable, "-c",
            "import torch; print(torch.cuda.get_device_name()); assert torch.cuda.device_count()==1; "
            "assert torch.cuda.get_device_properties(0).total_memory >= 44*1024**3; "
            "print(torch.ones(1,device='cuda').cpu())"], root, 30)
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
        groot = checkout / "submodules/Isaac-GR00T"
        run("groot-install", [*uv, "sync", "--python", "3.10"], groot, 600)
        run("groot-editable", [*uv, "pip", "install", "--no-deps", "-e", "."], groot, 30)
        policy_python = str(groot / ".venv/bin/python")
        run("groot-versions", [*uv, "pip", "freeze", "--python", policy_python], groot, 30)
        for src, dst in [(checkout / "uv.lock", "arena-uv.lock"),
                         (groot / "uv.lock", "groot-uv.lock"),
                         (groot / "pyproject.toml", "groot-pyproject.toml")]:
            shutil.copyfile(src, evidence / dst)

        base_model = root / "base-checkpoint"
        download_model = ("from huggingface_hub import snapshot_download; snapshot_download(" 
            "repo_id=" + repr(pins["checkpoint_repo"]) + ",revision=" + repr(pins["checkpoint_revision"]) +
            ",local_dir=" + repr(str(base_model)) + ",allow_patterns=['*.json','*.safetensors','README.md'])")
        run("checkpoint-download", [policy_python, "-c", download_model], groot, 300)
        dataset_root = root / "dataset"
        if payload.get("dataset_reference"):
            materialized_dataset = storage.download_run(payload["dataset_reference"], dataset_root)
            (evidence / "dataset-materialization.json").write_text(json.dumps(materialized_dataset, indent=2))
            training_dataset = dataset_root
            report["dataset_source"] = "caller-provided converted LeRobot dataset"
        else:
            download_data = ("from huggingface_hub import snapshot_download; snapshot_download("
                "repo_id=" + repr(pins["dataset_repo"]) + ",repo_type='dataset',revision=" +
                repr(pins["dataset_revision"]) + ",local_dir=" + repr(str(dataset_root)) +
                ",allow_patterns=['lerobot/*'])")
            run("dataset-download", [policy_python, "-c", download_data], groot, 300)
            training_dataset = dataset_root / "lerobot"
            report["dataset_source"] = "pinned NVIDIA published LeRobot dataset"

        output_dir = root / "candidate"
        output_dir.mkdir(exist_ok=False)
        train_command = [policy_python, "gr00t/experiment/launch_finetune.py",
            "--dataset-path", str(training_dataset),
            "--output-dir", str(output_dir),
            "--modality-config-path", "../../isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_config.py",
            "--global-batch-size", str(training["global_batch_size"]),
            "--max-steps", str(training["max_steps"]), "--num-gpus", "1",
            "--save-steps", str(training["save_steps"]), "--save-total-limit", "1",
            "--base-model-path", str(base_model), "--no-tune-llm", "--no-tune-visual",
            "--tune-projector", "--tune-diffusion-model", "--dataloader-num-workers", "0",
            "--learning-rate", str(training["learning_rate"]),
            "--gradient-accumulation-steps", "1", "--embodiment-tag", "NEW_EMBODIMENT",
            "--color-jitter-params", "brightness", "0.3", "contrast", "0.4",
            "saturation", "0.5", "hue", "0.08"]
        report["training_command"] = train_command
        run("native-training", train_command, groot, 720)

        checkpoint = output_dir / ("checkpoint-" + str(training["max_steps"]))
        state_path = checkpoint / "trainer_state.json"
        if not state_path.is_file():
            raise RuntimeError("native checkpoint has no trainer_state.json")
        state = json.loads(state_path.read_text())
        metrics = []
        for item in state.get("log_history", []):
            numeric = {key: value for key, value in item.items()
                       if key in {"loss", "train_loss", "grad_norm", "learning_rate"}
                       and isinstance(value, (int, float))}
            if numeric:
                metrics.append({**numeric, "step": item.get("step")})
        if not metrics:
            final_losses = re.findall(r"'train_loss':\s*([-+0-9.eE]+)",
                                      (evidence / "native-training.log").read_text(errors="replace"))
            if final_losses:
                metrics.append({"train_loss": float(final_losses[-1]),
                                "step": state.get("global_step"),
                                "source": "native trainer final summary"})
        if state.get("global_step") != training["max_steps"] or not metrics:
            raise RuntimeError("native trainer did not record the requested optimizer steps")
        if not all(math.isfinite(value) for item in metrics for key, value in item.items()
                   if key not in {"step", "source"}):
            raise RuntimeError("native training produced a non-finite metric")
        (evidence / "training-state-summary.json").write_text(json.dumps({
            "global_step": state["global_step"], "metrics": metrics,
            "best_model_checkpoint": state.get("best_model_checkpoint")}, indent=2))

        resume_only = {"optimizer.pt", "rng_state.pth", "scheduler.pt",
                       "trainer_state.json", "training_args.bin"}
        inference_checkpoint = root / "inference-checkpoint"
        shutil.copytree(checkpoint, inference_checkpoint, copy_function=os.link,
                        ignore=lambda _, names: [name for name in names if name in resume_only])
        files = {}
        total = 0
        for path in sorted(inference_checkpoint.rglob("*")):
            if path.is_file():
                relative = path.relative_to(inference_checkpoint).as_posix()
                size = path.stat().st_size
                total += size
                sha = sha256_file(path)
                files[relative] = {"sha256": sha, "bytes": size}
        repo_id = payload["checkpoint_hf_repo"]
        upload_receipt = root / "hf-upload.json"
        upload_code = (
            "import json,sys; from pathlib import Path; from huggingface_hub import HfApi; "
            "api=HfApi(); repo=sys.argv[1]; folder=sys.argv[2]; out=sys.argv[3]; run=sys.argv[4]; "
            "api.create_repo(repo_id=repo,private=True,exist_ok=True); "
            "commit=api.upload_folder(repo_id=repo,folder_path=folder,commit_message=run); "
            "Path(out).write_text(json.dumps({'repo_id':repo,'revision':commit.oid,'url':commit.repo_url}))")
        run("checkpoint-hf-upload", [policy_python, "-c", upload_code, repo_id,
            str(inference_checkpoint), str(upload_receipt), run_id], groot, 2400)
        receipt = json.loads(upload_receipt.read_text())
        checkpoint_ref = {"schema": "robot-learning-lab.hf-checkpoint.v1",
            "repo_id": repo_id, "revision": receipt["revision"], "private": True,
            "files": files, "bytes": total, "file_count": len(files)}
        (evidence / "checkpoint-hf-reference.json").write_text(json.dumps(checkpoint_ref, indent=2))
        report["checkpoint_hf_reference"] = checkpoint_ref
        report["training"] = {"status": "pass", "global_step": state["global_step"],
                              "finite_metrics": metrics, "checkpoint_bytes": total,
                              "checkpoint_files": len(files),
                              "persistence": "inference files only; resume state omitted"}
        report.update(status="training_checkpoint_persisted", task_success="not_run",
                      evaluation="not_run; use the separate arena_candidate worker",
                      checkpoint_hf_reference=checkpoint_ref)
    except Exception:
        report.update(status="fail", error=traceback.format_exc())
    finally:
        report["elapsed_s"] = time.monotonic() - started
        for path in evidence.rglob("*"):
            if path.is_file() and path.suffix in {".log", ".json", ".txt", ".html", ".yaml"}:
                path.write_text(redact(path.read_text(errors="replace"), secrets))
        report = redact(report, secrets)
        (evidence / "report.json").write_text(json.dumps(report, indent=2))
    ref = storage.retain_tree(evidence, payload["run_id"])
    return {"status": report["status"], "task_success": report["task_success"],
            "report": report, "s3_evidence": ref}
