"""Run the bounded native GR00T train/save/reload/Arena evaluation loop."""
import argparse
import json
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.probe_vla import finalize
from workers.serverless.native_checkpoint import validate_hf_checkpoint_reference


def endpoint_ids():
    result = subprocess.run(["runpodctl", "serverless", "list"], capture_output=True,
                            text=True, check=True, timeout=20)
    return {x["id"] for x in json.loads(result.stdout)
            if "robot-lab-arena-train" in x.get("name", "")}


def verify_hf_checkpoint(reference, token):
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    files = validate_hf_checkpoint_reference(reference)
    total = 0
    for name, item in files.items():
        metadata = get_hf_file_metadata(hf_hub_url(reference["repo_id"], name,
            revision=reference["revision"]), token=token, timeout=30)
        if metadata.size != item["bytes"]:
            raise ValueError("checkpoint object size mismatch")
        total += item["bytes"]
    if total != reference.get("bytes") or len(files) != reference.get("file_count"):
        raise ValueError("checkpoint manifest totals mismatch")
    return {"status": "pass", "files": len(files), "bytes": total,
            "repo_id": reference["repo_id"], "revision": reference["revision"],
            "verification": "private Hugging Face revision and remote file sizes; fresh-process model reload follows"}


def training_result_complete(worker):
    """Require a persisted native checkpoint; evaluation is a separate worker boundary."""
    report = worker.get("report", {})
    return (worker.get("status") == "training_checkpoint_persisted"
            and worker.get("task_success") == "not_run"
            and report.get("training", {}).get("status") == "pass"
            and isinstance(report.get("checkpoint_hf_reference"), dict))


def main():
    from huggingface_hub import get_token, get_hf_file_metadata, hf_hub_url
    from workers.s3_artifacts import Artifacts, CONFIG, profile_auth, client_from_auth, redact
    from lab.artifact_store import snapshot_file, retain_dependency

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--checkpoint-hf-repo", default="tobiagru/robot-learning-lab-g1-poc")
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 60:
        raise ValueError("--max-steps must be between 1 and 60 for this bounded POC")
    pins = json.loads((ROOT / "workers/arena/pins.json").read_text())
    token = get_token()
    for name in ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]:
        get_hf_file_metadata(hf_hub_url(pins["checkpoint_repo"], name,
                                       revision=pins["checkpoint_revision"]), token=token, timeout=15)
    if args.dataset_dir is None:
        get_hf_file_metadata(hf_hub_url(pins["dataset_repo"], "lerobot/meta/info.json",
                                       repo_type="dataset", revision=pins["dataset_revision"]),
                                 token=token, timeout=15)
    auth = profile_auth("runpod")
    secrets = dict(auth, hf_token=token)
    client = client_from_auth(CONFIG, auth)
    storage = Artifacts(client, CONFIG)
    run_id = "arena-training-" + uuid.uuid4().hex[:12]
    output = ROOT / "artifacts" / run_id
    output.mkdir()
    print(output, flush=True)
    sources = ["tools/probe_arena_train.py", "tools/probe_vla.py",
               "workers/serverless/arena_train.py", "workers/serverless/s3_artifacts.py",
               "workers/arena/pins.json"]
    for name in sources:
        snapshot_file(ROOT / name, output / "source" / name)
    baseline = ROOT / "artifacts/arena-reference-7441414d0c61"
    if baseline.exists():
        (output / "dependencies.json").write_text(json.dumps(
            {"unchanged_baseline_evaluation": retain_dependency(baseline)}, indent=2))
    payload = {"pins": pins, "run_id": run_id,
               "checkpoint_hf_repo": args.checkpoint_hf_repo,
               "training": {"global_batch_size": 1, "max_steps": args.max_steps, "save_steps": args.max_steps,
                            "learning_rate": 1e-5, "tuned_modules": ["projector", "diffusion_model"],
                            "frozen_modules": ["llm", "visual"]}}
    if args.dataset_dir is not None:
        dataset_dir = args.dataset_dir.resolve()
        if not (dataset_dir / "meta/info.json").is_file():
            raise ValueError("--dataset-dir must be a converted LeRobot dataset")
        dataset_ref = storage.retain_tree(dataset_dir, run_id + "-dataset")
        payload["dataset_reference"] = dataset_ref
        (output / "dataset-reference.json").write_text(json.dumps(dataset_ref, indent=2))
    (output / "request.json").write_text(json.dumps(payload, indent=2))
    before = endpoint_ids()
    process = None
    completed = False
    try:
        with socket.socket() as check:
            check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            check.bind(("127.0.0.1", 8907))
        with (output / "flash.log").open("x") as log:
            process = subprocess.Popen([str(ROOT / "tools/runpod-flash"), "dev", "--port", "8907",
                                        "--no-reload"], cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            base = None
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError("Flash exited before readiness")
                matches = re.findall(r"flash dev\s+localhost:(\d+)",
                                     (output / "flash.log").read_text(errors="replace"))
                if matches:
                    base = "http://127.0.0.1:" + matches[-1]
                if base is None:
                    time.sleep(0.5)
                    continue
                try:
                    with urllib.request.urlopen(base + "/openapi.json", timeout=1) as response:
                        schema = json.load(response)
                    if "/arena_train/runsync" not in schema["paths"]:
                        raise RuntimeError("missing Arena training route")
                    break
                except (OSError, TimeoutError):
                    time.sleep(0.5)
            else:
                raise TimeoutError("Flash readiness deadline")
            envelope = {"input": {"payload": payload, "storage_auth": auth, "model_auth": token}}
            request = urllib.request.Request(base + "/arena_train/runsync",
                data=json.dumps(envelope).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=3780) as response:
                result = redact(json.load(response), secrets)
            (output / "response.json").write_text(json.dumps(result, indent=2))
            worker = result.get("output", {})
            if "s3_evidence" in worker:
                verification = storage.download_run(worker["s3_evidence"], output / "evidence")
                (output / "download-verification.json").write_text(json.dumps(verification, indent=2))
            checkpoint_ref = worker.get("report", {}).get("checkpoint_hf_reference")
            if checkpoint_ref:
                verified = verify_hf_checkpoint(checkpoint_ref, token)
                (output / "checkpoint-verification.json").write_text(json.dumps(verified, indent=2))
            completed = training_result_complete(worker) and checkpoint_ref is not None
            print("Worker status:", worker.get("status", "missing"),
                  "checkpoint verified:", checkpoint_ref is not None, flush=True)
    except Exception as exc:
        (output / "client-error.json").write_text(json.dumps(
            {"status": "fail", "error": redact(str(exc), secrets), "error_type": type(exc).__name__},
            indent=2))
        print("Arena training attempt failed:", type(exc).__name__, flush=True)
    finally:
        if process is not None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=25)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
        cleanup = []
        try:
            for endpoint in endpoint_ids() - before:
                result = subprocess.run(["runpodctl", "serverless", "delete", endpoint],
                                        capture_output=True, timeout=30)
                cleanup.append({"endpoint": endpoint, "delete_exit_code": result.returncode})
            (output / "cleanup.json").write_text(json.dumps(
                {"deletions": cleanup, "remaining_owned_endpoints": sorted(endpoint_ids() - before)},
                indent=2))
        except Exception as exc:
            (output / "cleanup.json").write_text(json.dumps(
                {"status": "needs_check", "error_type": type(exc).__name__}))
        log = output / "flash.log"
        if log.exists():
            log.write_text(redact(log.read_text(errors="replace"), secrets))
        finalize(output)
    if not completed:
        raise SystemExit("Failed; retained evidence: " + str(output))


if __name__ == "__main__":
    main()
