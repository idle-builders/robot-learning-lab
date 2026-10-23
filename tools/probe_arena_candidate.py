"""Evaluate a persisted native GR00T checkpoint on the unchanged Arena task."""
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
from tools.probe_arena_train import verify_hf_checkpoint


def endpoint_ids():
    result = subprocess.run(["runpodctl", "serverless", "list"], capture_output=True,
                            text=True, check=True, timeout=20)
    return {item["id"] for item in json.loads(result.stdout)
            if "robot-lab-arena-candidate" in item.get("name", "")}


def main():
    from huggingface_hub import get_token
    from workers.s3_artifacts import Artifacts, CONFIG, profile_auth, client_from_auth, redact
    from lab.artifact_store import snapshot_file, retain_dependency

    parser = argparse.ArgumentParser()
    parser.add_argument("--training-run", type=Path, required=True)
    args = parser.parse_args()
    training_run = args.training_run.resolve()
    checkpoint_path = training_run / "evidence/checkpoint-hf-reference.json"
    checkpoint_reference = json.loads(checkpoint_path.read_text())
    pins = json.loads((ROOT / "workers/arena/pins.json").read_text())
    token = get_token()
    auth = profile_auth("runpod")
    secrets = dict(auth, hf_token=token)
    client = client_from_auth(CONFIG, auth)
    verified_input = verify_hf_checkpoint(checkpoint_reference, token)
    storage = Artifacts(client, CONFIG)

    run_id = "arena-candidate-" + uuid.uuid4().hex[:12]
    output = ROOT / "artifacts" / run_id
    output.mkdir()
    print(output, flush=True)
    for name in ["tools/probe_arena_candidate.py", "tools/probe_arena_train.py",
                 "tools/probe_vla.py", "workers/serverless/arena_candidate.py",
                 "workers/serverless/native_checkpoint.py", "workers/serverless/s3_artifacts.py",
                 "workers/arena/pins.json"]:
        snapshot_file(ROOT / name, output / "source" / name)
    (output / "dependencies.json").write_text(json.dumps(
        {"training_checkpoint": retain_dependency(training_run),
         "unchanged_baseline_evaluation": retain_dependency(
             ROOT / "artifacts/arena-reference-7441414d0c61")}, indent=2))
    (output / "checkpoint-input-verification.json").write_text(json.dumps(verified_input, indent=2))
    payload = {"pins": pins, "run_id": run_id, "checkpoint_reference": checkpoint_reference}
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
                    if "/arena_candidate/runsync" not in schema["paths"]:
                        raise RuntimeError("missing Arena candidate route")
                    break
                except (OSError, TimeoutError):
                    time.sleep(0.5)
            else:
                raise TimeoutError("Flash readiness deadline")
            envelope = {"input": {"payload": payload, "storage_auth": auth, "model_auth": token}}
            request = urllib.request.Request(base + "/arena_candidate/runsync",
                data=json.dumps(envelope).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=1980) as response:
                result = redact(json.load(response), secrets)
            (output / "response.json").write_text(json.dumps(result, indent=2))
            worker = result.get("output", {})
            if "s3_evidence" in worker:
                verification = storage.download_run(worker["s3_evidence"], output / "evidence")
                (output / "download-verification.json").write_text(json.dumps(verification, indent=2))
            completed = (worker.get("status") == "candidate_evaluation_executed" and
                         worker.get("task_success") is True)
            print("Worker status:", worker.get("status", "missing"),
                  "task success:", worker.get("task_success", "missing"), flush=True)
    except Exception as exc:
        (output / "client-error.json").write_text(json.dumps(
            {"status": "fail", "error": redact(str(exc), secrets),
             "error_type": type(exc).__name__}, indent=2))
        print("Arena candidate attempt failed:", type(exc).__name__, flush=True)
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
