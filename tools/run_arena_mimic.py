"""Launch the one-trial published-data Arena Mimic reference run."""
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


def endpoints():
    data = json.loads(subprocess.check_output(["runpodctl", "serverless", "list"], text=True))
    return {x["id"] for x in data if "robot-lab-arena-mimic" in x.get("name", "")}


def main():
    from huggingface_hub import get_token
    from workers.s3_artifacts import Artifacts, CONFIG, profile_auth, client_from_auth, redact
    from lab.artifact_store import snapshot_file

    token = get_token()
    if not token:
        raise SystemExit("Hugging Face login required")
    auth = profile_auth("runpod")
    secrets = dict(auth, hf_token=token)
    storage = Artifacts(client_from_auth(CONFIG, auth), CONFIG)
    run_id = "arena-mimic-" + uuid.uuid4().hex[:12]
    output = ROOT / "artifacts" / run_id
    output.mkdir()
    print(output, flush=True)
    for name in ["tools/run_arena_mimic.py", "workers/serverless/arena_mimic.py",
                 "workers/serverless/s3_artifacts.py", "workers/arena/pins.json"]:
        snapshot_file(ROOT / name, output / "source" / name)
    pins = json.loads((ROOT / "workers/arena/pins.json").read_text())
    payload = {"run_id": run_id, "pins": pins}
    (output / "request.json").write_text(json.dumps(payload, indent=2))
    before = endpoints()
    proc = None
    passed = False
    try:
        with socket.socket() as check:
            check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            check.bind(("127.0.0.1", 8907))
        with (output / "flash.log").open("x") as log:
            proc = subprocess.Popen([str(ROOT / "tools/runpod-flash"), "dev", "--port", "8907",
                                     "--no-reload"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            base = None
            for _ in range(120):
                if proc.poll() is not None:
                    raise RuntimeError("Flash exited before readiness")
                matches = re.findall(r"flash dev\s+localhost:(\d+)",
                                     (output / "flash.log").read_text(errors="replace"))
                if matches:
                    base = "http://127.0.0.1:" + matches[-1]
                    try:
                        with urllib.request.urlopen(base + "/openapi.json", timeout=1) as response:
                            if "/arena_mimic/runsync" in json.load(response)["paths"]:
                                break
                    except OSError:
                        pass
                time.sleep(.5)
            else:
                raise TimeoutError("Flash readiness deadline")
            request = urllib.request.Request(base + "/arena_mimic/runsync",
                data=json.dumps({"input": {"payload": payload, "storage_auth": auth,
                                            "model_auth": token}}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=1980) as response:
                result = redact(json.load(response), secrets)
            (output / "response.json").write_text(json.dumps(result, indent=2))
            worker = result.get("output", {})
            if worker.get("s3_evidence"):
                verification = storage.download_run(worker["s3_evidence"], output / "evidence")
                (output / "download-verification.json").write_text(json.dumps(verification, indent=2))
            passed = worker.get("status") == "published_mimic_trial_replayed"
            print("Worker status:", worker.get("status"), flush=True)
    except Exception as exc:
        (output / "client-error.json").write_text(json.dumps(
            {"status": "fail", "error_type": type(exc).__name__,
             "error": redact(str(exc), secrets)}, indent=2))
    finally:
        if proc is not None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=25)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=10)
        deleted = []
        for endpoint in endpoints() - before:
            result = subprocess.run(["runpodctl", "serverless", "delete", endpoint], timeout=30)
            deleted.append({"endpoint": endpoint, "exit_code": result.returncode})
        (output / "cleanup.json").write_text(json.dumps(
            {"deletions": deleted, "remaining_owned_endpoints": sorted(endpoints() - before)}, indent=2))
        if (output / "flash.log").exists():
            (output / "flash.log").write_text(redact((output / "flash.log").read_text(errors="replace"), secrets))
        finalize(output)
    if not passed:
        raise SystemExit("Mimic reference run failed; see " + str(output))


if __name__ == "__main__":
    main()

