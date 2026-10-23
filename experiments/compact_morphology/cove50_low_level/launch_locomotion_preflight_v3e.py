"""Source-bound V3e worker transport. Requires an already reviewed running guard.

Never creates a pod. The launch command runs one unscored diagnostic, relays
provider evidence throughout, retrieves source-matched results and stops the pod.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

import create_locomotion_preflight_guard_v3e as guard

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-locomotion-learning-v3-20261014/preflight-v3e"
BUNDLE = ART / "worker-source.tar.gz"
BUNDLE_MANIFEST = ART / "bundle-manifest.json"
REMOTE_ROOT = "/root/robot-learning-lab"
REMOTE_SOURCE_ART = REMOTE_ROOT + "/artifacts/ape-cove50-locomotion-learning-v3-20261014/preflight-v3e"
REMOTE_GUARD = REMOTE_SOURCE_ART + "/guard"
REMOTE_EVIDENCE = "/workspace/robot-learning-lab/ape-cove50-locomotion-learning-v3-20261014/preflight-v3e"
WORKER = REMOTE_ROOT + "/experiments/compact_morphology/cove50_low_level/run_locomotion_preflight_v3e.py"
BOOTSTRAP = REMOTE_ROOT + "/experiments/compact_morphology/cove50_low_level/bootstrap_locomotion_preflight_v3e.sh"
PYTHONPATH = REMOTE_ROOT + "/experiments/compact_morphology/mjlab_ape/src:" + REMOTE_ROOT
EVENTS = ART / "transport-log.jsonl"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(phase: str, **fields) -> None:
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a") as output:
        output.write(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(),
                                 "phase": phase, **fields}, sort_keys=True) + "\n")


def transport_contract() -> dict:
    packet_sha, manifest_sha = guard.creation_contract()
    bundle = json.loads(BUNDLE_MANIFEST.read_text())
    if (bundle.get("packet_sha256") != packet_sha
            or bundle.get("bundle_sha256") != sha(BUNDLE)
            or guard.read(guard.MANIFEST).get("launcher_source_sha256") != sha(Path(__file__))
            or manifest_sha != sha(guard.MANIFEST)):
        raise RuntimeError("V3e transport or frozen packet identity differs")
    if guard.read(guard.PLAN).get("persistent_evidence_root") != REMOTE_EVIDENCE:
        raise RuntimeError("V3e persistent output differs from frozen packet")
    return {"packet_sha256": packet_sha, "bundle_sha256": bundle["bundle_sha256"],
            "guard_manifest_sha256": manifest_sha,
            "launcher_sha256": sha(Path(__file__))}


def ssh_args(identity: Path) -> list[str]:
    return ["-i", str(identity), "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={ART / 'guard/worker-known-hosts-v3e'}",
            "-o", "ConnectTimeout=15"]


def remote(target: str, port: int, identity: Path, command: str,
           timeout: int = 45) -> str:
    result = subprocess.run(["ssh", *ssh_args(identity), "-p", str(port), target,
                             "bash -lc " + shlex.quote(command)],
                            text=True, capture_output=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"Remote command failed with exit {result.returncode}")
    return result.stdout.strip()


def copy_to(target: str, port: int, identity: Path, source: Path, destination: str) -> None:
    subprocess.run(["scp", *ssh_args(identity), "-P", str(port), str(source),
                    f"{target}:{destination}.incoming"], check=True, timeout=90,
                   stdout=subprocess.DEVNULL)
    remote(target, port, identity, f"mv {shlex.quote(destination)}.incoming {shlex.quote(destination)}")


def copy_from(target: str, port: int, identity: Path, source: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".incoming")
    subprocess.run(["scp", *ssh_args(identity), "-P", str(port),
                    f"{target}:{source}", str(temporary)], check=True, timeout=90,
                   stdout=subprocess.DEVNULL)
    temporary.replace(destination)


def fresh_guard(min_remaining_s: int) -> tuple[dict, float]:
    guard.check()
    binding = guard.live_attestation()
    remaining = (datetime.fromisoformat(binding["hard_deadline_utc"])
                 - datetime.now(timezone.utc)).total_seconds()
    if remaining < min_remaining_s:
        raise RuntimeError(f"V3e stage has only {remaining:.1f}s; requires {min_remaining_s}s")
    return binding, remaining


def relay_once(target: str, port: int, identity: Path, min_remaining_s: int) -> float:
    binding, remaining = fresh_guard(min_remaining_s)
    guard.write(guard.ATTESTATION, binding)
    guard.check()  # Keep the checked heartbeat later than the provider snapshot.
    state = guard.read(guard.STATE)
    if (state.get("status") != "running" or state.get("stop_requested") is not False
            or state.get("last_error") is not None
            or datetime.fromisoformat(state["last_check_at_utc"])
            < datetime.fromisoformat(binding["attested_at_utc"])):
        raise RuntimeError("V3e guard changed during relay")
    copy_to(target, port, identity, guard.STATE, REMOTE_GUARD + "/fresh-pod-create-guard.json")
    copy_to(target, port, identity, guard.ATTEMPT, REMOTE_GUARD + "/consumed-create-attempt.json")
    copy_to(target, port, identity, guard.ATTESTATION, REMOTE_GUARD + "/fresh-pod-attestation.json")
    copy_to(target, port, identity, guard.STATE, REMOTE_EVIDENCE + "/latest-provider-state.json")
    copy_to(target, port, identity, guard.ATTESTATION, REMOTE_EVIDENCE + "/latest-provider-attestation.json")
    record("relay", remaining_s=round(remaining, 1), pod_id=binding["pod_id"],
           attested_at_utc=binding["attested_at_utc"])
    return remaining


def run_stage(target: str, port: int, identity: Path, name: str, command: str,
              start_min_s: int, during_min_s: int) -> None:
    _, remaining = fresh_guard(start_min_s)
    record(name + "_start", remaining_s=round(remaining, 1))
    log = ART / (name + "-ssh.log")
    with log.open("w") as output:
        process = subprocess.Popen(["ssh", *ssh_args(identity), "-p", str(port), target,
                                    "bash -lc " + shlex.quote(command)],
                                   stdout=output, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                relay_once(target, port, identity, during_min_s)
                time.sleep(1)
            if process.returncode:
                raise RuntimeError(f"V3e {name} exited {process.returncode}; inspect logged output")
        except Exception:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
    record(name + "_complete", ssh_log_sha256=sha(log))


def launch(target: str, port: int, identity: Path) -> None:
    if (not re.fullmatch(r"root@[A-Za-z0-9.-]+", target)
            or not 1 <= port <= 65535 or not identity.is_file()):
        raise RuntimeError("Exact SSH target, port and identity are required")
    contract = transport_contract()
    if EVENTS.exists() or (ART / "retrieved").exists():
        raise RuntimeError("V3e launch already attempted; no replacement transaction")
    binding, remaining = fresh_guard(1500)
    record("launch_start", **contract, pod_id=binding["pod_id"],
           ssh_target=target, ssh_port=port, remaining_s=round(remaining, 1))
    try:
        remote(target, port, identity,
               f"test ! -e {REMOTE_ROOT} && mkdir -p {REMOTE_ROOT} {REMOTE_GUARD}")
        copy_to(target, port, identity, BUNDLE, "/root/v3e-worker-source.tar.gz")
        actual_sha = remote(target, port, identity,
                            "sha256sum /root/v3e-worker-source.tar.gz | cut -d ' ' -f 1")
        if actual_sha != contract["bundle_sha256"]:
            raise RuntimeError("Remote worker bundle differs from frozen source")
        remote(target, port, identity,
               f"tar xzf /root/v3e-worker-source.tar.gz -C {REMOTE_ROOT}")
        copy_to(target, port, identity, guard.MANIFEST,
                REMOTE_GUARD + "/creation-manifest.json")
        exact = remote(target, port, identity,
                       f"PYTHONPATH={shlex.quote(PYTHONPATH)} python3.12 {WORKER} contract",
                       timeout=90)
        if json.loads(exact)["packet_sha256"] != contract["packet_sha256"]:
            raise RuntimeError("Remote worker contract differs after extraction")
        record("source_verified", bundle_sha256=actual_sha)
        mount = remote(target, port, identity,
                       "findmnt -n -o TARGET --target /workspace")
        if mount != "/workspace":
            raise RuntimeError("/workspace is not the exact mounted persistent volume")
        remote(target, port, identity,
               f"test ! -e {REMOTE_EVIDENCE} && mkdir -p {REMOTE_EVIDENCE}")
        copy_to(target, port, identity, guard.PLAN,
                REMOTE_EVIDENCE + "/frozen-packet.json")
        copy_to(target, port, identity, guard.MANIFEST,
                REMOTE_EVIDENCE + "/creation-manifest.json")
        copy_to(target, port, identity, guard.ATTEMPT,
                REMOTE_EVIDENCE + "/consumed-create-attempt.json")
        record("persistent_volume_verified", mount=mount,
               evidence_root=REMOTE_EVIDENCE)
        relay_once(target, port, identity, 1200)
        run_stage(target, port, identity, "bootstrap",
                  f"bash {BOOTSTRAP} > {REMOTE_EVIDENCE}/bootstrap.log 2>&1",
                  start_min_s=1200, during_min_s=900)
        relay_once(target, port, identity, 900)
        run_stage(target, port, identity, "cuda-worker",
                  f"PYTHONPATH={shlex.quote(PYTHONPATH)} /root/cove50-v31b-venv/bin/python -u {WORKER} cuda --guard-attestation {REMOTE_GUARD}/fresh-pod-attestation.json > {REMOTE_EVIDENCE}/cuda-worker.log 2>&1",
                  start_min_s=900, during_min_s=120)
        fresh_guard(120)
        retrieved = ART / "retrieved"
        retrieved.mkdir()
        for name in ("cuda-preflight-result.json", "cuda-evaluate-sentinel.json",
                     "cuda-evaluate-sentinel.npz", "bootstrap.log", "cuda-worker.log"):
            copy_from(target, port, identity, REMOTE_EVIDENCE + "/" + name, retrieved / name)
        result = json.loads((retrieved / "cuda-preflight-result.json").read_text())
        actual = result["actual_evaluator_cli"]
        if (result.get("packet_sha256") != contract["packet_sha256"]
                or result.get("persistent_evidence_root") != REMOTE_EVIDENCE
                or result.get("evaluator_lineage") != guard.read(guard.PLAN)["evaluator_lineage"]
                or result.get("native_steps") != 204
                or result.get("optimizer_updates") != 0
                or actual["report_sha256"] != sha(retrieved / "cuda-evaluate-sentinel.json")
                or actual["trace_sha256"] != sha(retrieved / "cuda-evaluate-sentinel.npz")):
            raise RuntimeError("Retrieved V3e result does not match frozen native sources")
        record("retrieval_verified", result_sha256=sha(retrieved / "cuda-preflight-result.json"),
               report_sha256=actual["report_sha256"], trace_sha256=actual["trace_sha256"])
    except Exception as exc:
        record("launch_failed", reason=f"{type(exc).__name__}: {exc}")
        for name in ("bootstrap.log", "cuda-worker.log", "cuda-guard-abort.json"):
            try:
                copy_from(target, port, identity, REMOTE_EVIDENCE + "/" + name,
                          ART / "failure-retrieval" / name)
            except Exception:
                pass
        record("evidence_preserved_on_persistent_volume_pending_separate_recovery",
               evidence_root=REMOTE_EVIDENCE,
               note="If the watchdog stopped SSH before retrieval, do not rerun this packet; review volume recovery separately.")
        raise
    finally:
        try:
            guard.request_stop()
            record("provider_stop_requested_and_confirmed")
        except Exception as exc:
            record("provider_stop_pending", reason=f"{type(exc).__name__}: {exc}")
            # The independent launchd checker continues discovery and stop.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("contract", "launch"))
    parser.add_argument("--target")
    parser.add_argument("--port", type=int)
    parser.add_argument("--identity", type=Path)
    args = parser.parse_args()
    if args.phase == "contract":
        print(json.dumps(transport_contract(), sort_keys=True))
        return
    if not (args.target and args.port and args.identity):
        raise RuntimeError("V3e launch requires exact reviewed SSH endpoint and identity")
    launch(args.target, args.port, args.identity)


if __name__ == "__main__":
    main()
