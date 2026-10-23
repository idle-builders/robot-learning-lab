"""Relay V35's independently checked local provider guard to its worker.

The local launchd guard remains the authority that discovers and stops pods.
This relay only carries fresh read-only snapshots for the frozen worker gate.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time

import create_seated_bc_eval_guard_v35_a40_r3 as guard


def copy_file(source: Path, target: str, port: int, identity: Path, known_hosts: Path) -> None:
    common = ["-i", str(identity), "-o", "BatchMode=yes",
              "-o", "StrictHostKeyChecking=accept-new",
              "-o", f"UserKnownHostsFile={known_hosts}"]
    remote_dir = "/root/robot-learning-lab/artifacts/ape-cove50-stand-recovery/seated-bc-eval-v35-a40-r3"
    temporary = f"{remote_dir}/{source.name}.incoming"
    final = f"{remote_dir}/{source.name}"
    subprocess.run(["scp", *common, "-P", str(port), str(source),
                    f"{target}:{temporary}"], check=True, timeout=30)
    subprocess.run(["ssh", *common, "-p", str(port), target,
                    f"mv {temporary} {final}"], check=True, timeout=20)


def sync_once(target: str, port: int, identity: Path) -> None:
    # An attestation is made from a live provider GET, then a later checker
    # heartbeat makes the runner's attested <= checked time ordering true.
    # Carry a new state heartbeat before the slower provider GET. Otherwise a
    # route ending during the GET and two SSH copies can see a >30 s old state.
    guard.check()
    prior_state = guard.read(guard.STATE)
    prior_attestation = guard.read(guard.ATTESTATION)
    if (prior_state["status"] != "running" or prior_state.get("last_error") is not None
            or prior_state.get("pod_id") != prior_attestation["pod_id"]
            or datetime.fromisoformat(prior_state["last_check_at_utc"])
            < datetime.fromisoformat(prior_attestation["attested_at_utc"])):
        raise RuntimeError("V35 pre-attestation provider heartbeat is not ordered")
    known_hosts = guard.ART / "worker-known-hosts-v35"
    copy_file(guard.STATE, target, port, identity, known_hosts)
    guard.attest()
    guard.check()
    state = guard.read(guard.STATE)
    attestation = guard.read(guard.ATTESTATION)
    if (state["status"] != "running" or state.get("last_error") is not None
            or state.get("pod_id") != attestation["pod_id"]
            or datetime.fromisoformat(state["last_check_at_utc"])
            < datetime.fromisoformat(attestation["attested_at_utc"])):
        raise RuntimeError("V35 local guard did not produce ordered live provider evidence")
    # Publish state first: an old attestation is still valid against a newer
    # heartbeat. Publishing a new attestation first could reject a visit.
    copy_file(guard.STATE, target, port, identity, known_hosts)
    copy_file(guard.ATTESTATION, target, port, identity, known_hosts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("sync-once", "loop"))
    parser.add_argument("--target", required=True, help="SSH target for the reviewed V35 provider pod")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--identity", required=True, type=Path)
    args = parser.parse_args()
    if not args.identity.is_file() or not 1 <= args.port <= 65535:
        raise RuntimeError("V35 relay identity or SSH port is invalid")
    try:
        while True:
            sync_once(args.target, args.port, args.identity)
            if args.phase == "sync-once":
                return
            # A complete provider check plus two SSH copies takes roughly 16 s.
            # Keep its slack below the worker's 30 s stale-attestation limit.
            time.sleep(1)
    except Exception:
        try:
            guard.request_stop()
        except Exception:
            pass  # The independent launchd checker continues discovery/stop.
        raise


if __name__ == "__main__":
    main()
