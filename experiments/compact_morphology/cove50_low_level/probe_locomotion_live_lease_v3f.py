"""Offline timing and failure controls for a proposed atomic V3f lease.

Uses synthetic running state derived from the stopped V3e record. No provider
call, remote transport, CUDA step, optimizer or scored seed is performed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile

from locomotion_live_lease_v3f import LeaseFailure, atomic_install, build_lease, check_lease


ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "artifacts/ape-cove50-locomotion-learning-v3-20261014/preflight-v3e/guard"
OUT = ROOT / "artifacts/ape-cove50-locomotion-learning-v3-20261014/preflight-v3f-local-lease"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    att = json.loads((OLD / "fresh-pod-attestation.json").read_text())
    state = json.loads((OLD / "fresh-pod-create-guard.json").read_text())
    # This is deliberately fabricated control state, never a current pod claim.
    epoch = datetime(2026, 10, 14, 10, 29, 40, tzinfo=timezone.utc)
    att["attested_at_utc"] = epoch.isoformat()
    att["hard_deadline_utc"] = (epoch + timedelta(minutes=30)).isoformat()
    state["last_check_at_utc"] = (epoch + timedelta(milliseconds=200)).isoformat()
    state["status"] = "running"
    state["stop_requested"] = False
    state["last_error"] = None
    lease = build_lease(att, state)
    expected = {"packet_sha": att["packet_sha256"],
                "manifest_sha": sha(OLD / "creation-manifest.json"),
                "attempt_sha": sha(OLD / "consumed-create-attempt.json"),
                "guard_source_sha": att["guard_source_sha256"],
                "pod_id": att["pod_id"], "max_age_s": 40,
                "min_remaining_s": 120}
    if expected["manifest_sha"] != att["creation_manifest_sha256"]:
        raise RuntimeError("Old immutable manifest identity changed")
    if expected["attempt_sha"] != att["consumed_attempt_sha256"]:
        raise RuntimeError("Old immutable attempt identity changed")
    with tempfile.TemporaryDirectory(prefix="v3f-local-lease-") as directory:
        target = Path(directory) / "live-lease.json"
        atomic_install(target, lease)
        installed = json.loads(target.read_text())
        if installed != lease:
            raise RuntimeError("Atomic local lease differs after install")
        if target.with_name(target.name + ".incoming").exists():
            raise RuntimeError("Atomic local install left a partial lease")

    def expect(at_s: float, code: str | None = None, candidate: dict = lease) -> dict:
        try:
            details = check_lease(candidate, now=epoch + timedelta(seconds=at_s),
                                  **expected)
        except LeaseFailure as exc:
            if exc.code != code:
                raise RuntimeError(f"Expected {code}, got {exc.code}") from exc
            return {"at_s": at_s, "observed": exc.code, "details": exc.details}
        if code is not None:
            raise RuntimeError(f"Expected {code}, got accepted")
        return {"at_s": at_s, "observed": "accepted", "details": details}

    healthy_delivery = expect(23)
    healthy_monitor = expect(28)
    delayed_delivery = expect(35)
    stale_outage = expect(45, "attestation_stale")
    stopped = json.loads(json.dumps(lease))
    stopped["checked_state"]["status"] = "stopped"
    stopped_case = expect(23, "guard_stopped", stopped)
    wrong_pod = json.loads(json.dumps(lease))
    wrong_pod["attestation"]["pod_id"] = "synthetic-wrong-pod"
    wrong_pod_case = expect(23, "pod_binding", wrong_pod)
    wrong_hash = json.loads(json.dumps(lease))
    wrong_hash["attestation"]["packet_sha256"] = "0" * 64
    wrong_hash_case = expect(23, "packet_binding", wrong_hash)
    report = {
        "schema": "robot-learning-lab.v3f-local-live-lease-control.v1",
        "status": "local_only_prototype_not_frozen_or_independently_reviewed",
        "source_sha256": sha(Path(__file__)),
        "lease_source_sha256": sha(Path(__file__).with_name("locomotion_live_lease_v3f.py")),
        "source_basis": "stopped V3e immutable manifest/attempt hashes with synthetic running timestamps/state",
        "expected_lease_age_s": 40,
        "healthy_delivery": healthy_delivery,
        "healthy_monitor": healthy_monitor,
        "delayed_delivery": delayed_delivery,
        "stale_outage": stale_outage,
        "stopped_guard": stopped_case,
        "wrong_pod": wrong_pod_case,
        "wrong_packet": wrong_hash_case,
        "limits": ["No measured SSH relay latency for a new one-file transport",
                   "No CUDA worker integration or independent review",
                   "No native step, optimizer update, score visit or paid provider action"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "cases": 7,
                      "source_sha256": report["source_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
