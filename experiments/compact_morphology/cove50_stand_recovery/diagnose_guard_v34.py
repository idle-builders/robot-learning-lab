"""Read-only sampler for the V34 worker's live attestation predicate.

This is operational diagnosis. It does not run or score an original route.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time


ART = Path("/root/robot-learning-lab/artifacts/ape-cove50-stand-recovery/seated-trigger-v34")


def stamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def main() -> None:
    plan = json.loads((ART / "plan.json").read_text())
    request = stamp("2026-10-14T06:18:15.396848+00:00")
    bound = min(
        request + timedelta(seconds=plan["max_pod_wall_s"] - plan["stop_reserve_s"]),
        request + timedelta(seconds=plan["gpu_listed_spend_cap_usd"] * 3600 /
                            plan["gpu_listed_rate_usd_per_hour"] - plan["stop_reserve_s"]),
    )
    seen = set()
    for _ in range(300):
        now = datetime.now(timezone.utc)
        attestation = json.loads((ART / "fresh-pod-attestation.json").read_text())
        state = json.loads((ART / "fresh-pod-create-guard.json").read_text())
        attested = stamp(attestation["attested_at_utc"])
        checked = stamp(state["last_check_at_utc"])
        deadline = stamp(attestation["hard_deadline_utc"])
        failures = tuple(name for name, valid in {
            "schema": attestation["schema"] == "robot-learning-lab.cove50-seated-trigger.v34.guard-attestation"
                      and state["schema"] == "robot-learning-lab.cove50-seated-trigger.v34.guard-state",
            "plan": attestation["plan_sha256"] == state["plan_sha256"],
            "request": stamp(attestation["local_start_request_at_utc"]) == request,
            "pod": state["pod_id"] == attestation["pod_id"],
            "provider": attestation["provider_desired_status"] == "RUNNING"
                        and attestation["provider_runtime_status"] == "running"
                        and state["provider_desired_status"] == "RUNNING"
                        and state["provider_runtime_status"] == "running",
            "state": state["status"] == "running",
            "gpu": attestation["gpu_id"] == "NVIDIA GeForce RTX 4090",
            "rate": attestation["listed_rate_usd_per_hour"] == plan["gpu_listed_rate_usd_per_hour"],
            "clock_order": request <= attested <= checked <= now,
            "fresh": (now - checked).total_seconds() <= 30,
            "deadline": deadline <= bound and deadline > now,
        }.items() if not valid)
        if failures and failures not in seen:
            print(json.dumps({
                "at": now.isoformat(), "failures": failures,
                "attested": attested.isoformat(), "checked": checked.isoformat(),
                "checked_age_s": round((now - checked).total_seconds(), 3),
                "state_status": state["status"],
                "state_provider": [state["provider_desired_status"], state["provider_runtime_status"]],
                "attestation_provider": [attestation["provider_desired_status"],
                                         attestation["provider_runtime_status"]],
            }), flush=True)
            seen.add(failures)
        time.sleep(0.2)
    print(json.dumps({"completed_samples": 300, "distinct_failure_patterns": len(seen)}), flush=True)


if __name__ == "__main__":
    main()
