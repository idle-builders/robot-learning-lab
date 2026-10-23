"""Local prototype for one atomic live-provider lease in the next V3 preflight.

This module cannot create or start a pod. It does not authorize a V3f launch.
The live caller must obtain the attestation from the provider-backed guard,
copy the resulting single JSON file, and retain an independent local stop timer.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path


class LeaseFailure(RuntimeError):
    def __init__(self, code: str, details: dict):
        self.code = code
        self.details = details
        super().__init__(code)


def _time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            raise ValueError("timezone missing")
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError) as exc:
        raise LeaseFailure("invalid_timestamp", {"field_value_type": type(value).__name__}) from exc


def _fail(code: str, **details) -> None:
    raise LeaseFailure(code, details)


def build_lease(attestation: dict, state: dict) -> dict:
    """Bind one checked state and one provider attestation before transport."""
    stamped = _time(attestation.get("attested_at_utc"))
    checked = _time(state.get("last_check_at_utc"))
    if checked < stamped:
        _fail("state_precedes_attestation", state_at=checked.isoformat(),
              attested_at=stamped.isoformat())
    if state.get("pod_ids") != [attestation.get("pod_id")]:
        _fail("pod_binding_mismatch")
    if state.get("plan_sha256") != attestation.get("packet_sha256"):
        _fail("packet_binding_mismatch")
    if state.get("creation_manifest_sha256") != attestation.get("creation_manifest_sha256"):
        _fail("manifest_binding_mismatch")
    return {"schema": "robot-learning-lab.locomotion-v3f-live-lease.v1",
            "attestation": attestation, "checked_state": state}


def check_lease(lease: dict, *, packet_sha: str, manifest_sha: str,
                attempt_sha: str, guard_source_sha: str, pod_id: str,
                now: datetime, max_age_s: float, min_remaining_s: float,
                max_rate_usd_h: float = 0.89, max_cap_s: float = 2400) -> dict:
    """Return non-secret timing facts or a clause-specific fail-closed reason."""
    if lease.get("schema") != "robot-learning-lab.locomotion-v3f-live-lease.v1":
        _fail("lease_schema")
    att = lease.get("attestation")
    state = lease.get("checked_state")
    if not isinstance(att, dict) or not isinstance(state, dict):
        _fail("lease_content")
    now = now.astimezone(timezone.utc)
    stamped = _time(att.get("attested_at_utc"))
    checked = _time(state.get("last_check_at_utc"))
    deadline = _time(att.get("hard_deadline_utc"))
    age_att = (now - stamped).total_seconds()
    age_state = (now - checked).total_seconds()
    remaining = (deadline - now).total_seconds()
    facts = {"attestation_age_s": round(age_att, 3),
             "state_age_s": round(age_state, 3),
             "deadline_remaining_s": round(remaining, 3),
             "max_age_s": max_age_s}
    if checked < stamped:
        _fail("state_precedes_attestation", **facts)
    if (att.get("packet_sha256") != packet_sha
            or state.get("plan_sha256") != packet_sha):
        _fail("packet_binding", **facts)
    if (att.get("creation_manifest_sha256") != manifest_sha
            or state.get("creation_manifest_sha256") != manifest_sha):
        _fail("manifest_binding", **facts)
    if att.get("consumed_attempt_sha256") != attempt_sha:
        _fail("attempt_binding", **facts)
    if (att.get("guard_source_sha256") != guard_source_sha
            or state.get("guard_source_sha256") != guard_source_sha):
        _fail("guard_source_binding", **facts)
    if att.get("pod_id") != pod_id or state.get("pod_ids") != [pod_id]:
        _fail("pod_binding", **facts)
    if state.get("status") != "running" or state.get("stop_requested") is not False:
        _fail("guard_stopped", **facts)
    if state.get("last_error") is not None:
        _fail("guard_error", **facts)
    if (att.get("provider_desired_status") != "RUNNING"
            or att.get("provider_runtime_status") != "running"
            or att.get("watchdog_status") != "healthy"):
        _fail("provider_unhealthy", **facts)
    if att.get("gpu_id") != "NVIDIA GeForce RTX 4090" or att.get("gpu_count") != 1:
        _fail("gpu_mismatch", **facts)
    rate = att.get("listed_rate_usd_per_hour")
    cap = att.get("cap_s")
    if (not isinstance(rate, (int, float)) or not math.isfinite(rate)
            or rate < 0 or rate > max_rate_usd_h
            or not isinstance(cap, (int, float)) or not math.isfinite(cap)
            or cap < 0 or cap > max_cap_s):
        _fail("spend_bound", **facts)
    if not 0 <= age_att <= max_age_s:
        _fail("attestation_stale", **facts)
    if not 0 <= age_state <= max_age_s:
        _fail("state_stale", **facts)
    if remaining < min_remaining_s:
        _fail("deadline_reserve", **facts)
    return facts


def atomic_install(path: Path, lease: dict) -> None:
    """Install a single complete lease after transfer; no mixed file generation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incoming")
    with temporary.open("x") as output:
        json.dump(lease, output, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
