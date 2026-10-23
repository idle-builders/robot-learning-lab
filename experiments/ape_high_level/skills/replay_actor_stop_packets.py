"""Feed saved native 50 Hz body packets through the actor-side stop boundary.

The zero-twist tick is a replay fixture. This does not dispatch or cancel the
saved policy; it checks what high-level logic may infer from observed packets.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from cove_velocity_boundary import (BodyStopPacket, CoveVelocityBoundary,
                                    EntryObservation, MotionRequest, PolicyIdentity)


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-cove50-effort-envelope"
ARCHIVED_POINTER = ROOT / "studies/cove50-morphology/datum/rejected-design-pointer.json"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
CHECKPOINT = ROOT / "artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt"
BASELINE = ROOT / "artifacts/ape-stop-observability-v1/result.json"
OUTPUT = ROOT / "artifacts/ape-skill-contract-cove50-20261012/body-stop-packet-contract.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def yaw_from_quaternion(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q.T
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def run_case(case: str, world: int, *, imu_loss: bool = False,
             drop_sample_at_s: float | None = None,
             stale_sample_at_s: float | None = None,
             imu_loss_after_s: float | None = None) -> dict:
    plan = json.loads((SOURCE / f"{case}-plan.json").read_text())
    data = np.load(SOURCE / f"{case}.npz")
    identity = PolicyIdentity(sha(BODY), sha(CHECKPOINT), 21, 20)
    if (plan["sources"]["pointer"] != sha(ARCHIVED_POINTER) or
        plan["sources"]["body"] != identity.body_sha256 or
        plan["sources"]["checkpoint"] != identity.checkpoint_sha256):
        raise ValueError("Historical source identity changed")
    stop_s = float(plan["definition"]["stop"])
    stop_ms = round(stop_s * 1000)
    label = f"{case}-world{world}" + ("-imu-loss" if imu_loss else "")
    boundary = CoveVelocityBoundary(identity)
    entry = EntryObservation(0, "cove-50-direct-frame", identity.body_sha256,
                             True, True, True, True, .231)
    request = MotionRequest(label, "open_floor_forward", .35, 0, 0,
                            identity.body_sha256, identity.checkpoint_sha256, 8000,
                            runtime_mode="historical_reference_simulation")
    accepted = boundary.submit(request, entry, now_ms=0)
    if accepted["status"] != "accepted_for_historical_simulation":
        raise AssertionError(accepted)
    receipt = boundary.request_stop(request_id=label, now_ms=stop_ms,
                                    cause="replay_aligned_to_native_zero_twist")
    if receipt["status"] != "stop_command_requested":
        raise AssertionError(receipt)
    premature_capture = boundary.capture_after_stop(captured_at_ms=stop_ms + 20)
    if premature_capture["status"] != "reject_capture":
        raise AssertionError(premature_capture)
    time = data["time_s"].astype(float)
    q = data["root_pose"][:, world, 3:7].astype(float)
    yaw = yaw_from_quaternion(q)
    imu_rate = np.abs(np.diff(yaw, prepend=yaw[0])) / .02
    joints = data["joint_pos"][:, world].astype(float)
    encoder_rate = np.mean(np.abs(np.diff(joints, axis=0, prepend=joints[:1])), axis=1) / .02
    sole_force = np.linalg.norm(data["sole_contact_force"][:, world].astype(float), axis=2)
    events = []
    first_quiet_s = None
    first_capture = None
    revoked_capture = None
    for i, sample_s in enumerate(time):
        if sample_s <= stop_s:
            continue
        if drop_sample_at_s is not None and np.isclose(sample_s, drop_sample_at_s):
            events.append({"at_s": round(float(sample_s), 3), "status": "packet_missing"})
            continue
        at_ms = round(float(sample_s) * 1000)
        packet = BodyStopPacket(
            request_id=label, sampled_at_ms=at_ms,
            body_sha256=identity.body_sha256,
            checkpoint_sha256=identity.checkpoint_sha256,
            imu_yaw_rate_rad_s=float(imu_rate[i]),
            mean_abs_leg_encoder_velocity_rad_s=float(encoder_rate[i]),
            left_sole_force_n=float(sole_force[i, 0]),
            right_sole_force_n=float(sole_force[i, 1]),
            imu_healthy=(not imu_loss or sample_s < stop_s + .5) and
            (imu_loss_after_s is None or sample_s < imu_loss_after_s),
            joints_healthy=True, sole_forces_healthy=True)
        reply = boundary.observe_body_packet(
            packet, now_ms=at_ms + 120 if stale_sample_at_s is not None and
            np.isclose(sample_s, stale_sample_at_s) else at_ms)
        if reply["status"] not in ("body_settling", "body_feedback_unhealthy", "body_quiet_provisional"):
            events.append({"at_s": round(float(sample_s), 3), "status": reply["status"]})
        if reply["status"] == "body_quiet_provisional" and first_quiet_s is None:
            first_quiet_s = round(float(sample_s), 3)
            first_capture = boundary.capture_after_stop(captured_at_ms=at_ms + 20)
            events.append({"at_s": first_quiet_s, "status": reply["status"],
                           "capture": first_capture})
        if (first_quiet_s is not None and revoked_capture is None and
            reply["status"] == "body_feedback_unhealthy"):
            revoked_capture = boundary.capture_after_stop(captured_at_ms=at_ms + 20)
            events.append({"at_s": round(float(sample_s), 3),
                           "status": "body_quiet_revoked_on_health_loss",
                           "capture": revoked_capture})
    if boundary.active_request_id != label:
        raise AssertionError("Body-quiet packet released motion ownership")
    if imu_loss and first_quiet_s is not None:
        raise AssertionError("Unhealthy IMU licensed a post-stop capture")
    if first_quiet_s is not None and first_capture["status"] != "capture_allowed_pose_unverified":
        raise AssertionError(first_capture)
    if imu_loss_after_s is not None and (revoked_capture is None or
                                         revoked_capture["status"] != "reject_capture"):
        raise AssertionError("Late IMU loss did not revoke the capture state")
    return {"case": case, "world": world, "fixture": "imu_loss" if imu_loss else
            ("one_packet_gap" if drop_sample_at_s is not None else
             ("one_stale_packet" if stale_sample_at_s is not None else
              ("late_imu_loss" if imu_loss_after_s is not None else "healthy"))),
            "stop_request_s": stop_s, "first_body_quiet_s": first_quiet_s,
            "premature_capture": premature_capture,
            "capture_after_body_quiet": first_capture,
            "capture_after_revocation": revoked_capture,
            "final_boundary_state": boundary.state,
            "motion_owner_held": boundary.active_request_id == label,
            "metric_pose_valid": False,
            "events": events}


def main() -> None:
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("Revisit historical body classification")
    baseline = json.loads(BASELINE.read_text())
    expected = {(row["case"], row["world"]): row["gates"]["fused"]["first_accept_at_s"]
                for row in baseline["rows"]}
    rows = [run_case(case, world)
            for case in ("forward-stop-early", "forward-stop-late") for world in range(4)]
    for row in rows:
        if row["first_body_quiet_s"] != expected[(row["case"], row["world"])]:
            raise AssertionError("Skill boundary disagrees with sensor ablation")
    faults = [run_case("forward-stop-early", 0, imu_loss=True),
              run_case("forward-stop-early", 0, drop_sample_at_s=3.9),
              run_case("forward-stop-early", 0, stale_sample_at_s=3.9),
              run_case("forward-stop-early", 0, imu_loss_after_s=4.2)]
    result = {"schema": "robot-learning-lab.ape-actor-stop-boundary.v1",
              "classification": "historical rejected-body native-packet replay with interface fixtures",
              "source_body_sha256": sha(BODY), "source_checkpoint_sha256": sha(CHECKPOINT),
              "historical_design_pointer_sha256": sha(ARCHIVED_POINTER),
              "current_design_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "source_baseline_sha256": sha(BASELINE),
              "source_boundary_sha256": sha(ROOT / "experiments/ape_high_level/skills/cove_velocity_boundary.py"),
              "analysis_sha256": sha(Path(__file__)),
              "native_traces": {case: sha(SOURCE / f"{case}.npz") for case in
                                ("forward-stop-early", "forward-stop-late")},
              "healthy": rows, "faults": faults,
              "summary": {"healthy_body_quiet": sum(row["first_body_quiet_s"] is not None for row in rows),
                          "healthy_motion_owner_held": sum(row["motion_owner_held"] for row in rows),
                          "imu_loss_body_quiet": faults[0]["first_body_quiet_s"]},
              "limits": ["The stop request is aligned after the fact to the saved zero-twist tick; no asynchronous policy dispatch or cancellation is performed.",
                         "Idealized yaw/encoder/sole-force samples come from a historical rejected body simulation; no physical sensor noise or calibration is included.",
                         "Body quiet permits a motion-labelled camera capture only; metric pose and true physical stopping remain unverified."]}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
