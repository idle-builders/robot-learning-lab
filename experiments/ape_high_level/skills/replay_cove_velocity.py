"""Bind the high-level Cove velocity boundary to a saved historical stop rollout.

The historical Direct Frame policy received a zero-twist command at 4 s in
the retained evaluator and continued in the same physics episode. This replay
consumes that outcome; entry health, requests and fault cases are explicit
interface fixtures. It does not run the actor or certify physical motion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cove_velocity_boundary import (CoveVelocityBoundary, EntryObservation,
                                    MotionRequest, PolicyIdentity, StopObservation)


ROOT = Path(__file__).resolve().parents[3]
DESIGN = ROOT / "robot/ape-current-design.json"
ARCHIVED_POINTER = ROOT / "studies/cove50-morphology/datum/rejected-design-pointer.json"
TRANSFER = ROOT / "artifacts/ape-cove50-transfer-20261008"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sources() -> tuple[PolicyIdentity, dict, dict, dict]:
    design = json.loads(DESIGN.read_text())
    report = json.loads((TRANSFER / "report.json").read_text())
    matrix = json.loads((TRANSFER / "varied150-matrix.json").read_text())
    native = json.loads((TRANSFER / "varied150-walk-stop.json").read_text())
    if (design.get("status") != "rejected_no_selected_mechanical_design" or
        design.get("selected_design_id") is not None or
        design["id"] != "cove-50-direct-frame" or
        report["selected_design"]["id"] != design["id"]):
        raise ValueError("Historical design classification changed")
    if report["selected_design"]["pointer_sha256"] != sha(ARCHIVED_POINTER):
        raise ValueError("Archived design pointer changed")
    if report["selected_design"]["cad_generator_sha256"] != sha(ROOT / design["generator"]["path"]):
        raise ValueError("Selected CAD source changed")
    body_path = ROOT / report["body"]["training_xml"]
    body_hash = sha(body_path)
    checkpoint_path = ROOT / report["forward_continuation"]["checkpoint"]
    checkpoint_hash = sha(checkpoint_path)
    if (body_hash != report["body"]["training_xml_sha256"] or
        body_hash != native["body_sha256"] or
        checkpoint_hash != report["forward_continuation"]["checkpoint_sha256"] or
        checkpoint_hash != matrix["checkpoint_sha256"] or
        native["stop_at_s"] != 4.0 or
        next(row for row in matrix["cases"] if row["case"] == "walk-stop")["process_exit_code"] != 0):
        raise ValueError("Historical native policy evidence changed")
    config = (TRANSFER / "forward-env.yaml").read_text()
    decimation = int(config.splitlines()[0].removeprefix("decimation: "))
    period_ms = round(native["physics_dt_s"] * decimation * 1000)
    identity = PolicyIdentity(body_hash, checkpoint_hash,
                              len(native["first_action_world0"]), period_ms)
    sources = {"current_design_pointer_sha256": sha(DESIGN),
               "historical_design_pointer_sha256": sha(ARCHIVED_POINTER),
               "body_sha256": body_hash, "checkpoint_sha256": checkpoint_hash,
               "native_stop_report_sha256": sha(TRANSFER / "varied150-walk-stop.json"),
               "native_matrix_sha256": sha(TRANSFER / "varied150-matrix.json"),
               "evaluator_source_sha256": sha(
                   ROOT / "experiments/compact_morphology/mjlab_ape/evaluate.py"),
               "actor_action_dim": identity.actor_action_dim,
               "control_period_ms": identity.control_period_ms}
    return identity, native, sources, matrix


def run(identity: PolicyIdentity, native: dict, matrix: dict) -> dict:
    rows = []
    for world in native["worlds"]:
        index = world["world"]
        boundary = CoveVelocityBoundary(identity)
        request_id = f"open-floor-world-{index}"
        entry = EntryObservation(0, "cove-50-direct-frame", identity.body_sha256,
                                 True, True, True, True, 0.231)
        request = MotionRequest(request_id, "open_floor_forward", .35, 0, 0,
                                identity.body_sha256, identity.checkpoint_sha256,
                                deadline_ms=8000,
                                runtime_mode="historical_reference_simulation")
        accepted = boundary.submit(request, entry, now_ms=0)
        competing = boundary.submit(MotionRequest(
            "competing", "open_floor_forward", .35, 0, 0,
            identity.body_sha256, identity.checkpoint_sha256, 8000,
            runtime_mode="historical_reference_simulation"), entry, now_ms=1)
        premature_capture = boundary.capture_after_stop(captured_at_ms=3999)
        stop_command = boundary.request_stop(request_id=request_id, now_ms=4000)
        stopped = world["stop"]
        if stopped["command_time_s"] != 4.0 or world["duration_s"] != 8.0:
            raise ValueError("Native stop timing changed")
        stale_feedback = boundary.observe_stop(StopObservation(
            request_id, 3000, 3999, stopped["late_mean_planar_speed_m_s"],
            stopped["late_mean_abs_yaw_speed_rad_s"],
            stopped["late_double_support_fraction"], identity.body_sha256,
            identity.checkpoint_sha256, "historical_native_sim_evaluator"), now_ms=8000)
        observed = boundary.observe_stop(StopObservation(
            request_id, 6000, 8000, stopped["late_mean_planar_speed_m_s"],
            stopped["late_mean_abs_yaw_speed_rad_s"],
            stopped["late_double_support_fraction"], identity.body_sha256,
            identity.checkpoint_sha256, "historical_native_sim_evaluator"), now_ms=8000)
        post_stop_frame_contract = boundary.capture_after_stop(captured_at_ms=8001)
        if (accepted["status"] != "accepted_for_historical_simulation" or
            competing["reason"] != "another_motion_owner_active" or
            premature_capture["status"] != "reject_capture" or
            stop_command["observed_outcome"] != "pending" or
            stale_feedback["status"] != "rejected_feedback" or
            observed["status"] != "stopped_observed_in_simulation" or
            post_stop_frame_contract["status"] != "accept_fresh_stationary_frame_in_simulation"):
            raise AssertionError("Selected-body stop boundary replay changed")
        rows.append({"world": index, "request": accepted, "competing_request": competing,
                     "pre_stop_capture": premature_capture,
                     "stop_receipt": stop_command, "stale_feedback": stale_feedback,
                     "observed_stop": observed,
                     "hypothetical_post_stop_frame_contract": post_stop_frame_contract,
                     "net_displacement_after_stop_command_m": stopped["travel_after_command_m"],
                     "yaw_after_stop_command_rad": stopped["yaw_after_command_rad"],
                     "late_mean_planar_speed_m_s": stopped["late_mean_planar_speed_m_s"],
                     "late_double_support_fraction": stopped["late_double_support_fraction"]})

    gate = CoveVelocityBoundary(identity)
    entry = EntryObservation(0, "cove-50-direct-frame", identity.body_sha256,
                             True, True, True, True, .231)
    selected = gate.submit(MotionRequest(
        "selected", "open_floor_forward", .35, 0, 0,
        identity.body_sha256, identity.checkpoint_sha256, 8000,
        runtime_mode="selected_body_simulation"), entry, now_ms=0)
    unsupported = gate.submit(MotionRequest(
        "turn", "body_turn", 0, 0, .3, identity.body_sha256,
        identity.checkpoint_sha256, 8000,
        runtime_mode="historical_reference_simulation"), entry, now_ms=0)
    physical = gate.submit(MotionRequest(
        "physical", "open_floor_forward", .35, 0, 0,
        identity.body_sha256, identity.checkpoint_sha256, 8000,
        runtime_mode="physical_robot"), entry, now_ms=0)
    stale = gate.submit(MotionRequest(
        "stale", "open_floor_forward", .35, 0, 0,
        identity.body_sha256, identity.checkpoint_sha256, 8000,
        runtime_mode="historical_reference_simulation"), entry, now_ms=200)
    mismatch = gate.submit(MotionRequest(
        "wrong-checkpoint", "open_floor_forward", .35, 0, 0,
        identity.body_sha256, "0" * 64, 8000,
        runtime_mode="historical_reference_simulation"), entry, now_ms=0)
    timeout_gate = CoveVelocityBoundary(identity)
    timeout_gate.submit(MotionRequest(
        "deadline", "open_floor_forward", .35, 0, 0,
        identity.body_sha256, identity.checkpoint_sha256, 1000,
        runtime_mode="historical_reference_simulation"), entry, now_ms=0)
    deadline_stop = timeout_gate.tick(now_ms=1000)
    unresolved_stop = timeout_gate.tick(now_ms=6001)
    if [selected.get("reason"), unsupported.get("reason"),
        physical.get("reason"), stale.get("reason")] != [
        "no_selected_body_for_runtime",
        "skill_not_qualified_for_selected_body", "physical_motion_unqualified",
        "entry_state_stale_or_invalid"]:
        raise AssertionError("Unsupported/stale skill rejection changed")
    if (mismatch.get("reason") != "body_or_checkpoint_mismatch" or
        deadline_stop.get("status") != "stop_command_requested" or
        unresolved_stop.get("status") != "stop_unverified"):
        raise AssertionError("Mismatch/deadline gate changed")
    turn_reports = [json.loads((TRANSFER / f"varied150-turn-{side}.json").read_text())
                    for side in ("left", "right")]
    turn_evidence = {side: {
        "command_rad_s": report["angular_command_rad_s"],
        "target_8s_rad": report["worlds"][0]["target_yaw_change_rad"],
        "observed_yaw_rad": [w["net_yaw_rad"] for w in report["worlds"]],
    } for side, report in zip(("left", "right"), turn_reports)}
    return {"scope": "Historical rejected-body native walk-to-zero-twist outcome plus injected high-level requests/health faults; no current selected body, real robot, camera, physical stop time, or new actor rollout",
            "native_stop_worlds": rows,
            "native_turn_evidence": turn_evidence,
            "unsupported_turn_request": unsupported,
            "rejected_selected_body_request": selected,
            "physical_motion_request": physical,
            "stale_entry_request": stale,
            "checkpoint_mismatch_request": mismatch,
            "injected_deadline_stop": deadline_stop,
            "injected_unverified_stop": unresolved_stop,
            "max_observed_net_displacement_after_stop_command_m": max(
                row["net_displacement_after_stop_command_m"] for row in rows),
            "transfer_report_status": json.loads((TRANSFER / "report.json").read_text())["status"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    identity, native, sources, matrix = load_sources()
    result = {"historical_design": "cove-50-direct-frame", "sources": sources,
              **run(identity, native, matrix)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("native worlds", len(result["native_stop_worlds"]))
    print("largest post-command net displacement m",
          result["max_observed_net_displacement_after_stop_command_m"])
    print("turns", result["native_turn_evidence"])


if __name__ == "__main__":
    main()
