"""Replay timestamped Cove-50 stop feedback from retained native actor traces.

The actor and physics already ran in the saved MJLab episodes. This script
measures their stop path and feeds observed 50 Hz windows to the high-level
boundary; it does not create a new actor rollout or a physical stop claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from cove_velocity_boundary import (CoveVelocityBoundary, EntryObservation,
                                    MotionRequest, PolicyIdentity, StopObservation)


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-cove50-effort-envelope"
OUTPUT = ROOT / "artifacts/ape-skill-contract-cove50-20261012/historical-stop-trace.json"
ARCHIVED_POINTER = ROOT / "studies/cove50-morphology/datum/rejected-design-pointer.json"
CURRENT_POINTER = ROOT / "robot/ape-current-design.json"
CASES = ("forward-stop-early", "forward-stop-late")
WINDOW_FRAMES = 20  # 400 ms of 50 Hz feedback; an interface trial setting.


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def yaw_from_quaternion(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q.T
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def load_case(name: str) -> tuple[dict, dict, np.lib.npyio.NpzFile]:
    plan_path = SOURCE / f"{name}-plan.json"
    report_path = SOURCE / f"{name}.json"
    trace_path = SOURCE / f"{name}.npz"
    plan = json.loads(plan_path.read_text())
    report = json.loads(report_path.read_text())
    data = np.load(trace_path)
    sources = plan["sources"]
    if (sources["pointer"] != sha(ARCHIVED_POINTER) or
        sources["body"] != sha(ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml") or
        sources["checkpoint"] != sha(ROOT / "artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt") or
        report["body_sha256"] != sources["body"] or
        report["stop_at_s"] != plan["definition"]["stop"] or
        report["seed"] != plan["definition"]["seed"] or
        data["active"].shape[1] != len(report["worlds"]) or
        len(report["first_action_world0"]) != 21 or
        not np.isclose(float(data["step_dt_s"]), 0.02) or
        data["time_s"][-1] != 8.0):
        raise ValueError(f"Historical native evidence changed: {name}")
    return plan, report, data


def replay_world(name: str, plan: dict, report: dict, data: np.lib.npyio.NpzFile,
                 world: int) -> dict:
    dt = float(data["step_dt_s"])
    time = data["time_s"]
    stop_s = float(plan["definition"]["stop"])
    stop_index = int(np.argmin(np.abs(time - stop_s)))
    if not np.isclose(time[stop_index], stop_s):
        raise ValueError("Stop command falls between actor ticks")
    xy = data["root_pose"][:, world, :2].astype(float)
    yaw = yaw_from_quaternion(data["root_pose"][:, world, 3:7].astype(float))
    speed = np.linalg.norm(data["root_velocity"][:, world, :2], axis=1)
    yaw_speed = np.abs(np.diff(yaw, prepend=yaw[0])) / dt
    double_support = np.all(data["sole_found"][:, world] > 0, axis=1)
    if not np.all(data["active"][:, world]):
        raise ValueError("Stop trace has an early reset or inactive world")

    identity = PolicyIdentity(plan["sources"]["body"], plan["sources"]["checkpoint"], 21, 20)
    boundary = CoveVelocityBoundary(identity)
    request_id = f"{name}-world-{world}"
    # The evaluator has no timestamped health message at t=0. Entry health is
    # an explicit interface fixture; the stop dynamics below are native trace.
    entry = EntryObservation(0, "cove-50-direct-frame", identity.body_sha256,
                             True, True, True, True, 0.231)
    request = MotionRequest(request_id, "open_floor_forward", 0.35, 0.0, 0.0,
                            identity.body_sha256, identity.checkpoint_sha256,
                            deadline_ms=int(stop_s * 1000) if name.endswith("late") else 8000,
                            runtime_mode="historical_reference_simulation")
    accepted = boundary.submit(request, entry, now_ms=0)
    if accepted["status"] != "accepted_for_historical_simulation":
        raise AssertionError(accepted)
    # The saved native evaluator switches twist to zero at this tick. The
    # high-level cause is an injected owner interrupt or deadline fixture.
    stop_receipt = (boundary.tick(now_ms=int(stop_s * 1000)) if name.endswith("late")
                    else boundary.request_stop(request_id=request_id,
                                               now_ms=int(stop_s * 1000),
                                               cause="owner_interrupt"))
    if stop_receipt["status"] != "stop_command_requested":
        raise AssertionError(stop_receipt)
    premature_frame = boundary.capture_after_stop(captured_at_ms=int(stop_s * 1000) + 20)
    first_observed = None
    last_pending = None
    windows_checked = 0
    for end in range(stop_index + WINDOW_FRAMES, len(time)):
        start = end - WINDOW_FRAMES + 1
        sample = StopObservation(
            request_id=request_id,
            window_start_ms=round(float(time[start]) * 1000),
            window_end_ms=round(float(time[end]) * 1000),
            mean_planar_speed_m_s=float(speed[start:end + 1].mean()),
            mean_abs_yaw_speed_rad_s=float(yaw_speed[start:end + 1].mean()),
            double_support_fraction=float(double_support[start:end + 1].mean()),
            body_sha256=identity.body_sha256,
            checkpoint_sha256=identity.checkpoint_sha256,
            source="historical_native_sim_evaluator",
        )
        reply = boundary.observe_stop(sample, now_ms=sample.window_end_ms)
        windows_checked += 1
        if reply["status"] == "stopped_observed_in_simulation":
            first_observed = {"at_s": round(float(time[end]), 3),
                              "window_start_s": round(float(time[start]), 3),
                              "mean_planar_speed_m_s": sample.mean_planar_speed_m_s,
                              "mean_abs_yaw_speed_rad_s": sample.mean_abs_yaw_speed_rad_s,
                              "double_support_fraction": sample.double_support_fraction,
                              "feedback": reply}
            break
        if reply["status"] != "stopping":
            raise AssertionError(reply)
        last_pending = {"at_s": round(float(time[end]), 3),
                        "mean_planar_speed_m_s": sample.mean_planar_speed_m_s,
                        "mean_abs_yaw_speed_rad_s": sample.mean_abs_yaw_speed_rad_s,
                        "double_support_fraction": sample.double_support_fraction}

    stop_end = (int(np.argmin(np.abs(time - first_observed["at_s"])))
                if first_observed else len(time) - 1)
    path_to_observed = float(np.linalg.norm(np.diff(xy[stop_index:stop_end + 1], axis=0), axis=1).sum())
    path_to_end = float(np.linalg.norm(np.diff(xy[stop_index:], axis=0), axis=1).sum())
    radial = np.linalg.norm(xy[stop_index:] - xy[stop_index], axis=1)
    clip = np.abs(data["actuator_force"][:, world]) >= (
        0.995 * data["rated_effort_limits_nm"])
    capture = (boundary.capture_after_stop(captured_at_ms=round(first_observed["at_s"] * 1000) + 20)
               if first_observed else None)
    after_observed = slice(stop_end, None)
    remained_stationary = bool(
        first_observed and
        np.max(speed[after_observed]) < 0.01 and
        np.max(yaw_speed[after_observed]) < 0.01 and
        np.all(double_support[after_observed]))
    native_stop = report["worlds"][world]["stop"]
    final_net = float(np.linalg.norm(xy[-1] - xy[stop_index]))
    if abs(final_net - native_stop["travel_after_command_m"]) > 0.002:
        raise ValueError("Native report and trace stop displacement disagree")
    return {
        "case": name, "world": world, "seed": plan["definition"]["seed"],
        "stop_command_s": stop_s, "cause_fixture": stop_receipt["cause"],
        "command_receipt": stop_receipt, "premature_capture": premature_frame,
        "last_pending_window": last_pending, "first_observed_stop": first_observed,
        "windows_checked": windows_checked,
        "command_to_observed_stop_s": (round(first_observed["at_s"] - stop_s, 3)
                                       if first_observed else None),
        "planar_path_to_observed_stop_m": round(path_to_observed, 5),
        "planar_path_to_8s_m": round(path_to_end, 5),
        "max_radial_displacement_after_command_m": round(float(radial.max()), 5),
        "final_net_displacement_m": round(final_net, 5),
        "post_command_rated_leg_fraction": round(float(clip[stop_index + 1:].any(axis=1).mean()), 5),
        "remained_stationary_and_double_support_to_8s": remained_stationary,
        "post_observed_frame_contract": capture,
    }


def main() -> None:
    current = json.loads(CURRENT_POINTER.read_text())
    if (current.get("status") != "rejected_no_selected_mechanical_design" or
        current.get("selected_design_id") is not None):
        raise ValueError("Revisit historical-body replay classification")
    rows = []
    sources = {}
    for name in CASES:
        plan, report, data = load_case(name)
        sources[name] = {"plan_sha256": sha(SOURCE / f"{name}-plan.json"),
                         "report_sha256": sha(SOURCE / f"{name}.json"),
                         "native_trace_sha256": sha(SOURCE / f"{name}.npz")}
        rows.extend(replay_world(name, plan, report, data, world)
                    for world in range(len(report["worlds"])))
    if not all(row["first_observed_stop"] and row["remained_stationary_and_double_support_to_8s"]
               for row in rows):
        raise AssertionError("A native world did not produce lasting observed stop feedback")
    result = {
        "schema": "robot-learning-lab.cove50-stop-trace-contract.v1",
        "historical_design": "cove-50-direct-frame",
        "body_status": "historical_rejected_direct_frame",
        "source_pointer_sha256": sha(ARCHIVED_POINTER),
        "current_design_pointer_sha256": sha(CURRENT_POINTER),
        "source_body_sha256": sha(ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"),
        "source_checkpoint_sha256": sha(ROOT / "artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt"),
        "analysis_source_sha256": sha(Path(__file__)),
        "sources": sources,
        "method": "Retrospective playback of saved fresh-process native 50 Hz actor/physics traces. Entry health and owner interrupt/deadline causes are injected interface fixtures aligned with the actual zero-twist command tick; subsequent motion, support and effort are native observations.",
        "stop_gate": {"window_ms": WINDOW_FRAMES * 20,
                      "mean_planar_speed_m_s_at_most": 0.01,
                      "mean_abs_yaw_speed_rad_s_at_most": 0.01,
                      "double_support_fraction_at_least": 0.99},
        "summary": {"worlds": len(rows),
                    "observed_stops": sum(bool(row["first_observed_stop"]) for row in rows),
                    "observed_stop_delay_s_range": [min(row["command_to_observed_stop_s"] for row in rows),
                                                     max(row["command_to_observed_stop_s"] for row in rows)],
                    "path_to_observed_stop_m_range": [min(row["planar_path_to_observed_stop_m"] for row in rows),
                                                       max(row["planar_path_to_observed_stop_m"] for row in rows)],
                    "path_to_8s_m_range": [min(row["planar_path_to_8s_m"] for row in rows),
                                           max(row["planar_path_to_8s_m"] for row in rows)]},
        "rows": rows,
        "limits": [
            "The owner interruption and deadline are replay fixtures, not an asynchronous driver that caused the native command switch.",
            "This actor has near-continuous rated leg duty during gait and is not qualified for autonomous walking or physical stopping.",
            "No camera exposure, real hardware latency, obstacle distance or deceleration guarantee is measured.",
            "Both stop times use the same seed; these eight worlds diagnose timing variation, not broad statistical robustness.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    matrix = {
        "schema": "robot-learning-lab.cove50-stop-summary-matrix.v1",
        "note": "One row per native world. Owner interruption and deadline causes are interface fixtures aligned with the evaluator's real zero-twist switch. Both cases share seed 5501.",
        "cases": [{"case": row["case"], "world": row["world"],
                   "cause": row["cause_fixture"],
                   "stop_delay_s": row["command_to_observed_stop_s"],
                   "path_to_stop_m": row["planar_path_to_observed_stop_m"],
                   "path_to_8s_m": row["planar_path_to_8s_m"],
                   "net_displacement_m": row["final_net_displacement_m"],
                   "rated_leg_after_command": row["post_command_rated_leg_fraction"],
                   "remained_still": row["remained_stationary_and_double_support_to_8s"],
                   "report": str(OUTPUT.relative_to(ROOT))} for row in rows],
    }
    (OUTPUT.parent / "stop-summary-matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
