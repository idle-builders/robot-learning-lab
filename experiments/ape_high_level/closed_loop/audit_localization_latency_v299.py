"""V299: replay timestamped Direct Frame RGB-D motion under delayed delivery.

The images and visual deltas came from V298's MuJoCo run. This is an offline
queue/synchronization test, not a new camera model or dynamic obstacle test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_direct_localization_gate_v298 import (OUT as V298, motion_checked)
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v299"
DELAYS_STEPS = (0, 1, 2, 3)
STEP_S = .4


def simulate(rows: list[dict], delay: int) -> dict:
    commands = [np.asarray(row["motion_command_xy_m"], dtype=float)
                for row in rows]
    # Use precisely the same upper-camera deltas that the V298 actor produced
    # from RGB/depth; neither estimated path receives evaluator pose.
    measurements = [None]
    for row in rows[1:]:
        status = row["rgbd_status"]["upper_long"]
        measurements.append(np.asarray(status["translation_m"], dtype=float)
                            if status["status"] == "registered" else None)
    command_pose = np.zeros(2)
    naive_pose = np.zeros(2)
    aligned_capture_pose = np.zeros(2)
    timeline = []
    for current in range(len(rows)):
        command_pose += commands[current]
        naive_pose += commands[current]
        arriving = current - delay
        if arriving >= 1:
            # Naive fusion treats the arriving image's displacement as if it
            # described the *current* command, despite the capture timestamp.
            delta, naive_gate = motion_checked(commands[current],
                                               measurements[arriving])
            naive_pose += delta - commands[current]
            # Correct fusion updates the pose at that image's capture time,
            # then replays every later command up to this decision instant.
            delta, aligned_gate = motion_checked(commands[arriving],
                                                 measurements[arriving])
            aligned_capture_pose += delta
        else:
            naive_gate = aligned_gate = "no_visual_yet"
        aligned_pose = aligned_capture_pose + sum(
            commands[max(1, arriving + 1):current + 1], np.zeros(2))
        truth = np.asarray(rows[current]["evaluator_actual_xyyaw"][:2])
        timeline.append({
            "decision_time_s": rows[current]["capture_time_s"],
            "arriving_capture_step": arriving if arriving >= 0 else None,
            "latest_camera_age_s": None if arriving < 0 else round(
                (current - arriving) * STEP_S, 3),
            "naive_gate": naive_gate, "time_aligned_gate": aligned_gate,
            "actor_pose_xy_m": {
                "command": command_pose.round(5).tolist(),
                "naive_receive_time": naive_pose.round(5).tolist(),
                "capture_time_replay": aligned_pose.round(5).tolist()},
            "evaluator_actual_xy_m": truth.round(5).tolist(),
            "evaluator_pose_error_m": {
                "command": round(float(np.linalg.norm(command_pose - truth)), 5),
                "naive_receive_time": round(float(np.linalg.norm(naive_pose - truth)), 5),
                "capture_time_replay": round(float(np.linalg.norm(aligned_pose - truth)), 5)}})
    return {"delay_steps": delay, "delay_s": delay * STEP_S,
            "timeline": timeline,
            "final_errors_m": timeline[-1]["evaluator_pose_error_m"],
            "max_errors_m": {name: max(t["evaluator_pose_error_m"][name]
                                      for t in timeline)
                             for name in timeline[-1]["evaluator_pose_error_m"]}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prior_path = V298 / "localization-gate-audit.json"
    prior = json.loads(prior_path.read_text())
    cases = []
    for case in prior["heldout_cases"]:
        rows = json.loads((ROOT / case["trace_path"]).read_text())["rows"]
        cases.append({"seed": case["seed"], "kind": case["kind"],
                      "source_trace_sha256": case["trace_sha256"],
                      "delays": [simulate(rows, delay)
                                 for delay in DELAYS_STEPS]})
    summary = {}
    for delay in DELAYS_STEPS:
        samples = [case["delays"][delay] for case in cases]
        summary[str(delay)] = {name: {
            "mean_final_error_m": round(float(np.mean([
                sample["final_errors_m"][name] for sample in samples])), 5),
            "worst_final_error_m": max(
                sample["final_errors_m"][name] for sample in samples),
            "worst_at_any_step_m": max(
                sample["max_errors_m"][name] for sample in samples)}
            for name in samples[0]["final_errors_m"]}
    report = {"scope": "V299 offline delayed-delivery replay of the exact V298 selected Direct Frame heldout RGB-D visual deltas and 0.4s motion commands. Delays 0/0.4/0.8/1.2s are deliberately imposed, not measured camera latency. Compare a naive receive-time update against a capture-time update with command replay; both use the same V298 motion-check/half-weight rule and no true pose. Hidden pose scores after decisions. Visual deltas were measured under synchronous ideal render conditions, so this isolates timestamp logic, not motion blur, moving obstacles, or compute capacity. A time-aligned pose does not make aged obstacle observations safe for motion.",
              "source_sha256": sha(SOURCE), "v298_sha256": sha(prior_path),
              "delay_steps": DELAYS_STEPS, "step_s": STEP_S,
              "cases": cases, "summary": summary}
    (OUT / "localization-latency-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
