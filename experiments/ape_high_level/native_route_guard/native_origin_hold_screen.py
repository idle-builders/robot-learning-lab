"""V30: native zero-action origin wait against the V27 late side entry.

This is a feasibility and observability screen for a third high-level action.
It does not choose the action online or prove the person can be recognized.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

import held_body_intrusion_trial as late
import held_monitor_probe as monitor
import kitchen_native as kitchen
import kitchen_trial as trial
import native_early_hold_screen as early


OUT = kitchen.OUT / "v30-native-origin-hold"
SOURCE = early.TRACE_ROOT / "passive-zero.npz"
REPORT_SOURCE = SOURCE.with_suffix(".json")
SENSOR_END_S = 5.02


def overlay(frame: np.ndarray, case: str, mode: str,
            t: float, gap: float, unused_stop_s: float) -> np.ndarray:
    panel = np.full((83, frame.shape[1], 3), (29, 32, 37), dtype=np.uint8)
    for line, y, color in ((
        (f"RoboCasa layout 3 + box stool | native zero action | {case} | t={t:.2f}s",
         24, (227, 233, 238)),
        (f"zero action from t=0 | full-body person gap={gap:+.3f}m",
         51, (77, 96, 226) if gap < .05 else (227, 233, 238)),
        ("Historical Direct Frame; kinematic scene replay, no live collision response",
         76, (167, 184, 196)),
    )):
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .52, color, 1, cv2.LINE_AA)
    return np.concatenate([frame, panel], axis=0)


def sensor_trace(model: mujoco.MjModel, saved, world: int,
                 case: str, mode: str, bearings: np.ndarray,
                 rate_hz: int) -> dict:
    data = mujoco.MjData(model)
    rows = []
    for tick in range(round((SENSOR_END_S-.02)*rate_hz)+1):
        t = .02+tick/rate_hz
        model.body_pos[model.body("approaching_person").id, :2] = \
            monitor.hazard_xy(case, t)
        model.body_pos[model.body("low_pet").id, :2] = (10., -1.)
        index = min(len(saved["time_s"])-1, max(0, round(t/.02)-1))
        pose = early.direct_pose(model, data, saved, world, index)
        packet, person_hits = monitor.scan(model, data, pose, bearings, t)
        rows.append({"time_s": packet["captured_at_s"],
                     "actor_packet": packet,
                     "ideal_root_xy_m": pose[:2].round(5).tolist(),
                     "evaluator_person_ray_hits": person_hits})
    path = OUT / "sensor_traces" / f"{mode}-{case}-world{world}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")
    pre = [r for r in rows if r["time_s"] < late.ENTRY_START_S]
    post = [r for r in rows if r["time_s"] >= late.ENTRY_START_S]
    return {"mode": mode, "case": case, "world": world,
            "rate_hz": rate_hz, "ray_count": len(bearings),
            "packet_count": len(rows),
            "preentry_person_hit_packets": sum(
                r["evaluator_person_ray_hits"] > 0 for r in pre),
            "postentry_person_hit_packets": sum(
                r["evaluator_person_ray_hits"] > 0 for r in post),
            "first_person_hit_s": next((r["time_s"] for r in rows
                                        if r["evaluator_person_ray_hits"]), None),
            "trace": {"path": str(path), "sha256": trial.sha(path)}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_info = json.loads(REPORT_SOURCE.read_text())
    forward_info = json.loads(early.MODES["late_3p0s"][0].with_suffix(".json").read_text())
    if source_info["body_sha256"] != trial.sha(kitchen.BODY):
        raise RuntimeError("zero-action source is from another body")
    initial_state_matched = (source_info["initial_state_sha256"] ==
                             forward_info["initial_state_sha256"])
    scene = late.previous.OUT / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    original_out = early.OUT
    original_overlay = early.render_overlay
    early.OUT = OUT
    early.render_overlay = overlay
    results = []
    sensor_results = []
    initial_trace_difference = []
    try:
        with np.load(SOURCE, allow_pickle=False) as saved:
            with np.load(early.MODES["late_3p0s"][0], allow_pickle=False) as forward:
                for world in range(4):
                    initial_trace_difference.append({
                        "world": world,
                        "first_saved_root_pose_max_abs_difference": round(float(
                            np.max(np.abs(saved["root_pose"][0, world] -
                                          forward["root_pose"][0, world]))), 6),
                        "first_saved_joint_max_abs_difference_rad": round(float(
                            np.max(np.abs(saved["joint_pos"][0, world] -
                                          forward["joint_pos"][0, world]))), 6)})
            for world in range(4):
                for case in late.CASES:
                    row = early.one(model, saved, SOURCE, 0.0, world, case,
                                    "origin_zero_action", video=(world == 0))
                    row.pop("native_stop_command_s")
                    row.pop("root_command_progress_m")
                    row["zero_action_from_s"] = 0.0
                    results.append(row)
                    print(world, case, row["min_body_person_gap_m"], flush=True)
                for mode, (bearings, rate_hz) in monitor.SENSOR_MODES.items():
                    for case in monitor.CASES:
                        sensor_results.append(sensor_trace(
                            model, saved, world, case, mode, bearings, rate_hz))
    finally:
        early.OUT = original_out
        early.render_overlay = original_overlay
    old = json.loads((early.OUT / "report.json").read_text())
    report = {
        "schema": "robot-learning-lab.ape-native-origin-hold-screen.v1",
        "task": "T-9v3p3n",
        "body_status": "historical_rejected_direct_frame",
        "method": "Separate initial-condition feasibility case: retain the actual native zero-action trace from t=0 in the V27 box-stool RoboCasa room; replay the same late left/right side-person motion and sample articulated full-body gaps at20ms endpoints plus10ms midpoints. Take timed ideal 140deg front and 360deg all-around low-range packets during zero action for moving and stationary side-person pairs. Its initial-state hash differs from V29's forward traces, so this is not a matched third candidate or causal comparison. Person ray IDs score visibility only. No actor chooses a candidate, no social request or resumed locomotion, and no physical person collision response.",
        "scene_sha256": trial.sha(scene),
        "source_sha256": trial.sha(Path(__file__)),
        "replay_source_sha256": trial.sha(Path(early.__file__)),
        "range_source_sha256": trial.sha(Path(monitor.__file__)),
        "person_script_sha256": trial.sha(Path(late.__file__)),
        "body_sha256": trial.sha(kitchen.BODY),
        "zero_action_source": {"trace_sha256": trial.sha(SOURCE),
                               "report_sha256": trial.sha(REPORT_SOURCE),
                               "initial_state_sha256": source_info["initial_state_sha256"],
                               "first_action_sha256": source_info["first_action_sha256"],
                               "initial_state_matched_to_v29": initial_state_matched,
                               "first_saved_trace_difference": initial_trace_difference},
        "v29_report_sha256": trial.sha(early.OUT / "report.json"),
        "v29_candidate_summary": {
            "early_below_5cm": sum(r["below_5cm"] for r in old["results"]
                                    if r["mode"] == "early_2p5s"),
            "late_below_5cm": sum(r["below_5cm"] for r in old["results"]
                                   if r["mode"] == "late_3p0s")},
        "results": results, "sensor_results": sensor_results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
