"""Screen held low-range thresholds on three unchanged live Isaac streams.

This is a counterfactual packet screen, not a changed closed-loop trajectory.
The scripted cube and eye pose are used only to score first-alarm geometry.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random

from audit_viewpoint_route import box_clearance
from near_body_watchdog import NearBodyWatchdog
from raw_swept_gate import assess_swept_low_hits


ROOT = Path("artifacts/ape-realistic-people")
RUNS = {
    "intrusion": "isaac61-midroom-approaching-cube-held-probe-v215",
    "static_control": "isaac61-midroom-static-held-monitor-v213",
    "safe_moving_control": "isaac61-midroom-creeping-held-boundary-v214",
}


def read(run: Path) -> tuple[list[dict], dict]:
    evidence = run / "evidence"
    filenames = ("scene-info.json", "online-action-trace.json",
                 "range-fan-trace.json", "continuous-vision-monitor.json")
    data = {name: json.loads((evidence / name).read_text()) for name in filenames}
    source = {name: hashlib.sha256((evidence / name).read_bytes()).hexdigest()
              for name in filenames}
    actions = data["online-action-trace.json"]["actions"]
    captures = data["range-fan-trace.json"]["frames"]
    monitor = data["continuous-vision-monitor.json"]["frames"]
    scene = data["scene-info.json"]
    assert len(actions) == len(captures) == len(monitor)
    angles = [beam["angle_deg"] for beam in captures[0]["planes"][0]["beams"]]
    size = scene["route_obstacle_evaluator_only"]["size_xy_m"]
    rows = []
    for frame, (action, capture, watched) in enumerate(zip(actions, captures, monitor)):
        eye = (scene["camera_eye_xyz_m"][:2] if frame == 0 else
               actions[frame - 1]["proxy_eye_xyz_m_evaluator_only"][:2])
        previous = (actions[frame - 1]["command_for_next_frame"] if frame else {})
        readings = [(capture["timeline_time_s"],
                     [math.inf if beam["distance_m"] is None else beam["distance_m"]
                      for beam in capture["planes"][0]["beams"]],
                     action["route_cube_center_xy_m_evaluator_only"], True)]
        readings.extend((sample["time_s"],
            [math.inf if value is None else value
             for value in sample["range_m_by_plane"]["0"]],
            sample["route_cube_center_xy_m_evaluator_only"], False)
            for sample in watched["samples"])
        for instant, low, cube, first in readings:
            rows.append({"frame": frame, "time_s": instant,
                "low_ranges_m": low, "cube_gap_m_evaluator_only": box_clearance(
                    eye, cube, size),
                "preceding_translation_m": previous.get("forward_step_m", 0.)
                    if first else 0.,
                "preceding_body_turn_deg": previous.get("body_yaw_delta_deg", 0.)
                    if first else 0., "angles_deg": angles})
    return rows, source


def threshold(rows: list[dict], clearance_m: float) -> dict:
    alarms = []
    for row in rows:
        result = assess_swept_low_hits(row["low_ranges_m"], row["angles_deg"],
            step_m=0., step_heading_rel_deg=0., clearance_m=clearance_m)
        if result["veto"]:
            alarms.append({"frame": row["frame"], "time_s": row["time_s"],
                "cube_gap_m_evaluator_only": row["cube_gap_m_evaluator_only"],
                "coherent_hit_m": result["nearest_coherent_swept_hit_m"]})
    return {"first_alarm": alarms[0] if alarms else None,
            "alarm_packets": len(alarms)}


def stationary_closing(rows: list[dict], *, seed: int | None = None,
                       noise: float = 0., dropout: float = 0.) -> dict | None:
    watcher = NearBodyWatchdog()
    rng = random.Random(seed)
    for row in rows:
        if abs(row["preceding_body_turn_deg"]) > 2.:
            watcher.reset_motion_history()
        ranges = [None if not math.isfinite(value) else value
                  for value in row["low_ranges_m"]]
        if seed is not None:
            ranges = [None if value is None or rng.random() < dropout else
                      max(.001, value + rng.gauss(0., noise * value))
                      for value in ranges]
        result = watcher.observe(ranges, row["time_s"],
                                 row["preceding_translation_m"])
        if result["reason"] == "closing_while_stationary":
            return {"frame": row["frame"], "time_s": row["time_s"],
                "cube_gap_m_evaluator_only": row["cube_gap_m_evaluator_only"],
                "coherent_hit_m": result["patch_range_m"]}
    return None


def main() -> None:
    report = {"scope": "saved live packets only; no changed action or physical stop",
              "limitations": "Monitor nulls cannot prove measured free or unknown in the saved stream; only finite occupied-hit patches are screened. Decision-boundary scans have summaries but no saved per-beam readings, so this screen omits them.",
              "runs": {}}
    for label, dirname in RUNS.items():
        rows, source = read(ROOT / dirname)
        report["runs"][label] = {"run": dirname, "source_sha256": source,
            "packet_count": len(rows),
            "raw_thresholds": {str(value): threshold(rows, value)
                               for value in (.24, .30, .35)},
            "stationary_closing_clean": stationary_closing(rows),
            "stationary_closing_5pct_noise_dropout": [stationary_closing(
                rows, seed=seed, noise=.05, dropout=.05) for seed in range(20)]}
    output = ROOT / "held-low-threshold-screen-v215.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({label: {"thresholds": {
        value: item["first_alarm"] for value, item in data["raw_thresholds"].items()},
        "closing_clean": data["stationary_closing_clean"],
        "closing_fault_alarms": sum(item is not None for item in
            data["stationary_closing_5pct_noise_dropout"])}
        for label, data in report["runs"].items()}, indent=2))


if __name__ == "__main__":
    main()
