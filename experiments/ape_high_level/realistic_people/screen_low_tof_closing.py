"""Development screen for a body-stationary ToF closing cue on saved streams.

This exact-packet screen is not a changed live trajectory or physical sensor
noise model. It only applies while ego motion is independently confirmed zero.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path
import random
import statistics


ROOT = Path("artifacts/ape-realistic-people")
RUNS = {
    "incoming": ROOT / "isaac61-midroom-held-low-tof-v217b",
    "static": ROOT / "isaac61-midroom-held-low-tof-static-v218b",
}
LAG_S = .4
MIN_MATCHED_ZONES = 32
NOISE_STD_M = .015
DROPOUT = .15


def readings(trace: list[dict], seed: int | None,
             fault_mode: str) -> list[tuple[float, list[float | None]]]:
    rng = random.Random(seed)
    rows = []
    for sample in trace:
        common_bias = (rng.gauss(0., .01) if seed is not None and
                       fault_mode == "correlated" else 0.)
        zone_noise_std = .005 if fault_mode == "correlated" else NOISE_STD_M
        sensor = sample["sensors"]["front_left"]
        values = []
        for distance, height in zip(sensor["range_m_by_zone"],
                                    sensor["hit_height_above_floor_m_by_zone"]):
            if distance is None or height is None or height < .03:
                values.append(None)
            elif seed is not None and rng.random() < DROPOUT:
                values.append(None)
            else:
                values.append(max(.02, distance +
                    common_bias + (rng.gauss(0., zone_noise_std)
                                   if seed is not None else 0.)))
        rows.append((sample["sim_time_s"], values))
    return rows


def first_alarm(samples: list[dict], seed: int | None,
                threshold_m: float, persistence: int,
                fault_mode: str = "independent") -> dict | None:
    rows = readings(samples, seed, fault_mode)
    times = [time_s for time_s, _ in rows]
    streak = 0
    for index, (time_s, current) in enumerate(rows):
        previous_index = bisect.bisect_right(times, time_s - LAG_S) - 1
        if previous_index < 0:
            continue
        previous = rows[previous_index][1]
        matched = [old - now for old, now in zip(previous, current)
                   if old is not None and now is not None]
        closing = (len(matched) >= MIN_MATCHED_ZONES and
                   statistics.median(matched) >= threshold_m)
        streak = streak + 1 if closing else 0
        if streak >= persistence:
            return {"sim_time_s": time_s,
                    "cube_center_xy_m_evaluator_only": samples[index][
                        "cube_center_xy_m_evaluator_only"],
                    "matched_zones": len(matched),
                    "median_closing_m": statistics.median(matched)}
    return None


def main() -> None:
    source = {}
    traces = {}
    for name, run in RUNS.items():
        path = run / "evidence/low-tof-trace.json"
        source[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        traces[name] = json.loads(path.read_text())["samples"]
    rows = []
    for threshold in (.005, .01, .015):
        for persistence in (1, 2):
            moving = traces["incoming"]
            static = traces["static"]
            moving_clean = first_alarm(moving, None, threshold, persistence)
            static_clean = first_alarm(static, None, threshold, persistence)
            moving_faults = [first_alarm(moving, seed, threshold, persistence)
                             for seed in range(20)]
            static_faults = [first_alarm(static, seed, threshold, persistence)
                             for seed in range(20)]
            moving_correlated = [first_alarm(moving, seed, threshold, persistence,
                                            "correlated") for seed in range(20)]
            static_correlated = [first_alarm(static, seed, threshold, persistence,
                                            "correlated") for seed in range(20)]
            rows.append({"threshold_m": threshold, "persistence": persistence,
                "incoming_clean": moving_clean,
                "static_clean": static_clean,
                "incoming_fault_warnings": sum(x is not None for x in moving_faults),
                "static_fault_false_warnings": sum(x is not None for x in static_faults),
                "incoming_correlated_warnings": sum(x is not None
                                                     for x in moving_correlated),
                "static_correlated_false_warnings": sum(x is not None
                                                        for x in static_correlated),
                "incoming_fault_first_times_s": [x["sim_time_s"] if x else None
                                                 for x in moving_faults],
                "static_fault_first_times_s": [x["sim_time_s"] if x else None
                                               for x in static_faults],
                "static_correlated_first_times_s": [x["sim_time_s"] if x else None
                                                    for x in static_correlated]})
    report = {"scope": "revealed-scene saved-packet development screen; no changed action",
        "source_sha256": source,
        "rule": "front-left 8x8 matched nonfloor zone median range decrease over 0.4 s; at least 32 zones; fixed ego pose prerequisite",
        "synthetic_faults": "20 deterministic seeds per run: independent 15 mm Gaussian per-zone range std, or 10 mm frame-common plus 5 mm per-zone std, each with 15% per-zone dropout. Neither is a calibrated temporal/material model.",
        "rows": rows,
        "limits": ["Tuned on these same two trajectories; needs new objects, poses and body motion",
            "A closing warning cannot itself stop or escape an incoming object",
            "The rejected body's IMU/joint/contact evidence is absent in this Isaac eye-only stream"]}
    output = ROOT / "low-tof-closing-saved-screen-v217-v218.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps([{"threshold_m": row["threshold_m"],
        "persistence": row["persistence"],
        "incoming_clean_s": row["incoming_clean"]["sim_time_s"]
            if row["incoming_clean"] else None,
        "incoming_fault_warnings": row["incoming_fault_warnings"],
        "static_fault_false_warnings": row["static_fault_false_warnings"],
        "static_correlated_false_warnings": row[
            "static_correlated_false_warnings"]}
        for row in rows], indent=2))


if __name__ == "__main__":
    main()
