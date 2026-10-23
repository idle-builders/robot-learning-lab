"""Fault-screen the fixed stationary ToF intrusion cue on held-pose packets.

This replays unchanged saved packets. It does not change actions or model the
real optical response of a multizone sensor.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random

from tof_intrusion_response import StationaryTofIntrusionResponse


def alarm_time(samples: list[dict], seed: int | None,
               fault: str) -> float | None:
    rng = random.Random(seed)
    supervisor = StationaryTofIntrusionResponse()
    for row in samples:
        flank_sensors = {}
        for mount in ("front_left", "front_right"):
            sensor = row["sensors"][mount]
            distances = copy.copy(sensor["range_m_by_zone"])
            if seed is not None:
                common = rng.gauss(0., .01) if fault == "correlated" else 0.
                zone_std = .005 if fault == "correlated" else .015
                for index, value in enumerate(distances):
                    if value is None or rng.random() < .15:
                        distances[index] = None
                    else:
                        distances[index] = max(.02, value + common +
                                               rng.gauss(0., zone_std))
            flank_sensors[mount] = {
                "range_m_by_zone": distances,
                "hit_height_above_floor_m_by_zone": sensor[
                    "hit_height_above_floor_m_by_zone"]}
        actor_packet = {"sim_time_s": row["sim_time_s"],
            "sensors": flank_sensors}
        supervisor.observe(actor_packet)
        if supervisor.alarm is not None:
            return supervisor.alarm["time_s"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("moving_held_run", type=Path)
    parser.add_argument("static_held_run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    traces = {}
    hashes = {}
    for name, run in (("moving", args.moving_held_run),
                      ("static", args.static_held_run)):
        path = run / "evidence/low-tof-trace.json"
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        traces[name] = json.loads(path.read_text())["samples"]
    report = {"scope": "same-scene held-packet fault screen of one fixed cue; "
               "no changed action, no sensor optics or ego-motion faults",
              "source_sha256": hashes,
              "clean_alarm_s": {name: alarm_time(rows, None, "none")
                                for name, rows in traces.items()},
              "faults": {},
              "limits": ["Rule developed on the far incoming and near static cube cases",
                         "Fault amplitudes are synthetic and not a material/lighting model",
                         "Stationary commanded eye is assumed; no IMU confirmation"]}
    for fault in ("independent", "correlated"):
        times = {name: [alarm_time(rows, seed, fault) for seed in range(20)]
                 for name, rows in traces.items()}
        report["faults"][fault] = {"moving_alarm_count": sum(
            time_s is not None for time_s in times["moving"]),
            "static_false_alarm_count": sum(
                time_s is not None for time_s in times["static"]),
            "first_alarm_times_s": times}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"clean_alarm_s": report["clean_alarm_s"],
        "faults": {name: {key: row[key] for key in (
            "moving_alarm_count", "static_false_alarm_count")}
            for name, row in report["faults"].items()}}, indent=2))


if __name__ == "__main__":
    main()
