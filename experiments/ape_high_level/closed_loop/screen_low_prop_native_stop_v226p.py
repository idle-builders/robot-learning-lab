"""Overlay archived native stop tails on held-out low-prop alarm poses.

This is an evaluator-only response screen, not an actor rollout. It rotates
each 50 Hz historical Direct Frame stop displacement into the route heading,
then samples the moving prop at 20 ms for three possible command times.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from run import ROOT, sha
from run_robocasa_layout4_returning_prop_v460 import motion


SOURCE = Path(__file__)
OWNER_REPORT = ROOT / "artifacts/ape-robocasa-low-prop-held-pose-v224p/report.json"
NATIVE_REPORT = ROOT / "artifacts/ape-skill-contract-cove50-20261012/historical-stop-trace.json"
NATIVE_ROOT = ROOT / "artifacts/ape-cove50-effort-envelope"
OUT = ROOT / "artifacts/ape-robocasa-low-prop-native-stop-v226p"
COMMAND_TIMES_S = (10.8, 11.2, 11.6)
HORIZON_S = 4.0
BODY_PLUS_PROP_RADIUS_M = .17 + .24


def route_pose(trace, command_s):
    index = next(i for i, row in enumerate(trace)
                 if abs(row["time_s"] - command_s) < 1e-4)
    if index == 0:
        raise AssertionError("Need pre-command route direction")
    pose = np.asarray(trace[index]["actual_xy_m"], dtype=float)
    previous = np.asarray(trace[index - 1]["actual_xy_m"], dtype=float)
    heading = math.atan2(*(pose - previous)[::-1])
    return pose, heading


def overlay(case, command_s, name, world, native):
    route = case["route_episode"]["trace"]
    start_xy, route_heading = route_pose(route, command_s)
    native_time = native["time_s"]
    stop_s = 2.5 if name.endswith("early") else 3.0
    stop_i = int(np.argmin(abs(native_time - stop_s)))
    if abs(float(native_time[stop_i]) - stop_s) > 1e-6:
        raise AssertionError("Native command time is off actor tick")
    native_xy = native["root_pose"][:, world, :2].astype(float)
    native_forward = native_xy[stop_i] - native_xy[stop_i - 5]
    native_heading = math.atan2(native_forward[1], native_forward[0])
    rotate = route_heading - native_heading
    c, s = math.cos(rotate), math.sin(rotate)
    rotation = np.array(((c, -s), (s, c)))
    prop_at = motion(case["scenario"])
    samples = []
    for absolute_s in np.arange(command_s, command_s + HORIZON_S + .001, .02):
        relative_s = min(absolute_s - command_s, float(native_time[-1]) - stop_s)
        source_s = stop_s + relative_s
        source_xy = np.array([np.interp(source_s, native_time, native_xy[:, axis])
                              for axis in range(2)])
        ape_xy = start_xy + rotation @ (source_xy - native_xy[stop_i])
        prop_xy = prop_at(float(absolute_s))
        gap = float(np.linalg.norm(ape_xy - prop_xy) - BODY_PLUS_PROP_RADIUS_M)
        samples.append((float(absolute_s), gap, ape_xy.tolist(), prop_xy.tolist()))
    nearest = min(samples, key=lambda x: x[1])
    first_margin = next((x[0] for x in samples if x[1] < .05), None)
    first_contact = next((x[0] for x in samples if x[1] < 0), None)
    return {"scenario": case["scenario"]["id"], "command_time_s": command_s,
            "native_case": name, "native_world": world,
            "route_start_xy_m": start_xy.tolist(),
            "route_heading_rad": route_heading,
            "native_alignment_heading_rad": native_heading,
            "minimum_gap_m": nearest[1], "minimum_gap_at_s": nearest[0],
            "ape_xy_at_min_m": nearest[2], "prop_xy_at_min_m": nearest[3],
            "first_below_5cm_s": first_margin,
            "first_contact_s": first_contact}


def main():
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Historical body classification changed")
    owner = json.loads(OWNER_REPORT.read_text())
    historical = json.loads(NATIVE_REPORT.read_text())
    if historical["body_status"] != "historical_rejected_direct_frame":
        raise AssertionError("Stop source is no longer classified historical")
    cases = [case for case in owner["episodes"]
             if case["scenario"]["id"] != "near_stationary_control"]
    rows = []
    source_hashes = {}
    for name in ("forward-stop-early", "forward-stop-late"):
        path = NATIVE_ROOT / f"{name}.npz"
        source_hashes[name] = sha(path)
        if source_hashes[name] != historical["sources"][name]["native_trace_sha256"]:
            raise AssertionError("Native trace changed")
        with np.load(path, allow_pickle=False) as native:
            if not np.isclose(float(native["step_dt_s"]), .02):
                raise AssertionError("Native trace cadence changed")
            for case in cases:
                for command_s in COMMAND_TIMES_S:
                    for world in range(native["root_pose"].shape[1]):
                        rows.append(overlay(case, command_s, name, world, native))
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "robot-learning-lab.ape-low-prop-native-stop-screen.v1",
        "source_sha256": sha(SOURCE),
        "owner_report_sha256": sha(OWNER_REPORT),
        "historical_native_report_sha256": sha(NATIVE_REPORT),
        "native_trace_sha256": source_hashes,
        "command_times_s": COMMAND_TIMES_S,
        "horizon_s": HORIZON_S,
        "body_plus_prop_radius_m": BODY_PLUS_PROP_RADIUS_M,
        "method": "Evaluator-only rotation of each archived native 50Hz "
                  "zero-twist stop displacement into the V224p route heading "
                  "at selected alarm poses; moving prop is sampled every20ms. "
                  "The route actor did not execute these native responses.",
        "limits": "Historical rejected Direct Frame forward policy, different "
                  "body/task pose and no coupled contact dynamics. The source "
                  "stop traces use evaluator states, not actor-observed pose. "
                  "This screens response reserve, not a physical guarantee.",
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for case in cases:
        for command_s in COMMAND_TIMES_S:
            subset = [x for x in rows if x["scenario"] == case["scenario"]["id"]
                      and x["command_time_s"] == command_s]
            print(case["scenario"]["id"], command_s,
                  "gap_range", tuple(round(x, 3) for x in
                                     (min(z["minimum_gap_m"] for z in subset),
                                      max(z["minimum_gap_m"] for z in subset))),
                  "contacts", sum(z["first_contact_s"] is not None for z in subset),
                  "below_5cm", sum(z["first_below_5cm_s"] is not None
                                   for z in subset), flush=True)


if __name__ == "__main__":
    main()
