"""V191: exact selected-body clearance under hypothetical post-alert travel."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run import ROOT, Room, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v191"
V122 = ROOT / "artifacts/ape-closed-loop-cove50-v122/small-prop-clearance-audit.json"
V173 = ROOT / "artifacts/ape-closed-loop-cove50-v173/body-aware-range-audit.json"
TAILS_M = np.arange(0., .120001, .001)


def main() -> None:
    edge = json.loads(V122.read_text())["stopped_edge"]
    alarm = json.loads(V173.read_text())["v122_replays"][0]
    assert alarm["scan"] == "ideal" and alarm["stop_fraction"] == .75
    start = np.asarray(edge["evaluator_actual_stop_xyyaw"], dtype=float)
    delta = np.asarray(edge["next_command_delta_xy_m"], dtype=float)
    direction = delta / np.linalg.norm(delta)
    stop = start.copy()
    stop[:2] += .88 * alarm["stop_fraction"] * delta
    room = Room()
    rows = []
    try:
        room.reset()
        room.layout(85, absent=False, furnished=True)
        room.model.body_pos[room.model.body("loop_table").id] = (.40, .46, 0.)
        room.set_table_height(.37)
        for tail in TAILS_M:
            pose = stop.copy()
            pose[:2] += tail * direction
            room.actual[:] = pose
            room._forward()
            gap, body, obstacle = room._current_obstacle_clearance()
            rows.append({"tail_m": round(float(tail), 3),
                         "body_clearance_m": round(float(gap), 6),
                         "closest_robot_geom": body,
                         "closest_scene_geom": obstacle})
    finally:
        room.close()
    assert abs(rows[0]["body_clearance_m"] - alarm["stop_evaluator_gap_m"]) < .0001
    first_below_five = next((row for row in rows if row["body_clearance_m"] < .05), None)
    first_overlap = next((row for row in rows if row["body_clearance_m"] < 0), None)
    nominal = {str(int(100 * tail)): min(
        row["body_clearance_m"] for row in rows if row["tail_m"] <= tail + 1e-9)
        for tail in (0., .02, .05, .10)}
    result = {"scope": "Deterministic kinematic stop-tail sensitivity on the archived V122/V173 small-prop edge. Exact selected-body geometry and all loop solids, 1mm post-alert samples to12cm. Tail is an input, not a native selected-body stop measurement.",
              "source_sha256": sha(SOURCE), "v122_sha256": sha(V122),
              "v173_sha256": sha(V173),
              "alarm_fraction": alarm["stop_fraction"],
              "command_length_m": float(np.linalg.norm(delta)),
              "alarm_full_body_clearance_m": rows[0]["body_clearance_m"],
              "first_tail_below_5cm_m": None if first_below_five is None else first_below_five["tail_m"],
              "first_tail_overlap_m": None if first_overlap is None else first_overlap["tail_m"],
              "minimum_clearance_through_tail_m": nominal,
              "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stop-tail-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
