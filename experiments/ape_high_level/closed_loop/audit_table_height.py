"""Evaluator-only stress screen for vertical range coverage at nearby tables."""

from __future__ import annotations

from collections import Counter
import json

import numpy as np

from audit_footprint_calibration import elevated_scan_points, segmented_gap
from footprint_route import scan_points
from run import OUT, Room


def set_table_height(room: Room, height: float) -> None:
    top = room.model.geom("loop_table_top")
    top.pos[2] = height
    for xsign in (-1, 1):
        for ysign in (-1, 1):
            leg = room.model.geom(f"loop_table_leg_{xsign}_{ysign}")
            leg.pos[2] = height / 2
            leg.size[2] = height / 2


def main() -> None:
    room = Room()
    rows = []
    try:
        for seed in (85, 88):
            room.reset()
            room.layout(seed, absent=False, furnished=True)
            side = 1 if seed % 2 else -1
            for height in (.25, .29, .33, .37):
                set_table_height(room, height)
                for x in np.arange(.48, .961, .04):
                    for yy in np.arange(.16, .761, .04):
                        room.actual[:] = (x, side * yy, 0.)
                        room._forward()
                        truth, body, obstacle = room._current_obstacle_clearance()
                        if not obstacle.startswith("loop_table_"):
                            continue
                        low = scan_points(room.range_scan(), room.actual)
                        upper = elevated_scan_points(room)
                        low_gap = segmented_gap(low, room.actual[:2], tight=True)
                        dual_gap = segmented_gap(np.concatenate((low, upper)),
                                                 room.actual[:2], tight=True)
                        rows.append({"seed": seed, "table_top_height_m": height,
                                     "xy_m": [round(float(x), 3), round(float(side * yy), 3)],
                                     "true_gap_m": round(truth, 4),
                                     "closest_body": body,
                                     "closest_obstacle": obstacle,
                                     "low_tight_gap_m": round(low_gap, 4),
                                     "dual_tight_gap_m": round(dual_gap, 4)})
    finally:
        room.close()
    counts = {}
    examples = {}
    for height in (.25, .29, .33, .37):
        subset = [r for r in rows if r["table_top_height_m"] == height]
        counts[str(height)] = {}
        examples[str(height)] = {}
        for proxy in ("low_tight", "dual_tight"):
            failures = [r for r in subset if r["true_gap_m"] < .05 and
                        r[proxy + "_gap_m"] >= .055]
            counts[str(height)][proxy] = {
                "true_unsafe": sum(r["true_gap_m"] < .05 for r in subset),
                "false_safe": len(failures)}
            examples[str(height)][proxy] = failures[:5]
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"scope": "Neutral fixed-yaw furnished table positions; top and legs adjusted together; 11 cm and optional 33 cm virtual horizontal rays; full-body signed clearance evaluator-only.",
              "counts": counts, "false_safe_examples": examples, "rows": rows}
    (OUT / "table-height-coverage-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
