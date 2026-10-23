"""Ideal immediate straight-escape bound from the late layout-4 stop pose.

This is evaluator geometry only, with no acceleration, reaction delay, gait,
rotational sweep or actor knowledge of prop motion. It can reject an assumed
escape but cannot qualify a real skill.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from audit_robocasa_layout2_routes_v409 import clearance, sha256
from audit_robocasa_layout4_crossing_v448 import layout4_evaluator_scene
from audit_robocasa_dynamic_suite_v440 import prop_at


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-layout4-escape-v449"
SOURCE = ROOT / "artifacts/ape-robocasa-layout4-crossing-v447/report.json"
REQUIRED_GAP_M = .05


def main() -> None:
    source = json.loads(SOURCE.read_text())
    case = next(row for row in source["cases"]
                if row["scenario"]["id"] == "late_crossing")
    scenario, episode = case["scenario"], case["episode"]
    start = np.asarray(episode["end_true_xy_m"], dtype=float)
    t0 = float(episode["end_time_s"])
    model, data, ape_body, ape_geom = layout4_evaluator_scene(scenario["from_xy_m"])
    prop_body = model.body("route_obstacle").id
    samples = np.unique(np.r_[np.arange(t0, scenario["end_s"] + .005, .01),
                              scenario["end_s"]])
    rows = []
    for speed in (0., .25, .5, .75, 1.):
        bearings = [0.] if speed == 0 else [index * 22.5 for index in range(16)]
        for bearing in bearings:
            direction = np.array([math.cos(math.radians(bearing)),
                                  math.sin(math.radians(bearing))])
            best = {"gap_m_capped_at_10cm": .1, "geom": None, "time_s": None}
            for t in samples:
                xy = start + direction * speed * (t - t0)
                model.body_pos[prop_body, :2] = prop_at(scenario, float(t))
                mujoco.mj_forward(model, data)
                gap, geom = clearance(model, data, ape_body, ape_geom, xy)
                if gap < best["gap_m_capped_at_10cm"]:
                    best = {"gap_m_capped_at_10cm": gap,
                            "geom": geom, "time_s": round(float(t), 4)}
            rows.append({"ideal_speed_m_s": speed, "bearing_deg": bearing,
                         "minimum": best,
                         "5cm_margin_pass": best["gap_m_capped_at_10cm"] >= REQUIRED_GAP_M})
    viable = [row for row in rows if row["5cm_margin_pass"]]
    output = {"source_sha256": sha256(Path(__file__)),
              "actor_report_sha256": sha256(SOURCE),
              "scenario": scenario, "start_xy_m": start.tolist(),
              "start_time_s": t0, "end_time_s": scenario["end_s"],
              "required_gap_m": REQUIRED_GAP_M,
              "scope": "ideal instantaneous straight escape, 16 bearings and 0-1m/s, 10ms independent MuJoCo 50cm-cylinder checks",
              "viable_count": len(viable), "cases": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(output, indent=2) + "\n")
    print("cases", len(rows), "viable", len(viable), flush=True)
    for row in viable:
        print("viable", row["ideal_speed_m_s"], row["bearing_deg"],
              round(row["minimum"]["gap_m_capped_at_10cm"], 3), flush=True)


if __name__ == "__main__":
    main()
