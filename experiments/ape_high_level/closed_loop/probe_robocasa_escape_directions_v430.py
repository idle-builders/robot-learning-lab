"""Screen ideal straight escape directions at the V423 guard stop pose.

All directions and speeds are evaluator counterfactuals. They do not represent
an observed skill, an actor-selected route, or a body-qualified ability.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from audit_robocasa_layout2_routes_v409 import sha256
from probe_robocasa_escape_envelope_v429 import BASE, OUT as PREVIOUS_OUT, simulate


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-escape-directions-v430"


def main() -> None:
    base = json.loads(BASE.read_text())
    stop = base["episodes"][1]
    start_xy = np.asarray(stop["end_true_xy_m"], dtype=float)
    start_time = float(stop["end_time_s"])
    cases = []
    for index in range(16):
        bearing_deg = index * 22.5
        theta = math.radians(bearing_deg)
        direction = np.array([math.cos(theta), math.sin(theta)])
        for speed in (.5, .75, 1.):
            result = simulate(start_xy, start_time, speed, direction)
            cases.append({"bearing_deg": bearing_deg, "speed_m_s": speed,
                          "minimum_prop_gap_m": result["minimum_prop_gap_m"],
                          "minimum_fixed_gap_capped_m": result["minimum_fixed_gap_capped_m"],
                          "both_5cm_margins_pass": result["prop_margin_pass"] and result["fixed_margin_pass"],
                          "first_failed_margin": next(({"time_s": row["time_s"],
                                                         "prop_gap_m": row["prop_signed_gap_m"],
                                                         "fixture_gap_m": row["fixed_signed_gap_capped_m"],
                                                         "nearest_fixture": row["nearest_fixture"]}
                                                        for row in result["trace"]
                                                        if min(row["prop_signed_gap_m"],
                                                               row["fixed_signed_gap_capped_m"]) < .05), None)})
    viable = [case for case in cases if case["both_5cm_margins_pass"]]
    report = {"source_sha256": sha256(Path(__file__)),
              "replay_source_sha256": sha256(Path(__file__).with_name("probe_robocasa_escape_envelope_v429.py")),
              "base_report_sha256": sha256(BASE),
              "prior_report_sha256": sha256(PREVIOUS_OUT / "report.json"),
              "scope": "ideal immediate straight escapes, 16 bearings at 22.5 degree increments, 0.5/0.75/1.0m/s, 10ms MuJoCo geometry checks",
              "viable_count": len(viable), "cases": cases}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("cases", len(cases), "viable", len(viable), flush=True)
    for case in viable:
        print("viable", case["bearing_deg"], case["speed_m_s"],
              round(case["minimum_prop_gap_m"], 3),
              round(case["minimum_fixed_gap_capped_m"], 3), flush=True)


if __name__ == "__main__":
    main()
