"""Late low-prop approach during layout-2 owner-memory return."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_robocasa_layout2_owner_transfer_v695 as transfer
import run_robocasa_layout2_person_route_v697 as full
import run_robocasa_layout2_route_v407 as route


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-layout2-late-prop-v700"
START = np.array([.6, -4.7])
END = np.array([.6, -3.4])
BEGIN_S = 17.8
END_S = 21.8


def prop_at(t):
    fraction = np.clip((t - BEGIN_S) / (END_S - BEGIN_S), 0., 1.)
    return START + fraction * (END - START)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {"schema": "robot-learning-lab.layout2-late-prop-declaration.v1",
                   "scene": "RoboCasa layout2/style1, shifted owner from V697",
                   "prop_start_xy_m": START.tolist(),
                   "prop_end_xy_m": END.tolist(),
                   "motion_begin_s": BEGIN_S, "motion_end_s": END_S,
                   "speed_m_s": float(np.linalg.norm(END - START) / (END_S - BEGIN_S)),
                   "fixture_clearance_preflight": "15 points at x=0.6, y=-4.7 to -3.4: minimum signed fixture gap 0.06 m; evaluator-only admission",
                   "comparison": "Use the V697 image-grounded person station and memory controller, change the low prop from distant static to an authored fixture-clear late approach during return and slow re-recognition. Score body clearance throughout the full elapsed task time, including inference holds.",
                   "source_sha256": route.sha256(SOURCE),
                   "full_runner_source_sha256": route.sha256(full.SOURCE)}
    (OUT / "predeclared.json").write_text(json.dumps(declaration, indent=2) + "\n")
    previous = (full.OUT, full.PROP_SCRIPT_FACTORY, full.RUN_SCOPE,
                transfer.PROP_START.copy())
    try:
        full.OUT = OUT
        full.PROP_SCRIPT_FACTORY = lambda _: prop_at
        full.RUN_SCOPE = "actual layout2/style1 owner RGB-D station and full memory loop with authored late 0.325m/s low-prop approach; 50cm kinematic proxy; actor receives timed range and image observations only"
        transfer.PROP_START = START.copy()
        full.main()
    finally:
        full.OUT, full.PROP_SCRIPT_FACTORY, full.RUN_SCOPE, transfer.PROP_START = previous
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text())
    report["schema"] = "robot-learning-lab.layout2-late-prop-result.v1"
    report["late_prop_source_sha256"] = route.sha256(SOURCE)
    report["predeclaration_sha256"] = route.sha256(OUT / "predeclared.json")
    report["script"] = {"start_xy_m": START.tolist(), "end_xy_m": END.tolist(),
                        "begin_s": BEGIN_S, "end_s": END_S}
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
