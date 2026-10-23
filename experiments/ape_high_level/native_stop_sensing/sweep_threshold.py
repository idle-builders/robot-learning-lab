"""Development sweep of the near-body stop threshold on matched native tails.

This reuses the frozen v3 scene and actor, changing only the declared stop
threshold. The crossing-clear case is an additional moving, safe control.
These same four native traces are reused, so this is tuning evidence, not a
held-out or physical safety qualification.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import run


OUT = run.ROOT / "artifacts/ape-native-stop-sensing/v4-threshold-sweep"
THRESHOLDS_M = (.55, .65, .75)
CASES = ("approaching_person", "low_pet", "static_offpath",
         "false_return", "crossing_clear")


def crossing_hazard(case: str, t: float):
    if case == "crossing_clear":
        return "person", (2.2, -.25 - .375 * min(t, 4.0))
    return ORIGINAL_HAZARD(case, t)


ORIGINAL_HAZARD = run.hazard_xy


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    xml_path = OUT / "scene.xml"
    xml_path.write_text(run.make_scene())
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    saved = np.load(run.TRACE, allow_pickle=False)
    stop_report = json.loads(run.STOP_REPORT.read_text())
    delays = {r["world"]: r["command_to_observed_stop_s"]
              for r in stop_report["rows"] if r["case"] == "forward-stop-late"}
    if set(delays) != set(range(4)):
        raise ValueError("Missing native worlds")
    run.hazard_xy = crossing_hazard
    results = []
    # First establish the safe counterfactual for the new crossing case.
    run.OUT = OUT / "no-sensor-control"
    for world in range(4):
        result = run.run_case(model, saved, world, "crossing_clear", "none",
                              delays[world], video=False)
        results.append({"threshold_m": None, **result})
        if result["below_5cm_margin"]:
            raise ValueError("Crossing-clear no-sensor control was unsafe")
    for threshold in THRESHOLDS_M:
        run.STOP_THRESHOLD_M = threshold
        run.OUT = OUT / f"threshold-{threshold:.2f}"
        for world in range(4):
            for case in CASES:
                video = world == 0 and (
                    case == "low_pet" or
                    (threshold == .75 and case == "crossing_clear"))
                result = run.run_case(model, saved, world, case, "low15",
                                      delays[world], video=video)
                results.append({"threshold_m": threshold, **result})
                print(threshold, world, case, result["stop_command_s"],
                      result["min_modeled_body_gap_m"], flush=True)
    report = {
        "schema": "robot-learning-lab.native-stop-sensing.threshold-sweep.v1",
        "task": "T-7f9ef6",
        "purpose": "Development sensitivity of the same low-mounted 15 Hz stream; not held-out validation",
        "thresholds_m": THRESHOLDS_M,
        "cases": CASES,
        "crossing_clear_script": "upright person x=2.2m, y=-0.25-0.375*min(t,4)s m",
        "scene_sha256": run.sha(xml_path),
        "v3_runner_sha256": run.sha(Path(run.__file__)),
        "source_sha256": run.sha(Path(__file__)),
        "native_trace_sha256": run.sha(run.TRACE),
        "body_sha256": run.sha(run.BODY),
        "mujoco_version": mujoco.__version__,
        "results": results,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    saved.close()


if __name__ == "__main__":
    main()
