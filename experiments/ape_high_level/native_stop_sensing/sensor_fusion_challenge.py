"""Wider ray/depth stream check on the close crossing challenge."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import holdout_close
import motion_gate
import run


OUT = run.ROOT / "artifacts/ape-native-stop-sensing/v10-close-sensor-combos"
CASES = holdout_close.CASES
MODES = ("mid10", "mid_low_depth")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    xml_path = OUT / "scene.xml"
    xml_path.write_text(run.make_scene())
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    saved = np.load(run.TRACE, allow_pickle=False)
    stop_report = json.loads(run.STOP_REPORT.read_text())
    delays = {r["world"]: r["command_to_observed_stop_s"]
              for r in stop_report["rows"] if r["case"] == "forward-stop-late"}
    run.hazard_xy = holdout_close.hazard
    run.read_rays = motion_gate.forecast_read_rays
    run.STOP_THRESHOLD_M = motion_gate.THRESHOLD_M
    run.OUT = OUT
    results = []
    for world in range(4):
        for case in CASES:
            for mode in MODES:
                motion_gate.TRACKER = motion_gate.ScanTracker()
                video = world == 0 and case in ("cross_slow_close", "cross_fast_close")
                result = run.run_case(model, saved, world, case, mode,
                                      delays[world], video=video)
                results.append(result)
                print(world, case, mode, result["stop_command_s"],
                      result["min_modeled_body_gap_m"], flush=True)
    report = {
        "schema": "robot-learning-lab.native-stop-sensing.sensor-combos.v1",
        "task": "T-7f9ef6",
        "purpose": "Compare mid 10 Hz and mid+low+depth to low-only motion gate on frozen close crossing scripts",
        "threshold_m": motion_gate.THRESHOLD_M,
        "scene_sha256": run.sha(xml_path),
        "v3_runner_sha256": run.sha(Path(run.__file__)),
        "frozen_gate_sha256": run.sha(Path(motion_gate.__file__)),
        "close_case_source_sha256": run.sha(Path(holdout_close.__file__)),
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
