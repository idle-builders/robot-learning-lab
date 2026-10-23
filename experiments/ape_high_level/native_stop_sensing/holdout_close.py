"""Closer, consequential crossing challenge for the frozen v7 motion gate."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import motion_gate
import run


OUT = run.ROOT / "artifacts/ape-native-stop-sensing/v9-close-crossing"
CASES = ("cross_fast_close", "cross_slow_close", "cross_pause_close")


def hazard(case: str, t: float):
    if case == "cross_fast_close":
        return "person", (1.85, -.25-.52*min(t, 4.0))
    if case == "cross_slow_close":
        return "person", (1.85, -.25-.20*min(t, 4.0))
    if case == "cross_pause_close":
        return "person", (1.85, -.25-.375*min(t, 1.8))
    return motion_gate.ORIGINAL_HAZARD(case, t)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    xml_path = OUT / "scene.xml"
    xml_path.write_text(run.make_scene())
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    saved = np.load(run.TRACE, allow_pickle=False)
    stop_report = json.loads(run.STOP_REPORT.read_text())
    delays = {r["world"]: r["command_to_observed_stop_s"]
              for r in stop_report["rows"] if r["case"] == "forward-stop-late"}
    run.hazard_xy = hazard
    run.read_rays = motion_gate.forecast_read_rays
    run.STOP_THRESHOLD_M = motion_gate.THRESHOLD_M
    run.OUT = OUT
    results = []
    for world in range(4):
        for case in CASES:
            for mode in ("none", "low15"):
                motion_gate.TRACKER = motion_gate.ScanTracker()
                video = world == 0 and (case in ("cross_fast_close", "cross_pause_close"))
                result = run.run_case(model, saved, world, case, mode,
                                      delays[world], video=video)
                results.append(result)
                print(world, case, mode, result["stop_command_s"],
                      result["min_modeled_body_gap_m"], flush=True)
    report = {
        "schema": "robot-learning-lab.native-stop-sensing.close-crossing.v1",
        "task": "T-7f9ef6",
        "purpose": "New close crossing after gate freeze; fast no-sensor control clears while paused no-sensor control overlaps in all four native worlds",
        "case_scripts": {
            "cross_fast_close": "x=1.85; y=-0.25-0.52*min(t,4)",
            "cross_slow_close": "x=1.85; y=-0.25-0.20*min(t,4)",
            "cross_pause_close": "x=1.85; y=-0.25-0.375*min(t,1.8)",
        },
        "scene_sha256": run.sha(xml_path),
        "v3_runner_sha256": run.sha(Path(run.__file__)),
        "frozen_gate_sha256": run.sha(Path(motion_gate.__file__)),
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
