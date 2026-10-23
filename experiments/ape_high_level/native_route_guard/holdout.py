"""Changed crossing speeds and sensor loss against the frozen mapped-room guard."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import run as guard


OUT = guard.ROOT / "artifacts/ape-native-route-guard/v4-changed-crossing"
CASES = ("cross_fast_held", "cross_slow_held", "cross_pause_held", "low_dropout")


def hazard(case: str, t: float):
    if case == "cross_fast_held":
        return "person", (1.85, -.20-.60*min(t, 4.0))
    if case == "cross_slow_held":
        return "person", (1.85, -.20-.24*min(t, 4.0))
    if case == "cross_pause_held":
        return "person", (1.85, -.20-.42*min(t, 1.5))
    return guard.close_hazard(case, t)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    xml_path = OUT / "scene.xml"
    xml_path.write_text(guard.replay.make_scene())
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    saved = np.load(guard.replay.TRACE, allow_pickle=False)
    background = guard.map_static_room(model, saved)
    background_path = OUT / "background_xy.npy"
    np.save(background_path, background)
    stop = json.loads(guard.replay.STOP_REPORT.read_text())
    delays = {r["world"]: r["command_to_observed_stop_s"]
              for r in stop["rows"] if r["case"] == "forward-stop-late"}
    guard.replay.hazard_xy = hazard
    guard.replay.read_rays = guard.read_with_guard
    guard.replay.overlay = guard.overlay_guard
    guard.replay.STOP_THRESHOLD_M = .75
    guard.replay.OUT = OUT
    results = []
    for world in range(4):
        for case in CASES:
            for mode in ("none", "low15"):
                guard.GUARD = guard.FullScanGuard(background)
                video = world == 0 and mode == "low15"
                result = guard.replay.run_case(model, saved, world, case, mode,
                                               delays[world], video=video)
                results.append(result)
                print(world, case, mode, result["stop_command_s"],
                      result["min_modeled_body_gap_m"], flush=True)
    report = {"schema": "robot-learning-lab.native-route-guard.changed-crossing.v1",
              "task": "T-9v3p3n",
              "purpose": "Changed person motion and low-stream dropout against frozen V3 mapped-room guard",
              "case_scripts": {
                  "cross_fast_held": "x=1.85; y=-.20-.60*min(t,4)",
                  "cross_slow_held": "x=1.85; y=-.20-.24*min(t,4)",
                  "cross_pause_held": "x=1.85; y=-.20-.42*min(t,1.5)",
                  "low_dropout": "inherited low pet and low packet loss 1.2–2.4 s",
              },
              "scene_sha256": guard.replay.sha(xml_path),
              "background_map_sha256": guard.replay.sha(background_path),
              "frozen_guard_sha256": guard.replay.sha(Path(guard.__file__)),
              "source_sha256": guard.replay.sha(Path(__file__)),
              "native_trace_sha256": guard.replay.sha(guard.replay.TRACE),
              "body_sha256": guard.replay.sha(guard.replay.BODY),
              "mujoco_version": mujoco.__version__,
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    saved.close()


if __name__ == "__main__":
    main()
