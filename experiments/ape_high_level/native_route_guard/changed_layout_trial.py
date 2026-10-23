"""V19: unchanged low guard in a second upstream RoboCasa kitchen layout.

Layout 3 was chosen by a fixture-clearance screen before this guard run. The
person remains a stylized scripted capsule and the body a native trace splice.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import changed_speed_trial as left_cases
import command_route_trial as command
import dense_low_trial as dense
import kitchen_native as kitchen
import kitchen_trial as trial
import right_entry_trial as right_cases
import run as guard


OUT = kitchen.OUT / "v19-layout3"
CASES = left_cases.CASES + right_cases.CASES
MODES = ("none", "mapped_low_guard")
ORIGINAL_READ = guard.ORIGINAL_READ
ORIGINAL_POINTS = guard.low_ray_points
ORIGINAL_HAZARD = trial.hazard_xy


def make_scene() -> str:
    original = kitchen.KitchenArena

    def layout3(**kwargs):
        return original(layout_id=3, style_id=1, rng=kwargs["rng"], clutter_mode=0)

    kitchen.KitchenArena = layout3
    try:
        return kitchen.scene_xml()
    finally:
        kitchen.KitchenArena = original


def hazard_xy(case: str, t: float) -> tuple[float, float]:
    return (left_cases.hazard_xy(case, t) if case in left_cases.CASES
            else right_cases.hazard_xy(case, t))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    xml = OUT / "scene.xml"
    xml.write_text(make_scene())
    model = mujoco.MjModel.from_xml_path(str(xml))
    stop = json.loads(guard.replay.STOP_REPORT.read_text())
    delays = {row["world"]: row["command_to_observed_stop_s"]
              for row in stop["rows"] if row["case"] == "forward-stop-late"}
    trial.hazard_xy = hazard_xy
    guard.ORIGINAL_READ = dense.dense_read
    guard.low_ray_points = dense.dense_points
    guard.FullScanGuard = command.CommandRouteGuard
    results = []
    try:
        with np.load(kitchen.TRACE, allow_pickle=False) as saved:
            background = dense.initial_map(model, saved)
            map_path = OUT / "background_xy.npy"
            np.save(map_path, background)
            for mode in MODES:
                trial.OUT = OUT / mode
                for world in range(4):
                    for case in CASES:
                        video = world == 0 and (case in ("slow_cross_held",
                                                         "slow_right_held") or
                                                mode == "mapped_low_guard" and case in
                                                ("fast_clear_held", "fast_right_held"))
                        result = trial.run_episode(model, saved, background,
                                                   world, case, mode,
                                                   delays[world], video=video)
                        results.append(result)
                        print(mode, world, case, result["stop_command_s"],
                              result["min_modeled_body_person_gap_m"], flush=True)
    finally:
        trial.hazard_xy = ORIGINAL_HAZARD
        guard.ORIGINAL_READ = ORIGINAL_READ
        guard.low_ray_points = ORIGINAL_POINTS
        guard.FullScanGuard = command.OriginalGuard
    report = {
        "schema": "robot-learning-lab.ape-robocasa-layout3-low-guard.v1",
        "task": "T-9v3p3n",
        "body_status": "historical_rejected_direct_frame",
        "method": "Upstream RoboCasa layout3/style1 chosen by pre-run fixture clearance; unchanged historical native path, command-route guard, 73-ray front low 15Hz/1.2m sensor and V11 left plus repaired V17 right scripts. Fresh hazard-free initial-view static map in this layout only. Paired no-sensor controls. No guard tuning or new physical rollout.",
        "room": {"layout_id": 3, "style_id": 1,
                 "scene_sha256": trial.sha(xml),
                 "source_layout1_scene_sha256": trial.sha(kitchen.OUT / "scene.xml")},
        "sensor": {"rays": 73, "period_s": 1/15,
                   "yaw_range_deg": [-70., 70.], "max_range_m": dense.MAX_M,
                   "origin_local_m": [.08, 0., -.15],
                   "initial_map_points": len(background),
                   "initial_map_sha256": trial.sha(map_path),
                   "ideal_odom": True},
        "source_sha256": trial.sha(Path(__file__)),
        "scene_builder_sha256": trial.sha(Path(kitchen.__file__)),
        "dense_source_sha256": trial.sha(Path(dense.__file__)),
        "v11_case_source_sha256": trial.sha(Path(left_cases.__file__)),
        "v17_case_source_sha256": trial.sha(Path(right_cases.__file__)),
        "episode_runner_sha256": trial.sha(Path(trial.__file__)),
        "command_guard_sha256": trial.sha(Path(command.__file__)),
        "native_trace_sha256": trial.sha(kitchen.TRACE),
        "body_sha256": trial.sha(kitchen.BODY),
        "v18_report_sha256": trial.sha(kitchen.OUT / "v18-single-front/report.json"),
        "mujoco_version": mujoco.__version__,
        "results": results,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
