"""Interrupt held-phase range tracking in the complete owner memory task."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_route_v407 as route
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout4_wait_and_remap_v461 as timed_grid
import multi_novel_range_objects as multi_tracker


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-memory-held-dropout-v631"
DROP_WINDOW_S = (17.9, 18.9)
PROFILES = (("range", "stationary", None),
            ("range", "early", DROP_WINDOW_S),
            ("range", "late", None),
            ("range", "late", DROP_WINDOW_S),
            ("rgbd", "late", None),
            ("rgbd", "late", DROP_WINDOW_S))
SHIFTS_S = {"early": .4, "late": -.4}
POST_STOP_UNTIL_S = 20.8


def challenge_motion(label: str):
    if label == "stationary":
        return crossing.prop_motion("stationary")
    baseline = crossing.prop_motion("moving_across_return")
    return lambda time_s: baseline(time_s + SHIFTS_S[label])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    policy = json.loads(joined.CONFIG.read_text())
    case = {"id": "return_crossing_held_dropout", "from_xy_m": crossing.START,
            "to_xy_m": crossing.START, "begin_s": 14., "end_s": 14.4}
    declaration = OUT / "predeclared-held-dropout.json"
    declaration.write_text(json.dumps({
        "schema": "robot-learning-lab.robocasa-memory-held-dropout-declaration.v1",
        "profiles": PROFILES,
        "scene": case,
        "motion": {"base_waypoints_xy_m": crossing.ENTRY_WAYPOINTS,
                   "entry_start_s": crossing.ENTRY_START_S,
                   "entry_end_s": crossing.ENTRY_END_S,
                   "cross_end_s": crossing.CROSS_END_S,
                   "early_late_shifts_s": SHIFTS_S},
        "fault": "Withhold only 10 Hz held-phase 72-beam range packets from tracker during [17.9,18.9) s; preserve 20 ms raw near-body guard trace and initial route sensing. Missing is explicit and never mapped as free.",
        "frozen_reaction": "V625 1.2 s constant-velocity compact-track warning and bounded second escape",
        "post_stop_until_s": POST_STOP_UNTIL_S,
        "source_sha256": memory.digest(SOURCE),
        "memory_source_sha256": memory.digest(memory.SOURCE),
        "motion_source_sha256": memory.digest(crossing.SOURCE),
        "joined_source_sha256": memory.digest(Path(joined.__file__)),
        "timed_grid_source_sha256": memory.digest(Path(timed_grid.__file__)),
        "multi_tracker_source_sha256": memory.digest(Path(multi_tracker.__file__)),
        "prior_map_sha256": memory.digest(joined.PRIOR),
    }, indent=2) + "\n")
    prior = np.load(joined.PRIOR)
    vision = owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(owner.TEACH_CROP).convert("RGB")])[0]
    old_out, old_memory_out, old_hook, old_margin, old_motion = (
        joined.OUT, memory.OUT, joined.POST_OWNER_HOOK,
        route.MARGIN_M, joined.motion)
    joined.OUT = OUT
    memory.OUT = OUT
    route.MARGIN_M = .15
    episodes = []
    try:
        for source, label, dropout in PROFILES:
            joined.motion = lambda _, label=label: challenge_motion(label)

            def hook(*, source=source, label=label, dropout=dropout, **kwargs):
                return memory.callback(
                    source=source, item_state="retained",
                    run_label=f"{label}_{'drop' if dropout else 'healthy'}_held",
                    predeparture_watch_s=2.4,
                    predeparture_escape=True,
                    future_probe=True,
                    hold_monitor=True,
                    held_reactive_escape=True,
                    hold_scan_interval_s=.1,
                    hold_scan_dropout_window_s=dropout,
                    escape_distances_m=(.50, .45, .35),
                    escape_tolerance_m=.09,
                    post_stop_until_s=POST_STOP_UNTIL_S,
                    **kwargs)

            joined.POST_OWNER_HOOK = hook
            response = f"memory_{source}_retained_{label}_{'drop' if dropout else 'healthy'}_held"
            episode = joined.episode(case, response, True, prior, policy,
                                     vision, exemplar)
            episodes.append(episode)
            (OUT / "partial-report.json").write_text(json.dumps({
                "declaration_sha256": memory.digest(declaration),
                "episodes": episodes}, indent=2) + "\n")
            summary = episode["memory_return"]
            print(source, label, dropout, episode["outcome"],
                  None if summary is None else summary["route_status"],
                  None if summary is None else summary["held_reactive_escape_count"],
                  None if summary is None else summary["hold_missing_scan_count"],
                  None if summary is None else summary["minimum_proxy_gap_m_evaluator_only"],
                  flush=True)
    finally:
        joined.OUT, memory.OUT, joined.POST_OWNER_HOOK, joined.motion = (
            old_out, old_memory_out, old_hook, old_motion)
        route.MARGIN_M = old_margin
    (OUT / "report.json").write_text(json.dumps({
        "schema": "robot-learning-lab.robocasa-memory-held-dropout-result.v1",
        "declaration_sha256": memory.digest(declaration),
        "episodes": episodes,
    }, indent=2) + "\n")
    for source in (SOURCE, memory.SOURCE, crossing.SOURCE, Path(joined.__file__),
                   Path(timed_grid.__file__), Path(multi_tracker.__file__)):
        (OUT / source.name).write_bytes(source.read_bytes())


if __name__ == "__main__":
    main()
