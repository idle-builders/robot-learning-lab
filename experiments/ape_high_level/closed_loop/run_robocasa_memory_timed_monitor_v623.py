"""Compare 2.5 and 10 Hz held scans with real capture timestamps."""

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
OUT = ROOT / "artifacts/ape-robocasa-memory-timed-monitor-v623"
WATCH_S = 2.4
PROFILES = tuple((source, label, interval)
                 for interval in (.4, .1)
                 for label in ("early", "late")
                 for source in ("range", "rgbd"))
SHIFTS_S = {"early": .4, "late": -.4}
POST_STOP_UNTIL_S = 20.4


def challenge_motion(label: str):
    baseline = crossing.prop_motion("moving_across_return")
    return lambda time_s: baseline(time_s + SHIFTS_S[label])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    policy = json.loads(joined.CONFIG.read_text())
    case = {"id": "return_crossing_timed_monitor", "from_xy_m": crossing.START,
            "to_xy_m": crossing.START, "begin_s": 14., "end_s": 14.4}
    declaration = OUT / "predeclared-timed-monitor.json"
    declaration.write_text(json.dumps({
        "schema": "robot-learning-lab.robocasa-memory-timed-monitor-declaration.v1",
        "scene": case, "profiles": PROFILES,
        "timing_shifts_s": SHIFTS_S,
        "motion": {"entry_waypoints_xy_m": crossing.ENTRY_WAYPOINTS,
                   "entry_start_s": crossing.ENTRY_START_S,
                   "entry_end_s": crossing.ENTRY_END_S,
                   "cross_to_xy_m": crossing.END,
                   "cross_end_s": crossing.CROSS_END_S},
        "watch_s": WATCH_S,
        "escape_rule": "At two consecutive 0.4 s scans, >=3 adjacent beams in an eight-index span shorten >=7 cm to <1 m; if weighted centers drift >=0.5 beam, propose a 0.50/0.45/0.35 m known-free tangent in the direction of drift. Use 0.09 m arrival tolerance with a 0.10 m occupancy grid, execute bounded navigation, ask owner, and hold through 20.4 s.",
        "post_stop_until_s": POST_STOP_UNTIL_S,
        "monitor_rule": "After the initial escape outcome, keep 20 ms raw guard and compare 0.4 versus 0.1 s dual-height held range scans. Every grid/tracker update receives the actual capture time. Save actor packets; this diagnostic does not yet dispatch a second action.",
        "held_scan_intervals_s": [.4, .1],
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
        for source, motion_label, interval_s in PROFILES:
            joined.motion = lambda _, motion_label=motion_label: (
                challenge_motion(motion_label))

            def hook(*, source=source, motion_label=motion_label,
                     interval_s=interval_s, **kwargs):
                return memory.callback(
                    source=source, item_state="retained",
                    run_label=f"{motion_label}_{int(interval_s * 1000)}ms_timed",
                    predeparture_watch_s=WATCH_S,
                    predeparture_escape=True,
                    future_probe=True,
                    hold_monitor=True,
                    hold_scan_interval_s=interval_s,
                    escape_distances_m=(.50, .45, .35),
                    escape_tolerance_m=.09,
                    post_stop_until_s=POST_STOP_UNTIL_S,
                    **kwargs)

            joined.POST_OWNER_HOOK = hook
            response = f"memory_{source}_retained_{motion_label}_{int(interval_s * 1000)}ms_timed"
            episode = joined.episode(case, response, True, prior, policy,
                                     vision, exemplar)
            episodes.append(episode)
            (OUT / "partial-report.json").write_text(json.dumps({
                "declaration_sha256": memory.digest(declaration),
                "episodes": episodes}, indent=2) + "\n")
            summary = episode["memory_return"]
            print(source, motion_label, interval_s, episode["outcome"],
                  None if summary is None else summary["route_status"],
                  None if summary is None else summary["item_status"],
                  None if summary is None else summary["predeparture_watch_trigger"],
                  None if summary is None else summary["minimum_proxy_gap_m_evaluator_only"],
                  flush=True)
    finally:
        joined.OUT, memory.OUT, joined.POST_OWNER_HOOK, joined.motion = (
            old_out, old_memory_out, old_hook, old_motion)
        route.MARGIN_M = old_margin
    (OUT / "report.json").write_text(json.dumps({
        "schema": "robot-learning-lab.robocasa-memory-timed-monitor-result.v1",
        "declaration_sha256": memory.digest(declaration),
        "source_sha256": memory.digest(SOURCE),
        "memory_source_sha256": memory.digest(memory.SOURCE),
        "motion_source_sha256": memory.digest(crossing.SOURCE),
        "joined_source_sha256": memory.digest(Path(joined.__file__)),
        "timed_grid_source_sha256": memory.digest(Path(timed_grid.__file__)),
        "multi_tracker_source_sha256": memory.digest(Path(multi_tracker.__file__)),
        "episodes": episodes,
    }, indent=2) + "\n")
    for source in (SOURCE, memory.SOURCE, crossing.SOURCE, Path(joined.__file__),
                   Path(timed_grid.__file__), Path(multi_tracker.__file__)):
        (OUT / source.name).write_bytes(source.read_bytes())


if __name__ == "__main__":
    main()
