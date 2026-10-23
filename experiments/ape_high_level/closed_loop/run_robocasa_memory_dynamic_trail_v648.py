"""Pair permanent obstacle hits against observed-clear moving-track trail reopening."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import dynamic_trail_map
import held_pose_screen
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_route_v407 as route
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout4_wait_and_remap_v461 as timed_grid
import multi_novel_range_objects as multi_tracker


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-memory-dynamic-trail-v648"
POST_STOP_UNTIL_S = 20.8
PROFILES = tuple((source, "late", seed, reopen)
                 for seed in (6321, 6322)
                 for source in ("range", "rgbd")
                 for reopen in (False, True)) + (
                     ("range", "stationary", 6321, True),
                     ("rgbd", "stationary", 6321, True),
                     ("range", "early", 6321, True),
                     ("rgbd", "early", 6321, True))


def challenge_motion(label):
    if label == "stationary":
        return crossing.prop_motion("stationary")
    baseline = crossing.prop_motion("moving_across_return")
    return lambda time_s: baseline(time_s + ({"late": -.4, "early": .4}[label]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    policy = json.loads(joined.CONFIG.read_text())
    case = {"id": "return_crossing_dynamic_trail", "from_xy_m": crossing.START,
            "to_xy_m": crossing.START, "begin_s": 14., "end_s": 14.4}
    declaration = OUT / "predeclared-dynamic-trail.json"
    declaration.write_text(json.dumps({
        "schema": "robot-learning-lab.robocasa-memory-dynamic-trail-declaration.v1",
        "profiles": PROFILES,
        "scene": case,
        "motion": "V643 stationary, early(+0.4s) or late(-0.4s) shifted V603 scripted prop",
        "map_change": "Only during fresh held 10 Hz range scans, identify confirmed moving compact tracks, their previously confirmed centers and current envelope. Reopen old occupied trail cells only when two neighboring finite current beams reach beyond each cell by the 17 cm body plus 15 cm grid margin. Keep current object footprint occupied; missing beams never clear. Compare with permanent-occupancy baseline, retain full map at warning.",
        "actor_action": "Frozen V643 future-hold screen plus explicitly unqualified known-free emergency perpendicular response when unsupported. Both pose-source branches keep 20 ms guard, owner help and no item claim.",
        "post_stop_until_s": POST_STOP_UNTIL_S,
        "source_sha256": memory.digest(SOURCE),
        "memory_source_sha256": memory.digest(memory.SOURCE),
        "trail_source_sha256": memory.digest(Path(dynamic_trail_map.__file__)),
        "screen_source_sha256": memory.digest(Path(held_pose_screen.__file__)),
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
        for source, label, seed, reopen in PROFILES:
            joined.motion = lambda _, label=label: challenge_motion(label)

            def hook(*, source=source, label=label, seed=seed,
                     reopen=reopen, **kwargs):
                return memory.callback(
                    source=source, item_state="retained",
                    run_label=f"{label}_seed{seed}_{'reopen' if reopen else 'permanent'}_held",
                    predeparture_watch_s=2.4,
                    predeparture_escape=True,
                    future_probe=True,
                    hold_monitor=True,
                    held_reactive_escape=True,
                    hold_scan_interval_s=.1,
                    paired_scan_seed=seed,
                    future_hold_screen=True,
                    best_effort_if_unsupported=True,
                    dynamic_trail_reopen=reopen,
                    map_snapshot_at_hazard=True,
                    escape_distances_m=(.50, .45, .35),
                    escape_tolerance_m=.09,
                    post_stop_until_s=POST_STOP_UNTIL_S,
                    **kwargs)

            joined.POST_OWNER_HOOK = hook
            response = (f"memory_{source}_retained_{label}_seed{seed}_"
                        f"{'reopen' if reopen else 'permanent'}_held")
            episode = joined.episode(case, response, True, prior, policy,
                                     vision, exemplar)
            episodes.append(episode)
            (OUT / "partial-report.json").write_text(json.dumps({
                "declaration_sha256": memory.digest(declaration),
                "episodes": episodes}, indent=2) + "\n")
            summary = episode["memory_return"]
            decision = next((row for row in episode["trace"]
                             if row["stage"] == "memory_held_hazard_decision"), None)
            print(source, label, seed, reopen, summary["route_status"],
                  summary["dynamic_trail_reclaimed_cells"],
                  None if decision is None else sum(
                      option["plan_status"] == "known_goal"
                      for option in decision["actor_candidate_goals"]),
                  summary["minimum_proxy_gap_m_evaluator_only"], flush=True)
    finally:
        joined.OUT, memory.OUT, joined.POST_OWNER_HOOK, joined.motion = (
            old_out, old_memory_out, old_hook, old_motion)
        route.MARGIN_M = old_margin
    (OUT / "report.json").write_text(json.dumps({
        "schema": "robot-learning-lab.robocasa-memory-dynamic-trail-result.v1",
        "declaration_sha256": memory.digest(declaration),
        "episodes": episodes,
    }, indent=2) + "\n")
    for source in (SOURCE, memory.SOURCE, Path(dynamic_trail_map.__file__),
                   Path(held_pose_screen.__file__), crossing.SOURCE,
                   Path(joined.__file__), Path(timed_grid.__file__),
                   Path(multi_tracker.__file__)):
        (OUT / source.name).write_bytes(source.read_bytes())


if __name__ == "__main__":
    main()
