"""Challenge the held response with fixture-clear oblique prop crossings."""

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
OUT = ROOT / "artifacts/ape-robocasa-memory-bearing-v656"
SEED = 6321
POST_STOP_UNTIL_S = 20.8
# The shared V603 entry ends at (1.7, -3.65). Only the subsequent 1.4 s
# crossing direction changes; the owner show and earlier return are identical.
ENDS = {"toward": (.95, -3.35), "away": (.95, -3.95)}
PROFILES = tuple((bearing, source, reaction)
                 for bearing in ENDS
                 for source in ("range", "rgbd")
                 for reaction in (False, True))


def challenge_motion(bearing: str):
    baseline = crossing.prop_motion("moving_across_return")
    end = np.asarray(ENDS[bearing], dtype=float)
    stage = np.asarray(crossing.STAGE, dtype=float)

    def moved(time_s: float):
        original_time_s = time_s - .4  # Late V625 path through common entry.
        if original_time_s <= crossing.ENTRY_END_S:
            return baseline(original_time_s)
        fraction = min(1., max(0., (original_time_s - crossing.ENTRY_END_S) /
                               (crossing.CROSS_END_S - crossing.ENTRY_END_S)))
        return stage + fraction * (end - stage)

    return moved


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    policy = json.loads(joined.CONFIG.read_text())
    case = {"id": "return_crossing_bearing",
            "from_xy_m": crossing.START, "to_xy_m": crossing.START,
            "begin_s": 14., "end_s": 14.4}
    declaration = OUT / "predeclared-bearing.json"
    declaration.write_text(json.dumps({
        "schema": "robot-learning-lab.robocasa-memory-bearing-declaration.v1",
        "profiles": PROFILES,
        "paired_scan_seed": SEED,
        "scene": case,
        "common_entry": "V603 trajectory delayed 0.4 s, identical through its 18.2 s original entry end; then a 1.4 s straight crossing from (1.7,-3.65) toward the named endpoint",
        "endpoints_xy_m": ENDS,
        "comparison": "Freeze V654 one-warning range tracking, range/RGB-D pose, dynamic-trail map in held/reactive phases, future-hold screen, unqualified known-free emergency route and 20 ms guard. Compare response off/on within each changed bearing and pose source. Do not tune on these paths.",
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
        for bearing, source, reaction in PROFILES:
            joined.motion = lambda _, bearing=bearing: challenge_motion(bearing)

            def hook(*, bearing=bearing, source=source,
                     reaction=reaction, **kwargs):
                return memory.callback(
                    source=source, item_state="retained",
                    run_label=f"{bearing}_seed{SEED}_"
                              f"{'response' if reaction else 'no_action'}",
                    predeparture_watch_s=2.4,
                    predeparture_escape=True,
                    future_probe=True,
                    hold_monitor=True,
                    held_reactive_escape=reaction,
                    hold_scan_interval_s=.1,
                    paired_scan_seed=SEED,
                    future_hold_screen=True,
                    best_effort_if_unsupported=True,
                    dynamic_trail_reopen=True,
                    dynamic_trail_reopen_during_reactive=True,
                    held_warning_confirmations=1,
                    map_snapshot_at_hazard=True,
                    escape_distances_m=(.50, .45, .35),
                    escape_tolerance_m=.09,
                    post_stop_until_s=POST_STOP_UNTIL_S,
                    **kwargs)

            joined.POST_OWNER_HOOK = hook
            response = (f"memory_{source}_retained_{bearing}_seed{SEED}_"
                        f"{'response' if reaction else 'no_action'}")
            ep = joined.episode(case, response, True, prior, policy,
                                vision, exemplar)
            episodes.append(ep)
            (OUT / "partial-report.json").write_text(json.dumps({
                "declaration_sha256": memory.digest(declaration),
                "episodes": episodes}, indent=2) + "\n")
            summary = ep["memory_return"]
            decisions = [row for row in ep["trace"]
                         if row["stage"] == "memory_held_hazard_decision"]
            print(bearing, source, reaction, summary["route_status"],
                  None if not decisions else decisions[0]["time_s"],
                  summary["minimum_proxy_gap_m_evaluator_only"], flush=True)
    finally:
        joined.OUT, memory.OUT, joined.POST_OWNER_HOOK, joined.motion = (
            old_out, old_memory_out, old_hook, old_motion)
        route.MARGIN_M = old_margin
    (OUT / "report.json").write_text(json.dumps({
        "schema": "robot-learning-lab.robocasa-memory-bearing-result.v1",
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
