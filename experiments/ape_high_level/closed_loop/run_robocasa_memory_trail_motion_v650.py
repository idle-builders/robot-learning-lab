"""Keep observed-cleared moving trails through reactive motion replanning."""

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
OUT = ROOT / "artifacts/ape-robocasa-memory-trail-motion-v650"
POST_STOP_UNTIL_S = 20.8
PROFILES = (("rgbd", "early", 6321, False),
            ("rgbd", "early", 6321, True),
            ("range", "late", 6321, False),
            ("range", "late", 6321, True),
            ("rgbd", "late", 6321, False),
            ("rgbd", "late", 6321, True),
            ("rgbd", "stationary", 6321, True))


def challenge_motion(label):
    if label == "stationary":
        return crossing.prop_motion("stationary")
    baseline = crossing.prop_motion("moving_across_return")
    return lambda time_s: baseline(time_s + ({"late": -.4, "early": .4}[label]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    policy = json.loads(joined.CONFIG.read_text())
    case = {"id": "return_crossing_trail_motion", "from_xy_m": crossing.START,
            "to_xy_m": crossing.START, "begin_s": 14., "end_s": 14.4}
    declaration = OUT / "predeclared-trail-motion.json"
    declaration.write_text(json.dumps({
        "schema": "robot-learning-lab.robocasa-memory-trail-motion-declaration.v1",
        "profiles": PROFILES,
        "scene": case,
        "motion": "V648 shifted V603 prop: early +0.4s, late -0.4s or stationary",
        "comparison": "Both branches use V648 observed-clear dynamic trail reopening at every held scan. Integration branch applies the same rule after each fresh reactive-motion scan, before replanning; control reverts to permanent occupied hits inside execute_phase. Same finite future-hold screen, explicitly unqualified known-free emergency route, 20 ms guard and owner-help/no-item-claim state.",
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
        for source, label, seed, during_motion in PROFILES:
            joined.motion = lambda _, label=label: challenge_motion(label)

            def hook(*, source=source, label=label, seed=seed,
                     during_motion=during_motion, **kwargs):
                return memory.callback(
                    source=source, item_state="retained",
                    run_label=f"{label}_seed{seed}_{'motion' if during_motion else 'held_only'}_trail",
                    predeparture_watch_s=2.4,
                    predeparture_escape=True,
                    future_probe=True,
                    hold_monitor=True,
                    held_reactive_escape=True,
                    hold_scan_interval_s=.1,
                    paired_scan_seed=seed,
                    future_hold_screen=True,
                    best_effort_if_unsupported=True,
                    dynamic_trail_reopen=True,
                    dynamic_trail_reopen_during_reactive=during_motion,
                    map_snapshot_at_hazard=True,
                    escape_distances_m=(.50, .45, .35),
                    escape_tolerance_m=.09,
                    post_stop_until_s=POST_STOP_UNTIL_S,
                    **kwargs)

            joined.POST_OWNER_HOOK = hook
            response = (f"memory_{source}_retained_{label}_seed{seed}_"
                        f"{'motion' if during_motion else 'held_only'}_trail")
            ep = joined.episode(case, response, True, prior, policy,
                                vision, exemplar)
            episodes.append(ep)
            (OUT / "partial-report.json").write_text(json.dumps({
                "declaration_sha256": memory.digest(declaration),
                "episodes": episodes}, indent=2) + "\n")
            summary = ep["memory_return"]
            reactive_samples = sum(len(row["motion_substeps"])
                                   for row in ep["trace"]
                                   if row["stage"] == "memory_route_decision" and
                                   row["phase"] == "evade_reactive")
            print(source, label, during_motion, summary["route_status"],
                  reactive_samples, summary["minimum_proxy_gap_m_evaluator_only"],
                  flush=True)
    finally:
        joined.OUT, memory.OUT, joined.POST_OWNER_HOOK, joined.motion = (
            old_out, old_memory_out, old_hook, old_motion)
        route.MARGIN_M = old_margin
    (OUT / "report.json").write_text(json.dumps({
        "schema": "robot-learning-lab.robocasa-memory-trail-motion-result.v1",
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
