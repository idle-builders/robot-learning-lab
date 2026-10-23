"""Frozen V625 response against changed prop crossing speed, paired with no action."""

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
OUT = ROOT / "artifacts/ape-robocasa-memory-reactive-speed-v627"
PROFILES = tuple((speed, reaction) for speed in (.7, 1.5)
                 for reaction in (False, True))
POST_STOP_UNTIL_S = 21.2


def challenge_motion(speed: float):
    baseline = crossing.prop_motion("moving_across_return")

    def moved(time_s: float):
        original_time_s = time_s - .4  # V625 later-crossing start
        if original_time_s <= crossing.ENTRY_END_S:
            return baseline(original_time_s)
        return baseline(crossing.ENTRY_END_S + speed * (
            original_time_s - crossing.ENTRY_END_S))

    return moved


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    policy = json.loads(joined.CONFIG.read_text())
    case = {"id": "return_crossing_reactive_speed", "from_xy_m": crossing.START,
            "to_xy_m": crossing.START, "begin_s": 14., "end_s": 14.4}
    declaration = OUT / "predeclared-reactive-speed.json"
    declaration.write_text(json.dumps({
        "schema": "robot-learning-lab.robocasa-memory-reactive-speed-declaration.v1",
        "profiles": PROFILES,
        "scene": case,
        "motion": "V625 late crossing unchanged through entry end at base time 18.2s; multiply subsequent crossing clock by speed",
        "post_stop_until_s": POST_STOP_UNTIL_S,
        "response": "Frozen V625 actor-side 10 Hz held rule and bounded second escape, toggled off/on",
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
        for speed, reaction in PROFILES:
            joined.motion = lambda _, speed=speed: challenge_motion(speed)

            def hook(*, speed=speed, reaction=reaction, **kwargs):
                return memory.callback(
                    source="range", item_state="retained",
                    run_label=f"speed_{speed:.1f}_{'response' if reaction else 'no_action'}",
                    predeparture_watch_s=2.4,
                    predeparture_escape=True,
                    future_probe=True,
                    hold_monitor=True,
                    held_reactive_escape=reaction,
                    hold_scan_interval_s=.1,
                    escape_distances_m=(.50, .45, .35),
                    escape_tolerance_m=.09,
                    post_stop_until_s=POST_STOP_UNTIL_S,
                    **kwargs)

            joined.POST_OWNER_HOOK = hook
            response = f"memory_range_retained_speed_{speed:.1f}_{'response' if reaction else 'no_action'}"
            episode = joined.episode(case, response, True, prior, policy,
                                     vision, exemplar)
            episodes.append(episode)
            (OUT / "partial-report.json").write_text(json.dumps({
                "declaration_sha256": memory.digest(declaration),
                "episodes": episodes}, indent=2) + "\n")
            summary = episode["memory_return"]
            print(speed, reaction, episode["outcome"],
                  None if summary is None else summary["route_status"],
                  None if summary is None else summary["held_reactive_escape_count"],
                  None if summary is None else summary["minimum_proxy_gap_m_evaluator_only"],
                  flush=True)
    finally:
        joined.OUT, memory.OUT, joined.POST_OWNER_HOOK, joined.motion = (
            old_out, old_memory_out, old_hook, old_motion)
        route.MARGIN_M = old_margin
    (OUT / "report.json").write_text(json.dumps({
        "schema": "robot-learning-lab.robocasa-memory-reactive-speed-result.v1",
        "declaration_sha256": memory.digest(declaration),
        "episodes": episodes,
    }, indent=2) + "\n")
    for source in (SOURCE, memory.SOURCE, crossing.SOURCE, Path(joined.__file__),
                   Path(timed_grid.__file__), Path(multi_tracker.__file__)):
        (OUT / source.name).write_bytes(source.read_bytes())


if __name__ == "__main__":
    main()
