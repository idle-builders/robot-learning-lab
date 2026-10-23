"""Compare a fixed early hold route with revalidation at every action scan."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import dynamic_trail_map
import head_depth_upper_route
import held_pose_screen
import nominal_emergency_rank
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_route_v407 as route
import run_robocasa_memory_bearing_v656 as bearing_trial
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout4_wait_and_remap_v461 as timed_grid
import multi_novel_range_objects as multi_tracker


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-memory-early-hold-revalidate-v675"
SEEDS = (6323, 6324)
POST_STOP_UNTIL_S = 20.8
BRANCHES = ("early_route_hold", "revalidate_route_hold")
PROFILES = tuple((seed, direction, source, branch)
                 for seed in SEEDS
                 for direction in ("stationary", "toward", "away")
                 for source in ("range",)
                 for branch in BRANCHES)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    policy = json.loads(joined.CONFIG.read_text())
    case = {"id": "return_crossing_early_hold_revalidation",
            "from_xy_m": crossing.START, "to_xy_m": crossing.START,
            "begin_s": 14., "end_s": 14.4}
    declaration = OUT / "predeclared-early-hold-revalidation.json"
    declaration.write_text(json.dumps({
        "schema": "robot-learning-lab.robocasa-memory-early-hold-revalidation-declaration.v1",
        "profiles": PROFILES,
        "paired_scan_seeds": SEEDS,
        "scene": case,
        "oblique_endpoints_xy_m": bearing_trial.ENDS,
        "comparison": "At the same 15.8s predeparture warning both branches select the same best modeled route-plus-held-pose goal. The baseline executes that fixed goal. The changed branch recomputes support for the remaining observed-grid path and held pose from each fresh range track before its next 400ms motion command; on missing track or lost support it stops, asks the owner and monitors while held. Two seeds, stationary/toward/away prop, range pose. No evaluator-truth motion abort. Save complete owner videos and 20ms traces; compare exact predecision sensor pairs, invalidation time, task result and actual route-plus-held gap. This is a fail-closed revalidation experiment, not a safety certificate or deployable response.",
        "post_stop_until_s": POST_STOP_UNTIL_S,
        "source_sha256": memory.digest(SOURCE),
        "memory_source_sha256": memory.digest(memory.SOURCE),
        "upper_source_sha256": memory.digest(Path(head_depth_upper_route.__file__)),
        "rank_source_sha256": memory.digest(Path(nominal_emergency_rank.__file__)),
        "trail_source_sha256": memory.digest(Path(dynamic_trail_map.__file__)),
        "screen_source_sha256": memory.digest(Path(held_pose_screen.__file__)),
        "bearing_source_sha256": memory.digest(bearing_trial.SOURCE),
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
        for seed, direction, source, branch in PROFILES:
            joined.motion = lambda _, direction=direction: (
                crossing.prop_motion("stationary") if direction == "stationary"
                else bearing_trial.challenge_motion(direction))

            def hook(*, direction=direction, source=source,
                     seed=seed, branch=branch, **kwargs):
                return memory.callback(
                    source=source, item_state="retained",
                    run_label=f"{direction}_seed{seed}_{branch}",
                    predeparture_watch_s=2.4,
                    predeparture_escape=True,
                    predeparture_future_hold_probe=True,
                    predeparture_future_hold_select=True,
                    predeparture_future_hold_revalidate=(
                        branch == "revalidate_route_hold"),
                    future_probe=True,
                    hold_monitor=True,
                    held_reactive_escape=True,
                    hold_scan_interval_s=.1,
                    paired_scan_seed=seed,
                    future_hold_screen=True,
                    best_effort_if_unsupported=False,
                    emergency_candidate_strategy="nominal_ranked",
                    evaluator_abort_on_margin_breach=False,
                    dynamic_trail_reopen=True,
                    dynamic_trail_reopen_during_reactive=True,
                    held_warning_confirmations=1,
                    map_snapshot_at_hazard=True,
                    escape_distances_m=(.50, .45, .35),
                    escape_tolerance_m=.09,
                    post_stop_until_s=POST_STOP_UNTIL_S,
                    **kwargs)

            joined.POST_OWNER_HOOK = hook
            response = (f"memory_{source}_retained_{direction}_seed{seed}_"
                        f"{branch}")
            ep = joined.episode(case, response, True, prior, policy,
                                vision, exemplar)
            episodes.append(ep)
            (OUT / "partial-report.json").write_text(json.dumps({
                "declaration_sha256": memory.digest(declaration),
                "episodes": episodes}, indent=2) + "\n")
            summary = ep["memory_return"]
            decisions = [row for row in ep["trace"]
                         if row["stage"] == "memory_held_hazard_decision"]
            early = next((row for row in ep["trace"]
                          if row["stage"] == "predeparture_range_scan" and
                          "actor_early_future_hold_probe" in row), None)
            print(seed, direction, source, branch,
                  None if early is None else early[
                      "actor_early_future_hold_probe"]["supported_candidate_count"],
                  None if early is None else early.get(
                      "actor_early_selected_goal_xy_m"),
                  summary["route_status"],
                  summary["minimum_proxy_gap_m_evaluator_only"], flush=True)
    finally:
        joined.OUT, memory.OUT, joined.POST_OWNER_HOOK, joined.motion = (
            old_out, old_memory_out, old_hook, old_motion)
        route.MARGIN_M = old_margin
    (OUT / "report.json").write_text(json.dumps({
        "schema": "robot-learning-lab.robocasa-memory-early-hold-revalidation-result.v1",
        "declaration_sha256": memory.digest(declaration),
        "episodes": episodes,
    }, indent=2) + "\n")
    for source in (SOURCE, memory.SOURCE, Path(head_depth_upper_route.__file__),
                   Path(nominal_emergency_rank.__file__),
                   Path(dynamic_trail_map.__file__),
                   Path(held_pose_screen.__file__), bearing_trial.SOURCE,
                   crossing.SOURCE, Path(joined.__file__),
                   Path(timed_grid.__file__), Path(multi_tracker.__file__)):
        (OUT / source.name).write_bytes(source.read_bytes())


if __name__ == "__main__":
    main()
