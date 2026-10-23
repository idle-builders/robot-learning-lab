"""Test an explicitly unqualified emergency route when future hold is unsupported."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import held_pose_screen as held_screen
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_route_v407 as route
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout4_wait_and_remap_v461 as timed_grid
import multi_novel_range_objects as multi_tracker


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-memory-emergency-hold-v643"
POST_STOP_UNTIL_S = 20.8
SHIFTS_S = {"early": .4, "late": -.4}
PROFILES = (("range", "stationary", 6321, "screen_emergency"),
            ("rgbd", "stationary", 6321, "screen_emergency"),
            ("range", "early", 6321, "screen_emergency"),
            ("rgbd", "early", 6321, "screen_emergency")) + tuple(
                (source, "late", seed, mode)
                for seed in (6321, 6322)
                for source in ("range", "rgbd")
                for mode in ("baseline", "screen_emergency"))


def challenge_motion(label: str):
    if label == "stationary":
        return crossing.prop_motion("stationary")
    baseline = crossing.prop_motion("moving_across_return")
    return lambda time_s: baseline(time_s + SHIFTS_S[label])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    policy = json.loads(joined.CONFIG.read_text())
    case = {"id": "return_crossing_emergency_hold", "from_xy_m": crossing.START,
            "to_xy_m": crossing.START, "begin_s": 14., "end_s": 14.4}
    declaration = OUT / "predeclared-emergency-hold.json"
    declaration.write_text(json.dumps({
        "schema": "robot-learning-lab.robocasa-memory-emergency-hold-declaration.v1",
        "profiles": PROFILES,
        "scene": case,
        "motion": {"base_waypoints_xy_m": crossing.ENTRY_WAYPOINTS,
                   "entry_start_s": crossing.ENTRY_START_S,
                   "entry_end_s": crossing.ENTRY_END_S,
                   "cross_end_s": crossing.CROSS_END_S,
                   "early_late_shifts_s": SHIFTS_S},
        "screen": "Freeze V641's 24-path finite turn/latency/pose uncertainty screen and the same timestamped westward 5 cm warning.",
        "emergency": "If zero candidate paths and later held poses pass the V641 finite screen, keep that fact explicit but request the first currently known-free V625 perpendicular 0.35/0.45/0.25 m route as an unqualified emergency action. Keep 20 ms guard and 10 Hz monitoring; withhold item claim and owner-help state. If no current route exists, report no supported pose and do not invent free space.",
        "comparison": "Matched baseline uses V625 current-grid-only response. V641 no-emergency screen remains separately retained and is an adapted development result, not a held-out control.",
        "post_stop_until_s": POST_STOP_UNTIL_S,
        "source_sha256": memory.digest(SOURCE),
        "memory_source_sha256": memory.digest(memory.SOURCE),
        "screen_source_sha256": memory.digest(Path(held_screen.__file__)),
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
        for source, label, seed, mode in PROFILES:
            joined.motion = lambda _, label=label: challenge_motion(label)

            def hook(*, source=source, label=label, seed=seed, mode=mode,
                     **kwargs):
                return memory.callback(
                    source=source, item_state="retained",
                    run_label=f"{label}_seed{seed}_{mode}_held",
                    predeparture_watch_s=2.4,
                    predeparture_escape=True,
                    future_probe=True,
                    hold_monitor=True,
                    held_reactive_escape=True,
                    hold_scan_interval_s=.1,
                    paired_scan_seed=seed,
                    future_hold_screen=(mode == "screen_emergency"),
                    best_effort_if_unsupported=(mode == "screen_emergency"),
                    escape_distances_m=(.50, .45, .35),
                    escape_tolerance_m=.09,
                    post_stop_until_s=POST_STOP_UNTIL_S,
                    **kwargs)

            joined.POST_OWNER_HOOK = hook
            response = f"memory_{source}_retained_{label}_seed{seed}_{mode}_held"
            episode = joined.episode(case, response, True, prior, policy,
                                     vision, exemplar)
            episodes.append(episode)
            (OUT / "partial-report.json").write_text(json.dumps({
                "declaration_sha256": memory.digest(declaration),
                "episodes": episodes}, indent=2) + "\n")
            summary = episode["memory_return"]
            print(source, label, seed, mode, episode["outcome"],
                  summary["route_status"], summary["item_status"],
                  summary["minimum_proxy_gap_m_evaluator_only"], flush=True)
    finally:
        joined.OUT, memory.OUT, joined.POST_OWNER_HOOK, joined.motion = (
            old_out, old_memory_out, old_hook, old_motion)
        route.MARGIN_M = old_margin
    (OUT / "report.json").write_text(json.dumps({
        "schema": "robot-learning-lab.robocasa-memory-emergency-hold-result.v1",
        "declaration_sha256": memory.digest(declaration),
        "episodes": episodes,
    }, indent=2) + "\n")
    for source in (SOURCE, memory.SOURCE, Path(held_screen.__file__),
                   crossing.SOURCE, Path(joined.__file__),
                   Path(timed_grid.__file__), Path(multi_tracker.__file__)):
        (OUT / source.name).write_bytes(source.read_bytes())


if __name__ == "__main__":
    main()
