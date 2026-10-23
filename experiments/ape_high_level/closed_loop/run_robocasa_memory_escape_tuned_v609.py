"""Pair owner-memory controls with a scan-triggered longer predeparture escape."""

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


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-memory-escape-tuned-v609"
WATCH_S = 2.4
PROFILES = crossing.PROFILES


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    policy = json.loads(joined.CONFIG.read_text())
    case = {"id": "return_crossing_escape_tuned", "from_xy_m": crossing.START,
            "to_xy_m": crossing.START, "begin_s": 14., "end_s": 14.4}
    declaration = OUT / "predeclared-escape-tuned.json"
    declaration.write_text(json.dumps({
        "schema": "robot-learning-lab.robocasa-memory-escape-tuned-declaration.v1",
        "scene": case, "profiles": PROFILES,
        "motion": {"entry_waypoints_xy_m": crossing.ENTRY_WAYPOINTS,
                   "entry_start_s": crossing.ENTRY_START_S,
                   "entry_end_s": crossing.ENTRY_END_S,
                   "cross_to_xy_m": crossing.END,
                   "cross_end_s": crossing.CROSS_END_S},
        "watch_s": WATCH_S,
        "escape_rule": "At two consecutive 0.4 s scans, >=3 adjacent beams in an eight-index span shorten >=7 cm to <1 m; if weighted centers drift >=0.5 beam, propose a 0.50/0.45/0.35 m known-free tangent, with 0.05 m arrival tolerance in the direction of drift. Execute bounded navigation, then ask owner and hold through 19.8 s.",
        "post_stop_until_s": crossing.POST_STOP_UNTIL_S,
        "source_sha256": memory.digest(SOURCE),
        "memory_source_sha256": memory.digest(memory.SOURCE),
        "motion_source_sha256": memory.digest(crossing.SOURCE),
        "joined_source_sha256": memory.digest(Path(joined.__file__)),
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
        for source, motion_label in PROFILES:
            joined.motion = lambda _, motion_label=motion_label: (
                crossing.prop_motion(motion_label))

            def hook(*, source=source, motion_label=motion_label, **kwargs):
                return memory.callback(
                    source=source, item_state="retained",
                    run_label=f"{motion_label}_escape_tuned",
                    predeparture_watch_s=WATCH_S,
                    predeparture_escape=True,
                    escape_distances_m=(.50, .45, .35),
                    escape_tolerance_m=.05,
                    post_stop_until_s=crossing.POST_STOP_UNTIL_S,
                    **kwargs)

            joined.POST_OWNER_HOOK = hook
            response = f"memory_{source}_retained_{motion_label}_escape_tuned"
            episode = joined.episode(case, response, True, prior, policy,
                                     vision, exemplar)
            episodes.append(episode)
            (OUT / "partial-report.json").write_text(json.dumps({
                "declaration_sha256": memory.digest(declaration),
                "episodes": episodes}, indent=2) + "\n")
            summary = episode["memory_return"]
            print(source, motion_label, episode["outcome"],
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
        "schema": "robot-learning-lab.robocasa-memory-escape-tuned-result.v1",
        "declaration_sha256": memory.digest(declaration),
        "source_sha256": memory.digest(SOURCE),
        "memory_source_sha256": memory.digest(memory.SOURCE),
        "motion_source_sha256": memory.digest(crossing.SOURCE),
        "joined_source_sha256": memory.digest(Path(joined.__file__)),
        "episodes": episodes,
    }, indent=2) + "\n")
    for source in (SOURCE, memory.SOURCE, crossing.SOURCE, Path(joined.__file__)):
        (OUT / source.name).write_bytes(source.read_bytes())


if __name__ == "__main__":
    main()
