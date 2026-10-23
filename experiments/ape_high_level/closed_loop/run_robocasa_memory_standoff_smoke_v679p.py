"""Integration smoke: existing actor standoff before the full memory-return loop."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import robocasa_standoff_turn_v507 as standoff
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_route_v407 as route
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory
import run_robocasa_person_joined_v497 as joined


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-memory-standoff-smoke-v679p"
SOURCE = Path(__file__)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    policy = json.loads(joined.CONFIG.read_text())
    prior = np.load(joined.PRIOR)
    vision = owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(owner.TEACH_CROP).convert("RGB")])[0]
    script = crossing.prop_motion("stationary")
    case = {"id": "return_crossing_standoff_smoke",
            "from_xy_m": crossing.START, "to_xy_m": crossing.START,
            "begin_s": 14., "end_s": 14.4}
    old_out, old_memory_out, old_hook, old_turn, old_motion = (
        joined.OUT, memory.OUT, joined.POST_OWNER_HOOK,
        joined.owner_turn, joined.motion)
    old_prop, old_margin = standoff.PROP_SCRIPT, route.MARGIN_M
    joined.OUT = OUT
    memory.OUT = OUT
    joined.motion = lambda _: script
    standoff.PROP_SCRIPT = script
    joined.owner_turn = lambda *args: standoff.owner_turn(
        "stationary_standoff", *args[1:])
    route.MARGIN_M = .15

    def hook(**kwargs):
        return memory.callback(
            source="range", item_state="retained",
            run_label="standoff_stationary_smoke",
            predeparture_watch_s=2.4, predeparture_escape=True,
            predeparture_future_hold_probe=True,
            predeparture_future_hold_select=True,
            predeparture_future_hold_revalidate=True,
            predeparture_future_hold_replan_probe=True,
            future_probe=True, hold_monitor=True,
            held_reactive_escape=True, hold_scan_interval_s=.1,
            paired_scan_seed=6323, future_hold_screen=True,
            best_effort_if_unsupported=False,
            emergency_candidate_strategy="nominal_ranked",
            evaluator_abort_on_margin_breach=False,
            dynamic_trail_reopen=True,
            dynamic_trail_reopen_during_reactive=True,
            held_warning_confirmations=1,
            map_snapshot_at_hazard=True,
            escape_distances_m=(.50, .45, .35),
            escape_tolerance_m=.09,
            post_stop_until_s=20.8,
            **kwargs)

    joined.POST_OWNER_HOOK = hook
    try:
        episode = joined.episode(case, "standoff_stationary_smoke", True,
                                 prior, policy, vision, exemplar)
    finally:
        joined.OUT, memory.OUT, joined.POST_OWNER_HOOK, joined.owner_turn, joined.motion = (
            old_out, old_memory_out, old_hook, old_turn, old_motion)
        standoff.PROP_SCRIPT, route.MARGIN_M = old_prop, old_margin
    report = {"schema": "robot-learning-lab.robocasa-memory-standoff-smoke.v1",
              "source_sha256": memory.digest(SOURCE),
              "memory_source_sha256": memory.digest(memory.SOURCE),
              "standoff_source_sha256": memory.digest(Path(standoff.__file__)),
              "episode": episode}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(episode["outcome"], episode["memory_return"] and
          episode["memory_return"]["route_status"], flush=True)


if __name__ == "__main__":
    main()
