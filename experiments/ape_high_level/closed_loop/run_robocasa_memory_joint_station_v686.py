"""Full owner-memory episodes after joint station/refuge map admission."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import joint_meeting_refuge
import robocasa_standoff_turn_v507 as standoff
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_route_v407 as route
import run_robocasa_memory_bearing_v656 as bearing
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory
import run_robocasa_person_joined_v497 as joined


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-memory-joint-station-v686"
SEED = 6323
PROFILES = tuple((direction, source)
                 for direction in ("stationary", "toward", "away")
                 for source in ("range", "rgbd"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    policy = json.loads(joined.CONFIG.read_text())
    case = {"id": "return_crossing_joint_station",
            "from_xy_m": crossing.START, "to_xy_m": crossing.START,
            "begin_s": 14., "end_s": 14.4}
    declaration = OUT / "predeclared.json"
    declaration.write_text(json.dumps({
        "schema": "robot-learning-lab.robocasa-memory-joint-station-declaration.v1",
        "profiles": PROFILES,
        "paired_scan_seed": SEED,
        "station_offsets_xy_m": joint_meeting_refuge.MEETING_OFFSETS_XY_M,
        "refuge_offsets_xy_m": joint_meeting_refuge.REFUGE_OFFSETS_XY_M,
        "case": case,
        "comparison": "Try an actor-map-backed owner-show station only when a distinct refuge route is already observed free. The selector cannot see future scripted prop paths. Run complete stationary/toward/away owner-memory tasks under range/RGB-D pose, confirmed-track early screen, per-command revalidation and 20ms monitor. Score owner/item view, completed taught-item return, later proxy geometry, timing and help. This single-layout/seed cohort is a feasibility pilot, not a qualified sensor or body result.",
        "source_sha256": memory.digest(SOURCE),
        "selector_source_sha256": memory.digest(Path(joint_meeting_refuge.__file__)),
        "standoff_source_sha256": memory.digest(Path(standoff.__file__)),
        "memory_source_sha256": memory.digest(memory.SOURCE),
        "prior_map_sha256": memory.digest(joined.PRIOR),
    }, indent=2) + "\n")
    prior = np.load(joined.PRIOR)
    vision = owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(owner.TEACH_CROP).convert("RGB")])[0]
    old_out, old_memory_out, old_hook, old_margin, old_motion, old_turn = (
        joined.OUT, memory.OUT, joined.POST_OWNER_HOOK,
        route.MARGIN_M, joined.motion, joined.owner_turn)
    old_prop_script = standoff.PROP_SCRIPT
    joined.OUT = OUT
    memory.OUT = OUT
    route.MARGIN_M = .15
    episodes = []
    try:
        for direction, source in PROFILES:
            script = (crossing.prop_motion("stationary") if direction == "stationary"
                      else bearing.challenge_motion(direction))
            joined.motion = lambda _, script=script: script
            standoff.PROP_SCRIPT = script
            joined.owner_turn = lambda *args: standoff.owner_turn(
                "stationary_standoff", *args[1:],
                station_selector=joint_meeting_refuge.choose)

            def hook(*, direction=direction, source=source, **kwargs):
                return memory.callback(
                    source=source, item_state="retained",
                    run_label=f"{direction}_seed{SEED}_joint_station",
                    predeparture_watch_s=2.4,
                    predeparture_escape=True,
                    predeparture_future_hold_probe=True,
                    predeparture_future_hold_select=True,
                    predeparture_future_hold_revalidate=True,
                    predeparture_future_hold_replan_probe=True,
                    predeparture_confirmed_track_hold_select=True,
                    future_probe=True,
                    hold_monitor=True,
                    held_reactive_escape=True,
                    hold_scan_interval_s=.1,
                    paired_scan_seed=SEED,
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
                    post_stop_until_s=20.8,
                    **kwargs)

            joined.POST_OWNER_HOOK = hook
            response = f"memory_{source}_{direction}_joint_station"
            ep = joined.episode(case, response, True, prior, policy,
                                vision, exemplar)
            episodes.append(ep)
            (OUT / "partial-report.json").write_text(json.dumps({
                "declaration_sha256": memory.digest(declaration),
                "episodes": episodes}, indent=2) + "\n")
            chosen = next((r["chosen"] for r in ep["trace"] if
                           r["stage"] == "choose_owner_standoff"), None)
            summary = ep["memory_return"] or {}
            print(direction, source, ep["outcome"],
                  "station", None if chosen is None else chosen["target_xy_est_m"],
                  "refuge", None if chosen is None else chosen.get("selected_refuge"),
                  "return", summary.get("route_status"),
                  "gap", summary.get("minimum_proxy_gap_m_evaluator_only"), flush=True)
    finally:
        joined.OUT, memory.OUT, joined.POST_OWNER_HOOK, joined.motion, joined.owner_turn = (
            old_out, old_memory_out, old_hook, old_motion, old_turn)
        standoff.PROP_SCRIPT = old_prop_script
        route.MARGIN_M = old_margin
    (OUT / "report.json").write_text(json.dumps({
        "schema": "robot-learning-lab.robocasa-memory-joint-station-result.v1",
        "declaration_sha256": memory.digest(declaration),
        "episodes": episodes}, indent=2) + "\n")
    for source in (SOURCE, Path(joint_meeting_refuge.__file__),
                   Path(standoff.__file__), memory.SOURCE,
                   bearing.SOURCE, crossing.SOURCE, Path(joined.__file__)):
        (OUT / source.name).write_bytes(source.read_bytes())


if __name__ == "__main__":
    main()
