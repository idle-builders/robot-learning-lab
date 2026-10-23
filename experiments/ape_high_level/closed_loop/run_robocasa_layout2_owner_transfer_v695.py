"""Transfer the full sensor-driven owner task into a second RoboCasa room."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import robocasa_owner_person as person
import robocasa_person_interaction as interaction
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_route_v407 as route
import run_robocasa_memory_return_v599 as memory
import run_robocasa_person_joined_v497 as joined


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-layout2-owner-transfer-v695"
PRIOR = ROOT / "artifacts/ape-robocasa-layout2-survey-v477/sensor-derived-map.npz"
PREFLIGHT = ROOT / "artifacts/ape-robocasa-layout2-owner-scene-v694/report.json"
OWNER_SHIFT = np.array([-.3, 0.])
PROP_START = np.array([3., -1.2])


def transfer_scene(start):
    model = person.scene_with_person(start, layout_id=2, style_id=1,
                                     clutter_mode=0)
    person.set_owner_pose(model, interaction.TORSO_XY,
                          interaction.REST_HAND, show_can=False)
    return model


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    preflight = json.loads(PREFLIGHT.read_text())
    assert preflight["owner_candidates"][0]["placement_pass"]
    assert preflight["prop_candidates"][0]["placement_pass"]
    declaration = OUT / "predeclared.json"
    declaration.write_text(json.dumps({
        "schema": "robot-learning-lab.robocasa-layout2-owner-transfer-declaration.v1",
        "room": {"layout_id": 2, "style_id": 1, "clutter_mode": 0},
        "owner_shift_xy_m": OWNER_SHIFT.tolist(),
        "static_low_prop_xy_m": PROP_START.tolist(),
        "pose_source": "range",
        "comparison": "Transfer the frozen layout4 owner route, sensor-derived map/localizer, owner shown-item recognition, taught-item memory return and continuity-aware near-body supervision to actual RoboCasa layout2/style1. Only the evaluator-preflighted room, owner/prop placement and room-specific sensor-derived prior change. This is a stationary negative control and admission screen; no moving-path claim yet.",
        "source_sha256": memory.digest(SOURCE),
        "person_scene_source_sha256": memory.digest(Path(person.__file__)),
        "joined_source_sha256": memory.digest(Path(joined.__file__)),
        "memory_source_sha256": memory.digest(memory.SOURCE),
        "prior_map_sha256": memory.digest(PRIOR),
        "preflight_sha256": memory.digest(PREFLIGHT),
    }, indent=2) + "\n")
    prior = np.load(PRIOR)
    policy = json.loads(joined.CONFIG.read_text())
    vision = owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(owner.TEACH_CROP).convert("RGB")])[0]
    old = (joined.OUT, joined.PRIOR, joined.scene_with_owner_and_prop,
           joined.motion, joined.POST_OWNER_HOOK, memory.OUT,
           interaction.TORSO_XY, interaction.REST_HAND,
           interaction.SHOW_HAND, route.MARGIN_M)
    joined.OUT = OUT
    joined.PRIOR = PRIOR
    joined.scene_with_owner_and_prop = transfer_scene
    joined.motion = lambda _: lambda t: PROP_START.copy()
    memory.OUT = OUT
    interaction.TORSO_XY = old[6] + OWNER_SHIFT
    interaction.REST_HAND = old[7] + [*OWNER_SHIFT, 0.]
    interaction.SHOW_HAND = old[8] + [*OWNER_SHIFT, 0.]
    route.MARGIN_M = .15

    def hook(**kwargs):
        return memory.callback(
            source="range", item_state="retained",
            run_label="layout2_stationary_transfer",
            predeparture_watch_s=2.4,
            predeparture_escape=True,
            predeparture_future_hold_probe=True,
            predeparture_future_hold_select=True,
            predeparture_future_hold_revalidate=True,
            predeparture_future_hold_replan_probe=True,
            predeparture_confirmed_track_hold_select=True,
            predeparture_track_continuity_gate=True,
            future_probe=True,
            hold_monitor=True,
            held_reactive_escape=True,
            hold_scan_interval_s=.1,
            paired_scan_seed=6324,
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
    case = {"id": "layout2_stationary_owner_transfer",
            "from_xy_m": PROP_START.tolist(),
            "to_xy_m": PROP_START.tolist(),
            "begin_s": 14., "end_s": 14.4}
    try:
        ep = joined.episode(case, "memory_range_retained_layout2_static",
                            True, prior, policy, vision, exemplar)
    finally:
        (joined.OUT, joined.PRIOR, joined.scene_with_owner_and_prop,
         joined.motion, joined.POST_OWNER_HOOK, memory.OUT,
         interaction.TORSO_XY, interaction.REST_HAND,
         interaction.SHOW_HAND, route.MARGIN_M) = old
    (OUT / "report.json").write_text(json.dumps({
        "schema": "robot-learning-lab.robocasa-layout2-owner-transfer-result.v1",
        "declaration_sha256": memory.digest(declaration),
        "episode": ep}, indent=2) + "\n")
    for source in (SOURCE, Path(person.__file__), Path(joined.__file__),
                   memory.SOURCE, Path(interaction.__file__)):
        (OUT / source.name).write_bytes(source.read_bytes())
    summary = ep["memory_return"] or {}
    print(ep["route_state"], ep["outcome"], summary.get("route_status"),
          summary.get("item_status"), summary.get(
              "minimum_proxy_gap_m_evaluator_only"), flush=True)


if __name__ == "__main__":
    main()
