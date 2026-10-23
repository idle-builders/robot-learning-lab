"""Frozen changed-room and opposite-side prop challenge for V256p.

The actor is the existing medium-depth information-view/help loop. This file
declares scene variation before rendering or running any owner episode.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from depth_quality import reject_isolated_depth_spikes
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, sha
import run_blocked_forward_viewpoint_v321 as live
import run_heldout_moving_low_v305 as historical_room
import run_no_supported_view_owner_help_v256p as owner_help
from run_medium_depth_recovery_v246p import MediumWindowMap, profile_depth


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-articulated-depth-route-v258p"
# Seed, mode, near-side distance, arrival interval, departure interval.
CASES = (
    (178, "present", .30, 9.8, 10.4, 10.8, 12.8),
    (180, "present", .30, 9.8, 10.4, 10.8, 12.8),
    (182, "present", .30, 9.2, 9.8, 10.2, 11.8),
    (184, "present", .30, 10.6, 11.2, 12.0, 14.0),
)
# Declared before actor exposure. The sign changes which flank the moving
# prop approaches; furniture jitter is the same deterministic bound in all.
SCENE = {
    178: {"prop_side": 1, "prop_x_m": .02},
    180: {"prop_side": -1, "prop_x_m": -.02},
    182: {"prop_side": 1, "prop_x_m": .03},
    184: {"prop_side": -1, "prop_x_m": -.03},
}
FURNITURE_JITTER_XY_M = {
    "crate": (.025, .025), "box": (.025, .03),
    "shelf": (.02, .03), "sofa": (.03, .025),
    "lamp": (.02, .025),
}
COMPLIANT_SEEDS = (178, 182)
SENSOR_PROFILE = "virtual head depth 30cm-1m; same near-range and shin"
INITIAL_VETO_HOOK = None
ACTOR_POLICY = "V256p unchanged; no scene label enters decisions"
AFTER_FIRST_VIEW_HOOK = owner_help.after_first_view
OWNER_RESPONSES = None
POSE_SOURCE = "command"


class VariedMovingRoom(historical_room.HeldoutMovingRoom):
    def __init__(self, *, case: tuple, pose_source: str,
                 branch: Path) -> None:
        self.scene_spec = SCENE[case[0]]
        super().__init__(case=case, pose_source=pose_source, branch=branch)

    def layout(self, seed: int, absent: bool, *, furnished: bool = False) -> None:
        super().layout(seed, absent, furnished=furnished)
        if not furnished:
            return
        rng = np.random.default_rng(258000 + seed)
        for name, bounds in FURNITURE_JITTER_XY_M.items():
            body = self.model.body(f"loop_{name}")
            place = self.model.body_pos[body.id].copy()
            place[:2] += rng.uniform(-np.asarray(bounds), np.asarray(bounds))
            self.model.body_pos[body.id] = place
        for name, base in (("room_left_wall", (.73, .75, .72)),
                           ("room_right_wall", (.70, .73, .74))):
            rgb = np.clip(np.asarray(base) + rng.uniform(-.06, .06, 3),
                          0., 1.)
            self.model.geom_rgba[self.model.geom(name).id] = (*rgb, 1.)
        self._forward()

    def _update_moving_obstacle(self) -> None:
        if self.motion_side is None:
            return
        y = historical_room.scheduled_y(self.time_s, self.case)
        xy = (self.scene_spec["prop_x_m"],
              self.scene_spec["prop_side"] * y)
        self.obstacle_xy[3] = xy
        self.model.body_pos[self.model.body("loop_stool").id] = (
            *xy, historical_room.STOOL_SIZE[2])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    declaration_path = OUT / "predeclared-challenge.json"
    declaration_path.write_text(json.dumps({
        "schema": "robot-learning-lab.varied-owner-route-challenge.v1",
        "source_sha256": sha(SOURCE),
        "historical_visual_sha256": sha(selected_visual()),
        "cases": CASES, "scene": SCENE,
        "furniture_jitter_xy_m": FURNITURE_JITTER_XY_M,
        "compliant_seeds": COMPLIANT_SEEDS,
        "owner_responses": OWNER_RESPONSES,
        "sensor_profile": SENSOR_PROFILE,
        "actor_policy": ACTOR_POLICY,
    }, indent=2) + "\n")
    old_room = historical_room.HeldoutMovingRoom
    old_live_room = live.HeldoutMovingRoom
    old_cases = live.CASES
    old_depth = live.noisy_depth
    old_map = live.DepthVoxelMap
    old_compliance = owner_help.COMPLIANT_SEEDS
    try:
        historical_room.HeldoutMovingRoom = VariedMovingRoom
        live.HeldoutMovingRoom = VariedMovingRoom
        live.CASES = CASES
        live.noisy_depth = profile_depth("medium_30cm_1m", old_depth)
        live.DepthVoxelMap = MediumWindowMap
        owner_help.COMPLIANT_SEEDS = COMPLIANT_SEEDS
        preflight = [historical_room.screen_case(case) for case in CASES]
        (OUT / "geometry-preflight.json").write_text(json.dumps({
            "scope": "Evaluator-only V305 full-body and inter-prop geometry "
                     "screen at 20ms, now on the predeclared varied rooms. "
                     "Labels do not enter the actor.",
            "rows": preflight}, indent=2) + "\n")
        live.main(output_root=OUT, source=SOURCE,
                  map_depth_filter=reject_isolated_depth_spikes,
                  experiment_label="V258p changed-room owner route",
                  after_first_view=AFTER_FIRST_VIEW_HOOK,
                  max_recovery_steps=3,
                  on_initial_veto=INITIAL_VETO_HOOK,
                  pose_source=POSE_SOURCE)
    finally:
        historical_room.HeldoutMovingRoom = old_room
        live.HeldoutMovingRoom = old_live_room
        live.CASES = old_cases
        live.noisy_depth = old_depth
        live.DepthVoxelMap = old_map
        owner_help.COMPLIANT_SEEDS = old_compliance
    path = OUT / "blocked-forward-viewpoint-audit.json"
    report = json.loads(path.read_text())
    report["scope_override"] = (
        "Frozen V256p medium virtual head-depth actor plus changed furniture "
        "position/color, mirrored moving prop side and timing in four new "
        "historical Direct Frame owner episodes. V305 evaluator geometry "
        "preflight is scored separately. Actor receives only its own camera, "
        "near-range, shin and estimated state; all motion remains kinematic.")
    path.write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "report.json").write_text(json.dumps({
        "schema": "robot-learning-lab.varied-owner-route-result.v1",
        "source_sha256": sha(SOURCE),
        "declaration_sha256": sha(declaration_path),
        "geometry_preflight_sha256": sha(OUT / "geometry-preflight.json"),
        "runner_report_sha256": sha(path),
        "rows": [{"case": row["case"], "status": row["status"],
                  "recovery_gate": (None if row["recovery_gate"] is None
                                    else row["recovery_gate"]["status"]),
                  "base_moves": row["base_moves"],
                  "owner_questions": row["owner_questions"],
                  "minimum_clearance_m": row[
                      "evaluator_path_min_clearance_m"]}
                 for row in report["rows"]]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
