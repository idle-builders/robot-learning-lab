"""V191p: full owner turns with changed, collision-audited furniture.

These are diagnostic held-pose counterfactuals. The historical body cannot
enter this pose under the current pre-show unknown-space admission rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from articulated_person_scene import (FAR_HAND_XYZ_M, START_HAND_XYZ_M,
                                      add_person, set_person_pose)
from audit_direct_structural_entry_v271 import V149, V150
from build_scene_pairs_v133 import body_geoms, scene_geoms
import direct_structural_rooms_v270 as structural
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, Vision, sha
import run_occluded_camera_guard_v174p as v174p
import run_pre_show_reserve_screen_v100 as v100
import run_standoff_scene_variants_v160p as v160p
import run_standoff_yield_recovery_v153p as v153p
import run_live_articulated_owner_v343 as owner


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-varied-furniture-owner"
SEEDS = (170, 172)
RESPONSES_S = (.2, None)
LAYOUTS = {
    "open_corner": {"table_xyz_m": (.76, -.59, 0.),
                    "prop_variant": "fast_wide_turn"},
    "near_side_table": {"table_xyz_m": (-.38, .51, 0.),
                        "prop_variant": "fast_wide_turn"},
    "offset_side_table": {"table_xyz_m": (-.46, .58, 0.),
                          "prop_variant": "slow_narrow_offset"},
}


class FurnitureRoom(v160p.VariantRoom):
    layout_name = "open_corner"

    def layout(self, seed: int, absent: bool,
               *, furnished: bool = False) -> None:
        super().layout(seed, absent, furnished=furnished)
        self.model.body_pos[self.model.body("loop_table").id] = (
            LAYOUTS[self.layout_name]["table_xyz_m"])
        self._forward()


def geometry_preflight(room: FurnitureRoom, qpos: np.ndarray,
                       estimate: np.ndarray, seed: int) -> dict:
    """Evaluator-only continuous samples for the changed furniture."""
    room.layout(seed, absent=False, furnished=True)
    room.data.qpos[:] = qpos
    room.actual[:] = (qpos[0], qpos[1], 0.)
    room.estimated[:] = estimate
    room.head = float(qpos[room.head_addr])
    table = [room.model.geom(f"loop_table_{name}").id for name in (
        "top", "leg_-1_-1", "leg_-1_1", "leg_1_-1", "leg_1_1")]
    other = [room.model.geom(name).id for name in (
        "loop_stool_geom", "loop_crate_geom", "loop_box_geom",
        "loop_shelf_geom", "loop_TomatoSauce_geom",
        "loop_person_torso_geom", "loop_person_head_geom",
        "loop_person_upper_arm_geom", "loop_person_forearm_geom",
        "loop_person_hand_geom")]
    robot = body_geoms(room, scene_geoms(room))
    nearest = (float("inf"), None, None, None)
    table_body_nearest = (float("inf"), None, None, None)
    body_scene_nearest = (float("inf"), None, None, None)
    for time_s in np.arange(16.2, 19.0001, .02):
        room.time_s = float(time_s)
        shown = min(1., max(0., (time_s-16.6)/.4))
        hand = START_HAND_XYZ_M+shown*(FAR_HAND_XYZ_M-START_HAND_XYZ_M)
        set_person_pose(room.model, hand, carried_item="loop_TomatoSauce")
        if time_s >= 17.-1e-6:
            room.model.body_pos[room.model.body(
                "loop_TomatoSauce").id] = owner.OWNER_SHOW_XYZ_M
        room._forward()
        for a in table:
            for b in other:
                gap = float(mujoco.mj_geomDistance(
                    room.model, room.data, a, b, 10., None))
                if gap < nearest[0]:
                    nearest = (gap, round(float(time_s), 3),
                               room.model.geom(a).name,
                               room.model.geom(b).name)
            for b in robot:
                gap = float(mujoco.mj_geomDistance(
                    room.model, room.data, a, b, 10., None))
                if gap < table_body_nearest[0]:
                    table_body_nearest = (gap, round(float(time_s), 3),
                                          room.model.geom(a).name,
                                          room.model.geom(b).name)
        gap, body, scene = room._current_obstacle_clearance()
        if gap < body_scene_nearest[0]:
            body_scene_nearest = (float(gap), round(float(time_s), 3), body, scene)
    return {"table_other_minimum": nearest,
            "table_robot_minimum": table_body_nearest,
            "robot_scene_minimum": body_scene_nearest,
            "valid_furniture": nearest[0] >= .02 and
                               table_body_nearest[0] >= .05}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Historical-body scope changed")
    v98_file = v100.V98 / "live-moving-owner-offer-audit.json"
    v98 = json.loads(v98_file.read_text())
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    old_model = structural.structural_model
    old_room = v153p.ResponsiveRoom
    old_monitor = v153p.FrontMonitor
    old_out = v153p.OUT
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    v153p.ResponsiveRoom = FurnitureRoom
    v153p.FrontMonitor = v174p.CameraFrontMonitor
    rows = []
    preflights = []
    try:
        for layout_name, layout in LAYOUTS.items():
            FurnitureRoom.layout_name = layout_name
            FurnitureRoom.variant = v160p.VARIANTS[
                layout["prop_variant"]]
            v153p.OUT = OUT / layout_name
            for seed in SEEDS:
                prior = next(row for row in v98["rows"]
                             if row["condition"] == "moving" and
                             row["case"][0] == seed)
                source = (v100.V98 / "moving" / f"seed{seed}-present" /
                          f"seed{seed}-present/side-probe-00.npz")
                with np.load(source, allow_pickle=False) as packet:
                    qpos = packet["actor_qpos"].copy()
                    estimate = packet["estimated_pose_xyyaw"].copy()
                probe = FurnitureRoom(case=tuple(prior["case"]),
                                      pose_source="command", branch=OUT)
                try:
                    shoulder = probe.model.joint(
                        "left_shoulder_pitch").qposadr[0]
                    qpos[0] += v174p.STATION_X_M
                    estimate[0] += v174p.STATION_X_M
                    qpos[shoulder] = .6
                    preflight = geometry_preflight(probe, qpos,
                                                  estimate, seed)
                finally:
                    probe.close()
                preflights.append({"layout": layout_name, "seed": seed,
                                   **preflight})
                print("preflight", layout_name, seed, preflight, flush=True)
                if not preflight["valid_furniture"]:
                    continue
                for delay in RESPONSES_S:
                    row = v153p.run_one(
                        seed, v174p.STATION_X_M, delay,
                        tuple(prior["case"]), qpos, estimate,
                        vision, exemplar, owner_hist)
                    row.update({"layout": layout_name,
                                "prop_variant": layout["prop_variant"],
                                "staging_packet": str(source.relative_to(ROOT)),
                                "staging_packet_sha256": sha(source)})
                    rows.append(row)
                    print("episode", layout_name, seed, delay,
                          row["front_alarm"], row["task_result"],
                          row["evaluator_minimum_scene_gap_m"], flush=True)
    finally:
        structural.structural_model = old_model
        v153p.ResponsiveRoom = old_room
        v153p.FrontMonitor = old_monitor
        v153p.OUT = old_out
    report = {
        "schema": "robot-learning-lab.ape-varied-furniture-owner.v1",
        "source_sha256": sha(SOURCE),
        "v174_source_sha256": sha(Path(v174p.__file__)),
        "v153_source_sha256": sha(Path(v153p.__file__)),
        "v98_report_sha256": sha(v98_file),
        "historical_visual_sha256": sha(selected_visual()),
        "current_design_status": pointer["status"],
        "layouts": LAYOUTS, "preflights": preflights, "rows": rows,
        "scope": "Declared changed furniture positions plus two old moving-prop "
                 "profiles, with 20ms evaluator geometry preflight and complete "
                 "5.4s owner turns. Frozen V174 10Hz RGB-D camera warning, "
                 "front range monitor, scripted response, image worker and "
                 "decision logic. This is a diagnostic replay from a historical "
                 "held pose that pre-show unknown-space rules would defer. "
                 "The moved table has rigid ideal geometry; only evaluator "
                 "uses clearances. It does not qualify a real sensor or body.",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
