"""Challenge the frozen V153p owner loop with changed prop shape and path.

These two scene variants are declared before running the matrix. The actor's
front-left cue, stable-range gate, image worker and task decision are reused
unchanged. A cooperative stop is still a scene script, not robot motion.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from articulated_person_scene import add_person
from audit_direct_structural_entry_v271 import V149, V150
import direct_structural_rooms_v270 as structural
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, Vision, sha
import run_early_standoff_owner_v148p as v148p
import run_pre_show_reserve_screen_v100 as v100
import run_standoff_front_response_v150p as v150p
import run_standoff_yield_recovery_v153p as v153p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-scene-variants"
SEEDS = (170, 172)
STATIONS_X_M = (-.30, -.22)
RESPONSE_DELAYS_S = (.2, None)
VARIANTS = {
    "fast_wide_turn": {
        "approach_start_s": 17.2,
        "approach_end_s": 17.85,
        "end_xy_m": (-.20, .08),
        "prop_half_size_xyz_m": (.055, .05, .04),
    },
    "slow_narrow_offset": {
        "approach_start_s": 17.1,
        "approach_end_s": 18.25,
        "end_xy_m": (-.23, .15),
        "prop_half_size_xyz_m": (.035, .035, .035),
    },
}


class VariantRoom(v150p.ResponsiveRoom):
    variant: dict = VARIANTS["fast_wide_turn"]

    def layout(self, seed: int, absent: bool,
               *, furnished: bool = False) -> None:
        super().layout(seed, absent, furnished=furnished)
        self.model.geom("loop_stool_geom").size[:] = self.variant[
            "prop_half_size_xyz_m"]
        self._forward()

    def _forward(self) -> None:
        if hasattr(self, "model") and hasattr(self, "obstacle_xy"):
            variant = self.variant
            time_s = float(self.time_s)
            alarm_s = getattr(self, "front_alarm_s", None)
            if alarm_s is not None and self.response_delay_s is not None:
                time_s = min(time_s, alarm_s+self.response_delay_s)
            fraction = min(1., max(0.,
                (time_s-variant["approach_start_s"]) /
                (variant["approach_end_s"]-variant["approach_start_s"])))
            end_x, end_y = variant["end_xy_m"]
            x = end_x*fraction
            y = .306 + (end_y-.306)*fraction
            self.obstacle_xy[3] = (x, y)
            self.model.body_pos[self.model.body("loop_stool").id] = (
                x, y, variant["prop_half_size_xyz_m"][2])
        # Bypass the parent classes' nominal scripted low-prop path.
        super(v148p.StandoffRoom, self)._forward()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Historical-body scope changed")
    v98_path = v100.V98 / "live-moving-owner-offer-audit.json"
    v98 = json.loads(v98_path.read_text())
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    old_model = structural.structural_model
    old_room = v153p.ResponsiveRoom
    old_out = v153p.OUT
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    v153p.ResponsiveRoom = VariantRoom
    rows = []
    try:
        for seed in SEEDS:
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and
                         row["case"][0] == seed)
            source = (v100.V98 / "moving" / f"seed{seed}-present" /
                      f"seed{seed}-present/side-probe-00.npz")
            with np.load(source, allow_pickle=False) as packet:
                qpos = packet["actor_qpos"].copy()
                estimate = packet["estimated_pose_xyyaw"].copy()
            probe = VariantRoom(case=tuple(prior["case"]),
                                pose_source="command", branch=OUT)
            try:
                shoulder = probe.model.joint(
                    "left_shoulder_pitch").qposadr[0]
            finally:
                probe.close()
            for variant_name, variant in VARIANTS.items():
                VariantRoom.variant = variant
                v153p.OUT = OUT / variant_name
                for station_x in STATIONS_X_M:
                    held_qpos = qpos.copy()
                    held_estimate = estimate.copy()
                    held_qpos[0] += station_x
                    held_estimate[0] += station_x
                    held_qpos[shoulder] = .6
                    for delay in RESPONSE_DELAYS_S:
                        row = v153p.run_one(
                            seed, station_x, delay,
                            tuple(prior["case"]), held_qpos,
                            held_estimate, vision, exemplar, owner_hist)
                        row["variant"] = variant_name
                        row["staging_packet"] = str(source.relative_to(ROOT))
                        row["staging_packet_sha256"] = sha(source)
                        rows.append(row)
                        print(variant_name, seed, station_x, delay,
                              "alarm", row["front_alarm"],
                              "gap", row["evaluator_minimum_scene_gap_m"],
                              "task", row["task_result"], flush=True)
    finally:
        v153p.ResponsiveRoom = old_room
        v153p.OUT = old_out
        structural.structural_model = old_model
    report = {
        "schema": "robot-learning-lab.ape-standoff-scene-variants.v1",
        "source_sha256": sha(SOURCE),
        "v153_source_sha256": sha(Path(v153p.__file__)),
        "v98_report_sha256": sha(v98_path),
        "historical_visual_sha256": sha(selected_visual()),
        "current_design_status": pointer["status"],
        "variants": VARIANTS,
        "scope": "Complete furnished articulated-owner turns with two predeclared "
                 "prop half-size, speed and lateral-path changes. The "
                 "V153 front-left cue, stable-range gate and dual RGB-D "
                 "recognition are frozen. The response is a scripted prop "
                 "stop 0.2s after any observed front cue or an ignored "
                 "branch. Actor decisions do not use evaluator body gap. "
                 "This remains one historical kinematic body, two lighting "
                 "seeds and a fixed room/person geometry; it is not a "
                 "statistical or physical sensor validation.",
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
