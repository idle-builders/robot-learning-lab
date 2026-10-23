"""Test image-grounded person/range association before routing to a west owner view.

An observed person can cooperate in a short, freshly observed approach. Unknown
objects retain the original 0.60 m motion envelope. This is a hypothesis, not
person-motion or physical-body qualification.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_layout2_west_station_baseline_v729 as baseline
import run_robocasa_layout2_owner_transfer_v695 as transfer
import run_robocasa_layout2_person_route_v697 as full
import run_robocasa_layout2_route_v407 as route
import run_robocasa_layout2_side_prop_v701 as side
import run_robocasa_person_joined_v497 as joined
from run_robocasa_layout2_multitrack_v485 import grid_class as original_grid_class


SOURCE = Path(__file__)
OUT = side.ROOT / "artifacts/ape-robocasa-layout2-person-associated-station-v731"
PERSON_ASSOCIATION_M = .45
PERSON_MOTION_RESERVE_M = .10


def associated_grid_class(prior, policy):
    Base = original_grid_class(prior, policy)

    class PersonAssociatedGrid(Base):
        def plan(self, origin_xy, goal_xy):
            if self.mode != "navigate" or not self.multi_history:
                return super().plan(origin_xy, goal_xy)
            requested, status, path = route.SensorGrid.plan(self, origin_xy, goal_xy)
            if not path:
                return requested, status, path
            tracks = [track for track in self.multi_history[-1]["tracks"]
                      if track["hits"] >= 2]
            conflicts = []
            reserves = []
            for track in tracks:
                center = np.asarray(track["center_xy_m"])
                person_match = (full.FIRST_SURFACE is not None and
                                np.linalg.norm(center - full.FIRST_SURFACE) <=
                                PERSON_ASSOCIATION_M)
                motion = (PERSON_MOTION_RESERVE_M if person_match else
                          policy["assumed_object_max_total_displacement_m"])
                envelope = (track["radius_m"] + policy["actor_body_radius_m"] +
                            policy["required_gap_m"] +
                            policy["object_shape_allowance_m"] + motion)
                nearest = min(float(np.linalg.norm(self.xy(cell) - center))
                              for cell in path)
                reserve = float(np.linalg.norm(origin_xy - center) - envelope)
                reserves.append(reserve)
                if nearest < envelope:
                    conflicts.append({"track_id": track["id"],
                                      "actor_association": "rgbd_person" if person_match
                                                           else "unassociated_object",
                                      "route_nearest_m": nearest,
                                      "envelope_m": envelope,
                                      "track_age_s": track["age_s"]})
            decision = "continue_clear_route"
            if conflicts:
                decision = ("request_owner_show_here_safe_side"
                            if min(reserves) >= policy["minimum_staging_reserve_m"]
                            else "stop_current_pose_inside_object_region")
            self.multi_decisions.append({
                "time_s": self.multi_history[-1]["capture_time_s"],
                "confirmed_track_ids": [track["id"] for track in tracks],
                "person_surface_actor_xy_m": None if full.FIRST_SURFACE is None
                    else full.FIRST_SURFACE.tolist(),
                "conflicts": conflicts,
                "minimum_current_reserve_m": min(reserves) if reserves else None,
                "decision": decision})
            if conflicts:
                return None, decision, []
            return requested, status, path

    return PersonAssociatedGrid


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {
        "schema": "robot-learning-lab.layout2-person-associated-station.v1",
        "station_rule": {"standoff_m": baseline.STANDOFF_M,
                         "lateral_m": baseline.LATERAL_M},
        "actor_person_association_m": PERSON_ASSOCIATION_M,
        "person_motion_reserve_m": PERSON_MOTION_RESERVE_M,
        "unassociated_object_motion_reserve_m": .60,
        "profile": "stationary prop control; full owner task",
        "hypothesis": "A range object associated spatially with the first RGB-D person surface can use a cooperative-person approach reserve; unknown range tracks keep the full moving-object envelope.",
        "limits": "Person may move, association may be wrong, and hand/upper-body volume is not in the planar route screen.",
        "source_sha256": route.sha256(SOURCE)}
    (OUT / "predeclared.json").write_text(json.dumps(declaration, indent=2) + "\n")
    old = (full.OUT, full.RUN_SCOPE, full.STANDOFF_M, full.STATION_LATERAL_M,
           full.SEED_ROUTE_FROM_SURVEY, full.ACTIVE_VIEW_CORRECTION,
           joined.grid_class, transfer.PROP_START.copy())
    try:
        full.OUT = OUT
        full.RUN_SCOPE = "layout2 full owner static control, actor RGB-D/range-associated person approach to V728 west station; 50cm kinematic proxy"
        full.STANDOFF_M = baseline.STANDOFF_M
        full.STATION_LATERAL_M = baseline.LATERAL_M
        full.SEED_ROUTE_FROM_SURVEY = True
        full.ACTIVE_VIEW_CORRECTION = True
        joined.grid_class = associated_grid_class
        transfer.PROP_START = side.START.copy()
        full.main()
    finally:
        (full.OUT, full.RUN_SCOPE, full.STANDOFF_M, full.STATION_LATERAL_M,
         full.SEED_ROUTE_FROM_SURVEY, full.ACTIVE_VIEW_CORRECTION,
         joined.grid_class, transfer.PROP_START) = old
    report = json.loads((OUT / "report.json").read_text())
    ep = report["episode"]
    print(ep["route_state"], ep["outcome"], ep["route_episode"]["end_true_xy_m"],
          flush=True)


if __name__ == "__main__":
    main()
