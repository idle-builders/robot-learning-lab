"""Bound nearby wait-pose feasibility from the current all-around range view."""

from __future__ import annotations

import json
import math

import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from screen_item_approach_hazards import clearance
import run_post_confirmation_approach_v2 as source
import screen_20_final_wait_lanes as initial
import screen_prior_route_wait_stations as prior


OUT = ART / "local_wait_reposition_screen.json"
OFFSETS = (-90.,90.,180.,-135.,135.,-45.,45.)
DISTANCES_M = (.2,.4)
MAX_OBSERVED_TARGET_GAP_M = 1.0


def main() -> None:
    first = json.loads(initial.OUT.read_text())
    cases = {c["scene_key"]:c for c in json.loads((ART /
        "integrated_owner_20_conditional_precommit.json").read_text())["cases"]}
    failed = [r for r in first["rows"] if not r["supported_relative_bearings_deg"]]
    if len(failed) != 7:
        raise ValueError("Expected seven unsupported final views")
    rows = []
    for row in failed:
        case = cases[row["scene_key"]]
        trace = json.loads((ROOT / row["action_trace"]).read_text())
        screen = json.loads((ROOT / case["screen"]).read_text())
        placement = next(x for x in screen["scenes"] if x["scene_key"] ==
                         row["scene_key"])
        scene = source.base.make_scene(case, placement)
        start = np.asarray(row["endpoint_xz_m"],float)
        yaw = row["heading_deg"]
        target = np.asarray(trace["observed_target_world_xz_m"],float)
        actor = np.asarray(trace["actions"][-1]["actor_pose_world_xz_m"],float)
        eye = np.array([start[0],scene.floor_y+.42,start[1]])
        fan_angles = np.asarray(OFFSETS,np.float32)
        current_ranges = scene.range_fan(eye,yaw,owner_xz=None,
                                          angles_deg=fan_angles)
        candidates = []
        for index,offset in enumerate(OFFSETS):
            angle = math.radians(yaw+offset)
            direction = np.array([math.sin(angle),math.cos(angle)])
            near = np.asarray(current_ranges[:,index],float)
            for distance in DISTANCES_M:
                candidate = start+distance*direction
                gap = float(np.linalg.norm(actor+distance*direction-target))
                corridor_observed = bool(np.all(np.isfinite(near)) and
                    np.min(near) >= distance+first["body_radius_m"]+.05)
                view_distance = gap <= MAX_OBSERVED_TARGET_GAP_M
                supported = (prior.geometry(scene,candidate,yaw,
                    first["required_nearest_two_plane_m"])
                    if corridor_observed and view_distance else [])
                candidates.append({"move_relative_deg":offset,
                    "move_distance_m":distance,
                    "current_low_mid_range_m":near.tolist(),
                    "observed_move_corridor":corridor_observed,
                    "actor_observed_target_gap_m":gap,
                    "within_view_distance":view_distance,
                    "candidate_xz_m":candidate.tolist(),
                    "evaluator_supported_entry_bearings_deg":supported})
        feasible = [c for c in candidates if c[
            "evaluator_supported_entry_bearings_deg"]]
        rows.append({"scene_key":row["scene_key"],
            "candidate_count":len(candidates),
            "first_feasible":feasible[0] if feasible else None,
            "feasible_count":len(feasible),"candidates":candidates})
        print(row["scene_key"],len(feasible),
              None if not feasible else (feasible[0]["move_relative_deg"],
                  feasible[0]["move_distance_m"]),flush=True)
    OUT.write_text(json.dumps({"schema":"ape.local-wait-reposition-screen.v1",
        "source_final_screen":relative(initial.OUT),
        "source_final_screen_sha256":sha(initial.OUT),
        "source_prior_route_screen":relative(prior.OUT),
        "source_prior_route_screen_sha256":sha(prior.OUT),
        "move_relative_bearings_deg":OFFSETS,"move_distances_m":DISTANCES_M,
        "max_actor_observed_target_gap_m":MAX_OBSERVED_TARGET_GAP_M,
        "with_local_option":sum(r["first_feasible"] is not None for r in rows),
        "rows":rows,"limits":"Hybrid feasibility screen: move ray and target distance are actor-observed ideal range/map quantities; final lane/floor geometry is evaluator-only. No movement, fresh image, owner reply or actual sensor noise tested."},indent=2)+"\n")
    print("local option",sum(r["first_feasible"] is not None for r in rows),
          "/",len(rows))


if __name__ == "__main__":
    main()
