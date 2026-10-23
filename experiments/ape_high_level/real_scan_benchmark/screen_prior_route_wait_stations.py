"""Find whether failed final waits have a viable previously traversed station."""

from __future__ import annotations

import json
import math

import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from screen_item_approach_hazards import clearance
import run_post_confirmation_approach_v2 as source
import screen_20_final_wait_lanes as previous


OUT = ART / "prior_route_wait_station_screen.json"


def geometry(scene, xy: np.ndarray, yaw: float, required: float) -> list[float]:
    supported = []
    for offset in previous.OFFSETS:
        a = math.radians(yaw + offset)
        direction = np.array([math.sin(a), math.cos(a)])
        samples = [clearance(scene, xy + d * direction)
                   for d in previous.INBOUND_M]
        samples += [clearance(scene, xy - d * direction)
                    for d in previous.ESCAPE_M]
        if all(s["floor_supported"] and
               (s["nearest_two_plane_m"] is None or
                s["nearest_two_plane_m"] >= required) for s in samples):
            supported.append(offset)
    return supported


def main() -> None:
    initial = json.loads(previous.OUT.read_text())
    cases = {c["scene_key"]:c for c in json.loads((ART /
        "integrated_owner_20_conditional_precommit.json").read_text())["cases"]}
    failed = [r for r in initial["rows"] if not r["supported_relative_bearings_deg"]]
    if len(failed) != 7:
        raise ValueError("Expected seven unsupported current stops")
    rows = []
    for row in failed:
        case = cases[row["scene_key"]]
        trace = json.loads((ROOT / row["action_trace"]).read_text())
        screen = json.loads((ROOT / case["screen"]).read_text())
        placement = next(x for x in screen["scenes"] if x["scene_key"] ==
                         row["scene_key"])
        scene = source.base.make_scene(case, placement)
        final_xy = np.asarray(row["endpoint_xz_m"],float)
        target = np.asarray(trace["observed_target_world_xz_m"],float)
        stations = []
        seen = set()
        for action in reversed(trace["actions"][:-1]):
            xy = np.asarray(action["evaluator_actual_xz_m"],float)
            key = tuple(round(v,3) for v in xy)
            if key in seen:
                continue
            seen.add(key)
            distance = float(np.linalg.norm(xy-final_xy))
            if distance > .6:
                continue
            yaw = float(action["evaluator_actual_heading_deg"])
            observed_gap = float(np.linalg.norm(np.asarray(
                action["actor_pose_world_xz_m"],float)-target))
            supported = geometry(scene,xy,yaw,initial[
                "required_nearest_two_plane_m"])
            stations.append({"source_frame":action["frame"],
                "xz_m":xy.tolist(),"heading_deg":yaw,
                "backtrack_distance_m":distance,
                "actor_observed_target_gap_m":observed_gap,
                "supported_relative_bearings_deg":supported})
        viable = [s for s in stations if s["supported_relative_bearings_deg"]]
        rows.append({"scene_key":row["scene_key"],
            "candidate_count":len(stations),
            "nearest_viable_prior_station":viable[0] if viable else None,
            "stations":stations})
        print(row["scene_key"],len(stations),
              None if not viable else round(viable[0]["backtrack_distance_m"],3),
              flush=True)
    OUT.write_text(json.dumps({"schema":"ape.prior-route-wait-station-screen.v1",
        "source_screen":relative(previous.OUT),
        "source_screen_sha256":sha(previous.OUT),
        "source_action_report":relative(source.REPORT),
        "source_action_report_sha256":sha(source.REPORT),
        "unsupported_final_stops":len(rows),
        "with_viable_prior_station":sum(r["nearest_viable_prior_station"] is not None
                                       for r in rows),
        "rows":rows,
        "limits":"Evaluator-only static geometry at recorded prior route poses. Neither route backtracking, fresh visual confirmation nor sensor-derived corridor selection has been executed."},indent=2)+"\n")
    print("viable",sum(r["nearest_viable_prior_station"] is not None
                       for r in rows),"/",len(rows))


if __name__ == "__main__":
    main()
