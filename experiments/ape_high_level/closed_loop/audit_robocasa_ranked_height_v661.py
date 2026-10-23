"""Diagnose ideal low/mid horizontal ray coverage of the owner hand.

Evaluator-only: these rays were not available to V659's actor and cannot by
themselves authorize a physical sensor placement or response.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from audit_robocasa_layout2_routes_v409 import clearance
import robocasa_person_interaction as interaction
import run_robocasa_memory_bearing_v656 as bearing
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_ranked_emergency_v659 as trial
import run_robocasa_person_joined_v497 as joined
import run_robocasa_memory_return_v599 as memory


SOURCE = Path(__file__)
OUT = trial.OUT
REPORT = OUT / "height-diagnostic-v661.json"
HEIGHTS_M = (.30, .55)
BEAMS = 72
EPISODES = (8, 9)  # Away crossing, perpendicular and ranked response.


def ray_ring(model, data, xy, height_m: float) -> dict:
    origin = np.array([xy[0], xy[1], height_m], dtype=float)
    nearest = (float("inf"), None)
    hand = float("inf")
    for angle in np.linspace(-np.pi, np.pi, BEAMS, endpoint=False):
        ray = np.array([np.cos(angle), np.sin(angle), 0.], dtype=float)
        geom = np.array([-1], dtype=np.int32)
        distance = float(mujoco.mj_ray(model, data, origin, ray,
                                       None, 1, -1, geom))
        if distance < 0 or geom[0] < 0:
            continue
        name = model.geom(int(geom[0])).name
        if distance < nearest[0]:
            nearest = (distance, name)
        if name == "owner_hand_geom":
            hand = min(hand, distance)
    return {"nearest_m": None if not np.isfinite(nearest[0]) else
            round(nearest[0], 5),
            "nearest_geom": nearest[1],
            "hand_m": None if not np.isfinite(hand) else round(hand, 5)}


def main() -> None:
    report_path = OUT / "report.json"
    episodes = json.loads(report_path.read_text())["episodes"]
    result = {"schema": "robot-learning-lab.robocasa-ranked-height-diagnostic.v1",
              "report_sha256": memory.digest(report_path),
              "source_sha256": memory.digest(SOURCE),
              "ray_heights_m": HEIGHTS_M,
              "horizontal_beams_per_height": BEAMS,
              "role": "evaluator-only replay of ideal ray visibility; no actor access",
              "episodes": []}
    for index in EPISODES:
        ep = episodes[index]
        model = joined.scene_with_owner_and_prop(crossing.START)
        data = mujoco.MjData(model)
        interaction.show_pose(model, 1.)
        prop_body = model.body("route_obstacle").id
        ape_body = model.body("audit_ape").id
        ape_geom = model.geom("audit_ape_geom").id
        script = bearing.challenge_motion("away")
        rows = []
        for decision in ep["trace"]:
            if decision["stage"] != "memory_route_decision" or (
                    decision["phase"] != "evade_reactive"):
                continue
            for sub in decision["motion_substeps"]:
                if sub["stopped_before_move"]:
                    continue
                time_s = sub["time_s"]
                xy = np.asarray(sub["evaluator_actual_xy_m"])
                model.body_pos[prop_body, :2] = script(time_s)
                gap, nearest = clearance(model, data, ape_body, ape_geom, xy)
                assert abs(gap - sub["evaluator_proxy_gap_m"]) < .00003
                assert nearest == sub["evaluator_nearest_geom"]
                scans = {str(height): ray_ring(model, data, xy, height)
                         for height in HEIGHTS_M}
                rows.append({"time_s": time_s,
                             "proxy_gap_m": round(gap, 5),
                             "nearest_geom": nearest,
                             "actor_xy_m": xy.tolist(),
                             "ray_scans": scans})
        first_breach = next((row for row in rows if row["proxy_gap_m"] < .05),
                            None)
        first_mid_hand = next((row for row in rows if
                               row["ray_scans"]["0.55"]["hand_m"] is not None),
                              None)
        result["episodes"].append({
            "index": index,
            "run_label": ep["memory_return"]["run_label"],
            "samples_20ms": len(rows),
            "first_5cm_breach_s": None if first_breach is None else
                first_breach["time_s"],
            "first_mid_height_hand_return_s": None if first_mid_hand is None
                else first_mid_hand["time_s"],
            "low_hand_returns": sum(row["ray_scans"]["0.3"]["hand_m"] is not None
                                    for row in rows),
            "mid_hand_returns": sum(row["ray_scans"]["0.55"]["hand_m"] is not None
                                    for row in rows),
            "breach_packet": first_breach,
            "samples": rows})
    REPORT.write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps({"episodes": [
        {key: value for key, value in row.items() if key != "samples"}
        for row in result["episodes"]]}, indent=2))


if __name__ == "__main__":
    main()
