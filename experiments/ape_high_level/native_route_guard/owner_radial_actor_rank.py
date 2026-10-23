"""V85: extend V59's observation-only pose ranking to the V81 radial menu.

The chooser consumes only retained pre-move low/mid range history, current
pose and remembered owner station. V81 future-script gaps are appended to the
report only after the choice as evaluator diagnostics.
"""

from __future__ import annotations

import json
import math

import numpy as np

import owner_radial_wait_feasibility as v81


v59 = v81.v77.v59
ROOT = v81.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v85-radial-actor-rank"


def rank(packet, history, start):
    start = np.asarray(start, dtype=float)
    endpoints = v59.v51.v41.JointHazardResponse.endpoints(packet, start)
    surfaces = v59.dynamic_surface_points(history, start)
    rows = []
    for angle_deg in v81.ANGLES_DEG:
        direction = np.asarray([
            math.cos(math.radians(angle_deg)),
            math.sin(math.radians(angle_deg))])
        for distance in v81.DISTANCES_M:
            target = start+distance*direction
            gaps = [float(np.min(np.linalg.norm(
                endpoints-(start+fraction*distance*direction), axis=1)))
                for fraction in np.linspace(0., 1., 15)]
            approach = min((v59.time_indexed_surface_margin(
                start, target, distance, np.asarray(point["surface_xy_m"]))
                for point in surfaces), default=float("inf"))
            static = gaps[-1]-v59.BODY_PROXY_RADIUS_M
            station = float(np.linalg.norm(
                v59.v56.OWNER_SHOW_STATION_XY_M-target))
            eligible = bool(
                sum(value is not None for value in packet) >= 36 and
                min(gaps) >= v59.MIN_ENDPOINT_GAP_M and
                gaps[-1] >= v59.MIN_ENDPOINT_GAP_M and
                station <= v59.MAX_OWNER_STATION_RANGE_M)
            rows.append({"angle_deg": angle_deg, "distance_m": distance,
                         "target_xy_m": target.tolist(),
                         "minimum_path_endpoint_gap_m": min(gaps),
                         "held_endpoint_gap_m": gaps[-1],
                         "owner_station_range_m": station,
                         "actor_static_margin_m": static,
                         "actor_approach_margin_m": approach,
                         "actor_combined_margin_m": min(static, approach),
                         "eligible": eligible})
    choices = [row for row in rows if row["eligible"]]
    chosen = max(choices, key=lambda row: (
        row["actor_combined_margin_m"], -row["distance_m"])) \
        if choices and surfaces else None
    return chosen, rows, surfaces


def main():
    v60 = json.loads((ROOT / "artifacts/ape-native-route-guard/"
                       "v60-deferred-owner-show/report.json").read_text())
    trace = json.loads(open(v60["results"][0]["run"]["trace"]["path"]).read())
    history = [{"time_s": row["time_s"], "packet": row["actor_packet"]}
               for row in trace if row["stage"] == "post_alarm_range_watch"]
    assert len(history) == 14
    start = json.loads((v81.v77.v76.OUT / "report.json").read_text())[
        "start_xy_m"]
    chosen, rows, surfaces = rank(history[-1]["packet"], history, start)
    assert chosen is not None
    screen = json.loads((v81.OUT / "report.json").read_text())
    evaluator = next(row for row in screen["candidates"]
                     if row["angle_deg"] == chosen["angle_deg"] and
                     row["distance_m"] == chosen["distance_m"])
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "robot-learning-lab.ape-radial-actor-rank.v1",
        "source_sha256": v81.v77.v72.v52.sha(__file__),
        "v60_report_sha256": v81.v77.v72.v52.sha(
            ROOT / "artifacts/ape-native-route-guard/"
            "v60-deferred-owner-show/report.json"),
        "v81_report_sha256": v81.v77.v72.v52.sha(v81.OUT / "report.json"),
        "current_design_pointer_sha256": v81.v77.v72.v52.sha(
            ROOT / "robot/ape-current-design.json"),
        "start_xy_m": start,
        "history_rows": len(history),
        "actor_dynamic_surfaces": surfaces,
        "actor_candidate_count": len(rows),
        "actor_eligible_count": sum(row["eligible"] for row in rows),
        "actor_chosen": chosen,
        "actor_candidates": rows,
        "post_choice_evaluator": {
            "all_eight_clear": evaluator.get("evaluator_all_eight_clear"),
            "worst_prop_gap_m": evaluator.get("evaluator_worst_prop_gap_m"),
            "worst_person_gap_m": evaluator.get("evaluator_worst_person_gap_m"),
            "path_gaps": evaluator.get("evaluator_paths")},
        "scope": "V59's frozen actor-only tube score over an expanded uniform radial menu, followed by evaluator-only V81 path-gap comparison. No new actor belief or full episode."}
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path, chosen["angle_deg"], chosen["distance_m"],
          evaluator["evaluator_worst_prop_gap_m"])


if __name__ == "__main__":
    main()
