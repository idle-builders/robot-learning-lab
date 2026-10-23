"""V83: can fresh range observations reopen V82's blocked next step?

Hold the actual V82 stopped pose and replay the two fixture-clear prop paths
with their original person script. The actor-side result is the 10 Hz range
endpoint decision for one proposed 2.5 cm move; MuJoCo gaps are evaluator-only.
No motion or owner image is executed in this probe.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import owner_radial_oracle_pose_probe as v82


v81, v77 = v82.v81, v82.v77
ROOT = v82.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v83-blocked-step-reobserve"
PERIOD_S = .1
THREE_CLEAR_PACKETS = 3


def main():
    report = json.loads((v82.OUT / "report.json").read_text())
    source = v77.runner.original()
    summary = source["summary"]
    policy = json.loads(v77.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    model = mujoco.MjModel.from_xml_path(str(v77.runner.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    _, _, checkpoint_s, _, _ = v77.v72.v64.v51.v46.low_prop_geometry(
        v77.runner.geometry_at)(
            summary, source["result"]["result"]["episode"],
            scenario, model, data)
    scanner = v77.v72.v64.v51.v46.plane_packet(
        v77.runner.observed_packet, (.05, .30))
    results = []
    for item in report["results"]:
        path_name = item["path"]
        run = item["run"]
        trace = json.loads(Path(run["trace"]["path"]).read_text())
        steps = [row for row in trace if row["stage"] ==
                 "pre_show_restage_step"]
        decision = next(row for row in trace if row["stage"] ==
                        "pre_show_restage_decision")
        assert len(steps) == item["restage_steps"] == 20
        assert decision["actor_decision"]["action"] == "hold_step_blocked"
        held = np.asarray(steps[-1]["ape_xy_m"])
        proposed = held+.025*v82.DIRECTION
        endpoint = np.asarray(item["endpoint_xy_m"])
        start_s = float(steps[-1]["time_s"])
        end_s = checkpoint_s+v77.v72.v64.v60.WATCH_END_AFTER_CHECKPOINT_S
        rows = []
        clear_streak = 0
        first_single_clear_s = first_three_clear_s = None
        for index, time_s in enumerate(np.arange(
                start_s+PERIOD_S, end_s+.00001, PERIOD_S), start=1):
            person = v77.runner.human_xy(
                v77.v72.v64.v54.CASE, summary, float(time_s),
                run["request_time_s"], None)
            v77.runner.v35.two.set_distractor(model, person)
            model.body_pos[model.body("audit_ape").id, :2] = held
            model.body_pos[model.body("route_obstacle").id, :2] = \
                v77.v65.prop_at(float(time_s), checkpoint_s, endpoint,
                                item["finish_after_checkpoint_s"])
            mujoco.mj_forward(model, data)
            packet = scanner(model, data, held,
                             v77.v72.v64.v54.CASE, 500+len(steps)+index,
                             person_xy=person, hidden_return=False)
            healthy = v77.runner.side_healthy(packet)
            points = v77.v59.v51.v41.JointHazardResponse.endpoints(
                packet, held)
            gap = float(np.min(np.linalg.norm(points-proposed, axis=1)))
            clear = bool(healthy and gap >= v77.v59.v56.MIN_PATH_ENDPOINT_GAP_M)
            clear_streak = clear_streak+1 if clear else 0
            if clear and first_single_clear_s is None:
                first_single_clear_s = float(time_s)
            if clear_streak >= THREE_CLEAR_PACKETS and first_three_clear_s is None:
                first_three_clear_s = float(time_s)
            prop_gap = v77.runner.prop_gap(model, data, held)
            person_gap = v77.runner.geometry.gap(model, data, held, person)
            rows.append({"time_s": round(float(time_s), 5),
                         "actor_packet": packet,
                         "actor_healthy": healthy,
                         "actor_next_step_endpoint_gap_m": gap,
                         "actor_next_step_clear": clear,
                         "actor_clear_streak": clear_streak,
                         "evaluator_prop_gap_m": prop_gap,
                         "evaluator_person_gap_m": person_gap})
        results.append({"path": path_name,
                        "held_xy_m": held.tolist(),
                        "proposed_next_xy_m": proposed.tolist(),
                        "first_single_clear_s": first_single_clear_s,
                        "first_three_clear_s": first_three_clear_s,
                        "minimum_held_prop_gap_m": min(
                            row["evaluator_prop_gap_m"] for row in rows),
                        "minimum_held_person_gap_m": min(
                            row["evaluator_person_gap_m"] for row in rows),
                        "rows": rows})
        print(path_name, first_single_clear_s, first_three_clear_s,
              results[-1]["minimum_held_prop_gap_m"],
              results[-1]["minimum_held_person_gap_m"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    output = {"schema": "robot-learning-lab.ape-blocked-step-reobserve.v1",
              "source_sha256": v77.v72.v52.sha(__file__),
              "v82_report_sha256": v77.v72.v52.sha(v82.OUT / "report.json"),
              "scene_sha256": v77.v72.v52.sha(
                  v77.runner.v35.moving.SCENE_XML),
              "current_design_pointer_sha256": v77.v72.v52.sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "period_s": PERIOD_S,
              "three_clear_packets": THREE_CLEAR_PACKETS,
              "results": results,
              "scope": "Held blocked pose, one proposed next step, fresh ideal low/mid scans and evaluator gaps under two scripted future turns. No executed move, camera capture, selected body or physical stop qualification."}
    target = OUT / "report.json"
    target.write_text(json.dumps(output, indent=2)+"\n")
    print(target)


if __name__ == "__main__":
    main()
