"""Audit paired route prefixes, observed range decisions, geometry and videos."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

import run_owner_wait_loop as run


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scripted_xy(condition, start, target, t0, request_s, t):
    if condition == "wait_stationary" or t <= t0:
        return start
    fraction = min(1., max(0., (t-t0)/run.APPROACH_S))
    approaching = start+(target-start)*fraction
    if condition != "wait_complies" or request_s is None or t <= request_s+run.REACTION_S:
        return approaching
    reverse_s = request_s+run.REACTION_S
    anchor = start+(target-start)*min(1., (reverse_s-t0)/run.APPROACH_S)
    distance = float(np.linalg.norm(anchor-start))
    fraction = min(1., run.RETREAT_MPS*(t-reverse_s)/distance)
    return anchor+(start-anchor)*fraction


def main():
    path = run.OUT / "report.json"
    report = json.loads(path.read_text())
    checks = 0
    errors = []

    def check(ok, label):
        nonlocal checks
        checks += 1
        if not ok:
            errors.append(label)

    check(report["schema"] == "robot-learning-lab.ape-owner-wait-loop.v1", "schema")
    for key, source in (("source_sha256", Path(run.__file__)),
                        ("base_runner_sha256", Path(run.reteach.__file__)),
                        ("two_person_runner_sha256", Path(run.moving.__file__)),
                        ("scene_sha256", run.moving.SCENE_XML),
                        ("current_design_pointer_sha256",
                         run.ROOT / "robot/ape-current-design.json")):
        check(report[key] == sha(source), key)
    check(tuple(report["conditions"]) == run.CONDITIONS and
          len(report["results"]) == len(run.CONDITIONS), "paired matrix")
    by_condition = {}
    model = mujoco.MjModel.from_xml_path(str(run.moving.SCENE_XML))
    data = mujoco.MjData(model)
    for summary, result in zip(report["summaries"], report["results"]):
        condition = summary["condition"]
        ep = result["result"]["episode"]
        by_condition[condition] = (summary, ep)
        gate = [row for row in ep["trace"]
                if row["stage"] == "pre_show_bystander_gate"]
        check(len(gate) == summary["gate_packets"] and len(gate) > 0,
              f"gate packet count {condition}")
        check(all(abs(row["actor_packet"]["captured_at_s"]-row["time_s"]) < 1e-7
                  and len(row["actor_packet"]["ranges_m"]) == 72 for row in gate),
              f"timed range packet {condition}")
        start = np.asarray(summary["scripted_challenge"]["start_xy_evaluator_m"])
        target = np.asarray(summary["scripted_challenge"]["target_xy_evaluator_m"])
        t0 = summary["gate_start_s"]
        for row in gate:
            t = row["time_s"]
            bystander = scripted_xy(condition, start, target, t0,
                                     summary["request_s"], t)
            run.audit.set_people(model, run.moving.owner_xy_at(t0), bystander)
            measured = run.audit.distances(model, data,
                                           np.asarray(row["ape_xy_m"]))["distractor"]
            check(abs(measured-row["evaluator_bystander_gap_m"]) < 2e-4,
                  f"independent gate gap {condition} {t}")
        check(abs(min(row["evaluator_bystander_gap_m"] for row in gate)-
                  summary["gate_min_bystander_gap_m"]) < 1e-5,
              f"minimum gate gap {condition}")
        actions = [row for row in gate if row["actor_action"] != "observe"]
        check(actions[0]["actor_action"] == summary["gate_first_action"],
              f"first action {condition}")
        if summary["request_s"] is not None:
            requests = [row for row in actions
                        if row["actor_action"] == "suspend_show_and_request_space"]
            check(len(requests) == 1 and requests[0]["time_s"] ==
                  summary["request_s"] and requests[0]["actor_risk_streak"] >= 2,
                  f"request evidence {condition}")
        else:
            check(not any(row["actor_action"] == "suspend_show_and_request_space"
                          for row in gate), f"no request {condition}")
        media = ep["video"]
        movie = Path(media["path"])
        check(movie.is_file() and sha(movie) == media["sha256"],
              f"video hash {condition}")
        if movie.is_file():
            cap = cv2.VideoCapture(str(movie))
            frames = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames += 1
            cap.release()
            check(frames == media["frames"] and frames > 20,
                  f"decoded video {condition}")
        geometry = summary["full_geometry_audit"]
        if geometry is not None:
            check(geometry["all_passed"] and
                  geometry["outcome"] == ep["outcome"],
                  f"independent full-episode audit {condition}")
        expected_overall = min(x for x in (
            summary["gate_min_bystander_gap_m"],
            geometry["minimum_distractor_gap_m"] if geometry else None)
            if x is not None)
        check(abs(expected_overall-summary["overall_min_sampled_bystander_gap_m"]) < 1e-5,
              f"combined episode clearance {condition}")
    prefixes = [ep["route_episode"]["trace"]
                for _, ep in by_condition.values()]
    check(all(prefix == prefixes[0] for prefix in prefixes[1:]),
          "identical route prefix")
    packets = [[row for row in ep["trace"]
                if row["stage"] == "pre_show_bystander_gate"]
               for _, ep in by_condition.values()]
    check(all(rows[0]["actor_packet"] == packets[0][0]["actor_packet"]
              for rows in packets[1:]), "identical first gate packet")
    comply = [row for row in by_condition["wait_complies"][1]["trace"]
              if row["stage"] == "pre_show_bystander_gate"]
    ignored = [row for row in by_condition["wait_ignored"][1]["trace"]
               if row["stage"] == "pre_show_bystander_gate"]
    request_index = next(i for i, row in enumerate(comply)
                         if row["actor_action"] == "suspend_show_and_request_space")
    check(all(comply[i]["actor_packet"] == ignored[i]["actor_packet"]
              for i in range(request_index+1)), "paired pre-request observations")
    check(by_condition["wait_ignored"][1]["outcome"] ==
          "new_show_deferred_bystander_motion" and
          by_condition["wait_ignored"][0]["gate_min_bystander_gap_m"] < 0,
          "ignored outcome and overlap")
    check(by_condition["wait_stationary"][0]["request_s"] is None and
          by_condition["wait_stationary"][1]["outcome"] ==
          "new_show_metric_item_seen_unattributed", "stationary no-prompt task")
    output = {"schema": "robot-learning-lab.ape-owner-wait-loop-audit.v1",
              "report_sha256": sha(path), "checks": checks,
              "passed": not errors, "errors": errors}
    (run.OUT / "audit.json").write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
