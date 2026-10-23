"""Audit V32 causal prefixes, action evidence, gap summaries, and media."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

import body_aware_held_sensor as sensor
import held_space_request_loop as loop
import kitchen_native as kitchen
import kitchen_trial as trial
import online_cluster_guard as online


def main() -> None:
    report_path = loop.OUT / "report.json"
    report = json.loads(report_path.read_text())
    checks = 0
    errors = []

    def check(ok, message):
        nonlocal checks
        checks += 1
        if not ok:
            errors.append(message)

    check(report["schema"] == "robot-learning-lab.ape-held-space-request-loop.v1",
          "schema")
    for key, path in (("source_sha256", Path(loop.__file__)),
                      ("sensor_source_sha256", Path(sensor.__file__)),
                      ("tracker_source_sha256", Path(online.__file__)),
                      ("body_sha256", kitchen.BODY),
                      ("native_trace_sha256", kitchen.TRACE),
                      ("scene_sha256", loop.late.previous.OUT / "scene.xml")):
        check(report[key] == trial.sha(path), key)
    results = report["results"]
    check(len(results) == 48, "episode count")
    keyed = {(r["mode"], r["world"], r["case"], r["response"]): r
             for r in results}
    check(len(keyed) == len(results), "unique episode keys")
    traces = {}
    for key, row in keyed.items():
        mode, world, case, response = key
        trace_info = row["trace"]
        path = Path(trace_info["path"])
        check(path.is_file() and trial.sha(path) == trace_info["sha256"],
              f"trace hash {key}")
        if not path.is_file():
            continue
        trace = json.loads(path.read_text())
        traces[key] = trace
        check(len(trace) == row["packet_count"] and len(trace) > 20,
              f"packet count {key}")
        actions = [(i, x["actor_action"]) for i, x in enumerate(trace)
                   if x["actor_action"] != "observe"]
        requests = [(i, x) for i, x in enumerate(trace)
                    if x["actor_action"] == "suspend_show_and_request_space"]
        resumes = [(i, x) for i, x in enumerate(trace)
                   if x["actor_action"] == "resume_after_fresh_clearance"]
        check(len(requests) <= 1 and len(resumes) <= 1,
              f"single request/resume {key}")
        check(len(actions) == len(requests)+len(resumes), f"known actions {key}")
        check((row["request_s"] is None) == (not requests),
              f"request field {key}")
        check((row["resume_s"] is None) == (not resumes),
              f"resume field {key}")
        if requests:
            i, x = requests[0]
            check(x["time_s"] == row["request_s"] and
                  x["risk_streak"] >= 2 and x["actor_decision"]["risk"],
                  f"request evidence {key}")
        if resumes:
            i, x = resumes[0]
            window = trace[i-loop.CLEAR_PACKETS+1:i+1]
            check(len(window) == loop.CLEAR_PACKETS and
                  all(not p["actor_decision"]["risk"] and
                      p["tracked_clearance"]["clear"] for p in window) and
                  x["time_s"] == row["resume_s"],
                  f"fresh clearance for resume {key}")
        check(row["below_5cm"] == (row["min_body_person_gap_m"] < .05) and
              row["overlap"] == (row["min_body_person_gap_m"] <= 0),
              f"gap summary {key}")
        if case.startswith("static_"):
            check(row["request_s"] is None and row["resume_s"] is None,
                  f"static false alarm {key}")
        if row["video"] is not None:
            media = row["video"]
            movie = Path(media["path"])
            check(movie.is_file() and trial.sha(movie) == media["sha256"],
                  f"video hash {key}")
            if movie.is_file():
                capture = cv2.VideoCapture(str(movie))
                n = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    n += 1
                    check(frame.shape[0] >= 480 and frame.shape[1] >= 640,
                          f"video frame shape {key}")
                capture.release()
                check(n == media["frames"] and n > 10,
                      f"video frame count {key}")
    for mode in loop.MODES:
        for world in range(4):
            for case in loop.CASES[:2]:
                a = traces[(mode, world, case, "complies")]
                b = traces[(mode, world, case, "ignores")]
                request_idx = next((i for i, x in enumerate(a)
                                    if x["actor_action"] ==
                                    "suspend_show_and_request_space"), None)
                if request_idx is None:
                    request_idx = len(a)-1
                check(all(a[i]["actor_packet"] == b[i]["actor_packet"]
                          and a[i]["actor_decision"] == b[i]["actor_decision"]
                          for i in range(request_idx+1)),
                      f"paired pre-request packets {mode} {world} {case}")
    sensitivity_path = loop.OUT / "response-sensitivity" / "report.json"
    if sensitivity_path.is_file():
        sensitivity = json.loads(sensitivity_path.read_text())
        check(len(sensitivity["results"]) == 9, "sensitivity grid count")
        baseline_times = {(r["world"], r["case"]): r["request_s"]
                          for r in results if r["mode"] == "split_body_10hz"
                          and r["response"] == "complies"
                          and r["case"].startswith("late_")}
        for setting in sensitivity["results"]:
            episodes = setting["episodes"]
            check(len(episodes) == 8 and setting["requested"] == 8,
                  f"sensitivity episodes {setting['reaction_s']} {setting['retreat_mps']}")
            check(setting["below_5cm"] == sum(r["below_5cm"] for r in episodes)
                  and setting["resumed"] == sum(r["resume_s"] is not None
                                                for r in episodes),
                  f"sensitivity summary {setting['reaction_s']} {setting['retreat_mps']}")
            for row in episodes:
                key = (row["world"], row["case"])
                check(row["request_s"] == baseline_times[key],
                      f"same request timing {key} {setting['reaction_s']}")
                path = Path(row["trace"]["path"])
                check(path.is_file() and trial.sha(path) == row["trace"]["sha256"],
                      f"sensitivity trace hash {key} {setting['reaction_s']}")
    output = {"schema": "robot-learning-lab.ape-held-space-request-audit.v1",
              "report_sha256": trial.sha(report_path),
              "sensitivity_report_sha256": (trial.sha(sensitivity_path)
                                            if sensitivity_path.is_file() else None),
              "checks": checks, "passed": not errors, "errors": errors}
    (loop.OUT / "audit.json").write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
