"""Saved-evidence audit for the V17 right-entry sensor-placement test."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

import kitchen_native as kitchen
import kitchen_trial as trial
import right_entry_trial as right


def main() -> None:
    report = json.loads((right.OUT / "report.json").read_text())
    checks = {
        "source": report["source_sha256"] == trial.sha(Path(right.__file__)),
        "v15_source": report["v15_source_sha256"] == trial.sha(Path(right.left.__file__)),
        "runner": report["episode_runner_sha256"] == trial.sha(Path(trial.__file__)),
        "command_guard": report["command_guard_sha256"] ==
                         trial.sha(Path(right.command.__file__)),
        "v15_report": report["v15_report_sha256"] ==
                      trial.sha(right.left.OUT / "report.json"),
        "scene": report["scene_sha256"] == trial.sha(kitchen.OUT / "scene.xml"),
        "native_trace": report["native_trace_sha256"] == trial.sha(kitchen.TRACE),
        "body": report["body_sha256"] == trial.sha(kitchen.BODY),
        "map": report["sensor"]["initial_map_sha256"] ==
               trial.sha(right.OUT / "background_xy.npy"),
        "matrix": len(report["results"]) == 36 and
                  len({(r["configuration"], r["world"], r["case"])
                       for r in report["results"]}) == 36,
    }
    outcomes = {mode: {"fast_false_stops": 0, "threat_overlaps": 0,
                       "threats_below_5cm": 0, "right_ray_returns": 0}
                for mode in right.MODES}
    videos = 0
    for result in report["results"]:
        mode = result["configuration"]
        key = f"{mode}:{result['world']}:{result['case']}"
        stats = outcomes[mode]
        stats["fast_false_stops"] += int(result["case"] == "fast_right_held" and
                                         result["stop_command_s"] < 3.)
        stats["threat_overlaps"] += int(result["case"] != "fast_right_held" and
                                        result["modeled_person_overlap"])
        stats["threats_below_5cm"] += int(result["case"] != "fast_right_held" and
                                          result["below_5cm_person_margin"])
        trace = Path(result["trace"]["path"])
        checks[f"trace_hash:{key}"] = trial.sha(trace) == result["trace"]["sha256"]
        with np.load(trace, allow_pickle=False) as saved:
            gaps = np.r_[saved["gap_evaluator_m"], saved["midstep_gap_evaluator_m"]]
            checks[f"gap:{key}"] = (
                round(float(gaps.min()), 5) == result["min_modeled_body_person_gap_m"] and
                bool((gaps <= 0).any()) == result["modeled_person_overlap"] and
                bool((gaps < trial.MIN_MARGIN_M).any()) ==
                result["below_5cm_person_margin"])
            packets = [json.loads(p) for p in saved["packet_json"]]
            if mode == "none":
                checks[f"packets:{key}"] = len(packets) == 0
            else:
                expected = ("front_right_unavailable" if mode ==
                            "left_only_right_missing" else "right_flank_low")
                checks[f"packets:{key}"] = bool(packets) and all(
                    len(p["ranges_m"]) == len(p["ray_health"]) ==
                    len(p["ray_sources"]) == 73 and
                    sum(s == expected for s in p["ray_sources"]) == 26 and
                    all(value is None and not healthy
                        for value, healthy, source in zip(p["ranges_m"],
                                                          p["ray_health"],
                                                          p["ray_sources"])
                        if source == "front_right_unavailable")
                    for p in packets)
                stats["right_ray_returns"] += sum(
                    sum(source == "right_flank_low" and value is not None
                        for source, value in zip(p["ray_sources"], p["ranges_m"]))
                    for p in packets)
        video = result["video"]
        if video is not None:
            videos += 1
            movie = Path(video["path"])
            capture = cv2.VideoCapture(str(movie))
            frames = 0
            while True:
                okay, frame = capture.read()
                if not okay:
                    break
                frames += int(frame.size > 0)
            capture.release()
            checks[f"video:{key}"] = (trial.sha(movie) == video["sha256"] and
                                       frames == video["frames"])
    checks["six_videos"] = videos == 6
    model = mujoco.MjModel.from_xml_path(str(kitchen.OUT / "scene.xml"))
    data = mujoco.MjData(model)
    person = model.geom("person_capsule").id
    holder = np.zeros(6)
    fixtures = [j for j in range(model.ngeom)
                if j != person and model.geom_group[j] != 4 and
                model.geom(j).name != "pet_box" and
                model.geom_type[j] != mujoco.mjtGeom.mjGEOM_PLANE and
                "floor" not in model.geom(j).name.lower()]
    fixture_clearance = {}
    for case in right.CASES:
        minimum = (float("inf"), None, None)
        for t in np.arange(.02, 3.01, .10):
            model.body_pos[model.body("approaching_person").id, :2] = (
                right.hazard_xy(case, float(t)))
            mujoco.mj_forward(model, data)
            for geom in fixtures:
                gap = float(mujoco.mj_geomDistance(model, data, person,
                                                   geom, 2.0, holder))
                if gap < minimum[0]:
                    minimum = (gap, round(float(t), 3), model.geom(geom).name)
        fixture_clearance[case] = {"min_sampled_gap_m": round(minimum[0], 5),
                                   "time_s": minimum[1], "fixture": minimum[2]}
    checks["person_fixture_clear_10hz"] = all(
        value["min_sampled_gap_m"] > 0 for value in fixture_clearance.values())
    audit = {"schema": "robot-learning-lab.ape-robocasa-right-entry-fan-development-audit.v1",
             "report_sha256": trial.sha(right.OUT / "report.json"),
             "outcomes": outcomes, "videos_checked": videos,
             "person_fixture_clearance": fixture_clearance,
             "checks": checks, "pass": all(checks.values())}
    (right.OUT / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps({"pass": audit["pass"], "outcomes": outcomes,
                      "videos_checked": videos}, indent=2))
    if not audit["pass"]:
        raise SystemExit("V17 audit failed")


if __name__ == "__main__":
    main()
