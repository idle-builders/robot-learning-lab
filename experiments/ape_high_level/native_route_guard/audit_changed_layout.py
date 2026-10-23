"""Saved-evidence and scene-clearance audit for V19's changed kitchen."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

import changed_layout_trial as changed
import kitchen_native as kitchen
import kitchen_trial as trial


def main() -> None:
    report = json.loads((changed.OUT / "report.json").read_text())
    scene = changed.OUT / "scene.xml"
    checks = {
        "source": report["source_sha256"] == trial.sha(Path(changed.__file__)),
        "scene_builder": report["scene_builder_sha256"] ==
                         trial.sha(Path(kitchen.__file__)),
        "dense": report["dense_source_sha256"] ==
                 trial.sha(Path(changed.dense.__file__)),
        "left_cases": report["v11_case_source_sha256"] ==
                      trial.sha(Path(changed.left_cases.__file__)),
        "right_cases": report["v17_case_source_sha256"] ==
                       trial.sha(Path(changed.right_cases.__file__)),
        "runner": report["episode_runner_sha256"] == trial.sha(Path(trial.__file__)),
        "command": report["command_guard_sha256"] ==
                   trial.sha(Path(changed.command.__file__)),
        "scene": report["room"]["scene_sha256"] == trial.sha(scene),
        "different_layout": report["room"]["scene_sha256"] !=
                            report["room"]["source_layout1_scene_sha256"],
        "trace_source": report["native_trace_sha256"] == trial.sha(kitchen.TRACE),
        "body": report["body_sha256"] == trial.sha(kitchen.BODY),
        "v18": report["v18_report_sha256"] ==
               trial.sha(kitchen.OUT / "v18-single-front/report.json"),
        "map": report["sensor"]["initial_map_sha256"] ==
               trial.sha(changed.OUT / "background_xy.npy"),
        "matrix": len(report["results"]) == 48 and
                  len({(r["mode"], r["world"], r["case"])
                       for r in report["results"]}) == 48,
    }
    outcomes = {mode: {"fast_false_stops": 0, "threat_overlaps": 0,
                       "threats_below_5cm": 0, "min_threat_gap_m": float("inf"),
                       "min_sampled_body_fixture_gap_m": float("inf")}
                for mode in changed.MODES}
    videos = 0
    for result in report["results"]:
        mode = result["mode"]
        key = f"{mode}:{result['world']}:{result['case']}"
        group = outcomes[mode]
        is_fast = result["case"] in ("fast_clear_held", "fast_right_held")
        group["fast_false_stops"] += int(is_fast and result["stop_command_s"] < 3.)
        group["threat_overlaps"] += int(not is_fast and
                                        result["modeled_person_overlap"])
        group["threats_below_5cm"] += int(not is_fast and
                                          result["below_5cm_person_margin"])
        if not is_fast:
            group["min_threat_gap_m"] = min(group["min_threat_gap_m"],
                                             result["min_modeled_body_person_gap_m"])
        group["min_sampled_body_fixture_gap_m"] = min(
            group["min_sampled_body_fixture_gap_m"],
            result["min_sampled_body_fixture_gap_m"])
        path = Path(result["trace"]["path"])
        checks[f"trace_hash:{key}"] = trial.sha(path) == result["trace"]["sha256"]
        with np.load(path, allow_pickle=False) as saved:
            gaps = np.r_[saved["gap_evaluator_m"], saved["midstep_gap_evaluator_m"]]
            checks[f"gap:{key}"] = (
                round(float(gaps.min()), 5) == result["min_modeled_body_person_gap_m"] and
                bool((gaps <= 0).any()) == result["modeled_person_overlap"] and
                bool((gaps < trial.MIN_MARGIN_M).any()) ==
                result["below_5cm_person_margin"])
            packets = [json.loads(raw) for raw in saved["packet_json"]]
            checks[f"packet:{key}"] = (len(packets) == 0 if mode == "none" else
                bool(packets) and all(len(p["ranges_m"]) == 73 and
                                      p["valid_returns"] ==
                                      sum(value is not None for value in p["ranges_m"])
                                      for p in packets))
        video = result["video"]
        if video is not None:
            videos += 1
            movie = Path(video["path"])
            capture = cv2.VideoCapture(str(movie))
            count = 0
            while True:
                okay, frame = capture.read()
                if not okay:
                    break
                count += int(frame.size > 0)
            capture.release()
            checks[f"video:{key}"] = (trial.sha(movie) == video["sha256"] and
                                       count == video["frames"])
    checks["six_videos"] = videos == 6
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    person = model.geom("person_capsule").id
    holder = np.zeros(6)
    fixtures = [j for j in range(model.ngeom) if j != person and
                model.geom_group[j] != 4 and model.geom(j).name != "pet_box" and
                model.geom_type[j] != mujoco.mjtGeom.mjGEOM_PLANE and
                "floor" not in model.geom(j).name.lower()]
    person_fixture = {}
    for case in changed.CASES:
        best = (float("inf"), None, None)
        for t in np.arange(.02, 3.01, .1):
            model.body_pos[model.body("approaching_person").id, :2] = (
                changed.hazard_xy(case, float(t)))
            mujoco.mj_forward(model, data)
            for geom in fixtures:
                gap = float(mujoco.mj_geomDistance(model, data, person,
                                                   geom, 2., holder))
                if gap < best[0]:
                    best = (gap, round(float(t), 3), model.geom(geom).name)
        person_fixture[case] = {"min_sampled_gap_m": round(best[0], 5),
                                "time_s": best[1], "fixture": best[2]}
    checks["person_fixture_clear_10hz"] = all(
        x["min_sampled_gap_m"] > 0 for x in person_fixture.values())
    checks["ape_fixture_clear_10hz"] = all(
        x["min_sampled_body_fixture_gap_m"] > 0 for x in outcomes.values())
    audit = {"schema": "robot-learning-lab.ape-robocasa-layout3-low-guard-audit.v1",
             "report_sha256": trial.sha(changed.OUT / "report.json"),
             "outcomes": outcomes, "person_fixture_clearance": person_fixture,
             "videos_checked": videos, "checks": checks,
             "pass": all(checks.values())}
    (changed.OUT / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps({"pass": audit["pass"], "outcomes": outcomes,
                      "person_fixture_clearance": person_fixture,
                      "videos_checked": videos}, indent=2))
    if not audit["pass"]:
        raise SystemExit("V19 audit failed")


if __name__ == "__main__":
    main()
