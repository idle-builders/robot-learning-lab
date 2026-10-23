"""Saved-evidence and local-fixture audit for V20's changed near field."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

import kitchen_native as kitchen
import kitchen_trial as trial
import near_clutter_trial as near


def main() -> None:
    report = json.loads((near.OUT / "report.json").read_text())
    scene = near.OUT / "scene.xml"
    checks = {
        "source": report["source_sha256"] == trial.sha(Path(near.__file__)),
        "v19_source": report["v19_source_sha256"] ==
                      trial.sha(Path(near.layout3.__file__)),
        "dense": report["dense_source_sha256"] ==
                 trial.sha(Path(near.dense.__file__)),
        "runner": report["episode_runner_sha256"] == trial.sha(Path(trial.__file__)),
        "command": report["command_guard_sha256"] ==
                   trial.sha(Path(near.command.__file__)),
        "scene": report["room"]["scene_sha256"] == trial.sha(scene),
        "different_scene": report["room"]["scene_sha256"] !=
                           report["room"]["v19_scene_sha256"],
        "native_trace": report["native_trace_sha256"] == trial.sha(kitchen.TRACE),
        "body": report["body_sha256"] == trial.sha(kitchen.BODY),
        "v19": report["v19_report_sha256"] ==
               trial.sha(near.layout3.OUT / "report.json"),
        "map": report["sensor"]["initial_map_sha256"] ==
               trial.sha(near.OUT / "background_xy.npy"),
        "changed_map": report["sensor"]["initial_map_sha256"] !=
                       trial.sha(near.layout3.OUT / "background_xy.npy"),
        "matrix": len(report["results"]) == 48 and
                  len({(r["mode"], r["world"], r["case"])
                       for r in report["results"]}) == 48,
    }
    outcomes = {mode: {"fast_false_stops": 0, "threat_overlaps": 0,
                       "threats_below_5cm": 0, "min_threat_gap_m": float("inf"),
                       "min_sampled_body_fixture_gap_m": float("inf")}
                for mode in near.layout3.MODES}
    v19 = json.loads((near.layout3.OUT / "report.json").read_text())
    earlier = {(r["mode"], r["world"], r["case"]): r for r in v19["results"]}
    ray_differences = 0
    changed_guard_episodes = 0
    videos = 0
    for row in report["results"]:
        mode, world, case = row["mode"], row["world"], row["case"]
        key = f"{mode}:{world}:{case}"
        group = outcomes[mode]
        fast = case in ("fast_clear_held", "fast_right_held")
        group["fast_false_stops"] += int(fast and row["stop_command_s"] < 3.)
        group["threat_overlaps"] += int(not fast and row["modeled_person_overlap"])
        group["threats_below_5cm"] += int(not fast and
                                          row["below_5cm_person_margin"])
        if not fast:
            group["min_threat_gap_m"] = min(group["min_threat_gap_m"],
                                             row["min_modeled_body_person_gap_m"])
        group["min_sampled_body_fixture_gap_m"] = min(
            group["min_sampled_body_fixture_gap_m"],
            row["min_sampled_body_fixture_gap_m"])
        path = Path(row["trace"]["path"])
        checks[f"trace_hash:{key}"] = trial.sha(path) == row["trace"]["sha256"]
        with np.load(path, allow_pickle=False) as saved, np.load(
                earlier[mode, world, case]["trace"]["path"], allow_pickle=False) as old:
            gaps = np.r_[saved["gap_evaluator_m"], saved["midstep_gap_evaluator_m"]]
            checks[f"gap:{key}"] = (
                round(float(gaps.min()), 5) == row["min_modeled_body_person_gap_m"] and
                bool((gaps <= 0).any()) == row["modeled_person_overlap"] and
                bool((gaps < trial.MIN_MARGIN_M).any()) ==
                row["below_5cm_person_margin"])
            packets = [json.loads(raw) for raw in saved["packet_json"]]
            checks[f"packet:{key}"] = (len(packets) == 0 if mode == "none" else
                bool(packets) and all(len(p["ranges_m"]) == 73 and
                                      p["valid_returns"] ==
                                      sum(value is not None for value in p["ranges_m"])
                                      for p in packets))
            if mode == "none":
                checks[f"matched_control:{key}"] = np.array_equal(
                    saved["gap_evaluator_m"], old["gap_evaluator_m"])
            else:
                before = [json.loads(raw) for raw in old["packet_json"]]
                differences = sum(sum(a != b for a, b in
                                      zip(p["ranges_m"], q["ranges_m"]))
                                  for p, q in zip(packets, before))
                ray_differences += differences
                changed_guard_episodes += int(differences > 0)
        video = row["video"]
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
    checks["new_local_ray_signal"] = changed_guard_episodes > 0 and ray_differences > 0
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    new_geoms = [model.geom(name).id for name in ("near_stool_column",
                                                  "near_stool_seat")]
    ape_geoms = [j for j in range(model.ngeom) if model.geom_group[j] == 4]
    person = model.geom("person_capsule").id
    holder = np.zeros(6)
    ape_min = float("inf")
    with np.load(kitchen.TRACE, allow_pickle=False) as saved:
        for world in range(4):
            for t in np.arange(.02, 3.01, .1):
                kitchen.put_native_pose(model, data, saved, world, float(t), 3.)
                for a in ape_geoms:
                    for geom in new_geoms:
                        ape_min = min(ape_min, float(mujoco.mj_geomDistance(
                            model, data, a, geom, 2., holder)))
    person_min = {}
    for case in near.layout3.CASES:
        gap_min = float("inf")
        for t in np.arange(.02, 3.01, .1):
            model.body_pos[model.body("approaching_person").id, :2] = (
                near.layout3.hazard_xy(case, float(t)))
            mujoco.mj_forward(model, data)
            for geom in new_geoms:
                gap_min = min(gap_min, float(mujoco.mj_geomDistance(
                    model, data, person, geom, 2., holder)))
        person_min[case] = round(gap_min, 5)
    checks["new_stool_geometry_clear_10hz"] = (ape_min > 0 and
        all(value > 0 for value in person_min.values()))
    checks["ape_fixture_clear_10hz"] = all(
        group["min_sampled_body_fixture_gap_m"] > 0 for group in outcomes.values())
    audit = {"schema": "robot-learning-lab.ape-robocasa-near-clutter-low-guard-audit.v1",
             "report_sha256": trial.sha(near.OUT / "report.json"),
             "outcomes": outcomes, "changed_guard_episodes": changed_guard_episodes,
             "ray_differences_vs_v19_matched_packets": ray_differences,
             "min_sampled_ape_to_new_stool_m": round(ape_min, 5),
             "min_sampled_person_to_new_stool_m": person_min,
             "videos_checked": videos, "checks": checks,
             "pass": all(checks.values())}
    (near.OUT / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps({"pass": audit["pass"], "outcomes": outcomes,
                      "changed_guard_episodes": changed_guard_episodes,
                      "ray_differences": ray_differences,
                      "new_stool_ape_gap_m": audit["min_sampled_ape_to_new_stool_m"],
                      "videos_checked": videos}, indent=2))
    if not audit["pass"]:
        raise SystemExit("V20 audit failed")


if __name__ == "__main__":
    main()
