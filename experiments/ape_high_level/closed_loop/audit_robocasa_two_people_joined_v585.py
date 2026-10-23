"""Independent route/identity/geometry audit for the V584 joined episode."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

import robocasa_owner_person as person
import robocasa_person_interaction as interaction
import screen_robocasa_two_people_v582 as two


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-two-people-joined-v584"
REPORT = OUT / "report.json"
BASELINE = ROOT / "artifacts/ape-robocasa-first-return-concurrent-retry-v577/report.json"
SOURCE = Path(__file__).with_name("run_robocasa_two_people_joined_v584.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    report = json.loads(REPORT.read_text())
    ep = report["episode"]
    baseline = next(x for x in json.loads(BASELINE.read_text())["episodes"]
                    if x["response"] == ep["response"])
    model = mujoco.MjModel.from_xml_path(str(ROOT / "artifacts/ape-robocasa-two-people-v582/two-people.xml"))
    data = mujoco.MjData(model)
    person.set_owner_pose(model, interaction.TORSO_XY,
                          interaction.REST_HAND, show_can=False)
    two.set_distractor(model, two.DISTRACTOR_TORSO_XY)
    ape = model.geom("audit_ape_geom").id
    owner_geoms = [model.geom(name).id for name in person.PERSON_GEOMS]
    other_geoms = [model.geom(name.replace("owner_", "distractor_")).id
                   for name in person.PERSON_GEOMS]
    samples = []
    for row in ep["route_episode"]["trace"]:
        start = np.asarray(row["actual_xy_m"])
        estimate = np.asarray(row["estimated_xy_m"])
        requested = row["requested_xy_m"]
        delta = ((np.asarray(requested) - estimate) * (1. + .30 / 5.89)
                 if requested is not None else np.zeros(2))
        poses = [start]
        for index, step in enumerate(row["motion_substeps"], 1):
            fraction = (index - 1 if step["stopped_before_move"] else index) / 20
            poses.append(start + fraction * delta)
        for index, xy in enumerate(poses):
            model.body_pos[model.body("audit_ape").id, :2] = xy
            mujoco.mj_forward(model, data)
            gaps = {"owner": min(float(mujoco.mj_geomDistance(
                        model, data, ape, geom, 2., None)) for geom in owner_geoms),
                    "distractor": min(float(mujoco.mj_geomDistance(
                        model, data, ape, geom, 2., None)) for geom in other_geoms)}
            samples.append({"route_step": row["step"], "substep": index,
                            "ape_xy_m": xy.tolist(), **gaps})
    track = ep["multi_history"][-1]["tracks"][0]
    center = np.asarray(track["center_xy_m"])
    distance_to_distractor = float(np.linalg.norm(center - two.DISTRACTOR_TORSO_XY))
    distance_to_owner = float(np.linalg.norm(center - interaction.TORSO_XY))
    selected = next(r for r in ep["trace"] if r["stage"] == "choose_pre_show_standoff")
    video = ep["video"]
    cap = cv2.VideoCapture(video["path"])
    count = 0
    while cap.isOpened():
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    checks = {"runner_source_hash": report["source_sha256"] == sha(SOURCE),
              "scene_xml_hash": report["scene_xml_sha256"] == sha(ROOT / "artifacts/ape-robocasa-two-people-v582/two-people.xml"),
              "video_hash": video["sha256"] == sha(video["path"]),
              "video_frame_count": count == video["frames"],
              "route_geometry_sampled": len(samples) >= 20,
              "route_person_clearance_ge_10cm": min(min(x["owner"], x["distractor"]) for x in samples) >= .1,
              "range_track_matches_distractor_not_owner": distance_to_distractor < .05 and distance_to_owner > .5,
              "standoff_unsupported": selected["chosen"] is None and ep["outcome"] == "pre_show_standoff_not_supported",
              "no_show_or_find_claim": not any(r["stage"] in ("owner_show", "owner_confirmed_metric_find") for r in ep["trace"]),
              "matched_baseline_found": baseline["outcome"] == "owner_confirmed_metric_find"}
    result = {"scope": "independent V584 20ms route-proxy person geometry, anonymous track identity and video audit; extra person stationary and no actor identity cue",
              "source_sha256": sha(Path(__file__)), "report_sha256": sha(REPORT),
              "baseline_report_sha256": sha(BASELINE), "checks": checks,
              "all_passed": all(checks.values()),
              "route_sample_count": len(samples),
              "minimum_owner_gap_m": min(x["owner"] for x in samples),
              "minimum_distractor_gap_m": min(x["distractor"] for x in samples),
              "selected_range_track_distance_to_distractor_m": distance_to_distractor,
              "selected_range_track_distance_to_owner_m": distance_to_owner,
              "video_frames_decoded": count,
              "baseline_outcome": baseline["outcome"],
              "two_person_outcome": ep["outcome"]}
    (OUT / "audit-v585.json").write_text(json.dumps(result, indent=2) + "\n")
    print("pass", result["all_passed"], "samples", len(samples),
          "min_owner", result["minimum_owner_gap_m"],
          "min_distractor", result["minimum_distractor_gap_m"])
    if not result["all_passed"]:
        print([name for name, passed in checks.items() if not passed])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
