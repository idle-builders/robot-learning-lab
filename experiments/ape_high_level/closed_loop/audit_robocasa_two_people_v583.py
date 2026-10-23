"""Independent packet, identity-label and replay audit of V582."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-two-people-v582"
REPORT = OUT / "report.json"
SOURCE = Path(__file__).with_name("screen_robocasa_two_people_v582.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    report = json.loads(REPORT.read_text())
    checks = {"experiment_source_hash": report["source_sha256"] == sha(SOURCE),
              "scene_xml_hash": report["scene_xml_sha256"] == sha(OUT / "two-people.xml"),
              "model_weights_hash": report["person_detector_weights_sha256"] ==
                  sha(ROOT / ".cache/ape-high-level/models/grounding-dino-tiny/model.safetensors")}
    trials = {x["trial"]: x for x in report["trials"]}
    checks["matched_trials"] = set(trials) == {"distractor_only", "owner_and_distractor"}
    for trial in trials.values():
        for view in trial["views"]:
            key = f"{trial['trial']}_up{view['pitch_degrees']}"
            packet = np.load(view["raw_rgbd"]["path"])
            checks[f"{key}_raw_hash"] = sha(view["raw_rgbd"]["path"]) == view["raw_rgbd"]["sha256"]
            checks[f"{key}_image_hash"] = sha(view["image"]["path"]) == view["image"]["sha256"]
            checks[f"{key}_rgbd_shapes"] = (packet["rgb"].shape == (480, 640, 3) and
                                               packet["depth"].shape == (480, 640))
    alone = trials["distractor_only"]
    paired = trials["owner_and_distractor"]
    alone_first = [v["detections"][0] for v in alone["views"]]
    paired_first = [v["detections"][0] for v in paired["views"]]
    old = np.asarray(report["old_track_xy_m"])
    points = [np.asarray(d["point_xyz_m"]) for d in alone_first]
    candidate = np.mean(points, axis=0)[:2]
    displacement = float(np.linalg.norm(candidate - old))
    checks["distractor_only_two_pitch_consistency"] = float(np.linalg.norm(points[0] - points[1])) <= .15
    checks["distractor_only_old_rule_would_bind"] = (displacement <= .75 and
                                                       min(d["score"] for d in alone_first) >= .35)
    checks["distractor_only_gt_distractor"] = all(
        d["torso_iou_evaluator"]["distractor"] > .5 and
        d["torso_iou_evaluator"]["owner"] == 0.
        for d in alone_first)
    checks["paired_top_identity_switch"] = (
        paired_first[0]["torso_iou_evaluator"]["distractor"] > .5 and
        paired_first[1]["torso_iou_evaluator"]["owner"] > .4 and
        paired_first[1]["torso_iou_evaluator"]["distractor"] == 0.)
    checks["paired_two_pitch_rule_refuses"] = float(np.linalg.norm(
        np.asarray(paired_first[0]["point_xyz_m"]) -
        np.asarray(paired_first[1]["point_xyz_m"]))) > .15
    checks["distinct_person_clearance"] = report["evaluator_gaps"]["owner_distractor_m"] > .1
    video = report["camera_view_video"]
    checks["video_hash"] = sha(video["path"]) == video["sha256"]
    reader = cv2.VideoCapture(video["path"])
    count = 0
    while reader.isOpened():
        ok, frame = reader.read()
        if not ok:
            break
        count += 1
    reader.release()
    checks["video_decodes_32_frames"] = count == 32
    result = {"scope": "independent V582 packet/hash/label/association replay; four static rendered camera views, not a full dynamic episode",
              "source_sha256": sha(Path(__file__)), "report_sha256": sha(REPORT),
              "checks": checks, "all_passed": all(checks.values()),
              "independent_replay": {"distractor_only_candidate_xy_m": candidate.tolist(),
                                     "distractor_only_old_track_distance_m": displacement,
                                     "distractor_only_min_score": min(d["score"] for d in alone_first),
                                     "paired_top_boxes_are_different_evaluator_people": True,
                                     "video_frames": count}}
    (OUT / "audit-v583.json").write_text(json.dumps(result, indent=2) + "\n")
    print("all_passed", result["all_passed"], "checks", len(checks),
          "video_frames", count)
    if not result["all_passed"]:
        print([name for name, passed in checks.items() if not passed])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
