"""Trace/model/image consistency audit for the V21 actor-camera probe."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import torch

import camera_person_probe as probe
import kitchen_native as kitchen
import kitchen_trial as trial


def main() -> None:
    report = json.loads((probe.OUT / "report.json").read_text())
    checkpoint = (Path(torch.hub.get_dir()) / "checkpoints" /
                  "ssdlite320_mobilenet_v3_large_coco-a79551df.pth")
    rows = report["rows"]
    checks = {
        "source": report["source_sha256"] == trial.sha(Path(probe.__file__)),
        "scene": report["scene_sha256"] == trial.sha(kitchen.OUT / "scene.xml"),
        "trace": report["native_trace_sha256"] == trial.sha(kitchen.TRACE),
        "body": report["body_sha256"] == trial.sha(kitchen.BODY),
        "case_source": report["v11_case_source_sha256"] ==
                       trial.sha(Path(probe.held.__file__)),
        "checkpoint": report["detector"]["checkpoint_sha256"] ==
                      trial.sha(checkpoint),
        "matrix": len(rows) == 120 and len({(r["world"], r["case"],
                                            r["time_s"], r["camera"])
                                           for r in rows}) == 120,
        "frame_count": len(report["frames"]) == 12,
    }
    for row in rows:
        key = f"{row['world']}:{row['case']}:{row['time_s']}:{row['camera']}"
        accepted = [p for p in row["actor_proposals"] if p["score"] >=
                    report["detector"]["gate"]]
        matched = [p for p in accepted if p["evaluator_iou"] is not None and
                   p["evaluator_iou"] >= .25]
        checks[f"decision:{key}"] = (
            row["accepted_person_count"] == len(accepted) and
            row["evaluator_matched_detection"] == bool(matched) and
            all(0 <= p["local_depth_valid_fraction"] <= 1 and
                (p["local_depth_median_m"] is None or p["local_depth_median_m"] > 0)
                for p in row["actor_proposals"]))
        if row["case"] == "empty_room":
            checks[f"empty_truth:{key}"] = (row["evaluator_visible_pixels"] == 0
                                             and row["evaluator_box_xyxy"] is None)
        if "overlay" in row:
            image = Path(row["overlay"]["path"])
            pixels = cv2.imread(str(image))
            checks[f"image:{key}"] = (trial.sha(image) ==
                row["overlay"]["sha256"] and pixels is not None and
                list(pixels.shape[:2][::-1]) == report["image_size_px"])
    outcomes = {}
    for camera in probe.CAMERAS:
        scoped = [r for r in rows if r["camera"] == camera]
        positive = [r for r in scoped if r["case"] != "empty_room" and
                    r["evaluator_visible_pixels"] >= 500]
        empty = [r for r in scoped if r["case"] == "empty_room"]
        outcomes[camera] = {"visible_positive_views": len(positive),
                            "matched_visible_views": sum(
                                r["evaluator_matched_detection"] for r in positive),
                            "accepted_empty_views": sum(
                                r["accepted_person_count"] > 0 for r in empty),
                            "empty_views": len(empty)}
        for field, value in outcomes[camera].items():
            checks[f"summary:{camera}:{field}"] = report["summary"][camera][field] == value
    audit = {"schema": "robot-learning-lab.ape-robocasa-head-person-probe-audit.v1",
             "report_sha256": trial.sha(probe.OUT / "report.json"),
             "outcomes": outcomes, "images_checked": len(report["frames"]),
             "checks": checks, "pass": all(checks.values())}
    (probe.OUT / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps({"pass": audit["pass"], "outcomes": outcomes,
                      "images_checked": audit["images_checked"]}, indent=2))
    if not audit["pass"]:
        raise SystemExit("V21 audit failed")


if __name__ == "__main__":
    main()
