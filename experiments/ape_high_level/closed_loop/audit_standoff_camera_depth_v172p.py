"""Independent saved RGB-D projection, timing and cue audit for V171p."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from run import ROOT, sha
import screen_standoff_camera_depth_v171p as v171p


SOURCE = Path(__file__)
REPORT = v171p.OUT / "report.json"


def low_region_from_saved(data: dict) -> tuple[int, float | None, int]:
    depth = data["sampled_depth_m"].astype(float)
    height, width = map(int, data["full_shape_hw"])
    rows, cols = np.mgrid[v171p.DEPTH_OFFSET:height:v171p.DEPTH_STRIDE,
                          v171p.DEPTH_OFFSET:width:v171p.DEPTH_STRIDE]
    if depth.shape != rows.shape:
        raise AssertionError("Saved depth grid shape changed")
    focal = height/(2*math.tan(math.radians(float(data["fovy_deg"]))/2))
    valid = np.isfinite(depth) & (depth > .08) & (
        depth < v171p.MAX_DEPTH_M)
    directions = np.stack(((cols-width/2)/focal,
                           -(rows-height/2)/focal,
                           -np.ones_like(depth)), axis=-1)[valid]
    origin = data["camera_origin_xyz_m"]
    rotation = data["camera_rotation"]
    xyz = origin+(directions*depth[valid, None])@rotation.T
    estimate = data["estimated_pose_xyyaw"]
    delta = xyz[:, :2]-estimate[:2]
    c, s = math.cos(float(estimate[2])), math.sin(float(estimate[2]))
    x = c*delta[:, 0]+s*delta[:, 1]
    y = -s*delta[:, 0]+c*delta[:, 1]
    z = xyz[:, 2]
    low = x[(x > .08) & (x < .50) &
            (y > .05) & (y < .50) &
            (z > .015) & (z < .10)]
    p10 = (round(float(np.quantile(low, .1)), 5)
           if len(low) >= v171p.MIN_SUPPORT else None)
    return len(low), p10, int(np.count_nonzero(valid))


def main() -> None:
    report = json.loads(REPORT.read_text())
    checks = images = packets = 0

    def check(ok: bool, reason: str) -> None:
        nonlocal checks
        if not ok:
            raise AssertionError(reason)
        checks += 1

    check(sha(Path(v171p.__file__)) == report["source_sha256"],
          "RGB-D screen source changed")
    check(sha(v171p.v162p.OUT / "report.json") ==
          report["v162_report_sha256"],
          "Matched source case report changed")
    check(tuple(report["cameras"]) == v171p.CAMERAS and
          len(report["rows"]) == 32,
          "RGB-D camera matrix incomplete")
    source = json.loads((v171p.v162p.OUT / "report.json").read_text())
    cases = {case["id"]: case for case in source["source_cases"]}
    check({(r["source_case_id"], r["camera"])
           for r in report["rows"]} ==
          {(case, camera) for case in cases
           for camera in v171p.CAMERAS},
          "RGB-D case/camera matrix changed")
    for row in report["rows"]:
        case = cases[row["source_case_id"]]
        check(row["class"] == case["class"] and
              row["evaluator_first_below_5cm_s"] ==
              case["first_below_5cm_s"],
              "Evaluator label joined to wrong camera case")
        samples = row["samples"]
        check([s["capture_time_s"] for s in samples] ==
              list(v171p.TIMES_S),
              "RGB-D capture timeline changed")
        for index, sample in enumerate(samples):
            rgb_file = ROOT / sample["rgb_path"]
            depth_file = ROOT / sample["depth_packet_path"]
            check(sha(rgb_file) == sample["rgb_sha256"] and
                  sha(depth_file) == sample["depth_packet_sha256"],
                  "Saved RGB or depth packet changed")
            with Image.open(rgb_file) as image:
                check(image.width > 0 and image.height > 0,
                      "RGB frame malformed")
            images += 1
            with np.load(depth_file, allow_pickle=False) as packet:
                support, p10, valid = low_region_from_saved(packet)
                time_s = float(packet["capture_time_s"])
            check(time_s == sample["capture_time_s"] and
                  support == sample["front_left_low_depth_support"] and
                  p10 == sample["front_left_low_p10_x_m"] and
                  valid == sample["valid_depth_samples"],
                  "Saved metric depth does not reproduce actor statistic")
            packets += 1
            if index >= 2 and p10 is not None:
                older = samples[index-2]["front_left_low_p10_x_m"]
                if older is not None:
                    closing = round(older-p10, 5)
                    check(sample["closing_0p2s_m"] == closing and
                          sample["cue_now"] == (
                              closing >= report[
                                  "frozen_closing_threshold_m_per_0p2s"]),
                          "Camera cue differs from saved depth statistic")
        first = next((s["capture_time_s"] for s in samples
                      if s["cue_now"]), None)
        check(first == row["first_pose_compensated_depth_cue_s"],
              "First camera cue changed")
        check(row["timely_hazard_warning"] == (
            case["class"] == "hazard" and first is not None and
            first <= case["first_below_5cm_s"]),
            "Timely camera warning classification changed")
        check(row["false_control_warning"] == (
            case["class"] == "control" and first is not None),
            "Camera control warning classification changed")
    for camera in v171p.CAMERAS:
        group = [r for r in report["rows"] if r["camera"] == camera]
        hazard = [r for r in group if r["class"] == "hazard"]
        control = [r for r in group if r["class"] == "control"]
        item = report["summary"][camera]
        check(item["hazards"] == len(hazard) == 8 and
              item["timely_hazard_warnings"] == sum(
                  r["timely_hazard_warning"] for r in hazard) and
              item["controls"] == len(control) == 8 and
              item["false_control_warnings"] == sum(
                  r["false_control_warning"] for r in control),
              "Camera summary differs from saved rows")
    result = {
        "schema": "robot-learning-lab.ape-standoff-camera-depth-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "checks_passed": checks,
        "saved_depth_packets_recomputed": packets,
        "saved_rgb_images_decoded": images,
        "outcome": "pass",
    }
    (v171p.OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
