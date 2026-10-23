"""Audit alternate-arm episode, targeted range packets and videos."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from run import ROOT, sha
import compare_early_arm_orientation_v133p as v133p
import validate_early_forward_arm_v134p as v134p
import screen_rear_left_up_probe_v135p as v135p
import render_early_forward_arm_v137p as v137p


SOURCE = Path(__file__)
OUT = v134p.OUT / "audit.json"


def main() -> None:
    paths = ((v133p.OUT / "report.json", v133p.SOURCE),
             (v134p.OUT / "report.json", v134p.SOURCE),
             (v135p.OUT / "report.json", v135p.SOURCE),
             (v137p.OUT / "report.json", v137p.SOURCE))
    reports = []
    checks = 0
    for path, source in paths:
        report = json.loads(path.read_text())
        key = ("wrapper_source_sha256" if source == v137p.SOURCE
               else "source_sha256")
        assert report[key] == sha(source)
        checks += 1
        reports.append(report)
    arm, episode, sensors, media = reports
    assert len(arm["geometry"]) == 4 and len(arm["coverage"]) == 2
    assert len(episode["rows"]) == 8 and len(sensors["rows"]) == 2
    assert len(media["rows"]) == 2
    checks += 3
    for row in arm["geometry"]:
        assert row["minimum_scene_gap_m"] >= .05
        checks += 1
    for row in arm["coverage"]:
        plus = row["plus_swept"]
        minus = row["minus_swept"]
        assert plus["upper"]["unknown"] < minus["upper"]["unknown"]
        assert plus["lower"]["unknown"] == 0
        checks += 2
    for row in episode["rows"]:
        assert row["travel"]["scene"]["minimum_gap_m"] >= .05
        assert row["travel"]["self"]["minimum_gap_m"] >= 0.
        assert row["owner_show"]["minimum_scene_gap_m"] >= .05
        assert row["owner_show"]["owner_start_s"] == 16.6
        assert row["owner_show"]["image_time_s"] == 17.4
        assert row["actor_accepted_box"] is not None
        assert row["evaluator_box_iou"] >= .95
        frame = ROOT / row["owner_show"]["owner_image"]["actor_frame_path"]
        assert sha(frame) == row["owner_show"]["owner_image"][
            "actor_frame_sha256"]
        with np.load(frame) as packet:
            assert abs(float(packet["capture_time_s"])-17.4) < 1e-6
        checks += 9
    supported = 0
    for row in sensors["rows"]:
        path = ROOT / row["new_packet_path"]
        assert sha(path) == row["new_packet_sha256"]
        packets = json.loads(path.read_text())
        for rays in packets["ray_packets"].values():
            assert all(ray["capture_time_s"] == 8.0 for ray in rays)
            checks += 1
        for layout in row["layouts"].values():
            supported += layout["coverage_status"] == "observed_free"
            checks += 1
        checks += 1
    assert supported == 0
    checks += 1
    decoded = {}
    for row in media["rows"]:
        path = ROOT / row["video_path"]
        assert sha(path) == row["video_sha256"]
        reader = imageio.get_reader(path)
        try:
            count = sum(1 for _ in reader)
        finally:
            reader.close()
        assert count == 34
        decoded[row["case"]] = count
        checks += 2
    audit = {"schema": "robot-learning-lab.ape-early-forward-arm-audit.v1",
             "source_sha256": sha(SOURCE),
             "report_sha256": {str(path.relative_to(ROOT)): sha(path)
                                for path, _ in paths},
             "checks": checks,
             "supported_sensor_layouts": supported,
             "decoded_video_frames": decoded,
             "scope": "Consistency audit of alternate-arm geometry, rendered "
             "generic item gate, source/image/range/media hashes, 8.0s "
             "range times, unsupported pose-sampled route and video frames. "
             "Not physical safety or sensor qualification."}
    OUT.write_text(json.dumps(audit, indent=2)+"\n")
    print(OUT, checks)


if __name__ == "__main__":
    main()
