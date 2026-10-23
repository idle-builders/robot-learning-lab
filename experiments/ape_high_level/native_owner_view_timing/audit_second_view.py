"""Check equal-time branch provenance, decision evidence, and video frames."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-native-owner-view-timing/v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    report = json.loads((OUT / "report.json").read_text())
    assert report["schema"] == "robot-learning-lab.owner-second-view-timing.v1"
    assert report["seed"] == 86
    assert report["first_chest"]["actor_observation"]["candidate"] is None
    assert sha(ROOT / report["owner_trace"]) == report["owner_trace_sha256"]
    sources = {
        "owner": ROOT / "experiments/ape_high_level/closed_loop/run.py",
        "recovery_library": ROOT / "experiments/ape_high_level/native_owner_recovery/run_recovery.py",
        "timing_runner": ROOT / "experiments/ape_high_level/native_owner_view_timing/compare_second_view.py",
        "native_qpos": ROOT / "artifacts/ape-cove50-transfer-20261008/varied150-forward-seed144.npz",
        "native_odometry": ROOT / "artifacts/ape-sensors-50cm/native-proprioceptive-odometry-v1/varied150-forward-seed144-trace.json",
    }
    for key, path in sources.items():
        assert sha(path) == report["source_sha256"][key], key
    for row in [report["first_chest"]] + list(report["branches"].values()):
        observation = row["actor_observation"]
        assert sha(ROOT / observation["rgb"]) == observation["rgb_sha256"]
        assert sha(ROOT / observation["depth"]) == observation["depth_sha256"]
        assert len(row["range_samples"]) == 5
    chest = report["branches"]["second_chest_wait"]
    head = report["branches"]["head_look"]
    assert chest["elapsed_cumulative_recovery_s"] == head["elapsed_cumulative_recovery_s"] == .8
    assert chest["actor_observation"]["capture_time_s"] == head["actor_observation"]["capture_time_s"]
    assert chest["actor_observation"]["candidate"] is None
    assert head["actor_observation"]["candidate"] is not None
    assert head["evaluator"]["candidate_true_target_iou_evaluator_only"] >= .9
    assert chest["evaluator"]["true_target_pixel_area_evaluator_only"] == 0
    assert not chest["interrupted"] and not head["interrupted"]
    assert min(row["path_min_clearance_evaluator_only_m"]
               for row in (chest, head)) > 0
    movie = ROOT / report["comparison_video"]
    assert sha(movie) == report["comparison_video_sha256"]
    cap = cv2.VideoCapture(str(movie))
    assert cap.isOpened()
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape == (520, 1600, 3)
        count += 1
    cap.release()
    assert count == report["comparison_video_frames"] and count > 10
    audit = {"schema": "robot-learning-lab.owner-second-view-timing-audit.v1",
             "passed": True, "report_sha256": sha(OUT / "report.json"),
             "decoded_video_frames": count,
             "equal_cumulative_time_s": .8,
             "second_chest_candidate": False,
             "head_candidate": True,
             "head_target_iou_evaluator_only": head["evaluator"]["candidate_true_target_iou_evaluator_only"]}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("PASS equal-time second-view audit", count, "frames")


if __name__ == "__main__":
    main()
