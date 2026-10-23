"""Aggregate and verify matched owner post-stop view comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCES = (
    ROOT / "artifacts/ape-native-owner-reobserve/pilot-seed85-v1/report.json",
    ROOT / "artifacts/ape-native-owner-reobserve/pilot-v1/report.json",
)
DEFAULT_OUT = ROOT / "artifacts/ape-native-owner-reobserve/v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot decode {path}")
    frames = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if image.shape[:2] != (520, 1600):
            raise ValueError(f"Unexpected frame shape: {path}")
        frames += 1
    capture.release()
    if not frames:
        raise ValueError(f"Empty video: {path}")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    expected_code = ROOT / "experiments/ape_high_level/native_owner_reobserve/compare_post_stop_views.py"
    actor_code = ROOT / "experiments/ape_high_level/closed_loop/run.py"
    adapter_code = ROOT / "experiments/ape_high_level/native_owner_bridge/replay_owner_packets.py"
    cases = []
    sources = []
    for source in args.sources:
        path = source.resolve()
        report = json.loads(path.read_text())
        for key, current in (("analysis_code_sha256", expected_code),
                             ("owner_actor_code_sha256", actor_code),
                             ("native_adapter_code_sha256", adapter_code)):
            if report[key] != sha(current):
                raise ValueError(f"Code provenance changed: {current}")
        if len(report["cases"]) != 1:
            raise ValueError(f"Expected one case per source: {path}")
        case = report["cases"][0]
        video = ROOT / case["comparison_video"]
        if sha(video) != case["comparison_video_sha256"]:
            raise ValueError("Comparison video hash mismatch")
        if decode(video) != case["comparison_video_frames"]:
            raise ValueError("Comparison video frame count mismatch")
        for key, hashed in (("present_episode_trace", "present_episode_trace_sha256"),
                            ("absent_episode_trace", "absent_episode_trace_sha256")):
            if sha(ROOT / case[key]) != case[hashed]:
                raise ValueError(f"Episode trace hash mismatch: {key}")
        for prefix in ("present", "absent"):
            movie = (ROOT / case[f"{prefix}_episode_trace"]).with_name("episode.mp4")
            if sha(movie) != case[f"{prefix}_episode_video_sha256"]:
                raise ValueError(f"Episode video hash mismatch: {movie}")
        present_trace = json.loads((ROOT / case["present_episode_trace"]).read_text())
        absent_trace = json.loads((ROOT / case["absent_episode_trace"]).read_text())
        if present_trace["outcome"] != case["present_owner_outcome"]:
            raise ValueError("Present outcome mismatch")
        if absent_trace["outcome"] != case["absent_owner_outcome"]:
            raise ValueError("Absent outcome mismatch")
        if not case["absent_no_look_without_memory"] or "absent_abstained" not in case[
                "absent_owner_outcome"]:
            raise ValueError("Absent control did not abstain")
        branch_times = []
        for branch in case["branches"].values():
            for content, digest in (("rgb", "rgb_sha256"),
                                    ("depth", "depth_sha256")):
                if sha(ROOT / branch[content]) != branch[digest]:
                    raise ValueError(f"View packet hash mismatch: {content}")
            if len(branch["action_samples"]) != 4:
                raise ValueError("Expected four synchronized 0.1 s action samples")
            branch_times.append(branch["capture_time_s"])
        if abs(branch_times[0] - branch_times[1]) > 1e-6 or abs(
                branch_times[0] - case["stop_time_s"] - .4) > 1e-6:
            raise ValueError("The two views were not time matched")
        memory = case["actor_remembered_xy_m"]
        pose = case["stop_estimated_pose_xyyaw"]
        bearing = math.atan2(memory[1] - pose[1], memory[0] - pose[0])
        intended = math.atan2(math.sin(bearing - pose[2]), math.cos(bearing - pose[2]))
        actual = case["branches"]["head_lower"]["requested_head_yaw_rad"]
        if abs(actual - intended) > 1e-4:
            raise ValueError("Head aim was not derived from actor memory")
        cases.append(case)
        sources.append({"report": str(path.relative_to(ROOT)), "sha256": sha(path)})
    if sorted(case["seed"] for case in cases) != [85, 86]:
        raise ValueError("Expected seed 85 and 86")
    result = {"schema": "robot-learning-lab.native-owner-post-stop-views-audit.v1",
              "task": "T-theejd", "audit_code_sha256": sha(Path(__file__)),
              "source_reports": sources,
              "video_and_packet_hashes_verified": True,
              "post_stop_times_and_actor_memory_aim_verified": True,
              "cases": cases}
    path = out / "report.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    for case in cases:
        print(case["seed"], case["branches"]["timed_chest"][
            "true_target_pixel_area_evaluator_only"],
              case["branches"]["head_lower"][
                  "true_target_pixel_area_evaluator_only"],
              bool(case["branches"]["timed_chest"]["actor_fresh_metric_candidate"]),
              bool(case["branches"]["head_lower"]["actor_fresh_metric_candidate"]))


if __name__ == "__main__":
    main()
