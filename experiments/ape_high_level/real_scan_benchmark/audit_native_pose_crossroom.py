"""Score the frozen three-room pose-sensor comparison at capture time."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_native_pose_fallback import decoded_frames, sha


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/ape-real-room-next-captures-20261014/native_pose_crossroom_owner_screen"
PLAN = ROOT / "artifacts/ape-real-room-next-captures-20261014/native_pose_crossroom_precommit.json"
OUTPUT = ROOT / "artifacts/ape-real-room-next-captures-20261014/native_pose_crossroom_report.json"


def score(entry: dict) -> dict:
    trace_path = ROOT / entry["trace"]
    report = json.loads(trace_path.read_text())
    frames = report["trace"]
    source_route = json.loads((ROOT / report["source_capture"]).read_text())
    start_xz = source_route.get("start_xz")
    if start_xz is None:
        eye = source_route["frames"][0]["eye_m"]
        start_xz = [eye[0], eye[2]]
    start = np.asarray(start_xz, dtype=float)
    previous_actual = start.copy()
    errors = []
    for frame in frames:
        actual = np.asarray(frame.get("evaluator_actual_xz_at_capture_m",
                                      previous_actual), dtype=float)
        errors.append(float(np.linalg.norm(
            np.asarray(frame["actor_pose_xy_m"]) - (actual - start))))
        previous_actual = np.asarray(frame["evaluator_actual_xz_m"])
    video = ROOT / entry["video"]
    count = decoded_frames(video)
    if count != len(frames) or count != len(list(trace_path.parent.glob("rgb_*.png"))) or \
            count != len(list(trace_path.parent.glob("depth_*.png"))):
        raise RuntimeError(f"Misaligned media: {entry['trace']}")
    final = frames[-1]
    near = report["observed_standoff_band_met"]
    far = report.get("observed_far_conversation_band_met", False)
    range_m = final["decision_owner_actor"].get("nearest_forward_range_m")
    qualified_view = bool((near or (far and range_m is not None and range_m >= .9))
                          and max(errors) <= .08)
    fallback_frames = [f["frame"] for f in frames if
                       (f.get("pose_evidence") or {}).get("used") ==
                       "bounded_native_prediction"]
    return {"name": entry["name"], "scene_id": entry["scene_id"],
            "pose_mode": report["pose_mode"],
            "native_source": report.get("native_pose_source"),
            "outcome": report["outcome"],
            "stop_reason": final["planner_decision"].get("reason"),
            "frames": count, "accepted_travel_m": report["accepted_travel_m"],
            "max_pose_error_at_capture_m": max(errors),
            "final_pose_error_at_capture_m": errors[-1],
            "reported_max_post_action_pose_error_m": report["max_pose_error_m"],
            "observed_near_viewpoint": near,
            "observed_far_viewpoint": far,
            "final_owner_depth_m": report["final_observed_owner_depth_m"],
            "final_forward_range_m": range_m,
            "native_fallback_frames": fallback_frames,
            "qualified_view_under_precommit": qualified_view,
            "video": entry["video"], "video_sha256": sha(video),
            "trace": entry["trace"], "trace_sha256": sha(trace_path)}


def main() -> None:
    plan = json.loads(PLAN.read_text())
    source = plan["native_source"]
    batches = {
        "rgbd_yaw": BASE / "owner_batch_rgbd_yaw_arbitrated_crosschecked_personmasked_rangeprojected_farconverse_v1.json",
        "native_shaped": BASE / f"owner_batch_native_shaped_arbitrated_rangeprojected_farconverse_{source}_v1.json",
        "rgbd_native_fallback": BASE / f"owner_batch_rgbd_native_fallback_arbitrated_crosschecked_personmasked_rangeprojected_farconverse_{source}_v1.json",
    }
    rows = []
    for mode, path in batches.items():
        batch = json.loads(path.read_text())
        if len(batch["scenes"]) != len(plan["selected_rooms"]):
            raise RuntimeError(f"Incomplete batch: {path}")
        for entry in batch["scenes"]:
            if entry["task_status"] != "completed":
                raise RuntimeError(f"Uncompleted run: {entry}")
            row = score(entry)
            if row["pose_mode"] != mode:
                raise RuntimeError(f"Wrong pose mode: {entry}")
            rows.append(row)
    if {r["name"] for r in rows} != set(plan["selected_rooms"]):
        raise RuntimeError("Room selection changed")
    output = {"schema": "ape.real-capture-native-pose-crossroom-audit.v1",
              "precommit": str(PLAN.relative_to(ROOT)),
              "precommit_sha256": sha(PLAN),
              "scope": "Three development real-capture rooms and one historical rejected-body error trace; captured RGB-D, scanned person and kinematic ape. No selected-body or physical sensor qualification.",
              "capture_error_definition": "Actor estimate at image capture versus simulator position at the same capture; truth is evaluator-only.",
              "post_action_error_warning": "Legacy max_pose_error_m scores after the current command and includes one-step latency.",
              "batch_sha256": {mode: sha(path) for mode, path in batches.items()},
              "results": rows,
              "qualified_view_counts": {mode: sum(r["qualified_view_under_precommit"]
                                                  for r in rows if r["pose_mode"] == mode)
                                        for mode in batches}}
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["qualified_view_counts"]))


if __name__ == "__main__":
    main()
