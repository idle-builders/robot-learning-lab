"""Run the frozen upright RGB-D versus separate range owner comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRECOMMIT = ROOT / "assets/real_indoor_scenes/scansceneverse/rgbd_vs_range_owner_precommit_v2_20261014.json"
OUTPUT = ROOT / "assets/real_indoor_scenes/scansceneverse/rgbd_vs_range_owner_v2_20261014"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_count(path: Path) -> int:
    stream = cv2.VideoCapture(str(path))
    count = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    stream.release()
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precommit", type=Path, default=PRECOMMIT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    precommit = args.precommit.resolve()
    output = args.output.resolve()
    frozen = json.loads(precommit.read_text())
    settings = frozen["shared_settings"]
    scripts = [HERE / "run_real_capture_owner_loop.py",
               HERE / "rgbd_range_projection.py",
               HERE / "real_capture_dynamic_scene.py",
               ROOT / "experiments/ape_high_level/sensors/mounted_camera_geometry.py",
               ROOT / "experiments/ape_high_level/sensors/mounted_owner_rig.py",
               ROOT / "experiments/ape_high_level/sensors/mounted_low_scan.py",
               ROOT / "experiments/ape_high_level/realistic_people/online_owner_actor.py",
               ROOT / "experiments/ape_high_level/realistic_people/fan_view_route.py"]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in scripts}
    rows = []
    output.mkdir(parents=True, exist_ok=True)
    for scene in frozen["development_scenes"]:
        for mode in frozen["paired_conditions"]:
            range_mode = ("mounted_body_rgbd" if mode in
                          ("mounted_body_rgbd_prefix", "mounted_body_rgbd_strict",
                           "mounted_body_rgbd_low_scan")
                          else mode)
            folder = output / scene / mode
            folder.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(HERE / "run_real_capture_owner_loop.py"),
                       "--scene", scene, "--range-mode", range_mode,
                       "--person-style", settings["person_style"],
                       "--pose-mode", settings["pose_mode"],
                       "--steps", str(settings["steps_max"]), "--device", "mps",
                       "--simulated-inference-delay-s",
                       str(settings["simulated_inference_delay_s"]),
                       "--output-dir", str(folder)]
            if mode == "mounted_body_rgbd_strict":
                command.append("--no-camera-free-prefix")
            if mode == "mounted_body_rgbd_low_scan":
                command.append("--mounted-low-scan")
            if frozen.get("mount_screen"):
                command.extend(("--mount-screen", str(ROOT / frozen["mount_screen"])))
            for key, flag in (("gaze_route_arbitration", "--gaze-route-arbitration"),
                              ("pose_crosscheck", "--pose-crosscheck"),
                              ("person_masked_odometry", "--person-mask-odometry"),
                              ("spatial_range_association", "--spatial-range-association"),
                              ("far_conversation_viewpoint", "--far-conversation-viewpoint"),
                              ("depth_separated_overlap", "--depth-separated-overlap")):
                if settings.get(key):
                    command.append(flag)
            print(f"run {scene} {mode}", flush=True)
            run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                 timeout=300)
            if run.returncode:
                row = {"scene": scene, "mode": mode, "status": "failed",
                       "stderr_tail": run.stderr[-2000:]}
            else:
                report = json.loads((folder / "trace.json").read_text())
                frames = report["frames"]
                aligned = (frames == len(report["trace"]) ==
                           frame_count(folder / "first_person.mp4") ==
                           len(list(folder.glob("rgb_*.png"))) ==
                           len(list(folder.glob("depth_*.png"))))
                if range_mode in ("dual_rgbd_projection", "mounted_body_rgbd"):
                    aligned = (aligned and frames ==
                               frame_count(folder / "downward_view.mp4") ==
                               len(list(folder.glob("down_rgb_*.png"))) ==
                               len(list(folder.glob("down_depth_*.png"))))
                if not aligned:
                    raise RuntimeError(f"Unaligned media for {scene} {mode}")
                trace = report["trace"]
                row = {"scene": scene, "mode": mode, "status": "completed",
                       "outcome": report["outcome"],
                       "stop_reason": trace[-1]["planner_decision"].get("reason"),
                       "viewpoint_reached": bool(report["observed_standoff_band_met"] or
                                                 report["observed_far_conversation_band_met"]),
                       "frames": frames,
                       "accepted_travel_m": report["accepted_travel_m"],
                       "max_pose_error_m": report["max_pose_error_m"],
                       "rejected_steps": sum(not f["command_accepted"] for f in trace),
                       "mean_known_range_beams": sum(f["known_range_beam_count"]
                                                      for f in trace) / frames,
                       "mean_unknown_range_beams": sum(f["unknown_range_beam_count"]
                                                        for f in trace) / frames,
                       "mean_known_low_beams": sum(f["known_low_beam_count"]
                                                   for f in trace) / frames,
                       "mean_known_mid_beams": sum(f["known_mid_beam_count"]
                                                   for f in trace) / frames,
                       "min_evaluator_forward_range_at_action_m": min(
                           f["evaluator_nearest_forward_range_at_action_m"]
                           for f in trace),
                       "video": str((folder / "first_person.mp4").relative_to(ROOT)),
                       "downward_video": (str((folder / "downward_view.mp4").relative_to(ROOT))
                                          if range_mode in ("dual_rgbd_projection", "mounted_body_rgbd") else None),
                       "trace": str((folder / "trace.json").relative_to(ROOT))}
            rows.append(row)
            (output / "summary.json").write_text(json.dumps({
                "schema": "ape.paired-owner-range-result.v1",
                "precommit": str(precommit.relative_to(ROOT)),
                "precommit_sha256": sha256(precommit), "runner_hashes": hashes,
                "note": "All four scenes are exposed development rooms; no camera/range hardware selected.",
                "rows": rows}, indent=2) + "\n")
            print(row["status"], row.get("outcome", ""),
                  row.get("viewpoint_reached", ""), flush=True)
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
