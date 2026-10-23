"""Replay short-horizon RGB-D flow on saved Isaac views; score truth separately."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from audit_social_association import projected_x
from fast_owner_flow import FastOwnerFlow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--seed-frame", type=int, default=1)
    args = parser.parse_args()
    evidence = args.run / "evidence"
    inputs = json.loads((args.run / "inputs.json").read_text())
    fov = inputs.get("control_fov_deg", 90)
    action_path = evidence / "online-action-trace.json"
    actions = json.loads(action_path.read_text())["actions"] if action_path.exists() else None
    people = json.loads((evidence / "character-motion-trace.json").read_text())
    frozen_analysis = None
    if actions is None:
        analysis = json.loads((evidence / "owner-video-analysis.json").read_text())
        frozen_analysis = next(view["trace"] for view in analysis["views"]
                               if view["fov_deg"] == fov)
    depth = np.load(evidence / f"ape_social_{fov}deg-depth-sequence-stride3.npz")["depth_m"]
    capture = cv2.VideoCapture(str(evidence / f"ape_social_{fov}deg.mp4"))
    frames = []
    for index in range(len(people)):
        okay, bgr = capture.read()
        if not okay:
            raise RuntimeError(f"Video ended before saved view {index}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    assert len(people) == len(depth) == len(frames)
    if actions is not None:
        assert len(actions) == len(people)
        source = actions[args.seed_frame]
        decision = source["actor"]
        selected = decision.get("chosen_proposal")
        if selected is None or decision.get("local_depth_m") is None:
            raise RuntimeError("Seed frame has no detector-confirmed depth-supported person")
        seed_box = decision["proposals"][selected]["box_xyxy"]
        seed_depth = decision["local_depth_m"]
        seed_time = source["observation_time_s"]
    else:
        assert frozen_analysis is not None and len(frozen_analysis) == len(people)
        source = frozen_analysis[args.seed_frame]
        selected = source["chosen_proposal"]
        if selected is None:
            raise RuntimeError("Frozen seed has no chosen person")
        seed_box = source["boxes_xyxy"][selected]
        seed_depth = source["depth_health"]["candidate_support"][selected]["torso_patch_depth_m"]
        seed_time = people[args.seed_frame]["timeline_time_s"]
    tracker = FastOwnerFlow()
    seed = tracker.seed(frames[args.seed_frame], depth[args.seed_frame],
                        seed_box, seed_depth, time_s=seed_time, fov_deg=fov)
    rows = [{"frame": args.seed_frame, "track": seed,
             "source": "saved_actor_seed"}]
    for index in range(args.seed_frame + 1, len(people)):
        before = time.perf_counter()
        capture_time = (actions[index]["observation_time_s"] if actions else
                        people[index]["timeline_time_s"])
        yaw_delta = (actions[index - 1]["command_for_next_frame"]["yaw_delta_deg"]
                     if actions else 0.0)
        result = tracker.update(frames[index], depth[index],
            time_s=capture_time, fov_deg=fov, ego_yaw_delta_deg=yaw_delta)
        result["wall_s"] = time.perf_counter() - before
        row = {"frame": index, "track": result}
        if result["status"] == "tracked":
            if actions:
                truth = people[index]["characters"]
                eye = actions[index - 1]["proxy_eye_xyz_m_evaluator_only"]
                heading = actions[index - 1]["proxy_heading_deg_evaluator_only"]
                owner_x = projected_x(truth[0]["xyz_m"], eye, heading, fov)
                bystander_x = projected_x(truth[1]["xyz_m"], eye, heading, fov)
            else:
                owner_x, bystander_x = frozen_analysis[index][
                    "projected_character_x_evaluator_px"]
            x0, _, x1, _ = result["box_xyxy"]
            center = (x0 + x1) / 2
            label = "unscored"
            if owner_x is not None and bystander_x is not None:
                if abs(owner_x - bystander_x) < 80:
                    label = "projection_overlap"
                elif x0 <= owner_x <= x1 and x0 <= bystander_x <= x1:
                    label = "mixed_box"
                elif abs(center - owner_x) + 45 < abs(center - bystander_x):
                    label = "owner"
                elif abs(center - bystander_x) + 45 < abs(center - owner_x):
                    label = "bystander"
                else:
                    label = "ambiguous_projection"
            row["association_evaluator_only"] = label
            row["owner_root_x_px_evaluator_only"] = owner_x
            row["bystander_root_x_px_evaluator_only"] = bystander_x
        rows.append(row)
    counts = {key: sum(row.get("association_evaluator_only") == key for row in rows)
              for key in ("owner", "bystander", "mixed_box", "projection_overlap",
                          "ambiguous_projection", "unscored")}
    report = {"scope": "offline RGB-D flow from a saved detector seed; no control feedback; "
        "character roots enter evaluator only", "seed_frame": args.seed_frame,
        "source_camera_path": "online_control" if actions else "frozen_camera",
        "seed": seed, "frames": len(rows), "counts": counts,
        "uncertain": sum(row["track"]["status"] == "uncertain" for row in rows),
        "rows": rows}
    out = args.run / "fast-flow-replay.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
        ("seed_frame", "frames", "counts", "uncertain")}, indent=2))


if __name__ == "__main__":
    main()
