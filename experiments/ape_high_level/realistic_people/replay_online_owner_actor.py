"""Replay the streaming actor on retained Isaac packets as a regression control.

This checks observation-to-decision behavior without claiming action feedback;
the joined Isaac run is the separate test for that feedback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from online_owner_actor import OnlineOwnerActor


ROOT = Path(__file__).resolve().parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--fov", type=int, default=90)
    parser.add_argument("--click-x", type=int, required=True)
    parser.add_argument("--click-y", type=int, required=True)
    parser.add_argument("--person-detector", choices=("grounding_dino", "ssdlite"),
                        default="grounding_dino")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    video = evidence / f"ape_social_{args.fov}deg.mp4"
    depth_file = evidence / f"ape_social_{args.fov}deg-depth-sequence-stride3.npz"
    depth = np.load(depth_file)["depth_m"]
    motion = json.loads((evidence / "character-motion-trace.json").read_text())
    range_file = evidence / "range-fan-trace.json"
    range_frames = (json.loads(range_file.read_text())["frames"]
                    if range_file.exists() else None)
    action_file = evidence / "online-action-trace.json"
    executed_actions = (json.loads(action_file.read_text())["actions"]
                        if action_file.exists() else None)
    detector = ROOT / ".cache/ape-high-level/models/grounding-dino-tiny"
    features = ROOT / ".cache/ape-high-level/models/dinov2-small"
    import torch
    actor = OnlineOwnerActor(detector, features,
                             click_xy=(args.click_x, args.click_y),
                             device=("mps" if torch.backends.mps.is_available() and
                                     args.person_detector != "ssdlite" else "cpu"),
                             person_detector=args.person_detector)
    cap = cv2.VideoCapture(str(video))
    rows = []
    angles = np.arange(-90, 91, 5, dtype=np.float32)
    for index in range(len(depth)):
        okay, bgr = cap.read()
        if not okay:
            raise RuntimeError(f"Video ended before frame {index}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ranges = np.full((2, len(angles)), np.nan, dtype=np.float32)
        if range_frames is not None:
            range_row = range_frames[index]
            if range_row["frame"] != index or abs(
                    range_row["timeline_time_s"] - motion[index]["timeline_time_s"]) > 1e-6:
                raise RuntimeError("Range/motion time mismatch")
            ranges = np.array([[beam["distance_m"] if beam["distance_m"] is not None
                                else np.nan for beam in plane["beams"]]
                               for plane in range_row["planes"]], dtype=np.float32)
        rows.append({"frame": index, **actor.observe(
            rgb, depth[index], ranges, angles,
            time_s=motion[index]["timeline_time_s"], fov_deg=args.fov,
            ego_yaw_delta_deg=(executed_actions[index-1]["command_for_next_frame"]["yaw_delta_deg"]
                               if executed_actions is not None and index > 0 else 0.0))})
    cap.release()
    result = {"scope": "streaming actor decisions on saved observations; no action feedback",
              "fov_deg": args.fov, "click_xy": [args.click_x, args.click_y],
              "person_detector": args.person_detector,
              "source_sha256": {"video": sha(video), "depth": sha(depth_file),
                                "actor": sha(Path(__file__).with_name("online_owner_actor.py")),
                                "detector_weights": sha(detector / "model.safetensors")
                                    if args.person_detector == "grounding_dino" else sha(
                                        Path(torch.hub.get_dir()) / "checkpoints" /
                                        "ssdlite320_mobilenet_v3_large_coco-a79551df.pth"),
                                "feature_weights": sha(features / "model.safetensors")},
              "trace": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"intents": [row["intent"] for row in rows],
                      "reasons": [row["reason"] for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
