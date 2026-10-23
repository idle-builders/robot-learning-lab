"""Write the measured Berkeley button teacher rollouts with LeRobot's dataset writer."""

import argparse
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frames(path):
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    source = json.loads((args.source / "manifest.json").read_text())
    state_dim = source.get("state_dim", 19)
    if state_dim not in (19, 25):
        raise ValueError(f"Unexpected button observation width: {state_dim}")
    successful = [attempt for attempt in source["attempts"] if attempt["success"]]
    if len(successful) != source["successful_episodes"]:
        raise ValueError("Teacher manifest episode count does not match")
    if args.destination.exists():
        raise FileExistsError(args.destination)
    dataset = LeRobotDataset.create(
        repo_id="local/berkeley-lite-button-teacher",
        fps=source["fps"],
        root=args.destination,
        robot_type="berkeley_humanoid_lite_v1",
        use_videos=True,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", g=12, crf=23, preset="medium"),
        features={
            "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": None},
            "action": {"dtype": "float32", "shape": (8,), "names": None},
            "observation.images.external_view": {
                "dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]
            },
        },
    )
    written = 0
    for attempt in successful:
        stem = attempt["episode"]
        arrays_path = args.source / f"{stem}.npz"
        video_path = args.source / f"{stem}.mp4"
        if sha256(arrays_path) != attempt["state_action_sha256"] or sha256(video_path) != attempt["video_sha256"]:
            raise ValueError(f"Source hash mismatch: {stem}")
        with np.load(arrays_path) as arrays:
            state = arrays["state"]
            action = arrays["action"]
        count = 0
        for image in frames(video_path):
            dataset.add_frame({
                "observation.state": state[count],
                "action": action[count],
                "observation.images.external_view": image,
                "task": source["task"],
            })
            count += 1
        if count != len(state) or count != len(action):
            raise ValueError(f"Frame/state/action count mismatch: {stem}")
        dataset.save_episode()
        written += 1
        print(json.dumps({"episode": stem, "frames": count}), flush=True)
    print(json.dumps({"episodes": written, "frames": dataset.meta.total_frames, "destination": str(args.destination)}))


if __name__ == "__main__":
    main()
