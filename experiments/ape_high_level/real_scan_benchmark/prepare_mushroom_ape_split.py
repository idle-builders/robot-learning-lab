"""Reserve low-camera views in addition to MuSHRoom's published test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3] / "assets/real_indoor_scenes/mushroom"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", required=True)
    parser.add_argument("--extra", nargs="+", required=True)
    args = parser.parse_args()
    base = ROOT / args.room / "extracted" / "room_datasets" / args.room / "iphone/long_capture"
    source = (base / "test.txt").read_text().splitlines()
    frames = json.loads((base / "transformations_colmap.json").read_text())["frames"]
    available = {Path(f["file_path"]).stem for f in frames}
    if not set(args.extra).issubset(available):
        raise ValueError("Extra held-out frame absent from capture")
    selected = sorted(set(source) | set(args.extra))
    output = ROOT / args.room / "ape_heldout_test.txt"
    output.write_text("\n".join(selected) + "\n")
    (ROOT / args.room / "ape_heldout_test.json").write_text(json.dumps({
        "source_split": str(base / "test.txt"), "original_test_frames": len(source),
        "extra_low_camera_frames": args.extra, "final_test_frames": len(selected),
        "train_frames_remaining": len(frames) - len(selected), "output": str(output),
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
