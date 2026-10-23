"""Make an inspection gallery from pose-near, real VGA photographs.

The gallery is an appearance audit, not a simulated robot-camera render.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "assets/real_indoor_scenes/arkit_v1"
SPLIT = Path(__file__).with_name("scene_split.json")


def image_score(path: Path) -> float:
    rgb = cv2.imread(str(path))
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
    hist = np.bincount(small.ravel(), minlength=256).astype(float)
    hist /= hist.sum()
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    texture = np.mean(np.abs(cv2.Laplacian(small, cv2.CV_32F)))
    return float(entropy + 0.02 * min(texture, 60))


def turn_upright(photo: Image.Image, trajectory: np.ndarray, timestamp: float) -> Image.Image:
    pose = trajectory[np.argmin(np.abs(trajectory[:, 0] - timestamp))]
    up_in_camera = Rotation.from_rotvec(pose[1:4]).as_matrix() @ np.array([0., 0., 1.])
    x, y = up_in_camera[:2]
    if abs(x) >= abs(y):
        return photo.transpose(Image.Transpose.ROTATE_270 if x < 0 else Image.Transpose.ROTATE_90)
    return photo.transpose(Image.Transpose.ROTATE_180) if y > 0 else photo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--prefix", default="route_screen_recorded_gallery")
    args = parser.parse_args()
    scenes = [scene for scene in json.loads(args.split.read_text())["scenes"]
              if scene["tier"] in ("core", "narrow_stress")]
    font = ImageFont.load_default()
    tile_w, tile_h = 320, 270
    for page, start in enumerate(range(0, len(scenes), 7), 1):
        canvas = Image.new("RGB", (tile_w * 2, tile_h * 7), "#202020")
        draw = ImageDraw.Draw(canvas)
        for row, scene in enumerate(scenes[start : start + 7]):
            video_id = scene["video_id"]
            folder = DATA / video_id / "frames"
            ranked = sorted(
                ((image_score(folder / f"{index:02d}" / "rgb_vga.png"), index) for index in range(12)),
                reverse=True,
            )
            chosen = [ranked[0][1]]
            chosen.append(next(index for _, index in ranked[1:] if abs(index - chosen[0]) >= 2))
            scene_record = json.loads((DATA / video_id / "scene.json").read_text())
            labels = ", ".join(list(scene_record["annotation_labels"])[:4])
            trajectory = np.loadtxt(DATA / video_id / "camera.traj")
            for column, index in enumerate(chosen):
                photo = Image.open(folder / f"{index:02d}" / "rgb_vga.png").convert("RGB")
                photo = turn_upright(photo, trajectory, scene_record["selected_frames"][index]["timestamp_s"])
                photo.thumbnail((tile_w - 8, tile_h - 29))
                x = column * tile_w + (tile_w - photo.width) // 2
                y = row * tile_h + 24
                canvas.paste(photo, (x, y))
                fold = scene.get("agent_fold", scene.get("fold", ""))
                draw.text((column * tile_w + 5, row * tile_h + 5),
                          f"{video_id} | {scene['tier']} | {fold} | {index} | {labels}",
                          fill="white", font=font)
        output = DATA / f"{args.prefix}_{page}.jpg"
        canvas.save(output, quality=88)
        print(output)


if __name__ == "__main__":
    main()
