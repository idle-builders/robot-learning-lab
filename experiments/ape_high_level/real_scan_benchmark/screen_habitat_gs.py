"""Preview ungated real Habitat-GS reconstructions at 50 cm ape camera height.

Previewing a splat does not qualify an indoor scene, camera-height coverage,
collision mesh, or the owner task. Keep only reviewed indoor IDs in a later
frozen benchmark selection.
"""

import argparse
import gzip
import json
import math
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "assets/real_indoor_scenes/habitat_gs"
REPO = "RukawaY/gs_scenes"


def preview(index: int) -> dict:
    split = "train" if index <= 55 else "val"
    scene = f"scene{index:02d}"
    destination = ASSETS / split / scene
    destination.mkdir(parents=True, exist_ok=True)
    gs_path = Path(hf_hub_download(repo_id=REPO, repo_type="dataset",
        filename=f"{split}/{scene}/{scene}.gs.ply", local_dir=ASSETS))
    episode_path = Path(hf_hub_download(repo_id=REPO, repo_type="dataset",
        filename=f"episodes/pointnav/{split}/content/{scene}.json.gz",
        local_dir=ASSETS))
    episodes = json.load(gzip.open(episode_path, "rt"))["episodes"]
    selected = [episodes[i] for i in sorted({0, len(episodes) // 3,
                                              2 * len(episodes) // 3})]
    splats = o3d.t.io.read_point_cloud(str(gs_path))
    logits = splats.point.opacity.numpy()
    splats.point.opacity = o3d.core.Tensor(
        (1 / (1 + np.exp(-np.clip(logits, -30, 30)))).astype(np.float32))
    renderer = o3d.visualization.rendering.OffscreenRenderer(480, 360)
    renderer.scene.set_background([0, 0, 0, 1])
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "gaussianSplat"
    material.gaussian_splat_sh_degree = 0
    renderer.scene.add_geometry("real_capture", splats, material)
    page = Image.new("RGB", (480 * len(selected), 385), "#222222")
    draw = ImageDraw.Draw(page)
    sample_rows = []
    for slot, episode in enumerate(selected):
        start = np.asarray(episode["start_position"], dtype=float)
        goal = np.asarray(episode["goals"][0]["position"], dtype=float)
        eye = start + np.array([0., .42, 0.])
        direction = goal - start
        direction[1] = 0.
        if np.linalg.norm(direction) < .01:
            direction = np.array([0., 0., -1.])
        else:
            direction /= np.linalg.norm(direction)
        renderer.setup_camera(75., eye + direction, eye,
                              np.array([0., 1., 0.]))
        rgb = Image.fromarray(np.asarray(renderer.render_to_image())[:, :, :3])
        rgb.save(destination / f"preview_{slot}.jpg", quality=90)
        page.paste(rgb, (480 * slot, 25))
        draw.text((480 * slot + 7, 7),
                  f"{scene} | ape-height 0.42 m | sample {slot}", fill="white")
        sample_rows.append({"episode_id": episode["episode_id"],
                            "start_position_m": start.tolist(),
                            "goal_position_m": goal.tolist(),
                            "eye_position_m": eye.tolist()})
    page.save(destination / "preview_triptych.jpg", quality=90)
    record = {"schema": "ape.habitat-gs-ape-height-screen.v1",
              "source": f"https://huggingface.co/datasets/{REPO}",
              "source_commit": HfApi().repo_info(REPO, repo_type="dataset").sha,
              "scene_id": scene, "source_split": split,
              "gaussian_bytes": gs_path.stat().st_size,
              "episode_count": len(episodes), "samples": sample_rows,
              "status": "visual_screen_pending_not_qualified"}
    (destination / "screen.json").write_text(json.dumps(record, indent=2) + "\n")
    print(destination / "preview_triptych.jpg", flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-number", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.scene_number <= 65:
        parser.error("scene number must be 1–65")
    preview(args.scene_number)


if __name__ == "__main__":
    main()
