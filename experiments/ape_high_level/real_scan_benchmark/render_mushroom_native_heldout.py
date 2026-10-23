"""Render withheld MuSHRoom views with the training model's own gsplat rasterizer.

Use on temporary CUDA compute. This bypasses an upstream DN-Splatter aggregate
evaluation bug when only one MuSHRoom evaluation split is populated, while
keeping the trained checkpoint and camera intrinsics unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from nerfstudio.utils.eval_utils import eval_setup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame", action="append", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def choose_within(config):
        config.pipeline.datamanager.dataparser.eval_mode = "within"
        return config

    _, pipeline, checkpoint, _ = eval_setup(
        args.config, test_mode="test", update_config_callback=choose_within
    )
    dataset = pipeline.datamanager.eval_dataset
    wanted = set(args.frame)
    results = []
    for index, filename in enumerate(dataset.image_filenames):
        if filename.stem not in wanted:
            continue
        camera = dataset.cameras[index : index + 1].to("cpu")
        output = pipeline.model.get_outputs_for_camera(camera=camera)
        rgb = output["rgb"].detach().cpu().numpy()
        predicted = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
        real = np.asarray(Image.open(filename).convert("RGB"))
        if predicted.shape != real.shape:
            raise ValueError(f"Predicted/recorded size mismatch for {filename}: {predicted.shape}, {real.shape}")
        path = args.output / f"{filename.stem}_native.png"
        Image.fromarray(predicted).save(path)
        Image.fromarray(real).save(args.output / f"{filename.stem}_recorded.png")
        error = predicted.astype(np.float32) - real.astype(np.float32)
        mse = float(np.mean(error * error))
        results.append({"frame": filename.stem, "render": str(path),
                        "recorded": str(filename),
                        "psnr_db": float(10 * np.log10(255**2 / max(mse, 1e-12))),
                        "mean_abs_rgb_255": float(np.mean(np.abs(error)))})
    missing = wanted - {r["frame"] for r in results}
    if missing:
        raise ValueError(f"Frames not in withheld split: {sorted(missing)}")
    report = {"schema": "ape.mushroom-native-heldout.v1", "checkpoint": str(checkpoint),
              "evaluation_protocol": "within (withheld long-capture frames)", "views": results}
    (args.output / "native_heldout.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
