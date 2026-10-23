"""Audit partial camera-observed body-origin free rays in four captured rooms."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from mounted_camera_geometry import observed_free_prefix_body_fan


ROOT = Path(__file__).resolve().parents[3]
PLAN = ROOT / "artifacts/ape-sensors-50cm/captured-room-mounted-dual-v1/free_prefix_plan.json"
OUT = PLAN.with_name("free_prefix_report.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def depth(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path), dtype=np.float32) / 1000.
    image[image <= 0.] = np.nan
    return image


def main() -> None:
    plan = json.loads(PLAN.read_text())
    screen_path = ROOT / plan["input_screen"]
    screen = json.loads(screen_path.read_text())
    cameras = screen["camera_contract"]
    angles = np.arange(-90., 91., 5., dtype=np.float32)
    rows = []
    for scene_id in plan["development_scenes"]:
        selected = [row for row in screen["rows"] if row["scene_id"] == scene_id]
        if len(selected) != 1:
            raise ValueError(f"Expected exactly one mounted screen row: {scene_id}")
        row = selected[0]
        upper_path = ROOT / row["upper_depth"]
        lower_path = ROOT / row["lower_depth"]
        views = (
            (depth(upper_path), tuple(cameras["upper"]["position_m"]),
             float(cameras["upper"]["pitch_up_deg"])),
            (depth(lower_path), tuple(cameras["lower"]["position_m"]),
             -float(cameras["lower"]["pitch_down_deg"])),
        )
        prefix, reason = observed_free_prefix_body_fan(
            views, angles, vfov_deg=float(cameras["vfov_deg"]))
        strict = np.asarray(row["body_frame_free_mask_by_height"], dtype=bool)
        if strict.shape != prefix.shape:
            raise ValueError(f"Different strict/prefix fan shape: {scene_id}")
        if not np.array_equal(strict, prefix >= 2. - 1e-6):
            raise ValueError(f"Partial-prefix and strict full-clear disagree: {scene_id}")
        planes = []
        for index, height in enumerate(plan["geometry"]["body_origin_heights_m"]):
            values = prefix[index]
            positive = values[values > 0]
            planes.append({
                "height_m": height,
                "strict_full_clear_beams": int(strict[index].sum()),
                "prefix_ge_0p5m_beams": int(np.count_nonzero(values >= .5)),
                "prefix_ge_1p0m_beams": int(np.count_nonzero(values >= 1.)),
                "prefix_ge_2p0m_beams": int(np.count_nonzero(values >= 2. - 1e-6)),
                "nonzero_prefix_beams": int(len(positive)),
                "nonzero_prefix_median_m": (float(np.median(positive))
                                              if len(positive) else None),
                "nonzero_prefix_min_m": (float(np.min(positive))
                                           if len(positive) else None),
                "first_loss_reason_counts": dict(Counter(reason[index].tolist())),
                "prefix_by_angle_m": [round(float(v), 3) for v in values],
                "first_loss_by_angle": reason[index].tolist(),
            })
        rows.append({"scene_id": scene_id,
                     "upper_depth_sha256": digest(upper_path),
                     "lower_depth_sha256": digest(lower_path),
                     "planes": planes})
        print(scene_id, [(p["height_m"], p["strict_full_clear_beams"],
                          p["prefix_ge_0p5m_beams"], p["prefix_ge_1p0m_beams"])
                         for p in planes], flush=True)
    OUT.write_text(json.dumps({
        "schema": "ape.mounted-camera-free-prefix-result.v1",
        "plan": str(PLAN.relative_to(ROOT)),
        "plan_sha256": digest(PLAN),
        "screen": str(screen_path.relative_to(ROOT)),
        "screen_sha256": digest(screen_path),
        "source_hashes": {
            str(Path(__file__).relative_to(ROOT)): digest(Path(__file__)),
            "experiments/ape_high_level/sensors/mounted_camera_geometry.py":
                digest(ROOT / "experiments/ape_high_level/sensors/mounted_camera_geometry.py"),
        },
        "rows": rows,
    }, indent=2) + "\n")
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
