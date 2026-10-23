"""Compare one and three clicked teaching views on saved 50 cm search frames.

The extra views rotate a synthetic object in place. They are not a person
showing it, and evaluator IoU is used only for reporting after ranking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

try:
    from .active_scan import TEACH_POSITIONS, Vision
    from .scene import HeadScene, OBJECTS
except ImportError:
    from active_scan import TEACH_POSITIONS, Vision
    from scene import HeadScene, OBJECTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Directory holding a completed active_scan result and frames")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.source / "result.json").read_text())
    if source["models"]["teach_selector"] != "sam2-click":
        raise ValueError("Use a SAM 2 click-teaching source run")
    vision = Vision("sam2-click")
    scene = HeadScene()
    taught = {}
    try:
        for target in OBJECTS:
            episode = next(e for e in source["episodes"] if e["target"] == target)
            click = episode["teach_click_px"]
            views = []
            for heading in [-.55, 0.0, .55]:
                scene.place(TEACH_POSITIONS, {target: heading})
                scene.look(0, .75)
                image = Image.fromarray(scene.image())
                box, info = vision.select_click(image, click, [])
                if box is None:
                    views.append({"heading_rad": heading, "error": info})
                    continue
                feature = vision.feature([image.crop(box)])[0]
                views.append({"heading_rad": heading, "box": box, "selection": info,
                              "evaluator_iou": vision_iou(box, scene.evaluator_box(target)),
                              "feature": feature})
            taught[target] = views
    finally:
        scene.close()

    rows = []
    for episode in source["episodes"]:
        target, absent = episode["target"], episode["target_absent"]
        available = [v for v in taught[target] if "feature" in v]
        feature_sets = {
            "single": [v["feature"] for v in available if v["heading_rad"] == 0],
            "three": [v["feature"] for v in available],
        }
        obs_rows = []
        for observation in episode["observations"]:
            yaw = observation["yaw_rad"]
            image = Image.open(args.source / f"{target}-{'absent' if absent else 'moved'}-yaw{yaw:+.2f}.png")
            candidates = [p for p in observation["proposals"]
                          if not p["border_crop"] and p["floor_xy_m"] is not None]
            if not candidates:
                continue
            features = vision.feature([image.crop(tuple(round(v) for v in p["box"]))
                                       for p in candidates])
            for proposal, feature in zip(candidates, features):
                obs_rows.append({"yaw_rad": yaw,
                                 "evaluator_target_iou": proposal["evaluator_target_iou"],
                                 "single": max(float(feature @ ref) for ref in feature_sets["single"]),
                                 "three": max(float(feature @ ref) for ref in feature_sets["three"])})
        comparison = {}
        for mode in feature_sets:
            ranking = sorted(obs_rows, key=lambda row: row[mode], reverse=True)
            true = [r for r in ranking if r["evaluator_target_iou"] >= .25]
            false = [r for r in ranking if r["evaluator_target_iou"] < .25]
            comparison[mode] = {"top_score": ranking[0][mode] if ranking else None,
                                "top_correct": bool(ranking and ranking[0]["evaluator_target_iou"] >= .25),
                                "best_true": true[0][mode] if true else None,
                                "best_distractor": false[0][mode] if false else None}
        rows.append({"target": target, "target_absent": absent,
                     "candidate_count": len(obs_rows), "comparison": comparison})
    result = {"scope": "synthetic in-place object rotations, same click and saved search frames; "
                       "not a person teaching or a calibrated recognition decision",
              "source": str(args.source), "teaching_views": {
                  target: [{key: val for key, val in view.items() if key != "feature"}
                           for view in views] for target, views in taught.items()},
              "episodes": rows}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        print(row)


def vision_iou(a: list[int], b: list[int] | None) -> float:
    if b is None:
        return 0.0
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return overlap / (area_a + area_b - overlap) if area_a + area_b > overlap else 0.0


if __name__ == "__main__":
    main()
