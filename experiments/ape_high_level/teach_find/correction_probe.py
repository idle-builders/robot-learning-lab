"""Replay an explicit owner correction against later saved teach/find views.

The correction is a click on a rendered RGB search frame, selected by a human
looking at that frame. The selected proposal becomes an additional appearance
example. Evaluator IoU is read only after every ranking has been computed.
This probes reuse of a correction, not speech, gesture inference or a robot run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

try:
    from .active_scan import Vision, _iou
except ImportError:
    from active_scan import Vision, _iou


def filename(target: str, absent: bool, yaw: float) -> str:
    return f"{target}-{'absent' if absent else 'moved'}-yaw{yaw:+.2f}.png"


def ranked(rows: list[dict], key: str) -> dict:
    order = sorted(rows, key=lambda row: row[key], reverse=True)
    true = [row for row in order if row["evaluator_target_iou"] >= .25]
    false = [row for row in order if row["evaluator_target_iou"] < .25]
    return {"top_score": order[0][key] if order else None,
            "top_correct": bool(order and order[0]["evaluator_target_iou"] >= .25),
            "best_true": true[0][key] if true else None,
            "best_distractor": false[0][key] if false else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correction", type=Path, required=True,
                        help="JSON with source, target, yaw_rad and click_px")
    parser.add_argument("--evaluate", type=Path, nargs="+", required=True,
                        help="Saved active_scan directories; correction source must be excluded")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    correction = json.loads(args.correction.read_text())
    source_dir = Path(correction["source"])
    target = correction["target"]
    yaw = float(correction["yaw_rad"])
    click = correction["click_px"]
    if any(path.resolve() == source_dir.resolve() for path in args.evaluate):
        raise ValueError("Correction source must not be included in evaluation")
    source = json.loads((source_dir / "result.json").read_text())
    episode = next(e for e in source["episodes"] if e["target"] == target
                   and not e["target_absent"])
    observation = next(o for o in episode["observations"]
                       if abs(o["yaw_rad"] - yaw) < 1e-6)
    image = Image.open(source_dir / filename(target, False, yaw)).convert("RGB")
    vision = Vision("generic-proposal")
    box, selection = vision.select_click(image, click, observation["proposals"])
    if box is None:
        raise ValueError(f"Correction click did not select a proposal: {selection}")
    corrected_feature = vision.feature([image.crop(box)])[0]

    results = []
    for directory in args.evaluate:
        record = json.loads((directory / "result.json").read_text())
        for heldout in record["episodes"]:
            if heldout["target"] != target:
                continue
            rows = []
            for view in heldout["observations"]:
                candidates = [p for p in view["proposals"]
                              if not p["border_crop"] and p["floor_xy_m"] is not None]
                if not candidates:
                    continue
                frame = Image.open(directory / filename(target, heldout["target_absent"],
                                                        view["yaw_rad"])).convert("RGB")
                crops = [frame.crop(tuple(round(v) for v in p["box"]))
                         for p in candidates]
                features = vision.feature(crops)
                for proposal, feature in zip(candidates, features):
                    new_score = float(feature @ corrected_feature)
                    rows.append({"yaw_rad": view["yaw_rad"], "box": proposal["box"],
                                 "original": proposal["similarity"],
                                 "correction_only": new_score,
                                 "max_of_both": max(proposal["similarity"], new_score),
                                 "evaluator_target_iou": proposal["evaluator_target_iou"]})
            results.append({"source": str(directory), "seed": heldout["seed"],
                            "target_absent": heldout["target_absent"],
                            "candidate_count": len(rows),
                            "ranking": {mode: ranked(rows, mode) for mode in
                                        ("original", "correction_only", "max_of_both")}})

    selected = max(observation["proposals"], key=lambda p: _iou(p["box"], box))
    correction_iou = selected["evaluator_target_iou"]
    result = {"scope": "Human-selected click on one synthetic search RGB frame; saved "
                       "candidate proposals from other synthetic scenes; no voice, moving base, "
                       "real owner or calibrated accept threshold.",
              "correction": {"source": str(source_dir), "target": target,
                             "yaw_rad": yaw, "click_px": click, "selected_box": box,
                             "selection": selection, "evaluator_target_iou": correction_iou},
              "episodes": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in results:
        print(row["source"], "absent" if row["target_absent"] else "moved",
              {key: value["top_correct"] if not row["target_absent"] else
               round(value["top_score"], 3) if value["top_score"] is not None else None
               for key, value in row["ranking"].items()})


if __name__ == "__main__":
    main()
