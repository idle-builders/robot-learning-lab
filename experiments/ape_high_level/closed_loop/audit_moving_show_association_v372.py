"""V372: compare image-overlap and 3D association for the moving show.

Only adjacent actor accepted boxes, timestamps and V371 robust long-depth
points enter the association decision. Evaluator correctness labels score
those decisions afterward. A 12cm/100ms point gate is a procedural bound,
not identity proof or a trained person/hand tracker.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from run import ROOT, iou, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v372"
V370 = ROOT / "artifacts/ape-closed-loop-cove50-v370/moving-distant-show.json"
V371 = ROOT / "artifacts/ape-closed-loop-cove50-v371/moving-show-depth-replay.json"
MAX_3D_STEP_M = .12
MAX_FRAME_GAP_S = .12
MIN_BOX_IOU = .6


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    visual = json.loads(V370.read_text())
    depth = json.loads(V371.read_text())
    depth_rows = {r["case"]: r for r in depth["rows"]}
    rows = []
    for original in visual["rows"]:
        replay = depth_rows[original["case"]]
        samples = []
        for earlier, later, a, b in zip(
                original["samples"][:-1], original["samples"][1:],
                replay["samples"][:-1], replay["samples"][1:]):
            dt = later["capture_time_s"] - earlier["capture_time_s"]
            box_a = earlier["actor_choice"]["accepted_box"]
            box_b = later["actor_choice"]["accepted_box"]
            image_overlap = (iou(box_a, box_b)
                             if box_a is not None and box_b is not None
                             else None)
            xyz_a = a["estimates"]["noisy_long"]["world_xyz_m"]
            xyz_b = b["estimates"]["noisy_long"]["world_xyz_m"]
            distance = (math.dist(xyz_a, xyz_b)
                        if xyz_a is not None and xyz_b is not None else None)
            samples.append({
                "earlier_time_s": earlier["capture_time_s"],
                "later_time_s": later["capture_time_s"],
                "actor_image_iou": None if image_overlap is None else round(
                    image_overlap, 5),
                "actor_image_associated": bool(image_overlap is not None and
                                               image_overlap >= MIN_BOX_IOU and
                                               dt <= MAX_FRAME_GAP_S),
                "actor_3d_step_m": None if distance is None else round(
                    distance, 5),
                "actor_3d_associated": bool(distance is not None and
                                            distance <= MAX_3D_STEP_M and
                                            dt <= MAX_FRAME_GAP_S),
                "evaluator_both_boxes_correct": (
                    earlier["evaluator_accepted_presented_can"] and
                    later["evaluator_accepted_presented_can"]),
            })
        rows.append({"case": original["case"], "pattern": original["pattern"],
                     "seed": original["seed"], "pairs": samples,
                     "image_associated_pairs": sum(
                         s["actor_image_associated"] for s in samples),
                     "robust_3d_associated_pairs": sum(
                         s["actor_3d_associated"] for s in samples),
                     "evaluator_both_correct_pairs": sum(
                         s["evaluator_both_boxes_correct"] for s in samples)})
    report = {
        "scope": "Actor-only adjacent-frame association replay of V370 10Hz exposed RGB and V371 robust long-depth positions. Compare IoU>=0.6 to 3D point displacement<=12cm within120ms, both using the same released accepted boxes. Evaluator correctness attached afterward. This is no-contact held-item motion in two similar procedural rooms; no distractor at the same 3D location, person/hand identity, uncertainty calibration or native active-gaze latency. A 3D point link alone cannot authorize a physical handoff.",
        "source_sha256": sha(SOURCE),
        "v370_report_sha256": sha(V370),
        "v371_report_sha256": sha(V371),
        "max_3d_step_m": MAX_3D_STEP_M,
        "min_box_iou": MIN_BOX_IOU,
        "rows": rows}
    (OUT / "moving-show-association.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["case"], "image", row["image_associated_pairs"],
              "3d", row["robust_3d_associated_pairs"],
              "both_correct", row["evaluator_both_correct_pairs"], flush=True)


if __name__ == "__main__":
    main()
