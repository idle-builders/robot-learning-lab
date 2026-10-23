"""V228: frozen owner RGB identity gate on full Pi Wide-field origin images."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from audit_entry_color_v150 import descriptor
from run import ROOT, Vision, iou, sha
from run_color_grounded_entry_v151 import DINO_MIN, HUE_MIN


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v228"
V227 = ROOT / "artifacts/ape-closed-loop-cove50-v227/full-pi-field-optical-audit.json"
V204 = ROOT / "artifacts/ape-closed-loop-cove50-v204/fourth-split-origin-audit.json"
V149 = ROOT / "artifacts/ape-closed-loop-cove50-v149/owner-taught-exemplar.npy"
V150 = ROOT / "artifacts/ape-closed-loop-cove50-v150/entry-color-audit.json"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    optics = json.loads(V227.read_text())["rows"]
    baseline = json.loads(V204.read_text())["rows"]
    old = {(r["seed"], r["absent"], round(float(v["head_yaw_rad"]), 3)): v
           for r in baseline if r["seed"] in (107, 111, 113)
           for v in r["rows"]}
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner = np.asarray(json.loads(V150.read_text())["owner_color"]["hue_histogram"])
    rows = []
    for row in optics:
        image_path = V227.parent / row["full_field_rgb_file"]
        rgb = Image.open(image_path).convert("RGB")
        proposals = vision.detect(rgb)
        features = vision.feature([rgb.crop(p["box"]) for p in proposals]) if proposals else []
        candidates = []
        truth = row["full_field_true_can_box_evaluator_only"]
        for proposal, feature in zip(proposals, features):
            score = float(feature @ exemplar)
            hue = float(np.minimum(owner, np.asarray(
                descriptor(rgb, proposal["box"])["hue_histogram"])).sum())
            if score >= DINO_MIN and hue >= HUE_MIN:
                candidates.append((score, hue, proposal["box"]))
        selected = max(candidates, key=lambda x: x[0]) if candidates else None
        previous = old[(row["seed"], row["absent"], row["head_yaw_rad"])]
        found = {"seed": row["seed"], "absent": row["absent"],
                 "head_yaw_rad": row["head_yaw_rad"],
                 "full_field_rgb_sha256": sha(image_path),
                 "full_field_proposals": len(proposals),
                 "full_field_accepted_box": None if selected is None else selected[2],
                 "full_field_similarity": None if selected is None else round(selected[0], 4),
                 "full_field_hue": None if selected is None else round(selected[1], 4),
                 "full_field_correct_evaluator_only": bool(
                     selected is not None and truth is not None and
                     iou(selected[2], truth) >= .25),
                 "full_field_wrong_evaluator_only": bool(
                     selected is not None and (truth is None or
                     iou(selected[2], truth) < .25)),
                 "cropped_field_accepted": previous["accepted_box"] is not None,
                 "cropped_field_correct_evaluator_only":
                     previous["accepted_correct_evaluator_only"],
                 "cropped_field_wrong_evaluator_only": bool(
                     previous["accepted_box"] is not None and
                     not previous["accepted_correct_evaluator_only"])}
        rows.append(found)
        print(found["seed"], found["absent"], found["head_yaw_rad"],
              found["full_field_proposals"],
              found["full_field_correct_evaluator_only"],
              found["full_field_wrong_evaluator_only"], flush=True)
    summary = {"views": len(rows)}
    for field in ("correct_evaluator_only", "wrong_evaluator_only"):
        for prefix in ("full_field", "cropped_field"):
            summary[f"{prefix}_{field}_views"] = sum(
                r[f"{prefix}_{field}"] for r in rows)
    summary["present_rooms_with_correct_full_field_acceptance"] = sum(
        any(r["seed"] == seed and r["full_field_correct_evaluator_only"]
            for r in rows) for seed in (107, 111, 113))
    result = {"scope": "V228 frozen V151 generic RGB proposal, DINO appearance >=.35 and owner hue >=.70 gate on V227 896x480 full-Pi-Wide-pinhole origin images. Same three revealed V203 paired present/near-memory absent rooms and five saved head poses; V204 cropped-image decisions as paired reference. Evaluator true-can mask box only scores after selection. No detector recalibration, metric location, live motion, physical lens calibration, or full-field depth actor loop.",
              "source_sha256": sha(SOURCE), "v227_sha256": sha(V227),
              "v204_sha256": sha(V204), "owner_exemplar_sha256": sha(V149),
              "owner_hue_audit_sha256": sha(V150),
              "dino_min": DINO_MIN, "hue_min": HUE_MIN,
              "rows": rows, "summary": summary}
    (OUT / "full-field-identity-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
