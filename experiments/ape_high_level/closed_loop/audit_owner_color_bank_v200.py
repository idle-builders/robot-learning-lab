"""V200: frozen V151 owner hue gate on V197 true and V199 absent crops."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from audit_entry_color_v150 import descriptor
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v200"
V150 = ROOT / "artifacts/ape-closed-loop-cove50-v150/entry-color-audit.json"
V197 = ROOT / "artifacts/ape-closed-loop-cove50-v197/view-bank-audit.json"
V199 = ROOT / "artifacts/ape-closed-loop-cove50-v199/nearby-absent-audit.json"
V196_ROOT = ROOT / "artifacts/ape-closed-loop-cove50-v196"
V199_ROOT = ROOT / "artifacts/ape-closed-loop-cove50-v199"
HUE_MIN = .70


def score(old: dict, rgb: Image.Image, owner_hist: np.ndarray,
          *, source: str) -> dict:
    proposals = []
    for proposal in old["proposals"]:
        color = descriptor(rgb, proposal["box"])
        overlap = float(np.minimum(owner_hist,
                                   np.asarray(color["hue_histogram"])).sum())
        similarities = proposal.get("similarities")
        if similarities is None:
            raise AssertionError("Expected frozen view-bank proposal scores")
        bank_pass = proposal.get("bank_pass", proposal.get("view_bank_pass"))
        passed = bool(bank_pass and overlap >= HUE_MIN)
        proposals.append({"box": proposal["box"],
                          "bank_similarity": round(max(similarities), 4),
                          "distance_to_memory_m": proposal["distance_to_memory_m"],
                          "bank_pass": bank_pass,
                          "hue_intersection": round(overlap, 4),
                          "saturated_pixels": color["saturated_pixels"],
                          "red_fraction": color["red_fraction"],
                          "owner_color_bank_pass": passed,
                          "evaluator_truth_iou": proposal.get("evaluator_truth_iou")})
    candidates = [row for row in proposals if row["owner_color_bank_pass"]]
    selected = max(candidates, key=lambda row: row["bank_similarity"]) if candidates else None
    return {"source": source,
            "case": old.get("case"),
            "label": old.get("label"),
            "yaw_rad": old["yaw_rad"],
            "bank_accepted_before_color": (old.get("accepted_box") is not None or
                                            old.get("view_bank_accepted_box") is not None),
            "owner_color_bank_accepted_box": None if selected is None else selected["box"],
            "accepted_correct_evaluator_only": bool(
                selected is not None and selected["evaluator_truth_iou"] is not None
                and selected["evaluator_truth_iou"] >= .25),
            "proposals": proposals}


def main() -> None:
    owner_hist = np.asarray(json.loads(V150.read_text())["owner_color"]["hue_histogram"])
    true_rows = []
    for old in json.loads(V197.read_text())["rows"]:
        path = V196_ROOT / f"{old['label']}-yaw{old['yaw_rad']:.1f}.npz"
        with np.load(path) as saved:
            rgb = Image.fromarray(saved["rgb"])
        true_rows.append(score(old, rgb, owner_hist, source="v196_v197"))
    nearby_rows = []
    for old in json.loads(V199.read_text())["rows"]:
        path = V199_ROOT / f"{old['case']}-yaw{old['yaw_rad']:.1f}-rgb.png"
        rgb = Image.open(path)
        nearby_rows.append(score(old, rgb, owner_hist, source="v199"))
    present = [row for row in true_rows if row["label"] == "present"]
    matched_absent = [row for row in true_rows if row["label"] == "absent"]
    correct = sum(row["accepted_correct_evaluator_only"] for row in present)
    wrong_present = sum(row["owner_color_bank_accepted_box"] is not None and
                        not row["accepted_correct_evaluator_only"] for row in present)
    absent_accepts = sum(row["owner_color_bank_accepted_box"] is not None
                         for row in matched_absent)
    nearby_accepts = sum(row["owner_color_bank_accepted_box"] is not None
                         for row in nearby_rows)
    result = {"scope": "V200 frozen V150/V151 owner saturated-hue histogram intersection>=0.70 applied alongside unchanged V197/V199 max bank similarity>=0.35 and memory<=12cm on raw V196 true/matched-absent and V199 near-memory absent crops. No detector/threshold/prototype tuning; evaluator identity only scores selected results.",
              "source_sha256": sha(SOURCE),
              "v150_sha256": sha(V150), "v197_sha256": sha(V197),
              "v199_sha256": sha(V199),
              "present_views": len(present),
              "correct_present_views": correct,
              "wrong_present_views": wrong_present,
              "matched_absent_views": len(matched_absent),
              "matched_absent_acceptances": absent_accepts,
              "nearby_absent_views": len(nearby_rows),
              "nearby_absent_acceptances": nearby_accepts,
              "screen_pass": correct >= 12 and wrong_present == 0 and
                             absent_accepts == 0 and nearby_accepts == 0,
              "rows": true_rows + nearby_rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "owner-color-bank-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
