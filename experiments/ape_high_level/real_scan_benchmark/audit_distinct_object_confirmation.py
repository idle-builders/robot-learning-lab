"""Compare frozen real-room teach/find episodes before and after a decision rule.

Both sides must use the same teaching image and first search views before a
paired outcome is counted. This is development evidence, not a held-out score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DONE = {"teach_find_completed", "teach_find_completed_far_viewpoint"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def correct_present(row: dict) -> bool:
    result = row.get("find", {}).get("find_present", {})
    return (result.get("action") == "candidate_to_owner_for_confirmation" and
            (result.get("verification_target_iou") or 0.) >= .25 and
            not result.get("wrong_confident_candidate", False))


def false_absent(row: dict) -> bool:
    return row.get("find", {}).get("find_absent", {}).get("action") == \
        "candidate_to_owner_for_confirmation"


def audit(pairs: list[tuple[Path, Path]]) -> dict:
    rows = []
    for baseline_path, variant_path in pairs:
        baseline = json.loads(baseline_path.read_text())
        variant = json.loads(variant_path.read_text())
        if baseline["placement_screen"] != variant["placement_screen"]:
            raise ValueError("Paired runs use different scenario placement screens")
        old = {r["scene_id"]: r for r in baseline["scenes"]}
        new = {r["scene_id"]: r for r in variant["scenes"]}
        if old.keys() != new.keys():
            raise ValueError("Paired batches have different scenes")
        for scene_id in old:
            before, after = old[scene_id], new[scene_id]
            row = {"scene_id": scene_id, "name": before["name"],
                   "baseline_status": before["task_status"],
                   "variant_status": after["task_status"]}
            if before["task_status"] not in DONE or after["task_status"] not in DONE:
                rows.append(row)
                continue
            before_dir = (ROOT / before["summary"]).parent
            after_dir = (ROOT / after["summary"]).parent
            before_summary = json.loads((before_dir / "summary.json").read_text())
            after_summary = json.loads((after_dir / "summary.json").read_text())
            same_teach = (before_summary["teaching"]["rgb_sha256"] ==
                          after_summary["teaching"]["rgb_sha256"])
            same_first_views = {
                condition: digest(before_dir / condition / "rgb_000.png") ==
                           digest(after_dir / condition / "rgb_000.png")
                for condition in ("find_present", "find_absent")}
            present_trace = json.loads((after_dir / "find_present/trace.json").read_text())
            row.update(same_teach_image=same_teach,
                       same_first_search_views=same_first_views,
                       same_image_stimulus=(same_teach and all(same_first_views.values())),
                       baseline_present_correct=correct_present(before),
                       variant_present_correct=correct_present(after),
                       baseline_absent_false_offer=false_absent(before),
                       variant_absent_false_offer=false_absent(after),
                       variant_distinct_candidate_count=present_trace.get(
                           "distinct_candidate_count"),
                       variant_raw_candidate_count=present_trace.get(
                           "raw_candidate_count"),
                       variant_distinct_margin=present_trace.get("margin"),
                       variant_verify_same_object_distance_m=(
                           present_trace["verification"].get("same_object_distance_m")
                           if present_trace.get("verification") else None),
                       baseline_summary=before["summary"],
                       variant_summary=after["summary"])
            rows.append(row)
    completed = [r for r in rows if "baseline_present_correct" in r]
    paired = [r for r in completed if r["same_image_stimulus"]]
    return {"schema": "ape.real-room-distinct-object-confirmation-audit.v1",
            "qualification": "scripted development rooms; only same-image cases support a paired decision-rule comparison; not independent held-out generalization",
            "baseline_batches": [str(a.relative_to(ROOT)) for a, _ in pairs],
            "variant_batches": [str(b.relative_to(ROOT)) for _, b in pairs],
            "scene_count": len(rows), "completed_both_count": len(completed),
            "paired_identical_image_count": len(paired),
            "all_baseline_correct_present_count": sum(r["baseline_present_correct"] for r in completed),
            "all_variant_correct_present_count": sum(r["variant_present_correct"] for r in completed),
            "baseline_correct_present_count": sum(r["baseline_present_correct"] for r in paired),
            "variant_correct_present_count": sum(r["variant_present_correct"] for r in paired),
            "baseline_false_absent_offer_count": sum(r["baseline_absent_false_offer"] for r in paired),
            "variant_false_absent_offer_count": sum(r["variant_absent_false_offer"] for r in paired),
            "scenes": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", nargs=2, type=Path, action="append", required=True,
                        metavar=("BASELINE_BATCH", "VARIANT_BATCH"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit([(a.resolve(), b.resolve()) for a, b in args.pair])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "scenes"}, indent=2))


if __name__ == "__main__":
    main()
