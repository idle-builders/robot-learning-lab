"""Render matched automatic and prescribed route videos after V152 source checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v152"
PLAN = BASE / "paired-route-training-plan.json"
MATRIX = BASE / "native-paired-routes/matrix.json"
SUMMARY = BASE / "source-checked-paired-route-summary.json"
PARTIAL_SUMMARY = BASE / "partial-source-checked-paired-route-summary.json"
MANIFEST = BASE / "representative-route-videos.json"
PARTIAL_MANIFEST = BASE / "partial-representative-route-videos.json"
RENDERER = (
    "experiments.compact_morphology.cove50_policy_transfer."
    "reach_image_goal.render_observable_video_poststep"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_cases(rows: list[dict], limit: int) -> list[dict]:
    ordered = sorted(rows, key=lambda row: row["seed"])
    chosen: dict[str, dict] = {}

    def add(row: dict | None) -> None:
        if row is not None and len(chosen) < limit:
            chosen.setdefault(row["id"], row)

    def first(predicate):
        return next((row for row in ordered if predicate(row)), None)

    for cohort in ("opened_roll_high_neighborhood", "opened_pitch_low_neighborhood"):
        add(first(lambda row: row["cohort"] == cohort and
                  not row["baseline_pass"] and row["prescribed_pass"]))
        add(first(lambda row: row["cohort"] == cohort and
                  row["baseline_pass"] and not row["prescribed_pass"]))
        add(first(lambda row: row["cohort"] == cohort and
                  not row["baseline_margin_1p2_s_met"] and
                  row["prescribed_margin_1p2_s_met"] and row["prescribed_pass"]))
        add(first(lambda row: row["cohort"] == cohort and
                  row["baseline_margin_1p2_s_met"] and
                  not row["prescribed_margin_1p2_s_met"]))
        add(first(lambda row: row["cohort"] == cohort and
                  not row["baseline_pass"] and not row["prescribed_pass"]))
        add(first(lambda row: row["cohort"] == cohort and
                  row["baseline_pass"] and row["prescribed_pass"]))
    add(first(lambda row: row["reused_v151_source"]))
    for route in ("v16_live", "v16_locked", "v16_specialist_locked",
                  "v21_scoped_y_locked"):
        add(first(lambda row: row["automatic_route"] == route))
    for row in sorted(ordered, key=lambda item: (
            item["prescribed_contact_s"] - item["baseline_contact_s"],
            item["seed"])):
        add(row)
    return list(chosen.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-partial", action="store_true")
    args = parser.parse_args()
    summary_path = PARTIAL_SUMMARY if args.verify_partial else SUMMARY
    manifest_path = PARTIAL_MANIFEST if args.verify_partial else MANIFEST
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    summary = json.loads(summary_path.read_text())
    if (matrix["case_count"] < 1 or matrix["case_count"] > 48 or
            (not args.verify_partial and
             (not matrix["complete"] or matrix["case_count"] != 48)) or
            matrix["plan_sha256"] != sha(PLAN) or
            summary["source_sha256"]["matrix"] != sha(MATRIX) or
            summary["source_sha256"]["plan"] != sha(PLAN) or
            summary["source_sha256"]["checker"] != sha(
                Path(__file__).with_name("summarize_v152_paired_route_training.py")) or
            summary["source_checked_pair_count"] != matrix["case_count"] or
            summary["source_checked_native_source_count"] !=
            2 * matrix["case_count"] or
            summary["body_role"] != "rejected_historical_direct_frame_sandbox"):
        raise ValueError("Source-checked V152 route pairs required")
    checked_files = {(item["id"], item["route"]): item
                     for item in summary["source_files"]}
    chosen = select_cases(matrix["cases"], 8 if args.verify_partial else 12)
    manifest_cases = []
    for row in chosen:
        videos = []
        for label in ("baseline", "prescribed"):
            report_rel = row[f"{label}_report"]
            trace_rel = row[f"{label}_trace"]
            report, trace = ROOT / report_rel, ROOT / trace_rel
            source = checked_files.get((row["id"], label))
            if (not source or not report.is_file() or not trace.is_file() or
                    source["report"] != report_rel or
                    source["trace"] != trace_rel or
                    source["report_sha256"] != sha(report) or
                    source["trace_sha256"] != sha(trace) or
                    row[f"{label}_report_sha256"] != sha(report) or
                    row[f"{label}_trace_sha256"] != sha(trace)):
                raise ValueError(f"V152 source mismatch for video: {row['id']} {label}")
            saved = json.loads(report.read_text())
            if (saved["seed"] != row["seed"] or
                    abs(saved["phases"]["reach"]["continuous_finger_contact_s"] -
                        row[f"{label}_contact_s"]) > 1e-7 or
                    saved["selection_mode"] !=
                    (row["automatic_route"] if label == "baseline"
                     else row["prescribed_route"])):
                raise ValueError(f"V152 report differs for video: {row['id']} {label}")
            video = report.with_name(report.stem + "-aligned.mp4")
            if not video.is_file():
                temporary = video.with_name(video.stem + ".tmp.mp4")
                temporary.unlink(missing_ok=True)
                try:
                    subprocess.run([sys.executable, "-m", RENDERER,
                                    "--report", str(report),
                                    "--trace", str(trace),
                                    "--video", str(temporary)], cwd=ROOT, check=True)
                    os.replace(temporary, video)
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            videos.append({
                "variant": label,
                "path": str(video.relative_to(ROOT)),
                "video_sha256": sha(video),
                "report": report_rel, "trace": trace_rel,
                "report_sha256": sha(report), "trace_sha256": sha(trace),
                "contact_s": row[f"{label}_contact_s"],
                "full_gate_pass": row[f"{label}_pass"],
                "margin_1p2_s_met": row[f"{label}_margin_1p2_s_met"],
                "faults": row[f"{label}_faults"],
            })
        manifest_cases.append({"id": row["id"], "seed": row["seed"],
                               "cohort": row["cohort"],
                               "automatic_route": row["automatic_route"],
                               "prescribed_route": row["prescribed_route"],
                               "reused_v151_source": row["reused_v151_source"],
                               "videos": videos})
    manifest = {
        "schema": ("robot-learning-lab.cove50-v152-partial-route-videos.v1"
                   if args.verify_partial else
                   "robot-learning-lab.cove50-v152-route-videos.v1"),
        "role": "source_checked_prescribed_route_training_not_independent_score",
        "body_role": matrix["body_role"],
        "source_sha256": {"plan": sha(PLAN), "matrix": sha(MATRIX),
                          "summary": sha(summary_path),
                          "selection": sha(Path(__file__)),
                          "renderer": sha(Path(__file__).with_name(
                              "render_observable_video_poststep.py"))},
        "source_checked_pair_count": matrix["case_count"],
        "video_alignment": ("Saved trace qpos precedes each physics step; "
                            "frame i renders qpos[i+1] beside post-step contact."),
        "cases": manifest_cases,
    }
    encoded = json.dumps(manifest, indent=2) + "\n"
    if not args.verify_partial and MANIFEST.exists() and MANIFEST.read_text() != encoded:
        raise ValueError("Sealed V152 route video manifest changed")
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(encoded)
    os.replace(temporary, manifest_path)
    print(f"V152 rendered {len(manifest_cases)} source-matched route pairs"
          + (" from checked partial results" if args.verify_partial else ""))


if __name__ == "__main__":
    main()
