"""Render saved V158 development contacts after independent source checks.

Post-step videos visualize the saved CUDA trace; they never rescore contact.
Invalid RGB or physical-start pairs retain the original video and label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v158"
PLAN = BASE / "new-seed-contact-plan.json"
MATRIX = BASE / "native-development-contact/matrix.json"
SUMMARY = BASE / "source-checked-development-contact-summary.json"
PARTIAL_SUMMARY = BASE / "partial-source-checked-development-contact-summary.json"
MANIFEST = BASE / "development-contact-videos.json"
PARTIAL_MANIFEST = BASE / "partial-development-contact-videos.json"
RENDERER = (
    "experiments.compact_morphology.cove50_policy_transfer."
    "reach_image_goal.render_observable_video_poststep"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-partial", action="store_true")
    args = parser.parse_args()
    summary_path = PARTIAL_SUMMARY if args.verify_partial else SUMMARY
    manifest_path = PARTIAL_MANIFEST if args.verify_partial else MANIFEST
    if not all(path.is_file() for path in (PLAN, MATRIX, summary_path)):
        raise ValueError("Source-checked V158 development contact is not available")
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    summary = json.loads(summary_path.read_text())
    refs = summary["source_sha256"]
    if (not matrix["cases"] or
            refs["plan"] != sha(PLAN) or refs["matrix"] != sha(MATRIX) or
            refs["checker"] != sha(Path(__file__).with_name(
                "summarize_v158_new_seed_development_contact.py")) or
            summary["source_checked_family_count"] != matrix["case_count"] or
            matrix["held_contact_visit_count"] != 0 or
            summary["held_contact_visit_count"] != 0 or
            plan["contains_contact_results"] or
            (not args.verify_partial and not matrix["complete"])):
        raise ValueError("V158 source check or development boundary changed")
    checked_files = {(item["id"], item["route"]): item
                     for item in summary["source_files"]}
    prior_path = (manifest_path if manifest_path.is_file() else
                  PARTIAL_MANIFEST if not args.verify_partial and
                  PARTIAL_MANIFEST.is_file() else None)
    prior = json.loads(prior_path.read_text()) if prior_path else {}
    prior_videos = {
        (case["id"], item["variant"]): item
        for case in prior.get("cases", []) for item in case.get("videos", [])
    }
    cases = []
    for row in matrix["cases"]:
        if row["split"] != "development":
            raise ValueError("V158 held seed appeared in development videos")
        videos = []
        labels = (["automatic", "prescribed"] if row["prescribed_report"] else
                  ["automatic"])
        for label in labels:
            source = checked_files.get((row["id"], label))
            report_rel, trace_rel = row[f"{label}_report"], row[f"{label}_trace"]
            report, trace = ROOT / report_rel, ROOT / trace_rel
            if (not source or not report.is_file() or not trace.is_file() or
                    source["report"] != report_rel or
                    source["trace"] != trace_rel or
                    source["report_sha256"] != sha(report) or
                    source["trace_sha256"] != sha(trace) or
                    row[f"{label}_report_sha256"] != sha(report) or
                    row[f"{label}_trace_sha256"] != sha(trace)):
                raise ValueError(f"V158 video source changed: {row['id']} {label}")
            saved = json.loads(report.read_text())
            if (saved["seed"] != row["seed"] or
                    saved["selection_mode"] != row[f"{label}_route"] or
                    abs(saved["phases"]["reach"]["continuous_finger_contact_s"] -
                        row[f"{label}_contact_s"]) > 1e-7):
                raise ValueError(f"V158 video report differs: {row['id']} {label}")
            video = report.with_name(report.stem + "-aligned.mp4")
            earlier = prior_videos.get((row["id"], label), {})
            reusable = (
                video.is_file() and earlier.get("path") ==
                str(video.relative_to(ROOT)) and
                earlier.get("report_sha256") == sha(report) and
                earlier.get("trace_sha256") == sha(trace) and
                earlier.get("video_sha256") == sha(video))
            if not reusable:
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
                "variant": label, "path": str(video.relative_to(ROOT)),
                "video_sha256": sha(video),
                "report": report_rel, "trace": trace_rel,
                "report_sha256": sha(report), "trace_sha256": sha(trace),
                "contact_s": row[f"{label}_contact_s"],
                "full_gate_pass": row[f"{label}_native_pass"],
                "margin_1p2_s_met": row[f"{label}_margin_1p2_s_met"],
                "faults": row[f"{label}_faults"],
            })
        cases.append({"id": row["id"], "seed": row["seed"],
                      "stratum": row["stratum"],
                      "status": row["status"],
                      "matched_pair_valid": row["matched_pair_valid"],
                      "failure_reasons": row["failure_reasons"],
                      "automatic_route": row["automatic_route"],
                      "prescribed_route": row["prescribed_route"],
                      "videos": videos})
    manifest = {
        "schema": "robot-learning-lab.cove50-v158-development-contact-videos.v1",
        "role": "source_checked_new_seed_development_videos_not_held_score",
        "body_role": matrix["body_role"],
        "source_sha256": {"plan": sha(PLAN), "matrix": sha(MATRIX),
                          "summary": sha(summary_path),
                          "selection": sha(Path(__file__)),
                          "renderer": sha(Path(__file__).with_name(
                              "render_observable_video_poststep.py"))},
        "source_checked_family_count": matrix["case_count"],
        "held_contact_visit_count": 0,
        "video_alignment": ("Saved trace qpos precedes each physics step; "
                            "frame i renders qpos[i+1] beside post-step contact."),
        "cases": cases,
    }
    encoded = json.dumps(manifest, indent=2) + "\n"
    if not args.verify_partial and MANIFEST.exists() and MANIFEST.read_text() != encoded:
        raise ValueError("Sealed V158 development video manifest changed")
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(encoded)
    os.replace(temporary, manifest_path)
    print(f"V158 rendered {len(cases)} source-matched development families")


if __name__ == "__main__":
    main()
