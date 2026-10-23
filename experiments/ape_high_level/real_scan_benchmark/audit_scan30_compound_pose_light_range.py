"""Audit whole-task outcomes, range recovery and actual pose/standoff geometry."""

from __future__ import annotations

from collections import Counter
import json
import statistics

import cv2

import audit_scan30_range_recovery as range_audit
from audit_scan30_pose_bias import episode_metrics
from freeze_scan20_shared_policy_sources import sha
import run_scan30_compound_pose_light_range as run


AUDIT = run.ART / "scan30_compound_pose_light_range_audit.json"
COMPARISON = run.ART / "scan30_compound_pose_light_range_comparison.json"


def decoded_frames(path) -> int:
    """Decode every frame locally, where ffprobe is not installed."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("video could not be opened")
    frames = 0
    try:
        while True:
            ok, _frame = capture.read()
            if not ok:
                break
            frames += 1
    finally:
        capture.release()
    if frames < 1:
        raise ValueError("no decoded frames")
    return frames


def main() -> None:
    run.configure()
    range_audit.whole_task.video_frames = decoded_frames
    range_audit.AUDIT = AUDIT
    range_audit.COMPARISON = COMPARISON
    range_audit.main()
    result = json.loads(COMPARISON.read_text())
    frozen = json.loads(run.PRECOMMIT.read_text())
    report = json.loads(run.REPORT.read_text())
    # The reused mixed-run freeze seals its own lighting and gap references,
    # not the two descriptive comparisons selected by this wrapper. Preserve
    # that provenance limitation explicitly; do not call these paired controls.
    comparator_issues = [issue for issue in result["issues"] if issue in (
        f"frozen comparator changed: {run.LIGHT_RANGE.name}",
        f"frozen comparator changed: {run.POSE_CORRECTED.name}")]
    issues = [issue for issue in result["issues"] if issue not in
              comparator_issues]
    for path, key in ((run.mixed.PRECOMMIT,
                       "mixed_light_range_precommit_sha256"),
                      (run.pose.PRECOMMIT, "pose_bias_precommit_sha256")):
        if sha(path) != frozen[key]:
            issues.append(f"frozen source changed: {path.name}")
    band = tuple(frozen["evaluator_standoff_band_m"])
    metrics = [episode_metrics(row, band) for row in report["rows"]]
    for case, metric in zip(frozen["cases"], metrics):
        if not metric["all_increments_have_frozen_scale"]:
            issues.append(f"wrong motion-bias scale: {case['scene_key']}")
    exposed = [m for m in metrics if m["accepted_biased_increments"]]
    pose_errors = [m["max_evaluator_pose_error_m"] for m in metrics
                   if m["max_evaluator_pose_error_m"] is not None]
    result.update(schema="ape.scan30-compound-pose-light-range-comparison.v1",
        issues=issues,
        comparator_provenance_limitation=(
            "The precommit inherited SHA fields for the older lighting-only "
            "and range-only reports, not the mixed-light-range and pose-bias "
            "reports used as descriptive references here. Their current "
            "content hashes are recorded below but were not prospectively "
            "frozen by this combined run; no same-runtime causal comparison "
            "is claimed."),
        inherited_comparator_hash_mismatches=comparator_issues,
        descriptive_reference_sha256={"light_plus_range": sha(
            run.LIGHT_RANGE), "pose_bias_plus_correction": sha(
            run.POSE_CORRECTED)},
        references={"light_plus_range": str(run.LIGHT_RANGE.relative_to(
            run.mixed.unified.ROOT)),
            "pose_bias_plus_correction": str(run.POSE_CORRECTED.relative_to(
                run.mixed.unified.ROOT))},
        compound_outcomes=dict(Counter(m["result"] for m in metrics)),
        evaluator_standoff_band_m=band,
        biased_motion_exposed=len(exposed),
        accepted_biased_increments=sum(m["accepted_biased_increments"]
                                       for m in metrics),
        nonzero_range_pose_corrections=sum(m["nonzero_range_corrections"]
                                           for m in metrics),
        completed_in_actual_standoff_band=sum(m["result"] == "completed" and
            m["actual_standoff_in_band"] for m in metrics),
        completed_outside_actual_standoff_band=sum(m["result"] == "completed"
            and not m["actual_standoff_in_band"] for m in metrics),
        median_max_evaluator_pose_error_m=statistics.median(pose_errors)
            if pose_errors else None,
        pose_and_standoff=[{"scene_key": case["scene_key"], **metric}
            for case, metric in zip(frozen["cases"], metrics)])
    COMPARISON.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "matched_rooms", "outage_exposed", "compound_outcomes",
        "biased_motion_exposed", "completed_in_actual_standoff_band",
        "completed_outside_actual_standoff_band", "wrong_candidate_offers",
        "issues")}, indent=2))


if __name__ == "__main__":
    main()
