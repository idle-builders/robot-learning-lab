"""Render source-matched V150 paired training comparisons after source checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v139_reserved_contact as score_runner,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v150"
PLAN = BASE / "paired-contact-training-plan.json"
MATRIX = BASE / "native-paired-contact/matrix.json"
SUMMARY = BASE / "source-checked-paired-summary.json"
MANIFEST = BASE / "representative-videos.json"
PARTIAL_SUMMARY = BASE / "partial-source-checked-paired-summary.json"
PARTIAL_MANIFEST = BASE / "partial-representative-videos.json"
RENDERER = (
    "experiments.compact_morphology.cove50_policy_transfer."
    "reach_image_goal.render_observable_video_poststep"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_cases(rows: list[dict]) -> list[dict]:
    ordered = sorted(rows, key=lambda row: row["seed"])
    chosen: dict[str, dict] = {}

    def add(row: dict | None) -> None:
        if row is not None and len(chosen) < 10:
            chosen.setdefault(row["id"], row)

    def first(predicate):
        return next((row for row in ordered if predicate(row)), None)

    add(first(lambda row: row["screen_to_full_entry_faults"] or
              any(choice["entry_parity_faults"] for choice in
                  row["choices"].values())))
    add(first(lambda row: not row["baseline_pass"] and
              not row["choices"]["neg2"]["pass"] and
              not row["choices"]["pos2"]["pass"]))
    add(first(lambda row: not row["baseline_pass"] and
              row["choices"]["neg2"]["pass"]))
    for route in ("roll-high", "pitch-low"):
        add(first(lambda row: row["route_variant"] == route and
                  not row["baseline_pass"] and
                  row["choices"]["neg2"]["pass"]))
    add(first(lambda row: not row["baseline_pass"] and
              row["choices"]["pos2"]["pass"]))
    add(first(lambda row: row["baseline_pass"] and
              not row["choices"]["neg2"]["pass"]))
    add(first(lambda row: row["baseline_pass"] and
              not row["baseline_margin_1p2_s_met"] and
              row["choices"]["neg2"]["margin_1p2_s_met"]))
    add(first(lambda row: row["baseline_pass"] and
              not row["baseline_margin_1p2_s_met"] and
              row["choices"]["pos2"]["margin_1p2_s_met"]))
    add(first(lambda row: row["baseline_pass"] and
              row["choices"]["neg2"]["pass"] and
              not row["choices"]["neg2"]["margin_1p2_s_met"]))
    add(first(lambda row: row["baseline_pass"] and
              row["choices"]["pos2"]["pass"] and
              not row["choices"]["pos2"]["margin_1p2_s_met"]))
    add(first(lambda row: row["baseline_pass"] and
              not row["choices"]["pos2"]["pass"]))
    add(min(ordered, key=lambda row: (row["choices"]["neg2"]["contact_delta_s"],
                                      row["seed"])))
    add(min(ordered, key=lambda row: (row["choices"]["pos2"]["contact_delta_s"],
                                      row["seed"])))
    for route, observed in (("roll-high", "v21_scoped_y_locked"),
                            ("pitch-low", "v16_specialist_locked")):
        add(first(lambda row: row["route_variant"] == route and
                  row["selection_mode"] == observed))
    for route in ("roll-high", "pitch-low"):
        add(first(lambda row: row["route_variant"] == route))
    for row in ordered:
        if row["baseline_pass"] and not row["choices"]["pos2"]["pass"]:
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
            summary["matrix_sha256"] != sha(MATRIX) or
            summary["plan_sha256"] != sha(PLAN) or
            summary["source_checked_seed_count"] != matrix["case_count"] or
            summary["source_checked_native_visit_count"] !=
            3 * matrix["case_count"] or
            summary["body_role"] != "rejected_historical_direct_frame_sandbox"):
        raise ValueError("Source-checked V150 paired training required")
    manifest_rows = []
    selected = select_cases(matrix["cases"])
    if args.verify_partial:
        selected = selected[:8]
    for row in selected:
        base_report = json.loads((ROOT / row["baseline_result"]).read_text())
        observed = score_runner.observations.observed_features(base_report)
        if (observed != row["full_entry_observed_features"] or
                score_runner.entry_parity(row["entry_observed_features"], observed) !=
                row["screen_to_full_entry_faults"]):
            raise ValueError(f"V150 saved RGB entry differs: {row['id']}")
        videos = []
        for variant, report_rel, trace_rel, contact, passed, margin, faults, shift in (
                ("unchanged", row["baseline_result"], row["baseline_trace"],
                 row["baseline_contact_s"], row["baseline_pass"],
                 row["baseline_margin_1p2_s_met"], row["baseline_faults"], 0),
                *((label, choice["result"], choice["trace"], choice["contact_s"],
                   choice["pass"], choice["margin_1p2_s_met"], choice["faults"],
                   choice["shift_body_y_m"])
                  for label, choice in row["choices"].items())):
            report, trace = ROOT / report_rel, ROOT / trace_rel
            saved = json.loads(report.read_text())
            parity = score_runner.entry_parity(
                observed, score_runner.observations.observed_features(saved))
            expected_parity = ([] if variant == "unchanged" else
                               row["choices"][variant]["entry_parity_faults"])
            if (not trace.is_file() or saved["seed"] != row["seed"] or
                    saved["body_sha256"] != plan["source_sha256"]["body"] or
                    saved["selected_pointer_sha256"] !=
                    plan["source_sha256"]["pointer"] or
                    saved["reach_checkpoint_sha256"] !=
                    plan["source_sha256"]["actor"] or
                    abs(saved["phases"]["reach"]["continuous_finger_contact_s"] -
                        contact) > 1e-7 or
                    abs(saved.get("diagnostic_goal_shift_body_y_m", 0) - shift) > 1e-9 or
                    parity != expected_parity):
                raise ValueError(f"V150 video source differs: {row['id']} {variant}")
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
            videos.append({"variant": variant,
                           "path": str(video.relative_to(ROOT)),
                           "sha256": sha(video),
                           "report": report_rel, "trace": trace_rel,
                           "report_sha256": sha(report),
                           "trace_sha256": sha(trace),
                           "contact_s": contact,
                           "full_gate_pass": passed,
                           "margin_1p2_s_met": margin,
                           "faults": faults})
        manifest_rows.append({"id": row["id"], "seed": row["seed"],
                              "route_variant": row["route_variant"],
                              "observed_route": row["selection_mode"],
                              "videos": videos})
    manifest = {
        "schema": ("robot-learning-lab.cove50-v150-partial-videos.v1"
                   if args.verify_partial else
                   "robot-learning-lab.cove50-v150-representative-videos.v1"),
        "role": ("source_checked_partial_paired_training_not_independent_score"
                 if args.verify_partial else
                 "observed_entry_selected_paired_training_not_independent_score"),
        "body_role": matrix["body_role"],
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "summary_sha256": sha(summary_path),
        "source_checked_seed_count": matrix["case_count"],
        "renderer_source_sha256": sha(Path(__file__).with_name(
            "render_observable_video_poststep.py")),
        "selection_source_sha256": sha(Path(__file__)),
        "video_alignment": ("Saved trace qpos precedes each physics step; "
                            "frame i renders qpos[i+1] with post-step "
                            "contact and hand position checked within 0.02 mm."),
        "cases": manifest_rows,
    }
    encoded = json.dumps(manifest, indent=2) + "\n"
    if not args.verify_partial and MANIFEST.exists() and MANIFEST.read_text() != encoded:
        raise ValueError("Saved V150 representative video manifest changed")
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(encoded)
    os.replace(temporary, manifest_path)
    print(f"V150 rendered {len(manifest_rows)} source-matched paired video sets"
          + (" from source-checked partial results" if args.verify_partial else ""))


if __name__ == "__main__":
    main()
