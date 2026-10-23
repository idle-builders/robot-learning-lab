"""Render source-matched paired examples from complete opened V144 variance."""

from __future__ import annotations

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
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v144"
PLAN = BASE / "supported-opposite-direction-variance-plan.json"
MATRIX = BASE / "native-supported-opposite-direction-variance/matrix.json"
SUMMARY = BASE / "paired-contact-summary.json"
MANIFEST = BASE / "representative-videos.json"
RENDERER = "experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.render_observable_video"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    matrix = json.loads(MATRIX.read_text())
    summary = json.loads(SUMMARY.read_text())
    if (not matrix["complete"] or matrix["case_count"] != 16 or
            matrix["plan_sha256"] != sha(PLAN) or
            summary["matrix_sha256"] != sha(MATRIX) or
            summary["source_checked_native_visit_count"] != 48):
        raise ValueError("Complete source-checked V144 variance required")
    rows = matrix["cases"]
    selected = []
    for seed in (24111, 24119):
        group = [row for row in rows if row["seed"] == seed]
        failed = [row for row in group if not row["baseline_pass"] or
                  not row["baseline_margin_1p2_s_met"]]
        selected.append(min(failed or group, key=lambda row: row["baseline_contact_s"]))
        positive_rescues = [row for row in failed if row["choices"]["pos2"]["pass"]]
        if positive_rescues:
            selected.append(max(positive_rescues, key=lambda row:
                                row["choices"]["pos2"]["contact_s"] -
                                row["baseline_contact_s"]))
        negative_harms = [row for row in group if
                          not row["choices"]["neg2"]["pass"] and
                          row["choices"]["pos2"]["pass"]]
        if negative_harms:
            selected.append(min(negative_harms, key=lambda row:
                                row["choices"]["neg2"]["contact_s"]))
        selected.append(max(group, key=lambda row:
                            abs(row["choices"]["pos2"]["contact_s"] -
                                row["choices"]["neg2"]["contact_s"])))
    selected = list({row["id"]: row for row in selected}.values())
    manifest_rows = []
    for row in selected:
        base = json.loads((ROOT / row["baseline_result"]).read_text())
        entry = score_runner.observations.observed_features(base)
        videos = []
        for label, path, trace, contact, passed, faults, shift in (
                ("unchanged", row["baseline_result"], row["baseline_trace"],
                 row["baseline_contact_s"], row["baseline_pass"],
                 row["baseline_faults"], 0),
                *((label, choice["result"], choice["trace"],
                   choice["contact_s"], choice["pass"], choice["faults"],
                   choice["shift_body_y_m"])
                  for label, choice in row["choices"].items())):
            report_path, trace_path = ROOT / path, ROOT / trace
            report = json.loads(report_path.read_text())
            if (not trace_path.is_file() or report["seed"] != row["seed"] or
                    abs(report["phases"]["reach"]["continuous_finger_contact_s"] -
                        contact) > 1e-7 or
                    abs(report.get("diagnostic_goal_shift_body_y_m", 0) - shift) > 1e-9 or
                    score_runner.entry_parity(
                        entry, score_runner.observations.observed_features(report))):
                raise ValueError(f"V144 video source differs: {row['id']} {label}")
            video = report_path.with_suffix(".mp4")
            if not video.is_file():
                temporary = video.with_name(video.stem + ".tmp.mp4")
                temporary.unlink(missing_ok=True)
                try:
                    subprocess.run([sys.executable, "-m", RENDERER,
                                    "--report", str(report_path),
                                    "--trace", str(trace_path),
                                    "--video", str(temporary)],
                                   cwd=ROOT, check=True)
                    os.replace(temporary, video)
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            videos.append({"variant": label, "path": str(video.relative_to(ROOT)),
                           "sha256": sha(video),
                           "report_sha256": sha(report_path),
                           "trace_sha256": sha(trace_path),
                           "contact_s": contact, "full_gate_pass": passed,
                           "faults": faults})
        manifest_rows.append({
            "id": row["id"], "seed": row["seed"],
            "route_variant": row["route_variant"],
            "baseline_contact_s": row["baseline_contact_s"],
            "neg2_contact_s": row["choices"]["neg2"]["contact_s"],
            "pos2_contact_s": row["choices"]["pos2"]["contact_s"],
            "videos": videos,
        })
    manifest = {
        "schema": "robot-learning-lab.cove50-v144-representative-videos.v1",
        "role": "opened_training_variance_not_independent_score",
        "body_role": matrix["body_role"],
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "summary_sha256": sha(SUMMARY),
        "renderer_source_sha256": sha(Path(__file__).with_name("render_observable_video.py")),
        "selection_source_sha256": sha(Path(__file__)),
        "cases": manifest_rows,
    }
    encoded = json.dumps(manifest, indent=2) + "\n"
    if MANIFEST.exists() and MANIFEST.read_text() != encoded:
        raise ValueError("Saved V144 representative manifest changed")
    MANIFEST.write_text(encoded)
    print(f"V144 rendered {len(manifest_rows)} paired physical conditions")


if __name__ == "__main__":
    main()
