"""Render paired opened V141 score-failure diagnostics with exact sources."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    build_v118_observable_pairs as source_check,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v141"
PLAN = BASE / "opened-score-failure-plan.json"
MATRIX = BASE / "native-opened-failure-shifts/matrix.json"
MANIFEST = BASE / "paired-videos.json"
RENDERER = "experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.render_observable_video"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    if (not matrix["complete"] or matrix["plan_sha256"] != sha(PLAN) or
            matrix["case_count"] != plan["case_count"] or
            {row["id"] for row in matrix["cases"]} !=
            {case["id"] for case in plan["cases"]}):
        raise ValueError("Complete source-matched V141 diagnostic required")
    source_cases = {case["id"]: case for case in plan["cases"]}
    manifest_rows = []
    for row in matrix["cases"]:
        source_check.make_pair(
            "V141_opened_diagnostic_video_source_check", row, row,
            row["baseline_result"], row["diagnostic_result"],
            row["baseline_contact_s"], row["diagnostic_contact_s"],
            source_cases[row["id"]]["source_baseline_native_pass"],
            row["diagnostic_native_pass"])
        videos = []
        for variant in ("baseline", "diagnostic"):
            report = ROOT / row[f"{variant}_result"]
            trace = ROOT / row[f"{variant}_trace"]
            if not report.is_file() or not trace.is_file():
                raise ValueError(f"Incomplete V141 trace: {row['id']} {variant}")
            video = report.with_suffix(".mp4")
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
                           "report_sha256": sha(report),
                           "trace_sha256": sha(trace)})
        manifest_rows.append({
            "id": row["id"], "seed": row["seed"],
            "baseline_contact_s": row["baseline_contact_s"],
            "diagnostic_contact_s": row["diagnostic_contact_s"],
            "diagnostic_pass": row["diagnostic_pass"],
            "baseline_faults": row["baseline_faults"],
            "diagnostic_faults": row["diagnostic_faults"],
            "videos": videos,
        })
    manifest = {
        "schema": "robot-learning-lab.cove50-v141-paired-diagnostic-videos.v1",
        "role": "post_score_opened_diagnosis_not_independent_score",
        "body_role": matrix["body_role"],
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "renderer_source_sha256": sha(Path(__file__).with_name("render_observable_video.py")),
        "selection_source_sha256": sha(Path(__file__)),
        "cases": manifest_rows,
    }
    encoded = json.dumps(manifest, indent=2) + "\n"
    if MANIFEST.exists() and MANIFEST.read_text() != encoded:
        raise ValueError("Saved V141 diagnostic video manifest changed")
    MANIFEST.write_text(encoded)
    print(f"V141 rendered {len(manifest_rows)} opened diagnostic pairs")


if __name__ == "__main__":
    main()
