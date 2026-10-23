"""Render paired native traces that show V130 rescues and limitations."""

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
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v130"
PLAN = BASE / "opened-failure-perturbation-plan.json"
MATRIX = BASE / "native-opened-failure-perturbations/matrix.json"
RENDERER = "experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.render_observable_video"
MANIFEST = BASE / "representative-videos.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    matrix = json.loads(MATRIX.read_text())
    plan = json.loads(PLAN.read_text())
    if (not matrix["complete"] or matrix["case_count"] != 24 or
            matrix["plan_sha256"] != sha(PLAN) or
            {row["id"] for row in matrix["cases"]} !=
            {row["id"] for row in plan["cases"]}):
        raise ValueError("Complete source-matched V130 matrix required")
    selected = []
    for seed in sorted({row["seed"] for row in matrix["cases"]}):
        cohort = [row for row in matrix["cases"] if row["seed"] == seed]
        rescue = [row for row in cohort
                  if not row["baseline_pass"] and row["shifted_pass"]]
        if rescue:
            selected.append(max(rescue, key=lambda row: row["contact_delta_s"]))
        loss = [row for row in cohort if row["baseline_pass"] and
                row["shifted_pass"] and row["contact_delta_s"] < 0]
        if loss:
            selected.append(min(loss, key=lambda row: row["contact_delta_s"]))
        unresolved = [row for row in cohort if not row["shifted_pass"]]
        if unresolved:
            selected.append(min(unresolved, key=lambda row: row["shifted_contact_s"]))
    selected = list({row["id"]: row for row in selected}.values())
    manifest_rows = []
    for row in selected:
        source_check.make_pair(
            "V130_representative_video_source_check", row, row,
            row["baseline_result"], row["shifted_result"],
            row["baseline_contact_s"], row["shifted_contact_s"],
            row["baseline_pass"], row["shifted_pass"])
        videos = []
        for variant in ("baseline", "shifted"):
            report = ROOT / row[f"{variant}_result"]
            trace = ROOT / row[f"{variant}_trace"]
            if not report.is_file() or not trace.is_file():
                raise ValueError(f"Incomplete native trace: {row['id']} {variant}")
            video = report.with_suffix(".mp4")
            if not video.is_file():
                temporary = video.with_name(video.stem + ".tmp.mp4")
                temporary.unlink(missing_ok=True)
                try:
                    subprocess.run([sys.executable, "-m", RENDERER,
                                    "--report", str(report),
                                    "--trace", str(trace),
                                    "--video", str(temporary)],
                                   cwd=ROOT, check=True)
                    os.replace(temporary, video)
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            videos.append({"variant": variant,
                           "path": str(video.relative_to(ROOT)),
                           "sha256": sha(video),
                           "report_sha256": sha(report),
                           "trace_sha256": sha(trace)})
        manifest_rows.append({"id": row["id"], "seed": row["seed"],
                              "baseline_contact_s": row["baseline_contact_s"],
                              "shifted_contact_s": row["shifted_contact_s"],
                              "baseline_faults": row["baseline_faults"],
                              "shifted_faults": row["shifted_faults"],
                              "videos": videos})
    manifest = {
        "schema": "robot-learning-lab.cove50-v130-representative-videos.v1",
        "role": "source_matched_opened_training_clips_not_independent_score",
        "body_role": matrix["body_role"],
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "renderer_source_sha256": sha(Path(__file__).with_name("render_observable_video.py")),
        "selector_source_sha256": sha(Path(__file__)),
        "cases": manifest_rows,
    }
    encoded = json.dumps(manifest, indent=2) + "\n"
    if MANIFEST.exists() and MANIFEST.read_text() != encoded:
        raise ValueError("Saved representative video manifest changed")
    MANIFEST.write_text(encoded)
    print(f"V130 rendered {len(manifest_rows)} paired representative cases")


if __name__ == "__main__":
    main()
