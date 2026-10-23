"""Render source-matched pitch-low contact-direction examples from V142."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v142"
PLAN = BASE / "opened-pitch-low-y-sweep-plan.json"
MATRIX = BASE / "native-opened-pitch-low-y-sweep/matrix.json"
SUMMARY = BASE / "valid-visits-and-launch-failure.json"
MANIFEST = BASE / "direction-videos.json"
RENDERER = "experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.render_observable_video"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    summary = json.loads(SUMMARY.read_text())
    if (matrix["status"] != "stopped_evaluator_range_rejection" or
            matrix["case_count"] != 5 or matrix["plan_sha256"] != sha(PLAN) or
            summary["source_sha256"]["matrix"] != sha(MATRIX)):
        raise ValueError("Exact partial source-checked V142 diagnostic required")
    rows = {round(row["shift_body_y_m"] * 1000): row
            for row in matrix["cases"]}
    reference = [
        {"label": "frozen-unchanged", "shift_body_y_m": 0.0,
         "contact_s": plan["baseline_contact_s"],
         "report": plan["baseline_result"],
         "trace": plan["baseline_result"].removesuffix(".json") + ".npz"},
        {"label": "opened-minus2", "shift_body_y_m": -.002,
         "contact_s": plan["prior_neg2_contact_s"],
         "report": plan["prior_neg2_result"],
         "trace": plan["prior_neg2_result"].removesuffix(".json") + ".npz"},
    ]
    selected = reference + [
        {"label": f"opened-{shift:+d}mm",
         "shift_body_y_m": rows[shift]["shift_body_y_m"],
         "contact_s": rows[shift]["contact_s"],
         "report": rows[shift]["result"], "trace": rows[shift]["trace"]}
        for shift in (-1, 2, 3)]
    manifest_rows = []
    for item in selected:
        report = ROOT / item["report"]
        trace = ROOT / item["trace"]
        saved = json.loads(report.read_text())
        if (not trace.is_file() or saved["seed"] != 24119 or
                saved["phases"]["reach"]["continuous_finger_contact_s"] !=
                item["contact_s"] or
                saved["body_sha256"] != plan["source_sha256"]["body"] or
                saved["selected_pointer_sha256"] !=
                plan["source_sha256"]["pointer"]):
            raise ValueError(f"V142 video source mismatch: {item['label']}")
        video = report.with_suffix(".mp4")
        if not video.is_file():
            temp = video.with_name(video.stem + ".tmp.mp4")
            temp.unlink(missing_ok=True)
            try:
                subprocess.run([sys.executable, "-m", RENDERER,
                                "--report", str(report),
                                "--trace", str(trace),
                                "--video", str(temp)], cwd=ROOT, check=True)
                os.replace(temp, video)
            except BaseException:
                temp.unlink(missing_ok=True)
                raise
        manifest_rows.append({**item,
                              "video": str(video.relative_to(ROOT)),
                              "video_sha256": sha(video),
                              "report_sha256": sha(report),
                              "trace_sha256": sha(trace)})
    manifest = {
        "schema": "robot-learning-lab.cove50-v142-direction-videos.v1",
        "role": "partial_opened_diagnostic_not_policy_or_score",
        "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "summary_sha256": sha(SUMMARY),
        "renderer_source_sha256": sha(Path(__file__).with_name("render_observable_video.py")),
        "selection_source_sha256": sha(Path(__file__)),
        "cases": manifest_rows,
    }
    encoded = json.dumps(manifest, indent=2) + "\n"
    if MANIFEST.exists() and MANIFEST.read_text() != encoded:
        raise ValueError("Saved V142 video manifest changed")
    MANIFEST.write_text(encoded)
    print(f"V142 rendered {len(manifest_rows)} direction examples")


if __name__ == "__main__":
    main()
