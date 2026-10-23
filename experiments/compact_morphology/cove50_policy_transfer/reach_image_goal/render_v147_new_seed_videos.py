"""Render source-matched new-seed pass, rescue and failure comparisons."""

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
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v147"
PLAN = BASE / "new-seed-direction-training-plan.json"
MATRIX = BASE / "native-new-seed-direction-pairs/matrix.json"
SUMMARY = BASE / "new-seed-paired-summary.json"
MANIFEST = BASE / "representative-videos.json"
RENDERER = "experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.render_observable_video_poststep"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    summary = json.loads(SUMMARY.read_text())
    if (not matrix["complete"] or matrix["case_count"] != 40 or
            matrix["plan_sha256"] != sha(PLAN) or
            summary["matrix_sha256"] != sha(MATRIX) or
            summary["source_checked_native_visit_count"] != 120):
        raise ValueError("Complete source-checked V147 paired training required")
    rows = matrix["cases"]
    chosen = []

    def first(predicate):
        return next((row for row in rows if predicate(row)), None)

    chosen.extend([
        first(lambda r: not r["baseline_pass"] and r["choices"]["neg2"]["pass"]),
        first(lambda r: not r["baseline_pass"] and r["choices"]["pos2"]["pass"]),
        first(lambda r: not r["baseline_margin_1p2_s_met"] and
              r["choices"]["neg2"]["margin_1p2_s_met"]),
        first(lambda r: not r["baseline_margin_1p2_s_met"] and
              r["choices"]["pos2"]["margin_1p2_s_met"]),
        min(rows, key=lambda r: r["choices"]["neg2"]["contact_delta_s"]),
        min(rows, key=lambda r: r["choices"]["pos2"]["contact_delta_s"]),
        min(rows, key=lambda r: r["baseline_contact_s"]),
    ])
    for route in ("roll-high", "pitch-low"):
        chosen.append(max((row for row in rows if row["route_variant"] == route),
                          key=lambda r: r["baseline_contact_s"]))
    selected = list({row["id"]: row for row in chosen if row is not None}.values())[:8]
    manifest_rows = []
    for row in selected:
        base_report = json.loads((ROOT / row["baseline_result"]).read_text())
        observed = score_runner.observations.observed_features(base_report)
        videos = []
        for variant, report_rel, trace_rel, contact, passed, faults, shift in (
                ("unchanged", row["baseline_result"], row["baseline_trace"],
                 row["baseline_contact_s"], row["baseline_pass"],
                 row["baseline_faults"], 0),
                *((label, choice["result"], choice["trace"],
                   choice["contact_s"], choice["pass"], choice["faults"],
                   choice["shift_body_y_m"])
                  for label, choice in row["choices"].items())):
            report, trace = ROOT / report_rel, ROOT / trace_rel
            saved = json.loads(report.read_text())
            if (not trace.is_file() or saved["seed"] != row["seed"] or
                    saved["body_sha256"] != plan["source_sha256"]["body"] or
                    abs(saved["phases"]["reach"]["continuous_finger_contact_s"] -
                        contact) > 1e-7 or
                    abs(saved.get("diagnostic_goal_shift_body_y_m", 0) - shift) > 1e-9 or
                    score_runner.entry_parity(
                        observed, score_runner.observations.observed_features(saved))):
                raise ValueError(f"V147 video source differs: {row['id']} {variant}")
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
                           "report_sha256": sha(report),
                           "trace_sha256": sha(trace),
                           "contact_s": contact,
                           "full_gate_pass": passed, "faults": faults})
        manifest_rows.append({"id": row["id"], "seed": row["seed"],
                              "route_variant": row["route_variant"],
                              "videos": videos})
    manifest = {
        "schema": "robot-learning-lab.cove50-v147-representative-videos.v2",
        "role": "new_seed_paired_training_not_independent_score",
        "body_role": matrix["body_role"],
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "summary_sha256": sha(SUMMARY),
        "renderer_source_sha256": sha(Path(__file__).with_name("render_observable_video_poststep.py")),
        "selection_source_sha256": sha(Path(__file__)),
        "video_alignment": ("The saved trace records pre-step qpos; frame row i "
                            "renders qpos[i+1] beside its post-step contact "
                            "and hand-position report, checked to 0.02 mm."),
        "cases": manifest_rows,
    }
    encoded = json.dumps(manifest, indent=2) + "\n"
    if MANIFEST.exists() and MANIFEST.read_text() != encoded:
        raise ValueError("Saved V147 video manifest changed")
    MANIFEST.write_text(encoded)
    print(f"V147 rendered {len(manifest_rows)} source-matched paired video sets")


if __name__ == "__main__":
    main()
