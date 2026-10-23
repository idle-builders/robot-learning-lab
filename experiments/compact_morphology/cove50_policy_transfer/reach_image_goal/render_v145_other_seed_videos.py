"""Render source-matched direction risks on other opened seed groups."""

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
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v145"
PLAN = BASE / "other-opened-seed-direction-plan.json"
MATRIX = BASE / "native-other-opened-seed-directions/matrix.json"
SUMMARY = BASE / "other-opened-direction-summary.json"
MANIFEST = BASE / "representative-videos.json"
RENDERER = "experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.render_observable_video"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    matrix, summary = json.loads(MATRIX.read_text()), json.loads(SUMMARY.read_text())
    if (not matrix["complete"] or matrix["case_count"] != 11 or
            matrix["plan_sha256"] != sha(PLAN) or
            summary["matrix_sha256"] != sha(MATRIX) or
            summary["source_checked_seed_count"] != 11):
        raise ValueError("Complete source-checked V145 other-seed comparison required")
    rows = matrix["cases"]
    failed = [row for row in rows if
              not row["choices"]["neg2"]["pass"] or
              not row["choices"]["pos2"]["pass"] or
              not row["choices"]["neg2"]["margin_1p2_s_met"] or
              not row["choices"]["pos2"]["margin_1p2_s_met"]]
    selected = failed[:3]
    for route in ("pitch-low", "roll-high"):
        group = [row for row in rows if row["route_variant"] == route]
        selected.append(min(group, key=lambda row:
                            row["choices"]["pos2"]["contact_s"] -
                            row["baseline_contact_s"]))
    selected = list({row["id"]: row for row in selected}.values())[:5]
    manifest_rows = []
    for row in selected:
        base_report = json.loads((ROOT / row["baseline_result"]).read_text())
        observed = score_runner.observations.observed_features(base_report)
        videos = []
        for variant, report_rel, trace_rel, contact, passed, faults, shift, reused in (
                ("unchanged", row["baseline_result"], row["baseline_trace"],
                 row["baseline_contact_s"], row["baseline_pass"],
                 row["baseline_faults"], 0, True),
                *((label, choice["result"], choice["trace"],
                   choice["contact_s"], choice["pass"], choice["faults"],
                   choice["shift_body_y_m"], choice["reused_from_v139"])
                  for label, choice in row["choices"].items())):
            report, trace = ROOT / report_rel, ROOT / trace_rel
            saved = json.loads(report.read_text())
            if (not trace.is_file() or saved["seed"] != row["seed"] or
                    abs(saved["phases"]["reach"]["continuous_finger_contact_s"] -
                        contact) > 1e-7 or
                    abs(saved.get("diagnostic_goal_shift_body_y_m", 0) - shift) > 1e-9 or
                    score_runner.entry_parity(
                        observed, score_runner.observations.observed_features(saved))):
                raise ValueError(f"V145 video source differs: {row['id']} {variant}")
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
            videos.append({"variant": variant, "path": str(video.relative_to(ROOT)),
                           "sha256": sha(video),
                           "report_sha256": sha(report),
                           "trace_sha256": sha(trace),
                           "contact_s": contact, "full_gate_pass": passed,
                           "faults": faults, "reused_from_v139": reused})
        manifest_rows.append({"id": row["id"], "seed": row["seed"],
                              "route_variant": row["route_variant"],
                              "videos": videos})
    manifest = {
        "schema": "robot-learning-lab.cove50-v145-representative-videos.v1",
        "role": "opened_other_seed_direction_retention_not_independent_score",
        "body_role": matrix["body_role"],
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "summary_sha256": sha(SUMMARY),
        "renderer_source_sha256": sha(Path(__file__).with_name("render_observable_video.py")),
        "selection_source_sha256": sha(Path(__file__)),
        "cases": manifest_rows,
    }
    encoded = json.dumps(manifest, indent=2) + "\n"
    if MANIFEST.exists() and MANIFEST.read_text() != encoded:
        raise ValueError("Saved V145 representative manifest changed")
    MANIFEST.write_text(encoded)
    print(f"V145 rendered {len(manifest_rows)} other opened seed pairs")


if __name__ == "__main__":
    main()
