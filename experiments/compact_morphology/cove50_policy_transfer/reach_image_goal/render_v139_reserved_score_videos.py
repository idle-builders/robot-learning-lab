"""Render source-matched representative pairs from the independent V139 score."""

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
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v139"
PLAN = BASE / "reserved-contact-score-plan.json"
MATRIX = BASE / "native-reserved-contact/matrix.json"
MANIFEST = BASE / "representative-videos.json"
RENDERER = "experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.render_observable_video"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    if (not matrix["complete"] or matrix["case_count"] != 24 or
            matrix["plan_sha256"] != sha(PLAN) or
            {row["id"] for row in matrix["cases"]} !=
            {case["id"] for case in plan["cases"]}):
        raise ValueError("Complete source-matched V139 score required")
    rows = matrix["cases"]
    failures = [row for row in rows if not row["candidate_pass"] or
                not row["candidate_margin_1p2_s_met"]]
    showcase = list(failures)
    for route in ("roll-low", "roll-high", "pitch-low", "pitch-high", "center"):
        matches = [row for row in rows
                   if row["route_variant"] == route and
                   row["frozen_decision"]["shift_body_y_m"] < 0]
        if matches:
            showcase.append(min(matches, key=lambda row: row["candidate_contact_s"]))
    unchanged = [row for row in rows if
                 row["frozen_decision"]["shift_body_y_m"] == 0 and
                 row["candidate_pass"]]
    if unchanged:
        showcase.append(min(unchanged, key=lambda row: row["candidate_contact_s"]))
    selected = list({row["id"]: row for row in showcase}.values())
    manifest_rows = []
    for row in selected:
        shifted = row["frozen_decision"]["shift_body_y_m"] < 0
        if shifted:
            source_check.make_pair(
                "V139_reserved_video_source_check", row, row,
                row["baseline_result"], row["candidate_result"],
                row["baseline_contact_s"], row["candidate_contact_s"],
                row["baseline_native_pass"], row["candidate_native_pass"])
        elif (row["baseline_result"] != row["candidate_result"] or
              row["baseline_trace"] != row["candidate_trace"]):
            raise ValueError(f"Unchanged choice changed native source: {row['id']}")
        videos = []
        for variant in (("baseline", "candidate") if shifted else ("baseline",)):
            report = ROOT / row[f"{variant}_result"]
            trace = ROOT / row[f"{variant}_trace"]
            if not report.is_file() or not trace.is_file():
                raise ValueError(f"Incomplete V139 trace: {row['id']} {variant}")
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
        manifest_rows.append({
            "id": row["id"], "seed": row["seed"],
            "cohort": row["cohort"], "route_variant": row["route_variant"],
            "frozen_shift_body_y_m": row["frozen_decision"]["shift_body_y_m"],
            "baseline_contact_s": row["baseline_contact_s"],
            "candidate_contact_s": row["candidate_contact_s"],
            "candidate_pass": row["candidate_pass"],
            "candidate_margin_1p2_s_met": row["candidate_margin_1p2_s_met"],
            "entry_parity_faults": row["entry_parity_faults"],
            "candidate_faults": row["candidate_faults"],
            "videos": videos,
        })
    manifest = {
        "schema": "robot-learning-lab.cove50-v139-representative-videos.v1",
        "role": "independent_reserved_contact_score_on_rejected_historical_body",
        "body_role": matrix["body_role"],
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "renderer_source_sha256": sha(Path(__file__).with_name("render_observable_video.py")),
        "selection_source_sha256": sha(Path(__file__)),
        "cases": manifest_rows,
    }
    encoded = json.dumps(manifest, indent=2) + "\n"
    if MANIFEST.exists() and MANIFEST.read_text() != encoded:
        raise ValueError("Saved V139 representative manifest changed")
    MANIFEST.write_text(encoded)
    print(f"V139 rendered {len(manifest_rows)} representative reserved cases")


if __name__ == "__main__":
    main()
