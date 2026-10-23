"""Render verified V122 paired route contact visits for the local inspector."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v122"
PLAN = BASE / "near-route-pairs-plan.json"
MATRIX = BASE / "native-near-route-pairs/matrix.json"
INSPECTOR = BASE / "pair-inspector-matrix.json"
RENDERER = "experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.render_observable_video"


def main() -> None:
    if not MATRIX.is_file():
        return
    try:
        matrix = json.loads(MATRIX.read_text())
    except json.JSONDecodeError:
        return
    if matrix["plan_sha256"] != hashlib.sha256(PLAN.read_bytes()).hexdigest():
        raise ValueError("V122 route matrix differs from frozen plan")
    pairs = {pair["id"]: pair for pair in matrix["pairs"]}
    display = {"schema": "robot-learning-lab.cove50-v122-inspector.v1",
               "role": matrix["role"], "body_role": matrix["body_role"],
               "native_matrix_sha256": hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
               "plan_sha256": matrix["plan_sha256"],
               "complete": matrix["complete"],
               "case_count": matrix["case_count"],
               "pair_count": matrix["pair_count"],
               "decision_gate_met": matrix["decision_gate_met"],
               "cases": []}
    for row in matrix["cases"]:
        label, route = row["id"].rsplit("-", 1)
        pair = pairs.get(label)
        display["cases"].append({**row,
                                 "comparison_label": label,
                                 "route_choice": route,
                                 "paired_entry_matched": pair["entry_matched"] if pair else None,
                                 "pair_contact_delta_s": pair["contact_delta_s"] if pair else None,
                                 "opposite_contact_s": (pair["forced_contact_s"]
                                                        if route == "automatic" else
                                                        pair["automatic_contact_s"])
                                                        if pair else None})
    temporary_inspector = INSPECTOR.with_suffix(".json.tmp")
    temporary_inspector.write_text(json.dumps(display, indent=2) + "\n")
    os.replace(temporary_inspector, INSPECTOR)
    for row in matrix["cases"]:
        report, trace = ROOT / row["result"], ROOT / row["trace"]
        video = report.with_suffix(".mp4")
        if video.is_file() or not report.is_file() or not trace.is_file():
            continue
        temporary = video.with_name(video.stem + ".tmp.mp4")
        temporary.unlink(missing_ok=True)
        try:
            subprocess.run([sys.executable, "-m", RENDERER,
                            "--report", str(report), "--trace", str(trace),
                            "--video", str(temporary)], cwd=ROOT, check=True,
                           capture_output=True, text=True)
            os.replace(temporary, video)
            print(f"published {video.name}", flush=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            temporary.unlink(missing_ok=True)
            print(f"waiting for complete trace {row['id']}: {exc}", flush=True)


if __name__ == "__main__":
    main()
