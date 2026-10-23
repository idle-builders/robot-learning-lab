"""Render one V147 triplet with post-step video and contact rows aligned."""

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
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v147"
PLAN = BASE / "new-seed-direction-training-plan.json"
MATRIX = BASE / "native-new-seed-direction-pairs/matrix.json"
RENDERER = (
    "experiments.compact_morphology.cove50_policy_transfer."
    "reach_image_goal.render_observable_video_poststep"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    if matrix["plan_sha256"] != sha(PLAN):
        raise ValueError("V147 native matrix differs from its frozen plan")
    row = next((case for case in matrix["cases"]
                if case["seed"] == args.seed), None)
    if row is None or row["seed"] not in {case["seed"] for case in plan["cases"]}:
        raise ValueError("Requested V147 triplet is not complete")
    baseline = json.loads((ROOT / row["baseline_result"]).read_text())
    observed = score_runner.observations.observed_features(baseline)
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
        expected_evaluator = plan["source_sha256"][
            "evaluator" if variant == "unchanged" else "shift_evaluator"]
        if (not trace.is_file() or saved["seed"] != row["seed"] or
                saved["evaluator_sha256"] != expected_evaluator or
                saved["selected_pointer_sha256"] !=
                plan["source_sha256"]["pointer"] or
                saved["body_sha256"] != plan["source_sha256"]["body"] or
                abs(saved["phases"]["reach"]["continuous_finger_contact_s"] -
                    contact) > 1e-7 or
                abs(saved.get("diagnostic_goal_shift_body_y_m", 0) - shift) > 1e-9 or
                score_runner.entry_parity(
                    observed, score_runner.observations.observed_features(saved))):
            raise ValueError(f"V147 paired source differs: {args.seed} {variant}")
        video = report.with_name(report.stem + "-aligned.mp4")
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
        videos.append({
            "variant": variant, "path": str(video.relative_to(ROOT)),
            "sha256": sha(video), "report": report_rel,
            "report_sha256": sha(report), "trace": trace_rel,
            "trace_sha256": sha(trace), "contact_s": contact,
            "full_gate_pass": passed, "margin_1p2_s_met": margin,
            "faults": faults,
        })
    manifest = {
        "schema": "robot-learning-lab.cove50-v147-aligned-triplet-videos.v1",
        "role": "source_checked_partial_training_comparison_not_independent_score",
        "body_role": matrix["body_role"], "plan_sha256": sha(PLAN),
        "row_id": row["id"], "seed": args.seed,
        "row_sha256": hashlib.sha256(json.dumps(
            row, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "renderer_source_sha256": sha(Path(__file__).with_name(
            "render_observable_video_poststep.py")),
        "selection_source_sha256": sha(Path(__file__)),
        "video_alignment": ("The saved trace records pre-step qpos; frame row i "
                            "renders qpos[i+1] beside its post-step contact "
                            "and hand-position report, checked to 0.02 mm."),
        "note": ("Saved native CUDA traces rendered locally; this does not "
                 "create a CPU contact score. V147 training is incomplete "
                 "and its independent score reserve remains unopened."),
        "videos": videos,
    }
    output = BASE / f"aligned-comparison-{args.seed}-videos.json"
    encoded = json.dumps(manifest, indent=2) + "\n"
    if output.exists() and output.read_text() != encoded:
        raise ValueError("Saved V147 partial video manifest changed")
    output.write_text(encoded)
    print(f"V147 aligned source-matched triplet videos: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
