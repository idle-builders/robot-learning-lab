"""Source-check the bounded V6 touch pilot and publish compact inspector cases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from free_object_touch_env import ROOT

BASE = ROOT / "artifacts/ape-low-level-training-20261015/free-object-touch-bc-v6"
DEV = BASE / "development"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> None:
    teacher_path = BASE / "teacher/report.json"
    fit_path = BASE / "fit/report.json"
    score_path = DEV / "report.json"
    video_path = DEV / "video-verification.json"
    teacher = json.loads(teacher_path.read_text())
    fit = json.loads(fit_path.read_text())
    score = json.loads(score_path.read_text())
    video = json.loads(video_path.read_text())
    assert teacher["complete"] and teacher["attempted"] == 16
    assert teacher["plan_sha256"] == sha(BASE / "plan.json")
    assert teacher["collector_source_sha256"] == sha(Path(__file__).with_name("collect_touch_pad_relative_v6.py"))
    assert teacher["adapter_source_sha256"] == sha(Path(__file__).with_name("touch_pad_relative_v6.py"))
    assert fit["ledger_sha256"] == sha(teacher_path)
    assert fit["checkpoint_sha256"] == sha(ROOT / fit["checkpoint"])
    assert fit["source_sha256"] == sha(Path(__file__).with_name("fit_touch_pad_relative_v6.py"))
    assert score["source_sha256"] == sha(Path(__file__).with_name("evaluate_touch_pad_relative_v6.py"))
    assert score["plan_sha256"] == sha(BASE / "plan.json")
    assert score["scorer_amendment_sha256"] == sha(BASE / "scorer-amendment.json")
    assert score["checkpoint_sha256"] == fit["checkpoint_sha256"]
    assert score["old_checkpoint_sha256"] == sha(ROOT / "artifacts/ape-low-level-training-20261013/free-object-touch-bc-pilot/v2/actor.pt")
    assert video["report_sha256"] == sha(score_path) and video["videos_verified"] == 16
    assert len(score["records"]) == 16 and all(score["matched_initial_state"].values())
    assert len({(r["seed"], r["mode"]) for r in score["records"]}) == 16
    for row in teacher["records"]:
        assert sha(ROOT / row["episode"]) == row["episode_sha256"]
    for row in score["records"]:
        assert sha(ROOT / row["trace"]) == row["trace_sha256"]
        assert sha(ROOT / row["video"]) == row["video_sha256"]
        with np.load(ROOT / row["trace"]) as trace:
            assert len(trace["actions"]) == row["steps"]
            assert len(trace["physical_touch"]) == row["steps"]
        case_path = DEV / f"case-seed-{row['seed']}-{row['mode']}.json"
        case_path.write_text(json.dumps({"source_report": relative(score_path),
                                         "source_report_sha256": sha(score_path),
                                         "case": row}, indent=2) + "\n")
        row["case_report"] = relative(case_path)

    cases = []
    for seed in sorted({row["seed"] for row in score["records"]}):
        modes = {row["mode"]: row for row in score["records"] if row["seed"] == seed}
        candidate = modes["candidate"]
        reasons = []
        if candidate["longest_supported_touch_s"] < .2:
            reasons.append("under_200ms_supported_touch")
        if candidate["table_frames"] < 190 or not candidate["table_at_end"]:
            reasons.append("lost_table_support")
        if candidate["hand_floor_frames"]:
            reasons.append("hand_floor_contact")
        if candidate["early_end"]:
            reasons.append("early_termination")
        cases.append({"seed": seed, "cube_xy_m": candidate["cube_xy_m"],
                      "teacher_pass": modes["teacher"]["task_pass"],
                      "old_pass": modes["old-v2"]["task_pass"],
                      "zero_pass": modes["zero"]["task_pass"],
                      "candidate_pass": candidate["task_pass"],
                      "candidate_touch_s": candidate["longest_supported_touch_s"],
                      "candidate_table_frames": candidate["table_frames"],
                      "candidate_failure": ", ".join(reasons) if reasons else "none",
                      "paired_results": [{"label": name, "result": modes[name]["case_report"]}
                                         for name in ("teacher", "old-v2", "zero", "candidate")],
                      "result": candidate["case_report"]})
    matrix = {"schema": "robot-learning-lab.cove50-touch-v6.inspector-matrix.v1",
              "score_sha256": sha(score_path), "scorer_amendment_sha256": sha(BASE / "scorer-amendment.json"),
              "cases": cases}
    (DEV / "inspector-matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    assessment = {"schema": "robot-learning-lab.cove50-touch-v6.assessment.v1",
                  "teacher_source_sha256": sha(teacher_path), "fit_source_sha256": sha(fit_path),
                  "score_source_sha256": sha(score_path), "video_check_sha256": sha(video_path),
                  "whole_teacher_families": 16, "accepted_teacher_families": teacher["accepted"],
                  "accepted_train_families": len(fit["train_seeds"]),
                  "accepted_validation_families": len(fit["validation_seeds"]),
                  "train_frames": fit["train_frames"], "validation_frames": fit["validation_frames"],
                  "cpu_optimizer_updates": fit["optimizer_updates"], "gpu_hours": 0,
                  "gpu_rental_usd": 0, "fresh_matched_families": 4,
                  "fresh_passes_by_mode": score["task_passes_by_mode"],
                  "decision": "failed_challenger_rotate_portfolio",
                  "reason": "The reloaded actor achieved 0/4 fresh one-pad touch passes; its active actions left the teacher distribution and saturated early. The teacher passed only 1/4, so demonstration coverage is also weak. Retain the observation/data evidence and rotate the next learner slot rather than tuning this single pad further.",
                  "claim_limit": "Local CPU development only; no learned contact, two-pad acquisition, lift, CUDA score or mechanical transfer."}
    (BASE / "assessment.json").write_text(json.dumps(assessment, indent=2) + "\n")
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
