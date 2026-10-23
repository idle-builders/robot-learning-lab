"""Source-check four fresh matched V5 learner/teacher no-reset originals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from v4_action_map import processed


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Original score summary exists")
    plan = json.loads(args.plan.read_text())
    fit = json.loads(args.fit.read_text())
    if (plan["schema"] != "robot-learning-lab.broad-handoff-v5-fresh-score-plan.v1"
            or fit["schema"] != "robot-learning-lab.broad-handoff-v5-shared-bc-fit.v1"):
        raise RuntimeError("Score plan or fit changed")
    rows = []
    for case in plan["cases"]:
        seed = case["seed"]
        pair = {}
        for role in ("teacher", "student"):
            report_path = args.source / f"{role}-{seed}.json"
            trace_path = args.source / f"{role}-{seed}.npz"
            report = json.loads(report_path.read_text())
            if (report["seed"] != seed or report["body_sha256"] != plan["body_sha256"]
                    or report["rail_v"] != case["rail_v"]
                    or report["delay_ms"] != case["delay_ms"]
                    or report["walk_velocity_m_s"] != case["walk_velocity_m_s"]
                    or report["device"] != "cuda:0" or report["steps"] != len(report["timeseries"])):
                raise RuntimeError(f"{role} source identity changed: {seed}")
            if role == "student" and (report["checkpoint_schema"] != "robot-learning-lab.broad-handoff-v5-shared-bc.v1"
                                      or report["student_checkpoint_sha256"] != fit["checkpoint_sha256"]):
                raise RuntimeError("Student fit/checkpoint identity changed")
            with np.load(trace_path, allow_pickle=False) as trace:
                pre, post = trace["pre_qpos"], trace["post_qpos"]
                action, target = trace["applied_action"], trace["processed_target"]
                names = trace["target_names"].tolist()
                initial_qpos, initial_qvel = trace["initial_qpos"], trace["initial_qvel"]
            if (pre.shape[0] != report["steps"] or post.shape != pre.shape
                    or action.shape != (report["steps"], 21)
                    or target.shape != action.shape
                    or np.max(np.abs(pre[0] - initial_qpos)) > 1e-9
                    or np.max(np.abs(post[:-1] - pre[1:])) > 1e-9
                    or np.max(np.abs(processed(action, names) - target)) > 1e-5):
                raise RuntimeError(f"{role} trace parity changed: {seed}")
            pair[role] = {"report": report, "report_sha256": sha(report_path),
                          "trace_sha256": sha(trace_path), "initial_qpos": initial_qpos,
                          "initial_qvel": initial_qvel}
        qpos_gap = float(np.max(np.abs(pair["teacher"]["initial_qpos"] - pair["student"]["initial_qpos"])))
        qvel_gap = float(np.max(np.abs(pair["teacher"]["initial_qvel"] - pair["student"]["initial_qvel"])))
        if qpos_gap > 1e-4 or qvel_gap > 1e-4:
            raise RuntimeError(f"Matched start differs: {seed}")
        t, s = pair["teacher"]["report"], pair["student"]["report"]
        rows.append({"seed": seed, "case": case, "initial_qpos_max_gap": qpos_gap,
                     "initial_qvel_max_gap": qvel_gap,
                     "teacher_pass": t["full_continuity_gate_pass"],
                     "student_pass": s["full_continuity_gate_pass"],
                     "teacher_steps": t["steps"], "student_steps": s["steps"],
                     "teacher_contact_pass": t["no_hand_or_body_contact"],
                     "student_contact_pass": s["no_hand_or_body_contact"],
                     "teacher_duty_pass": t["rated_duty_windows_pass"],
                     "student_duty_pass": s["rated_duty_windows_pass"],
                     "teacher_squat_pass": t["squat_depth_return_pass"],
                     "student_squat_pass": s["squat_depth_return_pass"],
                     "teacher_arm_pass": t["arm_rise_return_pass"],
                     "student_arm_pass": s["arm_rise_return_pass"],
                     "source": {role: {key: pair[role][key] for key in ("report_sha256", "trace_sha256")}
                                for role in ("teacher", "student")}})
    result = {"schema": "robot-learning-lab.broad-handoff-v5-fresh-score.v1",
              "classification": "fresh paired learned-policy native rollout on experimental Direct Frame; no physical transfer",
              "plan_sha256": sha(args.plan), "fit_report_sha256": sha(args.fit),
              "completed_pairs": len(rows), "teacher_passes": sum(row["teacher_pass"] for row in rows),
              "student_passes": sum(row["student_pass"] for row in rows),
              "student_rescues": sum(row["student_pass"] and not row["teacher_pass"] for row in rows),
              "student_regressions": sum(row["teacher_pass"] and not row["student_pass"] for row in rows),
              "rows": rows}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("completed_pairs", "teacher_passes", "student_passes", "student_rescues", "student_regressions")}))


if __name__ == "__main__":
    main()
