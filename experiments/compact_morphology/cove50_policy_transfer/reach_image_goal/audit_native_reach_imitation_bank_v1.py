"""Audit saved V152/V158 successful native visits before any reach imitation fit."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
SOURCES = (
    BASE / "v152/native-paired-routes/matrix.json",
    BASE / "v158/source-checked-development-contact-inspector-matrix.json",
)
OUTPUT = BASE / "native-reach-imitation-bank-v1/audit.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    v152 = json.loads(SOURCES[0].read_text())
    v158 = json.loads(SOURCES[1].read_text())
    # V152 retains the first RGB-mismatched original as invalid: 47/48 valid.
    if (v152["complete"] or v152["planned_count"] != 48 or
            len(v152["cases"]) != 47 or not v158["complete"]):
        raise ValueError("Unexpected V152 invalid-original or V158 completion state")
    candidates = []
    for case in v152["cases"]:
        for prefix in ("baseline", "prescribed"):
            if case[f"{prefix}_pass"] and case[f"{prefix}_margin_1p2_s_met"]:
                candidates.append(("v152", case["seed"], case[f"{prefix}_report"],
                                   case[f"{prefix}_trace"],
                                   case[f"{prefix}_report_sha256"],
                                   case[f"{prefix}_trace_sha256"]))
    for case in v158["cases"]:
        if not case["matched_pair_valid"]:
            continue
        for prefix in ("automatic", "prescribed"):
            if case[f"{prefix}_native_pass"] and case[f"{prefix}_margin_1p2_s_met"]:
                candidates.append(("v158", case["seed"], case[f"{prefix}_report"],
                                   case[f"{prefix}_trace"], None, None))
    rows = []
    for cohort, seed, report_rel, trace_rel, report_sha, trace_sha in candidates:
        report_path, trace_path = ROOT / report_rel, ROOT / trace_rel
        if report_sha and sha(report_path) != report_sha:
            raise ValueError(f"Changed report: {report_rel}")
        if trace_sha and sha(trace_path) != trace_sha:
            raise ValueError(f"Changed trace: {trace_rel}")
        report = json.loads(report_path.read_text())
        with np.load(trace_path) as trace:
            qpos = trace["qpos"]
            action = trace["action"]
            phase = trace["phase"]
            dt = float(trace["step_dt_s"])
            if (qpos.shape[0] != action.shape[0] or
                    len(phase) != len(action) or
                    len(report["timeseries"]) != len(action) or
                    action.shape[1] != 21 or not np.isfinite(action).all() or
                    not np.isfinite(qpos).all()):
                raise ValueError(f"Trace shape or finiteness fault: {trace_rel}")
            reach_indices = np.flatnonzero(phase == "reach")
            if len(reach_indices) != 450 or not np.all(np.diff(reach_indices) == 1):
                raise ValueError(f"Incomplete reach: {trace_rel}")
            times = np.asarray([r["time_s"] for r in report["timeseries"]])
            if (np.max(np.abs(times - (np.arange(len(times)) + 1) * dt)) > 1e-6 or
                    any(report["timeseries"][i]["phase"] != phase[i] for i in range(len(phase)))):
                raise ValueError(f"Post-step report / pre-step trace index fault: {report_rel}")
            reach = [report["timeseries"][i] for i in reach_indices]
            error = np.asarray([r["controller_target_error_xy_m"] for r in reach])
            correction = np.asarray([r["learned_arm_residual_raw"] for r in reach])
            age = np.asarray([r["image_goal_age_s"] for r in reach])
            if (error.shape != (450, 2) or correction.shape != (450, 4) or
                    not np.isfinite(error).all() or not np.isfinite(correction).all() or
                    not np.isfinite(age).all()):
                raise ValueError(f"Missing actor-feature trajectory: {report_rel}")
            used_captures = [c for c in report["image_goal_capture_events"]
                             if c.get("used_for_control")]
            rows.append({"cohort": cohort, "seed": seed, "report": report_rel,
                         "report_sha256": sha(report_path), "trace": trace_rel,
                         "trace_sha256": sha(trace_path), "steps": len(action),
                         "reach_steps": len(reach_indices),
                         "reach_first_trace_index": int(reach_indices[0]),
                         "used_goal_captures": len(used_captures),
                         "max_goal_age_s": float(age.max()),
                         "selected_actor_sha256": report["selected_reach_checkpoint_sha256"],
                         "selection_mode": report["selection_mode"]})
    if len(rows) != 114 or len({r["seed"] for r in rows}) != 65:
        raise ValueError(f"Unexpected successful visit/family pool: {len(rows)} / {len({r['seed'] for r in rows})}")
    result = {"schema": "robot-learning-lab.native-reach-imitation-bank-audit.v1",
              "claim_limit": "source-aligned candidate donor visits only; no measured new actor, corrective teacher, raw RGB sequence, optimizer update or CUDA score",
              "source_matrices": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in SOURCES],
              "source_sha256": sha(Path(__file__)), "candidate_visits": len(rows),
              "seed_families": len({r["seed"] for r in rows}),
              "total_reach_pre_step_actions": sum(r["reach_steps"] for r in rows),
              "goal_capture_used_distribution": dict(Counter(r["used_goal_captures"] for r in rows)),
              "max_saved_goal_age_s": max(r["max_goal_age_s"] for r in rows),
              "selection_mode_distribution": dict(Counter(r["selection_mode"] for r in rows)),
              "visits": rows}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("candidate_visits", "seed_families",
                                                "total_reach_pre_step_actions",
                                                "goal_capture_used_distribution",
                                                "max_saved_goal_age_s")}, indent=2))


if __name__ == "__main__":
    main()
