"""Audit retained native handoff traces as possible whole-episode BC data."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-cove50-low-level-handoffs/walk-squat-arms-cuda-v3"
METHOD = ROOT / "artifacts/ape-cove50-low-level-handoffs/walk-squat-arms-cuda-method.json"
OUT = ROOT / "artifacts/ape-low-level-training-20261015/broad-handoff-bank-audit"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    verification = json.loads((SOURCE / "source-verification.json").read_text())
    if sha(METHOD) != verification["source_method_sha256"]:
        raise RuntimeError("Saved V3 method changed")
    rows = []
    for entry in verification["rows"]:
        if entry["unscored"]:
            continue
        report_path = Path(entry["report"])
        trace_path = Path(entry["trace"])
        if not report_path.is_absolute():
            report_path = ROOT / report_path
        if not trace_path.is_absolute():
            trace_path = ROOT / trace_path
        if sha(report_path) != entry["report_sha256"] or sha(trace_path) != entry["trace_sha256"]:
            raise RuntimeError(f"Saved report/trace source mismatch: {entry['seed']}")
        report = json.loads(report_path.read_text())
        with np.load(trace_path, allow_pickle=False) as trace:
            obs = trace["actor_obs"].copy()
            arm_obs = trace["arm_actor_obs"].copy()
            action = trace["applied_action"].copy()
            target = trace["processed_target"].copy()
        if (obs.shape != (1200, 72) or arm_obs.shape != obs.shape
                or action.shape != (1200, 21) or target.shape != action.shape
                or not all(np.isfinite(v).all() for v in (obs, arm_obs, action, target))
                or len(report["timeseries"]) != len(obs)):
            raise RuntimeError(f"Invalid full trace: {entry['seed']}")
        phases = np.asarray([step["phase"] for step in report["timeseries"]])
        arm = phases == "arms"
        raising = np.asarray([step["arm_requested"] for step in report["timeseries"]], dtype=bool)
        if not np.any(arm) or not np.all(obs[arm, 69] == 0):
            raise RuntimeError(f"Arm mode visibility assumption changed: {entry['seed']}")
        shoulders = report["shoulder_action_indices"]
        if len(shoulders) != 2 or not np.any(raising):
            raise RuntimeError(f"Shoulder/raise source changed: {entry['seed']}")
        rows.append({
            "seed": entry["seed"], "rail_v": entry["rail_v"],
            "delay_ms": entry["delay_ms"], "walk_velocity_m_s": entry["walk_velocity_m_s"],
            "full_physical_pass": bool(entry["pass"]),
            "motion_completed": bool(report["squat_depth_return_pass"] and report["arm_rise_return_pass"]),
            "failure_reasons": entry["failure_reasons"],
            "steps": len(obs), "phase_steps": dict(Counter(phases.tolist())),
            "actor_obs_width": obs.shape[1], "action_width": action.shape[1],
            "arm_mode_actor_obs_slot69_unique": np.unique(obs[arm, 69]).tolist(),
            "arm_mode_arm_actor_obs_slot69_max": float(np.max(arm_obs[arm, 69])),
            "max_abs_applied_action": float(np.max(np.abs(action))),
            "arm_raise_median_abs_shoulder_action": float(np.median(np.abs(action[raising][:, shoulders]))),
            "report": str(report_path.relative_to(ROOT)), "report_sha256": entry["report_sha256"],
            "trace": str(trace_path.relative_to(ROOT)), "trace_sha256": entry["trace_sha256"],
        })
    if len(rows) != 4:
        raise RuntimeError("Expected all four scored V3 episodes")
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "robot-learning-lab.broad-handoff-bank-audit.v1",
        "classification": "local source/label audit only; no new learner, native visit or skill score",
        "source_method_sha256": sha(METHOD),
        "source_verification_sha256": sha(SOURCE / "source-verification.json"),
        "complete_cuda_episodes": len(rows),
        "full_gate_accepted_whole_episodes": sum(r["full_physical_pass"] for r in rows),
        "motion_completed_episodes": sum(r["motion_completed"] for r in rows),
        "full_gate_accepted_frames": sum(r["steps"] for r in rows if r["full_physical_pass"]),
        "new_optimizer_updates": 0, "new_gpu_hours": 0,
        "observation_contract_issue": "A single 72-value actor observation does not expose the requested mode: slot 69 is gait velocity, then squat blend, and exactly zero during the arm stage, while the arm actor receives a separately injected prior raise flag. A new deployable mode/command interface is needed before a shared student fit.",
        "action_contract_issue": "Applied 21-channel action includes intentional shoulder values above 1 after physical-target conversion. A generic [-1,1] action clamp would change native targets.",
        "data_gate": "Two whole full-gate CUDA episodes are too few independent start families for a robust handoff learner; collect more source-valid varied no-reset trajectories and reserve whole seeds before fitting.",
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
