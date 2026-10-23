"""Build a source-checked, command-visible whole-episode handoff seed bank.

The saved V3 composition is a teacher diagnostic. This converts only its
physical-gate passes to labels; it does not fit or score a new policy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-cove50-low-level-handoffs/walk-squat-arms-cuda-v3"
AUDIT = ROOT / "artifacts/ape-low-level-training-20261015/broad-handoff-bank-audit/report.json"
OUT = ROOT / "artifacts/ape-low-level-training-20261015/broad-handoff-command-bank"
COMMANDS = ("requested_vx", "requested_yaw", "squat_request", "arm_request",
            "effective_previous_arm_request", "stop_request")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_features(row: dict, episode: dict, arm_obs: np.ndarray,
                   previous_action: np.ndarray) -> np.ndarray:
    command = np.column_stack((
        np.where(np.asarray([r["phase"] == "walk" for r in row]),
                 float(episode["walk_velocity_m_s"]), 0.0),
        np.zeros(len(row)),
        np.asarray([r["squat_requested"] for r in row], dtype=np.float32),
        np.asarray([r["arm_requested"] for r in row], dtype=np.float32),
        arm_obs[:, 69],
        np.asarray([r["phase"] == "stop" for r in row], dtype=np.float32),
    )).astype(np.float32)
    return np.concatenate((command, previous_action), axis=1)


def main() -> None:
    audit = json.loads(AUDIT.read_text())
    if sha(SOURCE / "source-verification.json") != audit["source_verification_sha256"]:
        raise RuntimeError("V3 verification changed after audit")
    if audit["complete_cuda_episodes"] != 4 or audit["full_gate_accepted_whole_episodes"] != 2:
        raise RuntimeError("V3 source pool changed")
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for item in audit["rows"]:
        report_path, trace_path = ROOT / item["report"], ROOT / item["trace"]
        if sha(report_path) != item["report_sha256"] or sha(trace_path) != item["trace_sha256"]:
            raise RuntimeError(f"Source changed for seed {item['seed']}")
        report = json.loads(report_path.read_text())
        with np.load(trace_path, allow_pickle=False) as trace:
            obs = trace["actor_obs"].copy()
            arm_obs = trace["arm_actor_obs"].copy()
            action = trace["applied_action"].copy()
            target = trace["processed_target"].copy()
            pre = trace["pre_qpos"].copy()
            post = trace["post_qpos"].copy()
        if len(report["timeseries"]) != len(obs) or obs.shape != (1200, 72):
            raise RuntimeError(f"Episode length/observation changed: {item['seed']}")
        if np.max(np.abs(post[:-1] - pre[1:])) > 1e-9:
            raise RuntimeError(f"Physics gap in seed {item['seed']}")
        if item["full_physical_pass"] != report["full_continuity_gate_pass"]:
            raise RuntimeError(f"Gate changed for seed {item['seed']}")
        previous_action = np.concatenate((np.zeros((1, 21), np.float32), action[:-1]), axis=0)
        extra = build_features(report["timeseries"], item, arm_obs, previous_action)
        features = np.concatenate((obs, extra), axis=1).astype(np.float32)
        if features.shape != (1200, 99) or target.shape != (1200, 21):
            raise RuntimeError(f"Feature/target shape changed: {item['seed']}")
        if not np.isfinite(features).all() or not np.isfinite(target).all():
            raise RuntimeError(f"Nonfinite label: {item['seed']}")
        bank_path = None
        if item["full_physical_pass"]:
            bank_path = OUT / f"seed-{item['seed']}.npz"
            np.savez_compressed(bank_path, features=features, physical_target=target,
                                applied_action=action, pre_qpos=pre, post_qpos=post,
                                seed=np.asarray(item["seed"]), dt_s=np.asarray(0.02))
        entries.append({
            "seed": item["seed"], "full_physical_pass": item["full_physical_pass"],
            "steps": len(obs), "failure_reasons": item["failure_reasons"],
            "max_abs_action": float(np.max(np.abs(action))),
            "max_abs_physical_target": float(np.max(np.abs(target))),
            "bank": str(bank_path.relative_to(ROOT)) if bank_path else None,
            "bank_sha256": sha(bank_path) if bank_path else None,
            "report_sha256": item["report_sha256"], "trace_sha256": item["trace_sha256"],
        })
    result = {
        "schema": "robot-learning-lab.broad-handoff-command-bank.v1",
        "classification": "offline supervised seed bank only; no new native visit, fit or skill score",
        "source_audit_sha256": sha(AUDIT),
        "feature_contract": {"base_actor_obs": 72, "commands": COMMANDS,
                             "previous_applied_action": 21, "total": 99,
                             "arm_effective_command_is_one_step_delayed": True},
        "label_contract": "21 physical processed joint targets after saved actor arbitration and action filtering; no generic unit clamp",
        "accepted_whole_episodes": sum(e["full_physical_pass"] for e in entries),
        "accepted_frames": sum(e["steps"] for e in entries if e["full_physical_pass"]),
        "ready_for_shared_fit": False,
        "optimizer_updates": 0, "gpu_hours": 0,
        "use_limit": "Only two independent passing seeds; use to check the learner interface, not to claim broad trained control or held evaluation.",
        "entries": entries,
    }
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
