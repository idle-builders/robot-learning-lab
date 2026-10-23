"""Source-check the unchanged posture reward on one opened pass/fall pair."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "artifacts/ape-cove50-stand-recovery/reward-counterfactual-v8"
BASELINE = ROOT / "artifacts/ape-cove50-stand-recovery/seat-command-development-v5"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = ARTIFACT / "plan.json"
    plan = json.loads(plan_path.read_text())
    assert digest(ARTIFACT / "source/evaluate_posture.py") == plan["source_snapshot_sha256"]
    rows = []
    for case in plan["cases"]:
        seed = case["seed"]
        report_path = ARTIFACT / f"seed{seed}.json"
        native_path = ARTIFACT / f"seed{seed}.npz"
        source = json.loads(report_path.read_text())
        old = json.loads((BASELINE / f"seed{seed}.json").read_text())
        assert source["initial_state_sha256"] == old["initial_state_sha256"]
        assert source["first_action_sha256"] == old["first_action_sha256"]
        with np.load(native_path) as new, np.load(BASELINE / f"seed{seed}.npz") as prior:
            assert all(np.array_equal(new[name], prior[name]) for name in prior.files)
            rate = new["reward_rate"].copy()
        names = source["reward_term_names"]
        assert rate.shape == (500, len(names))
        assert len(source["reward_term_weights"]) == len(names)
        sum_rate = rate.sum(axis=1) * 0.02
        recorded = np.asarray([sample["step_reward"] for sample in source["timeseries"]])
        # Text traces round each step to 4 decimals; retain the full-rate
        # arrays for the authoritative cumulative reward and term ranking.
        assert np.max(np.abs(sum_rate - recorded)) <= 5.1e-5
        original_fields = source["timeseries"][0].keys() & old["timeseries"][0].keys()
        assert all(all(new[key] == before[key] for key in original_fields if key != "body_contact")
                   for new, before in zip(source["timeseries"], old["timeseries"], strict=True))
        contribution = rate.sum(axis=0) * 0.02
        row = {
            "seed": seed,
            "physical_outcome": case["v5_outcome"],
            "source_report_sha256": digest(report_path),
            "native_trajectory_sha256": digest(native_path),
            "full_v5_action_and_state_arrays_exact": True,
            "unrounded_total_reward": float(contribution.sum()),
            "after_1s_total_reward": float(rate[50:].sum() * 0.02),
            "after_1s_mean_reward_per_step": float(rate[50:].sum(axis=1).mean() * 0.02),
            "positive_after_1s_reward_steps": int(np.count_nonzero(rate[50:].sum(axis=1) > 0)),
            "term_contributions": {name: float(value) for name, value in zip(names, contribution, strict=True)},
            "after_1s_term_contributions": {name: float(value) for name, value in zip(names, rate[50:].sum(axis=0) * 0.02, strict=True)},
            "native_floor_nonsole_body_contact_frames": sum(sample["floor_nonsole_body_contact"] for sample in source["timeseries"]),
        }
        rows.append(row)
    passed = next(row for row in rows if row["physical_outcome"] == "pass")
    fallen = next(row for row in rows if row["physical_outcome"] == "fall")
    output = {
        "schema": "robot-learning-lab.cove50-seated-reward-audit-report.v8",
        "status": "complete_one_opened_pass_fall_pair_no_reward_promotion",
        "plan_sha256": digest(plan_path),
        "source_snapshot_sha256": digest(ARTIFACT / "source/evaluate_posture.py"),
        "scorer_source_sha256": digest(Path(__file__)),
        "runtime": "two local CPU 10 s native replays, zero paid GPU time",
        "summary": {
            "pass_total_reward": passed["unrounded_total_reward"],
            "fall_total_reward": fallen["unrounded_total_reward"],
            "pass_minus_fall_total_reward": passed["unrounded_total_reward"] - fallen["unrounded_total_reward"],
            "fall_after_1s_reward": fallen["after_1s_total_reward"],
            "fall_after_1s_positive_steps": fallen["positive_after_1s_reward_steps"],
            "pass_ranks_above_fall": passed["unrounded_total_reward"] > fallen["unrounded_total_reward"],
        },
        "rows": rows,
        "interpretation": "Whole return ranks this pass above this fall, but the prone trajectory still earns positive reward at every step after 1s because leg-pose and counterbalance terms pay while the body is down. The task's fall/contact terminations are disabled. One paired audit is insufficient to approve the full reward/value contract or paid PPO.",
    }
    (ARTIFACT / "report.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
