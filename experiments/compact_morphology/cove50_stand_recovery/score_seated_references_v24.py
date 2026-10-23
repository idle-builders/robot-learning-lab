"""Score V24 fresh seated references with the retained V22 physical gate."""

import hashlib
import importlib.util
import json
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-rise-reset-repair-v24"
OUT = ART / "fresh-references"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
V22_SCORER = Path(__file__).with_name("score_seated_trainer_parity_v22.py")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads((ART / "plan.json").read_text())
    assert sha(BODY) == plan["body_sha256"]
    spec = importlib.util.spec_from_file_location("v22_physical_gate", V22_SCORER)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    scorer.ART = OUT
    gate = json.loads((ROOT / "artifacts/ape-cove50-stand-recovery/seat-command-development-v5/plan.json").read_text())["physical_gate"]
    model = mujoco.MjModel.from_xml_path(str(BODY))
    rows = [scorer.score(route, seed, model, gate)
            for seed in plan["fresh_environment_references"]["seeds"]
            for route in ("scripted", "parked")]
    pairs = []
    for seed in plan["fresh_environment_references"]["seeds"]:
        script, parked = [row for row in rows if row["seed"] == seed]
        assert script["initial_state_sha256"] == parked["initial_state_sha256"]
        assert script["initial_physics"] == parked["initial_physics"]
        pairs.append({"seed": seed, "matched_initial_state": True,
                      "script_pass": script["full_gate"],
                      "parked_pass": parked["full_gate"],
                      "script_anchor_s": script["anchor_s"],
                      "parked_anchor_s": parked["anchor_s"],
                      "script_minus_parked_reward": script["reward_total_unrounded"] - parked["reward_total_unrounded"]})
    zero_seed = plan["fresh_environment_references"]["separate_zero_seed"]
    zero_path = OUT / f"seed{zero_seed}-zero.json"
    zero = json.loads(zero_path.read_text())
    assert zero["seed"] == zero_seed and zero["device"] == "cpu"
    assert zero["seated_rise_stage"] and zero["zero_actions"] and zero["randomized_physics"]
    assert not zero["scripted_target_probe"]
    summary = {
        "paired_independent_starts": len(pairs),
        "source_matched_pairs": len(pairs),
        "scripted_full_gate_passes": sum(p["script_pass"] for p in pairs),
        "parked_full_gate_passes": sum(p["parked_pass"] for p in pairs),
        "scripted_anchor_by_0p2s": sum(p["script_anchor_s"] is not None and p["script_anchor_s"] <= 0.2 for p in pairs),
        "zero_ended_on_named_fall": bool(zero["ended_early"] and "fell_over" in (zero["end_termination_terms"] or [])),
        "reward_script_exceeds_parked_pairs": sum(p["script_minus_parked_reward"] > 0 for p in pairs),
    }
    summary["fresh_reference_gate_met"] = (
        summary["scripted_full_gate_passes"] == 6
        and summary["parked_full_gate_passes"] == 0
        and summary["scripted_anchor_by_0p2s"] == 6
        and summary["zero_ended_on_named_fall"]
    )
    report = {
        "schema": "robot-learning-lab.cove50-seated-fresh-references.v24",
        "plan_sha256": sha(ART / "plan.json"), "body_sha256": sha(BODY),
        "evaluator_sha256": sha(ROOT / "experiments/compact_morphology/mjlab_ape/evaluate_posture.py"),
        "physical_scorer_sha256": sha(V22_SCORER),
        "source_device": "cpu", "summary": summary, "pairs": pairs, "rows": rows,
        "zero_source": str(zero_path.relative_to(ROOT)), "zero_source_sha256": sha(zero_path),
        "zero_named_termination": zero["end_termination_terms"],
        "claim_limit": "Fresh source-matched CPU reference challenge; separate from CUDA reset load and not a learned PPO score.",
    }
    (ART / "fresh-reference-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
