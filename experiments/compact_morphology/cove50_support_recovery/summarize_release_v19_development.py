"""Verify four frozen V19 CUDA pairs and summarize the physical pilot outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from summarize_release_v19 import score, sha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--evaluation-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--body-xml", type=Path, required=True)
    args = parser.parse_args()
    method = json.loads(args.method.read_text())
    source = json.loads(args.evaluation_source.read_text())
    if (source["method_sha256"] != sha(args.method)
            or source["checkpoint_sha256"] != sha(args.checkpoint)
            or source["scorer_sha256"] != sha(Path(__file__).with_name("summarize_release_v19.py"))
            or source["runner_sha256"] != sha(Path(__file__).with_name("run_release_v19_development.py"))):
        raise RuntimeError("Frozen V19 evaluation source changed")
    rows, comparisons = [], []
    for family in method["development_families"]:
        pair = []
        for controller in ("reference", "checkpoint"):
            stem = f"seed{family['seed']}-{controller}"
            report = args.root / f"{stem}.json"
            trace = args.root / f"{stem}.npz"
            result = score(report, trace, args.body_xml)
            if (result["seed"] != family["seed"] or result["side"] != family["side"]
                    or result["controller"] != ("reference" if controller == "reference" else "ppo")
                    or result["eval_device"] != "cuda:0"
                    or result["checkpoint_sha256"] !=
                    (None if controller == "reference" else sha(args.checkpoint))):
                raise RuntimeError(f"Original {stem} source identity differs")
            pair.append(result)
            rows.append(result)
        if (pair[0]["initial_state_sha256"] != pair[1]["initial_state_sha256"]
                or pair[0]["initial_domain_sha256"] != pair[1]["initial_domain_sha256"]):
            raise RuntimeError(f"Original pair {family['seed']} physical/domain start differs")
        comparisons.append({
            "seed": family["seed"], "side": family["side"],
            "reference_full": pair[0]["full_gate_pass"],
            "checkpoint_full": pair[1]["full_gate_pass"],
            "reference_half_second": pair[0]["half_second_release"],
            "checkpoint_half_second": pair[1]["half_second_release"],
            "reference_clean_release_s": pair[0]["longest_clean_release_s"],
            "checkpoint_clean_release_s": pair[1]["longest_clean_release_s"],
            "reward_delta": pair[1]["reward_sum"] - pair[0]["reward_sum"],
            "same_initial_state_and_domain": True,
        })
    candidate = [row for row in rows if row["controller"] == "ppo"]
    result = {
        "schema": "robot-learning-lab.cove50-release-v19.development-report.v1",
        "method_sha256": sha(args.method),
        "evaluation_source_sha256": sha(args.evaluation_source),
        "checkpoint_sha256": sha(args.checkpoint),
        "body_sha256": sha(args.body_xml),
        "scorer_sha256": sha(Path(__file__).with_name("summarize_release_v19.py")),
        "valid_matched_cuda_pairs": len(comparisons),
        "training_updates": 4,
        "training_worlds": 512,
        "training_transitions": 49152,
        "candidate_half_second_clean_release": sum(x["half_second_release"] for x in candidate),
        "candidate_full_gate_passes": sum(x["full_gate_pass"] for x in candidate),
        "best_candidate_clean_release_s": max(x["longest_clean_release_s"] for x in candidate),
        "rows": rows, "comparisons": comparisons,
        "decision_rule": method["early_stop"],
        "claim_limit": "Four opened Direct Frame development families; no held robustness, physical transfer or single-step skill claim.",
    }
    path = args.root / "report.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: value for k, value in result.items()
                      if k not in ("rows", "comparisons")}, indent=2))


if __name__ == "__main__":
    main()
