"""Use source-verified one-update benchmarks after correcting RSL's saved-iteration convention."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import torch

from run_shared_ppo import ROOT, ART, METHOD, BASE, OUT, sha, run_branch


REPAIR = ART / "ppo-benchmark-repair-method.json"


def main() -> None:
    repair = json.loads(REPAIR.read_text())
    method = json.loads(METHOD.read_text())
    if (repair["training_method_sha256"] != sha(METHOD)
            or repair["original_benchmark_sha256"] != sha(OUT / "benchmark/report.json")
            or repair["bc_checkpoint_sha256"] != sha(BASE)):
        raise RuntimeError("Frozen benchmark repair identity mismatch")
    for item in method["sources"].values():
        path = ROOT / item["path"]
        if sha(path) != item["sha256"]:
            raise RuntimeError(f"Frozen training source changed: {path}")
    if sha(ROOT / repair["repair_runner"]["path"]) != repair["repair_runner"]["sha256"]:
        raise RuntimeError("Frozen repair runner changed")
    failed = json.loads((OUT / "benchmark/report.json").read_text())
    base = torch.load(BASE, map_location="cpu", weights_only=False)
    rows = []
    for row in failed["rows"]:
        path = Path(row["checkpoint"])
        if (row["exit_code"] != 0 or sha(path) != row["checkpoint_sha256"]
                or sha(Path(row["log"])) != row["log_sha256"]):
            raise RuntimeError("Original benchmark source changed or did not run")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        deltas = [(saved["actor_state_dict"][key] - base["actor_state_dict"][key]).abs().max().item()
                  for key in base["actor_state_dict"] if key.startswith("mlp.")]
        count_gain = int(saved["actor_state_dict"]["obs_normalizer.count"]
                         - base["actor_state_dict"]["obs_normalizer.count"])
        if (saved["iter"] != 844 or max(deltas) <= 1e-6
                or count_gain != row["nominal_transitions"]
                or len(saved["optimizer_state_dict"]["state"]) == 0):
            raise RuntimeError(f"One-update benchmark did not change actor: {row['label']}")
        rows.append({**row, "verified_mlp_max_delta": max(deltas),
                     "verified_normalizer_count_gain": count_gain,
                     "rsl_iteration_convention": "First resumed update from model_844 is saved as model_844."})
    chosen = max(rows, key=lambda row: row["nominal_transitions_per_s"])["worlds"]
    if chosen not in method["ppo_world_benchmark"]:
        raise RuntimeError("Benchmark choice outside frozen capacities")
    benchmark_repair = OUT / "benchmark-repaired.json"
    if benchmark_repair.exists():
        raise RuntimeError("Benchmark repair already exists")
    benchmark_repair.write_text(json.dumps({
        "schema": "robot-learning-lab.shared-forward-turn.benchmark-repair.v1",
        "repair_method_sha256": sha(REPAIR),
        "original_benchmark_sha256": sha(OUT / "benchmark/report.json"),
        "rows": rows, "chosen_worlds": chosen,
        "claim_limit": "Measured throughput and verified one PPO update per branch; no physical skill score.",
    }, indent=2) + "\n")
    folder = OUT / "train"
    if folder.exists():
        raise RuntimeError("Original bounded PPO train already exists")
    folder.mkdir()
    body = ROOT / method["sources"]["body"]["path"]
    row = run_branch("learn-r1", chosen, method["ppo_updates"], body)
    # RSL reports the start index on the first update and the last completed
    # iteration on the final checkpoint: 16 resumed updates end at model_859.
    target_iteration = 844 + method["ppo_updates"] - 1
    if (row["exit_code"] or row.get("saved_iteration") != target_iteration
            or not row["checkpoint"]):
        (folder / "failed.json").write_text(json.dumps(row, indent=2) + "\n")
        raise RuntimeError("Bounded PPO did not finish its frozen number of updates")
    candidate = folder / "candidate-final.pt"
    shutil.copyfile(row["checkpoint"], candidate)
    result = {"schema": "robot-learning-lab.shared-forward-turn.ppo-fit-r1.v1",
              "training_method_sha256": sha(METHOD),
              "repair_method_sha256": sha(REPAIR),
              "verified_benchmark_sha256": sha(benchmark_repair),
              "bc_checkpoint_sha256": sha(BASE),
              "worlds": chosen, "optimizer_updates": method["ppo_updates"],
              "nominal_transitions": row["nominal_transitions"],
              "elapsed_s": row["elapsed_s"], "saved_iteration": row["saved_iteration"],
              "candidate_checkpoint_sha256": sha(candidate), "source_row": row,
              "claim_limit": "Short native shared-command PPO fit; fresh physical replay is required."}
    (folder / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"worlds": chosen, "updates": result["optimizer_updates"],
                      "candidate_sha256": result["candidate_checkpoint_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
