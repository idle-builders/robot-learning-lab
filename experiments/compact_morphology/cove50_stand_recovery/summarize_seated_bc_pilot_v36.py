"""Source-check complete matched V36 A40 development families and physical gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from experiments.compact_morphology.cove50_stand_recovery import score_seated_reference_cuda_v26 as scorer


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-bc-pilot-v36"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads((ART / "plan.json").read_text())
    events = [json.loads(line) for line in (ART / "transport.jsonl").read_text().splitlines()]
    source = {(e["seed"], e["route"]): e for e in events
              if e["event"] == "original_complete"}
    model = mujoco.MjModel.from_xml_path(str(ROOT / plan["body"]))
    gate = json.loads((ROOT / "artifacts/ape-cove50-stand-recovery/seated-rise-ppo-pilot-v25/plan.json").read_text())["physical_gate"]
    scorer.ART = ART
    rows, pending = [], []
    for seed in plan["development_seeds"]:
        if any((seed, route) not in source for route in plan["routes_per_seed"]):
            pending.append(seed)
            continue
        reports, traces, scored = {}, {}, {}
        for route in plan["routes_per_seed"]:
            entry = source[(seed, route)]
            stem = ART / f"seed{seed}-{route}"
            paths = {suffix: stem.with_suffix("." + suffix)
                     for suffix in ("json", "npz", "mp4", "log")}
            if (sha(paths["json"]) != entry["report_sha256"]
                    or sha(paths["npz"]) != entry["trace_sha256"]
                    or sha(paths["mp4"]) != entry["video_sha256"]):
                raise RuntimeError(f"Original bytes differ: {seed} {route}")
            report = json.loads(paths["json"].read_text())
            checkpoint_sha = (plan["bc_checkpoint_sha256"] if route == "bc"
                              else plan["baseline_checkpoint_sha256"])
            if (report["seed"] != seed or report["intervention_route"] != route
                    or report["evaluator_source_sha256"] != plan["evaluator_sha256"]
                    or report["checkpoint_sha256"] != checkpoint_sha
                    or report["body_sha256"] != plan["body_sha256"]):
                raise RuntimeError(f"Report provenance differs: {seed} {route}")
            with np.load(paths["npz"], allow_pickle=False) as trace:
                first_actor = trace["actor_observation"][0].copy()
                first_critic = trace["critic_observation"][0].copy()
            row, _ = scorer.score(seed, route, model, gate, plan["body_sha256"])
            reports[route] = report
            traces[route] = (first_actor, first_critic)
            row["video"] = str(paths["mp4"].relative_to(ROOT))
            row["video_sha256"] = sha(paths["mp4"])
            scored[route] = row
        first = reports["bc"]
        for route in ("model25", "scripted"):
            other = reports[route]
            if (any(first[key] != other[key] for key in
                    ("initial_state_sha256", "initial_qpos", "initial_qvel",
                     "initial_physics", "initial_motors"))
                    or not np.array_equal(traces["bc"][0], traces[route][0])
                    or not np.array_equal(traces["bc"][1], traces[route][1])):
                raise RuntimeError(f"Matched physical start differs: {seed} {route}")
        rows.append({"seed": seed, "bc": scored["bc"],
                     "model25": scored["model25"], "scripted": scored["scripted"],
                     "matched_start": True,
                     "bc_rescue": scored["bc"]["full_gate"] and not scored["model25"]["full_gate"],
                     "bc_regression": scored["model25"]["full_gate"] and not scored["bc"]["full_gate"]})
    report = {"schema": "robot-learning-lab.seated-bc-pilot-v36.source-checked-development.v1",
              "plan_sha256": sha(ART / "plan.json"),
              "hardware": "NVIDIA A40",
              "complete_families": len(rows), "planned_families": len(plan["development_seeds"]),
              "pending_families": pending,
              "bc_full_gate": sum(r["bc"]["full_gate"] for r in rows),
              "model25_full_gate": sum(r["model25"]["full_gate"] for r in rows),
              "scripted_full_gate": sum(r["scripted"]["full_gate"] for r in rows),
              "bc_rescues": sum(r["bc_rescue"] for r in rows),
              "bc_regressions": sum(r["bc_regression"] for r in rows),
              "rows": rows,
              "limit": "Fresh A40 development on experimental Direct Frame, not held-out robustness or physical qualification"}
    (ART / "source-checked-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("complete_families", "planned_families", "bc_full_gate",
                       "model25_full_gate", "scripted_full_gate", "bc_rescues")}, sort_keys=True))


if __name__ == "__main__":
    main()
