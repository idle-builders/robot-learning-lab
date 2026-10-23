"""Resume the local V12 matrix after a zero-disk-space trace write failed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from probe_touch_corrective_feasibility_v12 import METHOD, OUT, ROOT, sha, visit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-visits", type=int, default=4)
    args = parser.parse_args()
    method = json.loads(METHOD.read_text())
    for entry in method["source"].values():
        if sha(ROOT / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"Frozen source changed: {entry['path']}")
    source = json.loads((ROOT / method["source"]["v8_report"]["path"]).read_text())
    records = {r["seed"]: r for r in source["records"]}
    rows = json.loads((OUT / "partial.json").read_text())
    done = {(r["seed"], r["cutoff_step"], r["sign"]) for r in rows}
    for row in rows:
        if sha(ROOT / row["trace"]) != row["trace_sha256"]:
            raise RuntimeError("Saved completed V12 trace changed")
    remaining = 0
    for seed in method["seeds"]:
        record = records[seed]
        path = ROOT / record["episode"]
        if sha(path) != record["episode_sha256"] or not record["accepted_gentle_whole_teacher"]:
            raise RuntimeError(f"V8 source episode invalid: {seed}")
        with np.load(path, allow_pickle=False) as data:
            saved = {k: data[k].copy() for k in
                     ("first_qpos", "first_qvel", "first_actor", "actor_obs", "action")}
        saved["first_finger_step"] = record["first_finger_step"]
        for stage, offset in (("precontact", -method["stages"]["precontact_steps_before_first_finger"]),
                              ("postcontact", method["stages"]["postcontact_steps_after_first_finger"])):
            for sign in method["perturbation"]["signs"]:
                cutoff = record["first_finger_step"] + offset
                key = (seed, cutoff, sign)
                if key in done:
                    continue
                output = OUT / f"seed{seed}-step{cutoff}-sign{'p' if sign > 0 else 'm'}.npz"
                if output.exists():
                    raise RuntimeError(f"Unattributed prior output exists: {output}")
                row = visit(seed, tuple(record["cube_xy_m"]), saved, cutoff, sign, method)
                if row["stage"] != stage:
                    raise RuntimeError("Stage label differs")
                rows.append(row)
                done.add(key)
                (OUT / "partial.json").write_text(json.dumps(rows, indent=2) + "\n")
                print(seed, stage, sign, row["gentle_whole_episode"],
                      round(row["longest_supported_touch_s"], 2),
                      round(row["cube_peak_xy_drift_mm"], 2), flush=True)
                remaining += 1
                if remaining >= args.max_visits:
                    break
            if remaining >= args.max_visits:
                break
        if remaining >= args.max_visits:
            break
    expected = len(method["seeds"]) * 2 * len(method["perturbation"]["signs"])
    if len(rows) == expected:
        result = {"schema": "robot-learning-lab.touch-corrective-feasibility-v12.result.v1",
                  "method_sha256": sha(METHOD),
                  "runner_sha256": method["source"]["runner"]["sha256"],
                  "recovery_orchestrator_sha256": sha(Path(__file__)),
                  "unscored_save_failures": [
                      "seed33007 postcontact sign -1: local disk full after physics, no scored trace; same development case rerun",
                      "seed33009 postcontact sign -1: local disk full after physics, no scored trace; same development case rerun",
                  ],
                  "device": "local CPU", "attempted": len(rows),
                  "basic_touch": sum(r["basic_touch"] for r in rows),
                  "gentle_whole_episodes": sum(r["gentle_whole_episode"] for r in rows),
                  "precontact_gentle": sum(r["gentle_whole_episode"] for r in rows if r["stage"] == "precontact"),
                  "postcontact_gentle": sum(r["gentle_whole_episode"] for r in rows if r["stage"] == "postcontact"),
                  "gpu_hours": 0, "optimizer_updates": 0,
                  "claim_limit": "CPU perturbed saved-teacher action feasibility only; no CUDA or learned contact score, and no correction guarantee for displaced cubes.",
                  "records": rows}
        (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"source_checked_completed": len(rows), "planned": expected}))


if __name__ == "__main__":
    main()
