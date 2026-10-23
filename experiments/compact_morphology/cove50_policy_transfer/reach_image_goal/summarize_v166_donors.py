"""Verify original V166 reach donors and summarize admitted whole examples."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-low-level-training-20261015/reach-own-trace-v166"
VERIFY = Path(__file__).with_name("verify_aligned_reach_v166.py")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    method_sha = sha(BASE / "method.json")
    rows = []
    batches = {}
    for name in ("pilot", "remaining"):
        path = BASE / "native-donors" / f"{name}-batch.json"
        if not path.exists():
            continue
        batch = json.loads(path.read_text())
        if batch["method_sha256"] != method_sha:
            raise ValueError(f"{name}: method changed")
        if len(batch["rows"]) != batch["completed"]:
            raise ValueError(f"{name}: row count differs")
        batches[name] = {"sha256": sha(path), "planned": batch["planned"],
                         "completed": batch["completed"], "accepted": batch["accepted"]}
        rows.extend(batch["rows"])
    if not rows or len({row["seed"] for row in rows}) != len(rows):
        raise ValueError("No donors or duplicate original seed")
    result_rows = []
    for row in rows:
        for key in ("report", "trace", "aligned_trace", "log"):
            if sha(ROOT / row[key]) != row[f"{key}_sha256"]:
                raise ValueError(f"{row['seed']}: {key} hash differs")
        check = subprocess.run(
            [sys.executable, str(VERIFY), "--report", row["report"],
             "--trace", row["trace"], "--aligned-trace", row["aligned_trace"]],
            cwd=ROOT, capture_output=True, text=True, check=True)
        verified = json.loads(check.stdout)
        if verified["seed"] != row["seed"] or not verified["source_alignment_pass"]:
            raise ValueError(f"{row['seed']}: synchronized source differs")
        if row["imitation_admit"] != bool(row["native_full_gate_pass"] and
                                          verified["full_contact_margin_pass"]):
            raise ValueError(f"{row['seed']}: admission differs")
        with np.load(ROOT / row["aligned_trace"]) as trace:
            steps = int(trace["step"].shape[0])
            feature_min = trace["actor_features_pre"].min(axis=0).tolist()
            feature_max = trace["actor_features_pre"].max(axis=0).tolist()
        report = json.loads((ROOT / row["report"]).read_text())
        result_rows.append({
            "seed": row["seed"], "stratum": row["stratum"],
            "selection_mode": row["selection_mode"],
            "admitted": row["imitation_admit"],
            "continuous_contact_s": row["continuous_contact_s"],
            "aligned_reach_steps": steps,
            "rail_v": report["rail_v"], "delay_ms": report["delay_ms"],
            "toy_offset_b_m": report["toy_offset_b_m"],
            "pitch_offset_rad": report["pitch_offset_rad"],
            "feature_min": feature_min, "feature_max": feature_max,
            "faults": row["faults"], "report": row["report"],
            "trace": row["trace"], "aligned_trace": row["aligned_trace"],
        })
    accepted = [row for row in result_rows if row["admitted"]]
    output = {
        "schema": "robot-learning-lab.reach-v166-donor-summary.v1",
        "claim": "own-trajectory source-valid development demonstrations; no learned score",
        "method_sha256": method_sha, "batches": batches,
        "completed": len(rows), "accepted": len(accepted),
        "accepted_reach_steps": sum(row["aligned_reach_steps"] for row in accepted),
        "accepted_by_stratum": dict(Counter(row["stratum"] for row in accepted)),
        "all_originals_source_verified": True,
        "rows": result_rows,
    }
    path = BASE / "donor-summary.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in
                      ("completed", "accepted", "accepted_reach_steps",
                       "accepted_by_stratum")}, indent=2))


if __name__ == "__main__":
    main()
