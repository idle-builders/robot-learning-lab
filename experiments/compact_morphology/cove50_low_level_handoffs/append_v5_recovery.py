"""Append only source-verified recoverable learner-state teacher corrections."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--recovery-check", type=Path, required=True)
    parser.add_argument("--recovery-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.out.exists():
        parser.error("Original correction bank exists")
    bank = json.loads(args.bank.read_text())
    check = json.loads(args.recovery_check.read_text())
    if (bank["schema"] != "robot-learning-lab.broad-handoff-v5-merged-teacher-bank.v1"
            or check["schema"] != "robot-learning-lab.broad-handoff-v5-recovery-source-check.v1"
            or not check["corrective_labels_allowed"]):
        raise RuntimeError("Need a source-verified recovered physical original")
    args.out.mkdir(parents=True)
    entries = list(bank["entries"])
    corrective_frames = 0
    for row in check["rows"]:
        if not row["motion_recovered"]:
            continue
        seed = row["seed"]
        report_path = args.recovery_source / f"recovery-{seed}.json"
        trace_path = args.recovery_source / f"recovery-{seed}.npz"
        if sha(report_path) != row["source_report_sha256"] or sha(trace_path) != row["source_trace_sha256"]:
            raise RuntimeError("Recovery source changed")
        report = json.loads(report_path.read_text())
        switch = row["teacher_recovery_step"]
        if report["steps"] != 1200:
            raise RuntimeError("Recovery physical gate changed")
        with np.load(trace_path, allow_pickle=False) as trace:
            x = trace["shared_features"][switch:].astype(np.float32)
            y = trace["processed_target"][switch:].astype(np.float32)
            names = trace["target_names"].tolist()
        if (x.shape != (1200-switch, 103) or y.shape != (1200-switch, 21)
                or names != bank["target_names"] or not np.isfinite(x).all()
                or not np.isfinite(y).all()):
            raise RuntimeError("Corrective feature/target identity changed")
        episode = args.out / f"recovery-{seed}.npz"
        np.savez_compressed(episode, features=x, physical_target=y,
                            target_names=np.asarray(names), dt_s=np.asarray(0.02))
        entries.append({"group": "recovery",
                        "tier": "corrective_full_gate" if row["full_gate_recovered"] else "corrective_duty_limited",
                        "seed": seed, "frames": len(x), "switch_step": switch,
                        "episode": str(episode.relative_to(ROOT)), "episode_sha256": sha(episode),
                        "source_trace_sha256": row["source_trace_sha256"]})
        corrective_frames += len(x)
    result = dict(bank)
    result.update({"schema": "robot-learning-lab.broad-handoff-v5-corrective-bank.v1",
                   "classification": "whole native teacher imitation plus source-verified recoverable student-state labels; no V5 learner score",
                   "base_bank_sha256": sha(args.bank), "recovery_check_sha256": sha(args.recovery_check),
                   "corrective_episodes": len(entries) - len(bank["entries"]),
                   "corrective_frames": corrective_frames,
                   "whole_episodes": len(entries), "frames": bank["frames"] + corrective_frames,
                   "entries": entries})
    (args.out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("whole_episodes", "corrective_episodes", "corrective_frames")}))


if __name__ == "__main__":
    main()
