"""Verify saved V5 batched teacher originals from metrics and aligned traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from collect_v5_batched_teacher import summary
from v4_action_map import processed


ROOT = Path(__file__).resolve().parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    args = parser.parse_args()
    batch = args.batch.resolve()
    report = json.loads((batch / "report.json").read_text())
    metrics = batch / "metrics.npz"
    source = ROOT / "experiments/compact_morphology/cove50_low_level_handoffs/collect_v5_batched_teacher.py"
    if (report["source_sha256"] != sha(source)
            or report["metrics_sha256"] != sha(metrics)
            or len(report["cases"]) != report["worlds"]):
        raise RuntimeError("V5 source, metrics, or world count differs")
    with np.load(metrics, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    series = {key: value for key, value in arrays.items()
              if key not in ("initial_qpos", "initial_qvel", "first_done", "continuity")}
    for index, case in enumerate(report["cases"]):
        if case["world_index"] != index:
            raise RuntimeError("World index changed")
        q = arrays["initial_qpos"][index, 3:7]
        yaw = float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                               1 - 2 * (q[2] ** 2 + q[3] ** 2)))
        rebuilt = summary(series, index, int(arrays["first_done"][index]),
                          yaw, float(arrays["continuity"][index]))
        for key in ("steps", "motion_complete", "strict_pass", "failure_reasons"):
            if case[key] != rebuilt[key]:
                raise RuntimeError(f"Original world {index} {key} changed")
        if not np.array_equal(np.asarray(case["initial_qpos"]), arrays["initial_qpos"][index]):
            raise RuntimeError(f"Original world {index} initial qpos changed")
        trace_name = case.get("trace")
        if trace_name:
            trace_path = batch / trace_name
            if not case["motion_complete"] or case["trace_sha256"] != sha(trace_path):
                raise RuntimeError(f"Original world {index} trace source changed")
            with np.load(trace_path, allow_pickle=False) as trace:
                pre, post = trace["pre_qpos"], trace["post_qpos"]
                obs, action = trace["actor_obs"], trace["applied_action"]
                target, names = trace["processed_target"], trace["target_names"].tolist()
            if (pre.shape[0] != 1200 or obs.shape != (1200, 72)
                    or action.shape != (1200, 21) or target.shape != (1200, 21)
                    or np.max(abs(pre[1:] - post[:-1])) > 1e-9
                    or np.max(abs(processed(action, names) - target)) > 1e-5
                    or not np.isfinite(obs).all()
                    or not np.isfinite(target).all()):
                raise RuntimeError(f"Original world {index} whole trace invalid")
    if (sum(row["motion_complete"] for row in report["cases"]) != report["motion_complete"]
            or sum(row["strict_pass"] for row in report["cases"]) != report["strict_passes"]):
        raise RuntimeError("Batch totals changed")
    result = {"schema": "robot-learning-lab.broad-handoff-v5-batched-source-check.v1",
              "batch_report_sha256": sha(batch / "report.json"),
              "metrics_sha256": sha(metrics), "worlds": report["worlds"],
              "motion_complete": report["motion_complete"],
              "strict_passes": report["strict_passes"],
              "retained_verified_traces": sum("trace" in row for row in report["cases"]),
              "learner_updates": 0}
    out = batch / "source-verification.json"
    if out.exists():
        raise RuntimeError("Original source verification exists")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
