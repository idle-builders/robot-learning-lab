"""Source-check and summarize the bounded native V4 teacher screen."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PLAN = ROOT / "artifacts/ape-low-level-training-20261015/broad-handoff-v4-l4-screen-method.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    path = source / ("screen-report.json" if (source / "screen-report.json").exists()
                     else "screen-partial.json")
    original = json.loads(path.read_text())
    plan = json.loads(PLAN.read_text())
    rows = original["rows"]
    if (original["plan_sha256"] != sha(PLAN)
            or len(rows) != original.get("visits", original.get("completed"))
            or [item["case"] for item in rows] != plan["screen"][:len(rows)]
            or (len(rows) != len(plan["screen"]) and not args.allow_partial)):
        raise RuntimeError("V4 frozen screen identity, order or completeness differs")
    matrix = []
    for item in rows:
        case = item["case"]
        report_path = source / Path(item["report"]).name
        trace_path = source / Path(item["trace"]).name
        log_path = source / Path(item["log"]).name
        if (sha(report_path) != item["report_sha256"]
                or sha(trace_path) != item["trace_sha256"]
                or sha(log_path) != item["log_sha256"]):
            raise RuntimeError(f"Native source artifact changed: {case['seed']}")
        report = json.loads(report_path.read_text())
        if (report["seed"] != case["seed"] or report["body_sha256"] != plan["exact_body_sha256"]
                or report["device"] != "cuda:0" or report["rail_v"] != case["rail_v"]
                or report["delay_ms"] != case["delay_ms"]
                or report["walk_velocity_m_s"] != case["walk_velocity_m_s"]
                or bool(report["full_continuity_gate_pass"]) != item["pass"]
                or any(report["checkpoint_sha256"][name] != spec["sha256"]
                       for name, spec in plan["actors"].items())):
            raise RuntimeError(f"Native report identity or gate differs: {case['seed']}")
        with np.load(trace_path, allow_pickle=False) as trace:
            pre, post = trace["pre_qpos"], trace["post_qpos"]
            obs = trace["actor_obs"]
            raw, target = trace["applied_action"], trace["processed_target"]
            if (len(pre) != report["steps"] or post.shape != pre.shape
                    or obs.shape != (len(pre), 72) or raw.shape != (len(pre), 21)
                    or target.shape != raw.shape or not all(np.isfinite(v).all()
                                                           for v in (pre, post, obs, raw, target))
                    or np.max(np.abs(pre[0] - report["initial_qpos"])) > 1e-8
                    or (len(pre) > 1 and np.max(np.abs(post[:-1] - pre[1:])) > 1e-8)):
                raise RuntimeError(f"Native trace state/shape differs: {case['seed']}")
            pelvis = np.asarray([[r["pelvis_x"], r["pelvis_y"], r["pelvis_z"]]
                                 for r in report["timeseries"]])
            if pelvis.shape != (len(pre), 3) or np.max(np.abs(post[:, :3] - pelvis)) > 1e-8:
                raise RuntimeError(f"Native trace/report position differs: {case['seed']}")
        if item["pass"] != (not item["failure_reasons"]):
            raise RuntimeError(f"Saved failure reasons differ: {case['seed']}")
        settled = report["windows"].get("settled", {})
        matrix.append({"seed": case["seed"], "walk_velocity_m_s": case["walk_velocity_m_s"],
                       "rail_v": case["rail_v"], "delay_ms": case["delay_ms"],
                       "physical_pass": item["pass"], "failure_reasons": item["failure_reasons"],
                       "steps": report["steps"], "walk_m": report["walk_distance_m"],
                       "stop_tail_m": report["stop_tail_xy_m"],
                       "settled_two_feet_fraction": settled.get("feet_both_fraction"),
                       "settled_rated_leg_fraction": settled.get("leg_duty_fraction"),
                       "elapsed_s": item["elapsed_s_including_constructor"],
                       "report_sha256": item["report_sha256"],
                       "trace_sha256": item["trace_sha256"]})
    failures = Counter(reason for item in matrix for reason in item["failure_reasons"])
    result = {"schema": "robot-learning-lab.broad-handoff-v4-l4-teacher-screen-summary.v1",
              "classification": "source-checked CUDA teacher demonstration screen; no new learner score",
              "plan_sha256": sha(PLAN), "native_summary_sha256": sha(path),
              "planned": len(plan["screen"]), "completed": len(matrix),
              "full_passes": sum(row["physical_pass"] for row in matrix),
              "accepted_whole_steps": sum(row["steps"] for row in matrix if row["physical_pass"]),
              "failure_counts": dict(failures),
              "total_native_visit_wall_s": sum(row["elapsed_s"] for row in matrix),
              "mean_native_steps_per_visit_s": sum(row["steps"] for row in matrix) /
                   max(1, sum(row["elapsed_s"] for row in matrix)),
              "rows": matrix,
              "learner_updates": 0, "learner_physical_score": None}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
