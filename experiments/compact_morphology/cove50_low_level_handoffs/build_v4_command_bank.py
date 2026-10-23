"""Convert verified whole V4 teacher passes into command-visible BC episodes.

This prepares labels only. Fresh closed-loop learner evaluation is separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from build_command_bank import build_features
from v4_action_map import canonical_raw, processed


ROOT = Path(__file__).resolve().parents[3]
PLAN = ROOT / "artifacts/ape-low-level-training-20261015/broad-handoff-v4-l4-screen-method.json"
CLOSURE = ROOT / "artifacts/ape-low-level-training-20261015/broad-handoff-v4-l4-closure-addendum.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    summary_path = source / "screen-report.json"
    if not summary_path.exists():
        if not args.allow_partial:
            raise RuntimeError("Complete V4 screen required; pass --allow-partial explicitly")
        summary_path = source / "screen-partial.json"
    summary = json.loads(summary_path.read_text())
    plan = json.loads(PLAN.read_text())
    closure = json.loads(CLOSURE.read_text())
    if summary["plan_sha256"] != sha(PLAN) or not CLOSURE.exists():
        raise RuntimeError("V4 plan or source closure changed")
    for item in closure["import_time_dependencies"]:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"Import-time dependency changed: {item['path']}")
    sentinel = json.loads((source / "sentinel-report.json").read_text())
    if (sentinel["plan_sha256"] != sha(PLAN) or sentinel["visits"] != 1
            or sentinel["rows"][0]["steps"] != 1):
        raise RuntimeError("Missing source-verified V4 sentinel")
    sentinel_row = sentinel["rows"][0]
    if (sha(source / Path(sentinel_row["report"]).name) != sentinel_row["report_sha256"]
            or sha(source / Path(sentinel_row["trace"]).name) != sentinel_row["trace_sha256"]):
        raise RuntimeError("V4 sentinel source changed")
    hardware = json.loads((source / "runtime-attestation.json").read_text())["hardware"]
    if (hardware["torch_device_name"] != "NVIDIA L4"
            or hardware["compute_capability"] != [8, 9]):
        raise RuntimeError("V4 hardware attestation differs")
    rows = summary["rows"]
    if len(rows) != summary.get("visits", summary.get("completed")):
        raise RuntimeError("V4 row count differs")
    if len(rows) != len(plan["screen"]) and not args.allow_partial:
        raise RuntimeError("V4 screen is incomplete")
    if [row["case"] for row in rows] != plan["screen"][:len(rows)]:
        raise RuntimeError("V4 original order/identity differs")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    accepted = []
    target_names = None
    for row in rows:
        report_path = source / Path(row["report"]).name
        trace_path = source / Path(row["trace"]).name
        if sha(report_path) != row["report_sha256"] or sha(trace_path) != row["trace_sha256"]:
            raise RuntimeError(f"Source changed: {row['case']['seed']}")
        report = json.loads(report_path.read_text())
        if (report["seed"] != row["case"]["seed"]
                or report["body_sha256"] != plan["exact_body_sha256"]
                or report["device"] != "cuda:0"
                or report["rail_v"] != row["case"]["rail_v"]
                or report["delay_ms"] != row["case"]["delay_ms"]
                or report["walk_velocity_m_s"] != row["case"]["walk_velocity_m_s"]
                or abs(report["dt_s"] - 0.02) > 1e-8
                or any(report["checkpoint_sha256"][name] != spec["sha256"]
                       for name, spec in plan["actors"].items())
                or bool(report["full_continuity_gate_pass"]) != row["pass"]):
            raise RuntimeError(f"Report identity or gate changed: {row['case']['seed']}")
        if not row["pass"]:
            continue
        with np.load(trace_path, allow_pickle=False) as trace:
            obs = trace["actor_obs"].astype(np.float32)
            arm_obs = trace["arm_actor_obs"].astype(np.float32)
            action = trace["applied_action"].astype(np.float32)
            target = trace["processed_target"].astype(np.float32)
            pre = trace["pre_qpos"]
            post = trace["post_qpos"]
            names = trace["target_names"].tolist()
        if (obs.shape != (1200, 72) or arm_obs.shape != (1200, 72)
                or action.shape != (1200, 21) or target.shape != (1200, 21)
                or np.max(np.abs(post[:-1] - pre[1:])) > 1e-9):
            raise RuntimeError(f"Whole episode or physical continuity invalid: {row['case']['seed']}")
        requested_arm = np.asarray([r["arm_requested"] for r in report["timeseries"]], dtype=np.float32)
        if (len(requested_arm) != 1200 or abs(float(arm_obs[0, 69])) > 1e-7
                or not np.allclose(arm_obs[1:, 69], requested_arm[:-1], atol=1e-7)):
            raise RuntimeError(f"Arm command lag differs: {row['case']['seed']}")
        if target_names is None:
            target_names = names
        elif names != target_names:
            raise RuntimeError("Target joint order differs")
        map_gap = float(np.max(np.abs(processed(action, names) - target)))
        if map_gap > 1e-5:
            raise RuntimeError(f"Saved native action map differs: {row['case']['seed']}: {map_gap}")
        canonical = canonical_raw(target, names).astype(np.float32)
        if float(np.max(np.abs(processed(canonical, names) - target))) > 1e-5:
            raise RuntimeError(f"Canonical inverse differs: {row['case']['seed']}")
        previous = np.concatenate((np.zeros((1, 21), np.float32), canonical[:-1]))
        extra = build_features(report["timeseries"], row["case"], arm_obs, previous)
        modes = ("walk", "stop", "posture", "arms")
        mode = np.asarray([[float(r["phase"] == name) for name in modes]
                           for r in report["timeseries"]], dtype=np.float32)
        if not np.all(mode.sum(axis=1) == 1):
            raise RuntimeError(f"Stage ownership ambiguous: {row['case']['seed']}")
        features = np.concatenate((obs, extra[:, :6], mode, extra[:, 6:]), axis=1).astype(np.float32)
        if features.shape != (1200, 103) or not np.isfinite(features).all() or not np.isfinite(target).all():
            raise RuntimeError(f"Nonfinite actor/label: {row['case']['seed']}")
        episode_path = out / f"seed-{row['case']['seed']}.npz"
        np.savez_compressed(episode_path, features=features, physical_target=target,
                            applied_action=action, canonical_applied_action=canonical,
                            target_names=np.asarray(names),
                            seed=np.asarray(row["case"]["seed"]), dt_s=np.asarray(0.02))
        accepted.append({"seed": row["case"]["seed"], "episode": str(episode_path.relative_to(ROOT)),
                         "episode_sha256": sha(episode_path), "report_sha256": row["report_sha256"],
                         "trace_sha256": row["trace_sha256"]})
    result = {
        "schema": "robot-learning-lab.broad-handoff-v4-l4-command-bank.v1",
        "classification": "source-checked whole teacher labels; no new learner or skill score",
        "screen_report_sha256": sha(summary_path), "plan_sha256": sha(PLAN),
        "closure_addendum_sha256": sha(CLOSURE),
        "source_coverage": len(rows), "planned_source_families": len(plan["screen"]),
        "accepted_whole_episodes": len(accepted), "accepted_frames": 1200 * len(accepted),
        "feature_width": 103, "physical_target_width": 21,
        "feature_contract": "72 native actor values; six explicit requested commands including one-step arm lag; four stage-owner flags walk/stop/posture/arms; prior canonical inverse of the processed 21-channel physical target",
        "target_names": target_names, "entries": accepted,
        "learner_updates": 0, "learner_score": None,
    }
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("source_coverage", "accepted_whole_episodes", "accepted_frames")}))


if __name__ == "__main__":
    main()
