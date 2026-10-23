"""Join source-checked whole V4/V5 teacher episodes for one quick BC cycle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from v4_action_map import canonical_raw, processed


ROOT = Path(__file__).resolve().parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--batches", type=Path, nargs=4, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.out.exists():
        parser.error("Original merged bank exists")
    base = json.loads(args.base.read_text())
    if base["schema"] != "robot-learning-lab.broad-handoff-v5-tiered-bank.v1" or base["whole_episodes"] != 17:
        raise RuntimeError("Expected source-checked 17-whole-episode base bank")
    entries = list(base["entries"])
    batch_sources = []
    expected = {(54520, 10, 0), (54521, 10, 20), (54522, 12, 0), (54523, 12, 20)}
    seen = set()
    args.out.mkdir(parents=True)
    for path in args.batches:
        report_path = path / "report.json"
        verify_path = path / "source-verification.json"
        report = json.loads(report_path.read_text())
        verify = json.loads(verify_path.read_text())
        setting = (report["seed"], report["rail_v"], report["delay_ms"])
        if (setting not in expected or setting in seen or report["device"] != "cuda:0"
                or report["body_sha256"] != base["body_sha256"]
                or report["checkpoint_sha256"] != base["checkpoint_sha256"]
                or not report["retained_motion_traces"] or report["steps_requested"] != 1200
                or verify["batch_report_sha256"] != sha(report_path)
                or verify["retained_verified_traces"] != report["motion_complete"]):
            raise RuntimeError(f"Changed or incomplete batch: {path}")
        seen.add(setting)
        batch_sources.append({"setting": setting, "report_sha256": sha(report_path),
                              "source_verification_sha256": sha(verify_path)})
        for case in report["cases"]:
            if not case["motion_complete"]:
                continue
            trace_path = path / case["trace"]
            if sha(trace_path) != case["trace_sha256"]:
                raise RuntimeError(f"Trace hash changed: {trace_path}")
            with np.load(trace_path, allow_pickle=False) as trace:
                obs = trace["actor_obs"].astype(np.float32)
                arm_obs = trace["arm_actor_obs"].astype(np.float32)
                target = trace["processed_target"].astype(np.float32)
                action = trace["applied_action"].astype(np.float32)
                pre, post = trace["pre_qpos"], trace["post_qpos"]
                names = trace["target_names"].tolist()
                dt = float(trace["dt_s"])
            if (obs.shape != (1200, 72) or arm_obs.shape != (1200, 72)
                    or target.shape != (1200, 21) or names != base["target_names"]
                    or abs(dt - 0.02) > 1e-8
                    or np.max(np.abs(post[:-1] - pre[1:])) > 1e-9
                    or np.max(np.abs(processed(action, names) - target)) > 1e-5):
                raise RuntimeError(f"Native trace parity failed: {trace_path}")
            t = np.arange(1200, dtype=np.float32) * 0.02
            walking, stopping = t < 2, (t >= 2) & (t < 4)
            squatting, arms = (t >= 6) & (t < 10), (t >= 16) & (t < 20)
            commands = np.column_stack((
                np.where(walking, case["walk_velocity_m_s"], 0), np.zeros(1200),
                squatting.astype(np.float32), arms.astype(np.float32),
                arm_obs[:, 69], stopping.astype(np.float32))).astype(np.float32)
            modes = np.column_stack((walking, stopping, (t >= 4) & (t < 16), t >= 16)).astype(np.float32)
            canonical = canonical_raw(target, names).astype(np.float32)
            previous = np.concatenate((np.zeros((1, 21), np.float32), canonical[:-1]))
            features = np.concatenate((obs, commands, modes, previous), axis=1)
            if features.shape != (1200, 103) or not np.isfinite(features).all():
                raise RuntimeError(f"Feature contract failed: {trace_path}")
            episode = args.out / f"seed-{report['seed']}-world-{case['world_index']:04d}.npz"
            np.savez_compressed(episode, features=features, physical_target=target,
                                target_names=np.asarray(names), dt_s=np.asarray(0.02))
            entries.append({"group": f"batch-{report['seed']}", "tier": "strict" if case["strict_pass"] else "motion_duty_limited",
                            "seed": report["seed"], "world_index": case["world_index"],
                            "rail_v": report["rail_v"], "delay_ms": report["delay_ms"],
                            "walk_velocity_m_s": case["walk_velocity_m_s"],
                            "episode": str(episode.relative_to(ROOT)), "episode_sha256": sha(episode),
                            "source_trace_sha256": case["trace_sha256"]})
    if seen != expected:
        raise RuntimeError("Missing retained setting")
    result = {"schema": "robot-learning-lab.broad-handoff-v5-merged-teacher-bank.v1",
              "classification": "whole native teacher imitation only; no V5 learner score",
              "body_sha256": base["body_sha256"], "target_names": base["target_names"],
              "base_report_sha256": sha(args.base), "batch_sources": batch_sources,
              "whole_episodes": len(entries), "strict_teacher_passes": sum(e["tier"] == "strict" for e in entries),
              "motion_duty_limited": sum(e["tier"] == "motion_duty_limited" for e in entries),
              "frames": 1200 * len(entries), "entries": entries}
    (args.out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("whole_episodes", "strict_teacher_passes", "motion_duty_limited", "frames")}))


if __name__ == "__main__":
    main()
