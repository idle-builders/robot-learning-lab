"""Source-check V4 whole handoffs for a broader, explicitly tiered V5 seed bank.

Duty-only misses remain labelled. They are usable imitation examples of completed
motion, never counted as strict physical passes or fresh learner scores.
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
SCREEN = ROOT / "artifacts/ape-low-level-training-20261015/broad-handoff-v4-l4-sentinel-r2"
SCORE = ROOT / "artifacts/ape-low-level-training-20261015/broad-handoff-v4-l4-score"
BODY_SHA = "9841b2e9f57dafe7842873e80c43431f506721d729cef98d92c574ca52dc1d0c"
ACTOR_SHAS = {"gait": "0e8c6862108bd880e330fd8c07d19bf012d4a034f8be6a2ef808de190e5f301f",
              "stop": "1d01d490e3754bf9cb14e19588d784cb7bd1c9d6e82ae0b216c3ce06590e27a2",
              "squat": "0b20cc71d765ce6913adb1f70587fcbf59442bcff25b7645acd64a9e643028f4",
              "arms": "d3e718e3632b4e002adf08f16028700e16be18c4392702d975c9c38f24005774"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_rows() -> tuple[list[dict], dict]:
    screen = json.loads((SCREEN / "screen-report.json").read_text())
    screen_check = json.loads((SCREEN / "screen-source-verification.json").read_text())
    score = json.loads((SCORE / "report.json").read_text())
    score_check = json.loads((SCORE / "source-verification.json").read_text())
    if (screen["visits"] != 16 or screen["accepted_whole_episodes"] != 9
            or screen_check["native_summary_sha256"] != sha(SCREEN / "screen-report.json")
            or score["completed"] != 4 or score["teacher_passes"] != 4
            or score_check["score_report_sha256"] != sha(SCORE / "report.json")
            or score_check["completed"] != 4):
        raise RuntimeError("V4 source coverage or score verification changed")
    # The screen verification schema retains row-level source hashes.
    if len(screen_check["rows"]) != 16:
        raise RuntimeError("V4 screen source verification incomplete")
    rows = []
    for row in screen["rows"]:
        reasons = row["failure_reasons"]
        if row["pass"] or reasons == ["rated_duty_windows_pass"]:
            rows.append({"pool": "screen", "case": row["case"], "source": row,
                         "tier": "strict" if row["pass"] else "motion_duty_limited"})
    for row in score["rows"]:
        teacher = row["teacher"]
        if not teacher["pass"]:
            raise RuntimeError("Changed V4 teacher score")
        rows.append({"pool": "opened_score_teacher", "case": row["case"],
                     "source": teacher, "tier": "strict"})
    return rows, {"screen_report_sha256": sha(SCREEN / "screen-report.json"),
                  "screen_verification_sha256": sha(SCREEN / "screen-source-verification.json"),
                  "score_report_sha256": sha(SCORE / "report.json"),
                  "score_verification_sha256": sha(SCORE / "source-verification.json")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("V5 bank output already exists; preserve original extraction")
    rows, source_hashes = source_rows()
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    target_names = None
    for item in rows:
        case, source = item["case"], item["source"]
        base = SCREEN if item["pool"] == "screen" else SCORE
        report_path = base / Path(source["report"]).name
        trace_path = base / Path(source["trace"]).name
        if sha(report_path) != source["report_sha256"] or sha(trace_path) != source["trace_sha256"]:
            raise RuntimeError(f"Native source changed for seed {case['seed']}")
        report = json.loads(report_path.read_text())
        if (report["body_sha256"] != BODY_SHA or report["device"] != "cuda:0"
                or report["seed"] != case["seed"] or report["steps"] != 1200
                or report["rail_v"] != case["rail_v"]
                or report["delay_ms"] != case["delay_ms"]
                or report["walk_velocity_m_s"] != case["walk_velocity_m_s"]
                or abs(report["dt_s"] - 0.02) > 1e-8
                or report["checkpoint_sha256"] != ACTOR_SHAS
                or bool(report["full_continuity_gate_pass"]) != (item["tier"] == "strict")
                or report["ended_early"]):
            raise RuntimeError(f"Native report identity changed for seed {case['seed']}")
        if item["tier"] == "motion_duty_limited":
            if not (report["squat_depth_return_pass"] and report["arm_rise_return_pass"]
                    and report["no_hand_or_body_contact"] and report["support_windows_pass"]
                    and report["walk_distance_m"] >= 0.06
                    and report["stop_tail_xy_m"] is not None
                    and report["stop_tail_xy_m"] <= 0.03
                    and not report["rated_duty_windows_pass"]):
                raise RuntimeError(f"Duty-only classification changed for seed {case['seed']}")
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
                or np.max(np.abs(post[:-1] - pre[1:])) > 1e-9
                or not all(np.isfinite(x).all() for x in (obs, arm_obs, action, target))):
            raise RuntimeError(f"Whole native trace changed for seed {case['seed']}")
        if target_names is None:
            target_names = names
        elif names != target_names:
            raise RuntimeError("Teacher target order differs")
        if np.max(np.abs(processed(action, names) - target)) > 1e-5:
            raise RuntimeError(f"Native target/action map changed for seed {case['seed']}")
        canonical = canonical_raw(target, names).astype(np.float32)
        previous = np.concatenate((np.zeros((1, 21), np.float32), canonical[:-1]))
        extra = build_features(report["timeseries"], case, arm_obs, previous)
        phases = ("walk", "stop", "posture", "arms")
        mode = np.asarray([[float(r["phase"] == phase) for phase in phases]
                           for r in report["timeseries"]], dtype=np.float32)
        features = np.concatenate((obs, extra[:, :6], mode, extra[:, 6:]), axis=1)
        if features.shape != (1200, 103) or not np.isfinite(features).all():
            raise RuntimeError(f"Feature contract changed for seed {case['seed']}")
        episode = out / f"seed-{case['seed']}.npz"
        np.savez_compressed(episode, features=features, physical_target=target,
                            target_names=np.asarray(names), seed=np.asarray(case["seed"]),
                            dt_s=np.asarray(0.02))
        entries.append({"seed": case["seed"], "pool": item["pool"], "tier": item["tier"],
                        "case": case, "episode": str(episode.relative_to(ROOT)),
                        "episode_sha256": sha(episode),
                        "native_report_sha256": source["report_sha256"],
                        "native_trace_sha256": source["trace_sha256"]})
    if len(entries) != 17 or len({entry["seed"] for entry in entries}) != 17:
        raise RuntimeError("Expected 13 screen plus four opened score whole families")
    result = {"schema": "robot-learning-lab.broad-handoff-v5-tiered-bank.v1",
              "classification": "source-checked whole native teacher data; no V5 learner or fresh score",
              "claim_limit": "Duty-limited complete motions are imitation data only; opened V4 score teachers are now training data and cannot be reused as V5 scores",
              "body_sha256": BODY_SHA, "checkpoint_sha256": ACTOR_SHAS,
              "extractor_sha256": sha(Path(__file__)), "source_hashes": source_hashes,
              "whole_episodes": len(entries), "strict_teacher_passes": sum(e["tier"] == "strict" for e in entries),
              "motion_duty_limited": sum(e["tier"] == "motion_duty_limited" for e in entries),
              "frames": 1200 * len(entries), "feature_width": 103,
              "physical_target_width": 21, "target_names": target_names,
              "optimizer_updates": 0, "fresh_learner_score": None, "entries": entries}
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("whole_episodes", "strict_teacher_passes", "motion_duty_limited", "frames")}))


if __name__ == "__main__":
    main()
