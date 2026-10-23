"""Audit frozen jaw-only touch rules on retained native reach episodes.

Finger/toy contact is hidden evaluator truth. These correlated traces were
generated for low-level reach experiments on the now-rejected Direct Frame;
this replay does not qualify a physical contact sensor or possession claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
REACH = ROOT / "artifacts/ape-cove50-side-reach-20261009"
PREVIOUS = ROOT / "artifacts/ape-cove50-contact-20261009/selected-reach-audit.json"
OUT = ROOT / "artifacts/ape-cove50-contact-20261012/native-cohort-audit.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def channel_result(values: np.ndarray, truth: np.ndarray, dt: float,
                   model: dict) -> dict:
    predicted = (values >= model["threshold"] if model["direction"] == "ge"
                 else values <= model["threshold"])
    first = np.flatnonzero(predicted)
    first_true = np.flatnonzero(truth)
    return {
        "first_detection_s": round(float(first[0] * dt), 3) if len(first) else None,
        "first_true_contact_s": (round(float(first_true[0] * dt), 3)
                                 if len(first_true) else None),
        "predicted_touch_s": round(float(predicted.sum() * dt), 3),
        "false_touch_s": round(float((predicted & ~truth).sum() * dt), 3),
        "missed_touch_s": round(float((~predicted & truth).sum() * dt), 3),
        "false_touch_event_without_contact": bool(predicted.any() and not truth.any()),
    }


def main() -> None:
    previous = json.loads(PREVIOUS.read_text())
    fit = ROOT / previous["fit_case"]
    fit_report = json.loads(fit.with_suffix(".json").read_text())
    body_sha = fit_report["body_sha256"]
    rows = []
    exclusions = []
    for path in sorted(REACH.rglob("*.npz")):
        report_path = path.with_suffix(".json")
        if not report_path.exists():
            exclusions.append({"source": str(path.relative_to(ROOT)),
                               "reason": "no_native_episode_report"})
            continue
        report = json.loads(report_path.read_text())
        if report.get("body_sha256") != body_sha:
            exclusions.append({"source": str(path.relative_to(ROOT)),
                               "reason": "different_body_or_no_body_hash"})
            continue
        if report.get("hand", "right") != "right":
            exclusions.append({"source": str(path.relative_to(ROOT)),
                               "reason": "left_hand_episode"})
            continue
        with np.load(path, allow_pickle=False) as data:
            required = {"actor_observation", "action_names", "finger_toy_contact",
                        "step_dt_s"}
            if not required.issubset(data.files):
                exclusions.append({"source": str(path.relative_to(ROOT)),
                                   "reason": "missing_required_native_trace_fields"})
                continue
            names = list(data["action_names"])
            if "right_gripper" not in names:
                exclusions.append({"source": str(path.relative_to(ROOT)),
                                   "reason": "no_right_gripper_channel"})
                continue
            action_name = ("executed_action" if "executed_action" in data.files else
                           "teacher_action" if "teacher_action" in data.files else None)
            if action_name is None:
                exclusions.append({"source": str(path.relative_to(ROOT)),
                                   "reason": "no_executed_or_teacher_action"})
                continue
            idx = names.index("right_gripper")
            truth = data["finger_toy_contact"].astype(bool)
            dt = float(data["step_dt_s"])
            position = data["actor_observation"][:, idx].astype(float)
            command = data[action_name][:, idx].astype(float)
            if not (len(position) == len(command) == len(truth)):
                raise ValueError(f"Mismatched trace lengths: {path}")
            signals = {
                "jaw_joint_position_rad": channel_result(
                    position, truth, dt, previous["thresholds"]["jaw_joint_position_rad"]),
                "jaw_command_raw": channel_result(
                    command, truth, dt, previous["thresholds"]["jaw_command_raw"]),
            }
            video = path.with_suffix(".mp4")
            rows.append({
                "source": str(path.relative_to(ROOT)),
                "source_sha256": sha(path),
                "native_report": str(report_path.relative_to(ROOT)),
                "native_report_sha256": sha(report_path),
                "video": str(video.relative_to(ROOT)) if video.exists() else None,
                "body_sha256": report["body_sha256"],
                "cohort": str(path.parent.relative_to(REACH)),
                "method": report.get("method"),
                "survived": report.get("survived"),
                "duration_s": round(len(truth) * dt, 3),
                "true_finger_toy_contact_s": round(float(truth.sum() * dt), 3),
                "signals": signals,
            })
    if not rows:
        raise RuntimeError("No native right-gripper trace with matching body")
    full_no_contact = [row for row in rows if row["duration_s"] >= 7.5 and
                       row["true_finger_toy_contact_s"] == 0]
    summary = {
        "native_matching_body_episodes": len(rows),
        "no_finger_toy_contact_episodes": sum(
            row["true_finger_toy_contact_s"] == 0 for row in rows),
        "complete_no_finger_toy_contact_episodes": len(full_no_contact),
        "complete_no_contact_false_touch_events": {
            channel: sum(row["signals"][channel]["false_touch_event_without_contact"]
                         for row in full_no_contact)
            for channel in previous["thresholds"]
        },
        "complete_no_contact_false_touch_sources": [
            row["source"] for row in full_no_contact
            if row["signals"]["jaw_joint_position_rad"]
                  ["false_touch_event_without_contact"]
        ],
    }
    result = {
        "schema": "robot-learning-lab.cove50-native-contact-cohort-audit.v1",
        "current_design_status": json.loads(
            (ROOT / "robot/ape-current-design.json").read_text())["status"],
        "historical_body_sha256": body_sha,
        "frozen_fit_source": previous["fit_case"],
        "frozen_fit_source_sha256": sha(fit),
        "frozen_thresholds": previous["thresholds"],
        "evaluator_truth": "Native MJLab finger-tip to cube contact; never an actor sensor",
        "scope": "Offline audit of correlated saved right-hand native reaches on the historical rejected Direct Frame; no empty-toy, possession, real current, tactile or social-touch measurement",
        "summary": summary,
        "episodes": rows,
        "exclusions": exclusions,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
