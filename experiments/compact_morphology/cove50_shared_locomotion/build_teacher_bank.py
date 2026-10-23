"""Accept complete physical teacher episodes and emit balanced BC source data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-low-level-training-20261015/shared-forward-turn"
METHOD = ART / "method.json"
NATIVE = Path("/workspace/robot-learning-lab/ape-low-level-training-20261015/shared-forward-turn/teachers")
OUT = Path("/workspace/robot-learning-lab/ape-low-level-training-20261015/shared-forward-turn/bank")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_world(case: dict, world: dict, diagnostic: dict, index: int) -> tuple[bool, list[str], float]:
    active = diagnostic["active"][:, index].astype(bool)
    root_z = diagnostic["root_pose"][active, index, 2]
    hand = diagnostic["hand_force"][active, index]
    body = diagnostic["body_contact"][active, index]
    leg_force = diagnostic["actuator_force"][active, index, :]
    rated = diagnostic["rated_effort_limits_nm"]
    duty = float((leg_force >= 0.98 * rated).any(axis=1).mean()) if len(leg_force) else 0.0
    failures = []
    if (not world["ended_by_timeout"] or not active.all()
            or world["duration_s"] < case["duration_s"] - 0.02):
        failures.append("incomplete episode")
    if len(root_z) == 0 or float(root_z.min()) < 0.10:
        failures.append("root below 0.10 m")
    if len(hand) and float(hand.max()) >= 1.0:
        failures.append("hand-floor contact")
    if body.any():
        failures.append("body contact")
    if case["bucket"] == "forward":
        if world["command_progress_m"] < 0.30:
            failures.append("forward travel below 0.30 m")
        if world["alternating_touchdowns"] < 2:
            failures.append("fewer than two alternating touchdowns")
    elif world["command_yaw_progress_rad"] < 0.50:
        failures.append("directed yaw below 0.50 rad")
    return not failures, failures, duty


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"One-shot teacher bank already exists: {OUT}")
    method = json.loads(METHOD.read_text())
    for name, item in method["sources"].items():
        path = ROOT / item["path"]
        if not path.is_file() or sha(path) != item["sha256"]:
            raise RuntimeError(f"Frozen {name} source mismatch")
    OUT.mkdir(parents=True)
    accepted = {split: {bucket: {"observation": [], "action": [], "episode": []}
                        for bucket in ("forward", "left", "right")}
                for split in ("train", "dev")}
    summary = []
    for split in ("train", "dev"):
        manifest = json.loads((NATIVE / split / "result.json").read_text())
        if manifest["method_sha256"] != sha(METHOD):
            raise RuntimeError(f"{split} collection used another method")
        for row in manifest["rows"]:
            case = row["case"]
            report_path = NATIVE / row["report"]
            trace_path = NATIVE / row["trace"]
            imitation_path = NATIVE / row["imitation"]
            for path in (report_path, trace_path, imitation_path):
                if sha(path) != row["files_sha256"][path.name]:
                    raise RuntimeError(f"Source hash mismatch: {path}")
            report = json.loads(report_path.read_text())
            with np.load(trace_path, allow_pickle=False) as source:
                diag = {key: source[key] for key in ("active", "root_pose", "hand_force",
                                                     "body_contact", "actuator_force",
                                                     "rated_effort_limits_nm")}
            with np.load(imitation_path, allow_pickle=False) as source:
                obs = source["actor_observation"]
                action = source["policy_action"]
                command = source["command"]
                qpos = source["qpos"]
                qvel = source["qvel"]
                active = source["active"]
            first_action_sha = hashlib.sha256(action[0].tobytes()).hexdigest()
            initial_sha = hashlib.sha256(qpos[0].tobytes() + qvel[0].tobytes()
                                          + command[0].tobytes()).hexdigest()
            if (first_action_sha != report["first_action_sha256"]
                    or initial_sha != report["initial_state_sha256"]
                    or obs.shape[:2] != active.shape
                    or not np.allclose(obs[..., -3:], command, atol=1e-5)):
                raise RuntimeError(f"Pre-action source identity mismatch: {case['name']}")
            reasons = {}
            duties = []
            good = 0
            for index, world in enumerate(report["worlds"]):
                passed, failure, duty = evaluate_world(case, world, diag, index)
                duties.append(duty)
                for reason in failure:
                    reasons[reason] = reasons.get(reason, 0) + 1
                if not passed:
                    continue
                if not active[:, index].all():
                    raise RuntimeError(f"Accepted world has inactive teacher frame: {case['name']} {index}")
                data = accepted[split][case["bucket"]]
                data["observation"].append(obs[:, index].astype(np.float32))
                data["action"].append(action[:, index].astype(np.float32))
                data["episode"].append(np.full(obs.shape[0], case["seed"] * 1000 + index,
                                               dtype=np.int64))
                good += 1
            summary.append({"name": case["name"], "split": split,
                            "bucket": case["bucket"], "worlds": case["worlds"],
                            "accepted_whole_episodes": good,
                            "rejected_reasons": reasons,
                            "mean_rated_leg_fraction": round(float(np.mean(duties)), 4),
                            "report_sha256": sha(report_path),
                            "imitation_sha256": sha(imitation_path)})
    counts = {}
    for split, buckets in accepted.items():
        for bucket, data in buckets.items():
            episodes = len(data["observation"])
            counts[f"{split}/{bucket}"] = episodes
            if episodes:
                path = OUT / f"{split}-{bucket}.npz"
                np.savez_compressed(path,
                                    observation=np.concatenate(data["observation"]),
                                    action=np.concatenate(data["action"]),
                                    episode=np.concatenate(data["episode"]))
    thresholds = method["minimum_accepted_teacher_episodes"]
    ready = all(counts[f"{split}/{bucket}"] >= thresholds[split]
                for split in ("train", "dev") for bucket in ("forward", "left", "right"))
    result = {"schema": "robot-learning-lab.shared-forward-turn.teacher-bank.v1",
              "method_sha256": sha(METHOD), "ready_for_bc": ready,
              "accepted_whole_episodes": counts, "rows": summary,
              "data_sha256": {p.name: sha(p) for p in OUT.glob("*.npz")},
              "claim_limit": "Teacher motion examples; rated effort is reported, not silently treated as safe."}
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"ready_for_bc": ready, "accepted": counts}), flush=True)
    if not ready:
        raise RuntimeError("Too few whole teacher episodes for balanced imitation")


if __name__ == "__main__":
    main()
