"""Fresh paired native motion and no-reset command-switch score for shared actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PLAN = ROOT / "artifacts/ape-low-level-training-20261015/shared-forward-turn/score-method.json"
TRAIN_METHOD = ROOT / "artifacts/ape-low-level-training-20261015/shared-forward-turn/method.json"
RUN = Path("/workspace/robot-learning-lab/ape-low-level-training-20261015/shared-forward-turn")
OUT = RUN / "score"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_for(mode: str, case: dict, train_method: dict) -> Path:
    if mode == "source":
        key = "turn" if case["kind"] == "turn" else "forward"
        return ROOT / train_method["sources"][key]["path"]
    if mode == "bc":
        return RUN / "bc/bc-best.pt"
    return RUN / "ppo/train/candidate-final.pt"


def score_case(case: dict, mode: str, folder: Path, method: dict, train_method: dict) -> dict:
    name = f"{case['name']}-{mode}"
    report, trace, video, log = (folder / f"{name}.{suffix}" for suffix in
                                 ("json", "npz", "mp4", "log"))
    if any(p.exists() for p in (report, trace, video, log)):
        raise RuntimeError(f"Original score visit already exists: {name}")
    checkpoint = checkpoint_for(mode, case, train_method)
    if mode in ("bc", "ppo"):
        source_report = RUN / ("bc/report.json" if mode == "bc" else "ppo/train/report.json")
        key = "candidate_checkpoint_sha256"
        if sha(checkpoint) != json.loads(source_report.read_text())[key]:
            raise RuntimeError(f"Learned checkpoint changed: {mode}")
    else:
        key = "turn" if case["kind"] == "turn" else "forward"
        if sha(checkpoint) != train_method["sources"][key]["sha256"]:
            raise RuntimeError(f"Source checkpoint changed: {name}")
    body = ROOT / train_method["sources"]["body"]["path"]
    evaluator = ROOT / train_method["sources"]["evaluator"]["path"]
    command = [sys.executable, str(evaluator), str(checkpoint),
               "--task", case["task"], "--body-xml", str(body),
               "--physics-model", "frame50", "--initial-root-z", "0.231",
               "--yaw-only-head", "--arms-down", "--worlds", "1",
               "--seed", str(case["seed"]),
               "--velocity-x", str(case["command"][0]),
               "--velocity-y", str(case["command"][1]),
               "--angular-z", str(case["command"][2]),
               "--motor-rail-v", str(case["motor_rail_v"]),
               "--command-delay-ms", str(case["command_delay_ms"]),
               "--ankle-target-margin-rad", "0.02", "--physics-dt", "0.002",
               "--duration-s", str(case["duration_s"]),
               "--out", str(report), "--diagnostic-trace", str(trace),
               "--video", str(video)]
    if case["kind"] == "handoff":
        command += ["--switch-at-s", str(case["switch_at_s"]),
                    "--switch-velocity-x", "0", "--switch-velocity-y", "0",
                    "--switch-angular-z", str(case["successor_yaw_rad_s"])]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "experiments/compact_morphology/mjlab_ape/src") + ":" + str(ROOT)
    env["MUJOCO_GL"] = "egl"
    with log.open("x") as stream:
        proc = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                              stderr=subprocess.STDOUT, timeout=300, check=False)
    if proc.returncode:
        raise RuntimeError(f"Native score visit failed: {name}, exit={proc.returncode}")
    saved = json.loads(report.read_text())
    if (saved["body_sha256"] != train_method["sources"]["body"]["sha256"]
            or saved["checkpoint"] != str(checkpoint)
            or saved["seed"] != case["seed"] or saved["task"] != case["task"]
            or len(saved["worlds"]) != 1):
        raise RuntimeError(f"Native score identity mismatch: {name}")
    with np.load(trace, allow_pickle=False) as values:
        steps = round(case["duration_s"] * 50)
        if (len(values["command"]) != steps or not values["active"][:, 0].all()
                or not np.isfinite(values["root_pose"]).all()):
            raise RuntimeError(f"Incomplete score trace: {name}")
        roots = values["root_pose"][:, 0, :3]
        hand_force = float(values["hand_force"][:, 0].max())
        body_contact = bool(values["body_contact"][:, 0].any())
        if case["kind"] == "handoff":
            switch = round(case["switch_at_s"] * 50)
            if (not np.allclose(values["command"][:switch, 0], case["command"], atol=1e-5)
                    or not np.allclose(values["command"][switch:, 0],
                                       [0, 0, case["successor_yaw_rad_s"]], atol=1e-5)):
                raise RuntimeError(f"Wrong no-reset command transition: {name}")
            forward_at_switch = float(roots[switch - 1, 0] - roots[0, 0])
        else:
            if not np.allclose(values["command"][:, 0], case["command"], atol=1e-5):
                raise RuntimeError(f"Wrong steady command: {name}")
            forward_at_switch = None
        minimum_root_z = float(roots[:, 2].min())
    world = saved["worlds"][0]
    checks = {
        "full_duration": world["ended_by_timeout"] and world["duration_s"] >= case["duration_s"] - 0.02,
        "upright": minimum_root_z >= method["physical_gate"]["minimum_root_z_m"],
        "no_forbidden_contact": (hand_force < method["physical_gate"]["maximum_hand_force_n"]
                                 and not body_contact),
    }
    if case["kind"] == "forward":
        checks["command_progress"] = world["command_progress_m"] >= method["physical_gate"]["forward_progress_m"]
        checks["alternating_touchdowns"] = world["alternating_touchdowns"] >= method["physical_gate"]["forward_alternating_touchdowns"]
    elif case["kind"] == "turn":
        checks["directed_yaw"] = world["command_yaw_progress_rad"] >= method["physical_gate"]["turn_directed_yaw_rad"]
    else:
        checks["predecessor_forward"] = forward_at_switch >= method["physical_gate"]["handoff_predecessor_forward_m"]
        directed = world["switch"]["yaw_after_command_rad"] * (1 if case["successor_yaw_rad_s"] > 0 else -1)
        checks["successor_turn"] = directed >= method["physical_gate"]["handoff_successor_directed_yaw_rad"]
    return {"case": case["name"], "mode": mode, "kind": case["kind"],
            "seed": case["seed"], "pass": all(checks.values()),
            "checks": checks, "failure_reasons": [name for name, ok in checks.items() if not ok],
            "forward_at_switch_m": forward_at_switch,
            "command_progress_m": world["command_progress_m"],
            "directed_yaw_rad": (world["command_yaw_progress_rad"] if case["kind"] == "turn"
                                 else directed if case["kind"] == "handoff" else None),
            "rated_leg_fraction": world["fraction_frames_with_saturated_leg"],
            "hand_force_n": hand_force, "body_contact": body_contact,
            "minimum_root_z_m": minimum_root_z,
            "initial_state_sha256": saved["initial_state_sha256"],
            "first_action_sha256": saved["first_action_sha256"],
            "report": str(report.relative_to(OUT)), "report_sha256": sha(report),
            "trace": str(trace.relative_to(OUT)), "trace_sha256": sha(trace),
            "video": str(video.relative_to(OUT)), "video_sha256": sha(video)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("development", "stress"), required=True)
    args = parser.parse_args()
    method = json.loads(PLAN.read_text())
    train_method = json.loads(TRAIN_METHOD.read_text())
    if method["training_method_sha256"] != sha(TRAIN_METHOD):
        raise RuntimeError("Training method differs")
    if sha(ROOT / method["scorer"]["path"]) != method["scorer"]["sha256"]:
        raise RuntimeError("Frozen scorer changed")
    for item in train_method["sources"].values():
        path = ROOT / item["path"]
        if sha(path) != item["sha256"]:
            raise RuntimeError(f"Frozen source changed: {path}")
    if args.phase == "stress":
        dev = json.loads((OUT / "development/report.json").read_text())
        if (dev["method_sha256"] != sha(PLAN)
                or not any(r["pass"] for r in dev["rows"]
                           if r["mode"] in ("bc", "ppo") and r["kind"] == "handoff")):
            raise RuntimeError("No learned development handoff signal; stress remains unopened")
    folder = OUT / args.phase
    if folder.exists():
        raise RuntimeError(f"One-shot score phase already exists: {folder}")
    folder.mkdir(parents=True)
    rows = []
    for case in method[f"{args.phase}_cases"]:
        pair = []
        for mode in ("source", "bc", "ppo"):
            row = score_case(case, mode, folder, method, train_method)
            rows.append(row)
            pair.append(row)
            (folder / "partial.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(case["name"], mode, row["pass"], flush=True)
        if len({row["initial_state_sha256"] for row in pair}) != 1:
            raise RuntimeError(f"Matched starting state differs: {case['name']}")
    result = {"schema": "robot-learning-lab.shared-forward-turn.native-score.v1",
              "phase": args.phase, "method_sha256": sha(PLAN),
              "training_method_sha256": sha(TRAIN_METHOD),
              "visits": len(rows), "rows": rows, "optimizer_updates": 0,
              "claim_limit": "Fresh exploratory native physical score on experimental Direct Frame only."}
    (folder / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"phase": args.phase, "visits": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
