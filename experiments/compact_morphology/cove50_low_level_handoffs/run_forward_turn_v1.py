"""One-shot native forward→turn handoff screen with matched saved-actor controls."""

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
METHOD = ROOT / "artifacts/ape-cove50-low-level-handoffs/forward-turn-v1/method.json"
OUT = Path("/workspace/robot-learning-lab/ape-cove50-low-level-handoffs/forward-turn-v1")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_sources(method: dict) -> None:
    for entry in method["source"].values():
        path = ROOT / entry["path"]
        if not path.is_file() or sha(path) != entry["sha256"]:
            raise RuntimeError(f"Source mismatch: {path}")


def execute(case: dict, mode: str, folder: Path, method: dict) -> dict:
    label = f"seed{case['seed']}-{mode}"
    report, trace, video, log = (folder / f"{label}.{suffix}" for suffix in
                                 ("json", "npz", "mp4", "log"))
    if any(path.exists() for path in (report, trace, video, log)):
        raise RuntimeError(f"Original visit already exists: {label}")
    evaluator = ROOT / method["source"]["evaluator"]["path"]
    actor = ROOT / method["source"]["actor"]["path"]
    body = ROOT / method["source"]["body"]["path"]
    sign = case["turn_sign"]
    command = [sys.executable, str(evaluator), str(actor),
               "--task", "Mjlab-Velocity-Flat-Ape-Elbows", "--body-xml", str(body),
               "--physics-model", "frame50", "--initial-root-z", "0.231",
               "--yaw-only-head", "--arms-down", "--worlds", "1",
               "--seed", str(case["seed"]), "--velocity-x",
               "0" if mode == "turn_control" else "0.25", "--velocity-y", "0",
               "--angular-z", str(sign * 0.3 if mode == "turn_control" else 0),
               "--motor-rail-v", str(case["motor_rail_v"]),
               "--command-delay-ms", str(case["command_delay_ms"]),
               "--ankle-target-margin-rad", "0.02", "--physics-dt", "0.002",
               "--duration-s", str(case["duration_s"]), "--out", str(report),
               "--diagnostic-trace", str(trace)]
    if mode in ("handoff", "forward_control", "turn_control"):
        command += ["--switch-at-s", str(case["switch_at_s"]),
                    "--switch-velocity-x", "0" if mode != "forward_control" else "0.25",
                    "--switch-velocity-y", "0",
                    "--switch-angular-z", str(sign * 0.3 if mode != "forward_control" else 0)]
    if case["record_video"]:
        command += ["--video", str(video)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "experiments/compact_morphology/mjlab_ape/src") + ":" + str(ROOT)
    env["MUJOCO_GL"] = "egl"
    with log.open("x") as stream:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, timeout=300, check=False)
    if process.returncode:
        raise RuntimeError(f"Native visit failed ({process.returncode}): {log}")
    saved = json.loads(report.read_text())
    if (saved["body_sha256"] != method["source"]["body"]["sha256"]
            or saved["seed"] != case["seed"]
            or len(saved["worlds"]) != 1
            or saved["candidate_motor_rail_v"] != case["motor_rail_v"]
            or saved["candidate_command_delay_ms"] != case["command_delay_ms"]
            or saved["switch_at_s"] != case["switch_at_s"]):
        raise RuntimeError(f"Native report identity mismatch: {label}")
    with np.load(trace, allow_pickle=False) as values:
        commands = values["command"][:, 0, :]
        roots = values["root_pose"][:, 0, :3]
        active = values["active"][:, 0].astype(bool)
        if len(commands) != round(case["duration_s"] * 50) or not active.all():
            raise RuntimeError(f"Incomplete trace: {label}")
        before = round(case["switch_at_s"] * 50)
        expected_first = [0, 0, sign * 0.3] if mode == "turn_control" else [0.25, 0, 0]
        if not np.allclose(commands[:before], expected_first, atol=1e-6):
            raise RuntimeError(f"Wrong initial command: {label}")
        expected_later = [0, 0, sign * 0.3] if mode in ("handoff", "turn_control") else [0.25, 0, 0]
        if not np.allclose(commands[before:], expected_later, atol=1e-6):
            raise RuntimeError(f"Wrong successor command: {label}")
        hand_force = float(values["hand_force"][active, 0].max())
        body_contact = bool(values["body_contact"][active, 0].any())
        min_root_z = float(roots[active, 2].min())
        forward_at_switch = float(roots[before - 1, 0] - roots[0, 0])
    world = saved["worlds"][0]
    gate = method["physical_screen"]
    directed_yaw = (float(world.get("switch", {}).get("yaw_after_command_rad", 0)) * sign)
    checks = ({"full_duration": world["ended_by_timeout"] and world["duration_s"] >= case["duration_s"] - 0.02,
               "predecessor_forward": forward_at_switch >= gate["minimum_forward_at_switch_m"],
               "successor_turn": directed_yaw >= gate["minimum_directed_post_switch_yaw_rad"],
               "no_forbidden_contact": hand_force < gate["maximum_hand_floor_force_n"] and not body_contact,
               "upright": min_root_z >= gate["minimum_root_z_m"]}
              if mode == "handoff" else {})
    return {"seed": case["seed"], "mode": mode, "conditions": case,
            "first_state_sha256": saved["initial_state_sha256"],
            "first_action_sha256": saved["first_action_sha256"],
            "duration_s": world["duration_s"], "checks": checks,
            "rough_handoff_pass": all(checks.values()) if checks else None,
            "forward_at_switch_m": round(forward_at_switch, 4),
            "directed_post_switch_yaw_rad": round(directed_yaw, 3),
            "hand_force_n": round(hand_force, 3), "body_contact": body_contact,
            "minimum_root_z_m": round(min_root_z, 4),
            "rated_leg_fraction": world["fraction_frames_with_saturated_leg"],
            "report": str(report.relative_to(OUT)), "report_sha256": sha(report),
            "trace": str(trace.relative_to(OUT)), "trace_sha256": sha(trace),
            "video": str(video.relative_to(OUT)) if case["record_video"] else None,
            "video_sha256": sha(video) if case["record_video"] else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("sentinel", "development", "held"), required=True)
    args = parser.parse_args()
    method = json.loads(METHOD.read_text())
    if method["schema"] != "robot-learning-lab.forward-turn-v1.method.v1":
        raise RuntimeError("Wrong method")
    check_sources(method)
    cases = ([method["sentinel"]] if args.phase == "sentinel"
             else method[f"{args.phase}_cases"])
    folder = OUT / args.phase
    if folder.exists():
        raise RuntimeError(f"One-shot phase already exists: {folder}")
    folder.mkdir(parents=True)
    rows = []
    for case in cases:
        modes = ("handoff",) if args.phase == "sentinel" else (
            "handoff", "forward_control", "turn_control")
        pair = []
        for mode in modes:
            row = execute(case, mode, folder, method)
            pair.append(row)
            rows.append(row)
            (folder / "partial.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(f"{case['seed']} {mode}: {row['rough_handoff_pass']}", flush=True)
        if len(pair) > 1:
            if pair[0]["first_state_sha256"] != pair[1]["first_state_sha256"]:
                raise RuntimeError(f"Forward control state differs: {case['seed']}")
            if pair[0]["first_action_sha256"] != pair[1]["first_action_sha256"]:
                raise RuntimeError(f"Forward control first action differs: {case['seed']}")
    result = {"schema": "robot-learning-lab.forward-turn-v1.result.v1",
              "phase": args.phase, "method_sha256": sha(METHOD),
              "source": method["source"], "visits": len(rows), "optimizer_updates": 0,
              "rough_handoff_passes": sum(r["rough_handoff_pass"] is True for r in rows),
              "rows": rows,
              "claim_limit": "Fresh saved-actor command-switch simulation only; no new learner fit, motor qualification or body transfer."}
    (folder / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"phase": args.phase, "visits": len(rows),
                      "passes": result["rough_handoff_passes"]}), flush=True)


if __name__ == "__main__":
    main()
