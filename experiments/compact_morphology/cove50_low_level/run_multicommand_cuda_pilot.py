"""Bounded source-checked native CUDA replay of independent squat/arm BC starts."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path("/root/robot-learning-lab")
OUT = Path("/workspace/multicommand-cuda-pilot")
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
SOURCE = {
    "body": BODY,
    "squat_student": ROOT / "artifacts/ape-low-level-training-20261015/multicommand-cpu-pilot-v1/student-squat-varied-history.pt",
    "arms_student": ROOT / "artifacts/ape-low-level-training-20261015/multicommand-cpu-pilot-v1/student-arms-only.pt",
    "squat_teacher": ROOT / "artifacts/ape-cove50-squat-recovery-20261009/squat-teacher-bc-blendmask.pt",
    "arms_teacher": ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-arms-varied149.pt",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(task: str, report: dict) -> tuple[bool, dict]:
    if task == "squat":
        a, b, c = (report[k] for k in ("initial_hold", "middle_hold", "final_hold"))
        depth = a["mean_pelvis_z_m"] - b["mean_pelvis_z_m"]
        return_error = abs(a["mean_pelvis_z_m"] - c["mean_pelvis_z_m"])
        passed = (report["duration_s"] >= 9.99 and not report["ended_early"]
                  and depth >= 0.04 and return_error <= 0.015
                  and all(x["both_feet_grounded_fraction"] >= 0.95
                          and x["hand_contact_frames"] == 0
                          and x["body_contact_fraction"] == 0
                          and x["saturated_leg_fraction"] <= 0.1
                          for x in (a, b, c)))
        return passed, {"depth_m": depth, "return_error_m": return_error,
                        "max_tilt_deg": max(x["max_tilt_deg"] for x in (a, b, c))}
    a, b, c = (report[k] for k in ("before", "raised", "lowered"))
    rise = min(b["hand_height_m"][i] - a["hand_height_m"][i] for i in range(2))
    return_error = max(abs(c["hand_height_m"][i] - a["hand_height_m"][i])
                       for i in range(2))
    passed = (report["duration_s"] >= 9.99 and rise >= 0.15
              and return_error <= 0.04 and report["peak_hand_force_n"] < 1.0
              and all(x["shoulder_saturation_fraction"] <= 0.1
                      for x in (a, b, c)))
    return passed, {"minimum_hand_rise_m": rise, "return_error_m": return_error,
                    "peak_hand_force_n": report["peak_hand_force_n"]}


def main() -> None:
    plan = json.loads((ROOT / "artifacts/ape-low-level-training-20261015/multicommand-cpu-pilot-v1/cuda-pilot-plan.json").read_text())
    for name, path in SOURCE.items():
        if sha(path) != plan["source"][name]["sha256"]:
            raise RuntimeError(f"source differs: {name}")
    if OUT.exists():
        raise RuntimeError("refusing to overlay existing CUDA pilot")
    OUT.mkdir(parents=True)
    (OUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'experiments/compact_morphology/mjlab_ape/src'}:{ROOT}"
    env["MUJOCO_GL"] = "egl"
    env["NVIDIA_DRIVER_CAPABILITIES"] = "compute,utility,graphics"
    all_results = []
    for case in plan["cases"]:
        task = case["task"]
        evaluator = ROOT / "experiments/compact_morphology/mjlab_ape" / (
            "evaluate_posture.py" if task == "squat" else "evaluate_arm.py")
        paired = []
        for route in case["routes"]:
            key = ("squat" if task == "squat" else "arms") + "_" + route
            label = f"{task}-{route}-seed{case['seed']}"
            report_path = OUT / f"{label}.json"
            trace_path = OUT / f"{label}.npz"
            video_path = OUT / f"{label}.mp4"
            command = [sys.executable, str(evaluator), str(SOURCE[key]),
                       "--out", str(report_path), "--native-trajectory-out", str(trace_path),
                       "--video", str(video_path), "--cove50-body-xml", str(BODY),
                       "--device", "cuda:0", "--seed", str(case["seed"]),
                       "--motor-rail-v", str(case["motor_rail_v"]),
                       "--command-delay-ms", str(case["command_delay_ms"]),
                       "--action-smoothing-alpha", str(case["action_smoothing_alpha"])]
            if route == "student":
                command.append("--student-bc")
            if task == "squat":
                command += ["--task", "Mjlab-Posture-Flat-Ape-Elbows-HeadingObserved",
                            "--mode", "cycle", "--squat-active-mask", "--action-abs-bound", "1",
                            "--expose-squat-blend", "--mask-action-history"]
            else:
                command += ["--shoulder-target-margin-rad", "0.05"]
            with (OUT / f"{label}.log").open("w") as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise RuntimeError(f"{label} exited {result.returncode}; see log")
            report = json.loads(report_path.read_text())
            if report["body_sha256"] != plan["source"]["body"]["sha256"]:
                raise RuntimeError(f"{label} body differs")
            with np.load(trace_path, allow_pickle=False) as trace:
                start = {name: trace[name].copy() for name in
                         ("initial_qpos", "initial_qvel", "actor_observation")}
            passed, metrics = score(task, report)
            item = {"task": task, "route": route, "seed": case["seed"],
                    "motor_rail_v": case["motor_rail_v"],
                    "command_delay_ms": case["command_delay_ms"],
                    "action_smoothing_alpha": case["action_smoothing_alpha"],
                    "pass": bool(passed), "metrics": metrics,
                    "report": report_path.name, "report_sha256": sha(report_path),
                    "trace": trace_path.name, "trace_sha256": sha(trace_path),
                    "video": video_path.name, "video_sha256": sha(video_path)}
            paired.append((item, start))
            all_results.append(item)
            (OUT / "partial-matrix.json").write_text(json.dumps(all_results, indent=2) + "\n")
            print(label, "PASS" if passed else "FAIL", flush=True)
        a, b = (start for _, start in paired)
        gaps = {name: float(np.max(np.abs(a[name][0] - b[name][0]
                                           if name == "actor_observation"
                                           else a[name] - b[name])))
                for name in a}
        paired[0][0]["paired_start_gaps"] = gaps
        paired[1][0]["paired_start_gaps"] = gaps
        if gaps["initial_qpos"] > 1e-4 or gaps["initial_qvel"] > 1e-4 or gaps["actor_observation"] > 1e-4:
            raise RuntimeError(f"invalid paired start: {task} {case['seed']} {gaps}")
    (OUT / "matrix.json").write_text(json.dumps(all_results, indent=2) + "\n")
    print("complete", len(all_results), flush=True)


if __name__ == "__main__":
    main()
