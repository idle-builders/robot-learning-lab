"""Collect a small source-labelled CPU imitation pilot; CUDA scores follow separately."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from multicommand_bc import load_episode


ROOT = Path(__file__).resolve().parents[3]
EVALUATORS = ROOT / "experiments/compact_morphology/mjlab_ape"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
TEACHERS = {
    "squat": ROOT / "artifacts/ape-cove50-squat-recovery-20261009/squat-teacher-bc-blendmask.pt",
    "both_arms": ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-arms-varied149.pt",
}
SEEDS = ((2144, "train"), (54101, "train"), (54102, "train"),
         (54103, "validation"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases-json", type=Path,
                        help="Explicit task/seed/split and supported motor/smoothing cases")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{EVALUATORS / 'src'}:{ROOT}"
    accepted = []
    rejected = []
    checker = ROOT / "experiments/compact_morphology/cove50_low_level/check_multicommand_trace.py"
    if args.cases_json:
        cases = json.loads(args.cases_json.read_text())["cases"]
    else:
        cases = [{"task": task, "seed": seed, "split": split,
                  "motor_rail_v": 12.0, "command_delay_ms": 0.0,
                  "action_smoothing_alpha": 0.2}
                 for task in TEACHERS for seed, split in SEEDS]
    for case in cases:
        task, seed, split = case["task"], case["seed"], case["split"]
        if task not in TEACHERS or split not in ("train", "validation"):
            raise ValueError(f"unsupported collection case: {case}")
        rail = float(case["motor_rail_v"])
        if not 10.0 <= rail <= 12.0:
            raise ValueError(f"rail outside source-supported interval: {case}")
        prefix = out / f"{task}-seed{seed}"
        report = prefix.with_suffix(".json")
        trace = prefix.with_suffix(".npz")
        check = out / f"{task}-seed{seed}-trace-check.json"
        log = prefix.with_suffix(".log")
        common = [sys.executable, str(EVALUATORS /
                  ("evaluate_posture.py" if task == "squat" else "evaluate_arm.py")),
                  str(TEACHERS[task]), "--out", str(report), "--seed", str(seed),
                  "--device", "cpu", "--cove50-body-xml", str(BODY),
                  "--motor-rail-v", str(rail),
                  "--command-delay-ms", str(case["command_delay_ms"]),
                  "--action-smoothing-alpha", str(case["action_smoothing_alpha"]),
                  "--native-trajectory-out", str(trace)]
        command = (common + ["--task", "Mjlab-Posture-Flat-Ape-Elbows-HeadingObserved",
                             "--mode", "cycle", "--squat-active-mask",
                             "--action-abs-bound", "1", "--expose-squat-blend",
                             "--mask-action-history"] if task == "squat" else
                   common + ["--shoulder-target-margin-rad", "0.05"])
        with log.open("w") as stream:
            run = subprocess.run(command, cwd=ROOT, env=env,
                                 stdout=stream, stderr=subprocess.STDOUT, check=False)
        item = {"task": task, "split": split, "conditions": case,
                "trace": str(trace),
                "report": str(report), "check": str(check)}
        if run.returncode:
            rejected.append({**item, "reason": f"evaluator_exit_{run.returncode}"})
            continue
        checked = subprocess.run([sys.executable, str(checker), str(trace), str(report),
                                  "--out", str(check)], cwd=ROOT, capture_output=True,
                                 text=True, check=False)
        if checked.returncode:
            rejected.append({**item, "reason": checked.stderr.strip()})
            continue
        try:
            load_episode(item)
        except ValueError as exc:
            rejected.append({**item, "reason": str(exc)})
            continue
        accepted.append(item)
        print(f"accepted {task} seed{seed} {split}", flush=True)
    manifest = {"schema": "robot-learning-lab.multicommand-cpu-pilot.v1",
                "claim_limit": "CPU teacher data for imitation; no CUDA skill score",
                "episodes": accepted, "rejected": rejected}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"accepted": len(accepted), "rejected": len(rejected)}, indent=2))


if __name__ == "__main__":
    main()
