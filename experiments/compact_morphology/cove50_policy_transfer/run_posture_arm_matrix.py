"""Fresh, reserved-seed native posture and arm replay matrix for Cove."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-cove50-transfer-20261008"
BODY = OUT / "cove50-direct-training.xml"
EVAL = ROOT / "experiments/compact_morphology/mjlab_ape"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--squat", type=Path)
    parser.add_argument("--arms", type=Path)
    parser.add_argument("--independent-arms", type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--lessons", choices=("all", "squat", "arms"), default="all")
    parser.add_argument("--case", action="append", help="Run only named cases")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--cove50-body-xml", type=Path, default=BODY)
    parser.add_argument("--motor-rail-v", type=float, default=12.0)
    parser.add_argument("--command-delay-ms", type=float, default=0.0)
    parser.add_argument("--shoulder-target-margin-rad", type=float)
    parser.add_argument("--leg-target-margin-rad", type=float)
    parser.add_argument("--action-smoothing-alpha", type=float)
    parser.add_argument("--squat-active-mask", action="store_true")
    parser.add_argument("--action-abs-bound", type=float)
    parser.add_argument("--expose-squat-blend", action="store_true")
    parser.add_argument("--mask-action-history", action="store_true")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    body = args.cove50_body_xml.resolve()
    checkpoints = {"squat": args.squat, "arms": args.arms,
                   "independent-arms": args.independent_arms}
    specs = [
        ("squat", "cycle-stand-start", 144, None),
        ("squat", "cycle-stand-reserved", 905, None),
        ("squat", "cycle-squat-start", 906, "squat"),
        ("arms", "both-arms", 144, None),
        ("arms", "both-arms-reserved", 905, None),
        ("independent-arms", "left-only", 144, "left"),
        ("independent-arms", "right-only", 905, "right"),
        ("independent-arms", "both-independent", 906, "both"),
    ]
    if args.lessons == "squat":
        specs = [spec for spec in specs if spec[0] == "squat"]
    elif args.lessons == "arms":
        specs = [spec for spec in specs if spec[0] != "squat"]
    if args.case:
        unknown = set(args.case) - {spec[1] for spec in specs}
        if unknown:
            parser.error(f"Unknown case names: {', '.join(sorted(unknown))}")
        specs = [spec for spec in specs if spec[1] in args.case]
    for lesson in {spec[0] for spec in specs}:
        if checkpoints[lesson] is None:
            parser.error(f"--{lesson} checkpoint is required for selected cases")
    rows = []
    for lesson, name, seed, option in specs:
        checkpoint = checkpoints[lesson].resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        prefix = f"{args.prefix}-{name}"
        report = out / f"{prefix}.json"
        log = out / f"{prefix}.log"
        evaluator = EVAL / ("evaluate_posture.py" if lesson == "squat"
                            else "evaluate_arm.py")
        command = [sys.executable, str(evaluator)]
        if lesson == "squat":
            command += [str(checkpoint), "--mode", "cycle", "--task",
                        "Mjlab-Posture-Flat-Ape-Elbows-HeadingObserved"]
            if option == "squat":
                command.append("--start-squat")
        else:
            command.append(str(checkpoint))
            if option:
                command += ["--independent-arm", option]
        command += ["--cove50-body-xml", str(body),
                    "--seed", str(seed + args.seed_offset), "--out", str(report)]
        command += ["--motor-rail-v", str(args.motor_rail_v),
                    "--command-delay-ms", str(args.command_delay_ms)]
        if lesson == "squat":
            if args.leg_target_margin_rad is not None:
                command += ["--leg-target-margin-rad",
                            str(args.leg_target_margin_rad)]
            if args.squat_active_mask:
                command.append("--squat-active-mask")
            if args.action_abs_bound is not None:
                command += ["--action-abs-bound", str(args.action_abs_bound)]
            if args.expose_squat_blend:
                command.append("--expose-squat-blend")
            if args.mask_action_history:
                command.append("--mask-action-history")
        else:
            if args.shoulder_target_margin_rad is not None:
                command += ["--shoulder-target-margin-rad",
                            str(args.shoulder_target_margin_rad)]
        if args.action_smoothing_alpha is not None:
            command += ["--action-smoothing-alpha",
                        str(args.action_smoothing_alpha)]
        if seed == 144 and not args.seed_offset:
            command += ["--state-out", str(out / f"{prefix}.npz")]
        with log.open("w") as stream:
            process = subprocess.run(command, cwd=ROOT, stdout=stream,
                                     stderr=subprocess.STDOUT,
                                     env={**os.environ, "MUJOCO_GL": "egl"},
                                     check=False)
        row = {"lesson": lesson, "case": name, "seed": seed + args.seed_offset,
               "checkpoint": str(checkpoint.relative_to(ROOT)),
               "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               "body_sha256": hashlib.sha256(body.read_bytes()).hexdigest(),
               "motor_rail_v": args.motor_rail_v,
               "command_delay_ms": args.command_delay_ms,
               "shoulder_target_margin_rad": args.shoulder_target_margin_rad,
               "leg_target_margin_rad": args.leg_target_margin_rad,
               "action_smoothing_alpha": args.action_smoothing_alpha,
               "squat_active_mask": args.squat_active_mask,
               "action_abs_bound": args.action_abs_bound,
               "expose_squat_blend": args.expose_squat_blend,
               "mask_action_history": args.mask_action_history,
               "exit_code": process.returncode,
               "report": str(report.relative_to(ROOT)),
               "log": str(log.relative_to(ROOT))}
        if process.returncode == 0:
            result = json.loads(report.read_text())
            row["duration_s"] = result["duration_s"]
            if lesson == "squat":
                row["net_yaw_rad"] = result["net_yaw_rad"]
                row["initial_hold"] = result["initial_hold"]
                row["middle_hold"] = result["middle_hold"]
                row["final_hold"] = result["final_hold"]
                row["net_xy_m"] = result["net_xy_m"]
                row["ended_early"] = result["ended_early"]
            else:
                row["before"] = result["before"]
                row["raised"] = result["raised"]
                row["lowered"] = result["lowered"]
                row["peak_hand_force_n"] = result["peak_hand_force_n"]
        else:
            row["error_tail"] = "\n".join(log.read_text(errors="replace").splitlines()[-12:])
        rows.append(row)
        (out / f"{args.prefix}-posture-arm-matrix.json").write_text(
            json.dumps({"cases": rows}, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    if any(row["exit_code"] != 0 for row in rows):
        raise SystemExit("One or more native posture/arm replays failed")


if __name__ == "__main__":
    main()
