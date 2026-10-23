"""Fresh-process closed-loop velocity probes for a Cove checkpoint.

Each subprocess builds a new MJLab world so seed and body sensitivity results
are independent of PPO training state.  This script reports outcomes; it does
not infer physical hardware qualification from simulated successes.
"""

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
EVALUATOR = ROOT / "experiments/compact_morphology/mjlab_ape/evaluate.py"


def summarize(report: dict) -> dict:
    worlds = report["worlds"]
    mean = lambda key: round(sum(w[key] for w in worlds) / len(worlds), 4)
    return {
        "worlds": len(worlds),
        "survived_8s": sum(w["ended_by_timeout"] for w in worlds),
        "one_meter_alternating": report["one_meter_with_alternating_steps"],
        "mean_forward_m": mean("forward_m"),
        "mean_lateral_m": mean("lateral_m"),
        "mean_abs_lateral_m": round(sum(abs(w["lateral_m"]) for w in worlds) / len(worlds), 4),
        "max_abs_lateral_m": round(max(abs(w["lateral_m"]) for w in worlds), 4),
        "mean_yaw_rad": mean("net_yaw_rad"),
        "mean_leg_saturated_fraction": mean("fraction_frames_with_saturated_leg"),
        "mean_moving_leg_saturated_fraction": (
            mean("fraction_moving_frames_with_saturated_leg")
            if all("fraction_moving_frames_with_saturated_leg" in w for w in worlds) else None
        ),
        "mean_stop_travel_after_command_m": (
            round(sum(w["stop"]["travel_after_command_m"] for w in worlds) / len(worlds), 4)
            if all("stop" in w for w in worlds) else None
        ),
        "mean_late_double_support_fraction": (
            round(sum(w["stop"]["late_double_support_fraction"] for w in worlds) / len(worlds), 4)
            if all("stop" in w for w in worlds) else None
        ),
        "mean_foot_slip_m_left_right": [
            round(sum(w["contact_slip_m_left_right"][side] for w in worlds) / len(worlds), 4)
            for side in (0, 1)
        ],
        "max_hand_force_n": max(w["hand_force_n"] for w in worlds),
        "max_body_contact_fraction": max(w["fraction_frames_with_body_contact"] for w in worlds),
        "world_results": [{"forward_m": w["forward_m"],
                           "lateral_m": w["lateral_m"],
                           "net_yaw_rad": w["net_yaw_rad"],
                           "ended_by_timeout": w["ended_by_timeout"]}
                          for w in worlds],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--case", action="append",
                        help="Run only named cases; repeat for a bounded diagnostic subset")
    parser.add_argument("--action-limit-margin-rad", type=float,
                        help="Clip leg position targets inside the Cove joint limits")
    parser.add_argument("--ankle-target-margin-rad", type=float,
                        help="Clip only ankle position targets inside the Cove joint limits")
    parser.add_argument("--out-dir", type=Path, default=OUT,
                        help="Keep recovery evaluations separate from retained transfer evidence")
    parser.add_argument("--body-dir", type=Path, default=OUT,
                        help="Directory containing the exact selected body and sensitivity variants")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="Use a new reserved seed set when selecting a later checkpoint")
    args = parser.parse_args()
    if args.action_limit_margin_rad is not None and args.ankle_target_margin_rad is not None:
        parser.error("Choose either all-leg or ankle-only target clipping")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    bodies = args.body_dir.resolve()
    base = bodies / "cove50-direct-training.xml"
    cases = [
        ("forward-seed144", base, 144, 0.35, 0.0, 12.0, 0.0, None, None),
        ("forward-seed901", base, 901, 0.35, 0.0, 12.0, 0.0, None, None),
        ("turn-left", base, 902, 0.0, 0.3, 12.0, 0.0, None, None),
        ("turn-right", base, 903, 0.0, -0.3, 12.0, 0.0, None, None),
        ("walk-stop", base, 904, 0.35, 0.0, 12.0, 0.0, 4.0, None),
        ("light-lowgrip-weak-delay", bodies / "cove50-direct-light-lowgrip.xml",
         905, 0.35, 0.0, 10.0, 10.0, None, 0.55),
        ("heavy-lowgrip-weak-delay", bodies / "cove50-direct-heavy-lowgrip.xml",
         906, 0.35, 0.0, 10.0, 10.0, None, 0.55),
        ("heavy-highgrip-weak-delay", bodies / "cove50-direct-heavy-highgrip.xml",
         907, 0.35, 0.0, 10.0, 10.0, None, 1.25),
    ]
    if args.case:
        unknown = set(args.case) - {case[0] for case in cases}
        if unknown:
            parser.error(f"Unknown case names: {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case[0] in args.case]
    results = []
    for name, body, seed, vx, omega, rail, delay, stop, friction in cases:
        seed += args.seed_offset
        report_path = out / f"{args.prefix}-{name}.json"
        log_path = out / f"{args.prefix}-{name}.log"
        command = [sys.executable, str(EVALUATOR), str(checkpoint),
                   "--task", "Mjlab-Velocity-Flat-Ape-Elbows",
                   "--body-xml", str(body), "--physics-model", "frame50",
                   "--yaw-only-head", "--arms-down", "--worlds", str(args.worlds),
                   "--seed", str(seed), "--velocity-x", str(vx),
                   "--velocity-y", "0", "--angular-z", str(omega),
                   "--motor-rail-v", str(rail), "--command-delay-ms", str(delay),
                   "--out", str(report_path)]
        if stop is not None:
            command += ["--stop-at-s", str(stop)]
        if args.action_limit_margin_rad is not None:
            command += ["--action-limit-margin-rad", str(args.action_limit_margin_rad)]
        if args.ankle_target_margin_rad is not None:
            command += ["--ankle-target-margin-rad", str(args.ankle_target_margin_rad)]
        if friction is not None:
            command += ["--contact-slide-friction", str(friction)]
        if name == "forward-seed144":
            command += ["--state-out", str(out / f"{args.prefix}-{name}.npz")]
        with log_path.open("w") as log:
            proc = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                  env=os.environ.copy(), check=False)
        row = {"case": name, "body_sha256": hashlib.sha256(body.read_bytes()).hexdigest(),
               "seed": seed, "velocity_x_m_s": vx, "yaw_command_rad_s": omega,
               "motor_rail_v": rail, "command_delay_ms": delay,
               "action_limit_margin_rad": args.action_limit_margin_rad,
               "ankle_target_margin_rad": args.ankle_target_margin_rad,
               "imported_contact_slide_friction": friction,
               "stop_at_s": stop, "process_exit_code": proc.returncode,
               "report": str(report_path.relative_to(ROOT)),
               "log": str(log_path.relative_to(ROOT))}
        if proc.returncode == 0:
            row.update(summarize(json.loads(report_path.read_text())))
        else:
            row["error_tail"] = "\n".join(log_path.read_text(errors="replace").splitlines()[-12:])
        results.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("world_results", "log", "report")}),
              flush=True)
        (out / f"{args.prefix}-matrix.json").write_text(json.dumps({
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "cases": results,
        }, indent=2) + "\n")
    if any(row["process_exit_code"] != 0 for row in results):
        raise SystemExit("One or more native rollouts failed; inspect the case logs")


if __name__ == "__main__":
    main()
