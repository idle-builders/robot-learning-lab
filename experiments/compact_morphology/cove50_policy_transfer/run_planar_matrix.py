"""Reserved signed, lateral, turn, stop and boundary replay matrix on Cove."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from run_velocity_matrix import ROOT, OUT, EVALUATOR, summarize


def case(name: str, seed: int, vx: float, vy: float, yaw: float,
         *, body: str = "training", rail: float = 12.0, delay: float = 0.0,
         friction: float | None = None, stop: float | None = None) -> dict:
    return vars().copy()


CASES = [
    case("forward", 1101, 0.35, 0.0, 0.0),
    case("backward", 1102, -0.35, 0.0, 0.0),
    case("left-side", 1103, 0.0, 0.2, 0.0),
    case("right-side", 1104, 0.0, -0.2, 0.0),
    case("left-turn", 1105, 0.0, 0.0, 0.4),
    case("right-turn", 1106, 0.0, 0.0, -0.4),
    case("left-turn-stop", 1115, 0.0, 0.0, 0.4, stop=4.0),
    case("right-turn-stop", 1116, 0.0, 0.0, -0.4, stop=4.0),
    case("left-curve", 1107, 0.25, 0.0, 0.4),
    case("right-curve", 1108, 0.25, 0.0, -0.4),
    case("forward-stop", 1109, 0.35, 0.0, 0.0, stop=4.0),
    case("backward-stop", 1110, -0.35, 0.0, 0.0, stop=4.0),
    case("heavy-lowgrip-forward", 1111, 0.35, 0.0, 0.0,
         body="heavy-lowgrip", rail=10.0, delay=10.0, friction=0.55),
    case("heavy-lowgrip-backward", 1112, -0.35, 0.0, 0.0,
         body="heavy-lowgrip", rail=10.0, delay=10.0, friction=0.55),
    case("heavy-lowgrip-turn", 1113, 0.0, 0.0, 0.4,
         body="heavy-lowgrip", rail=10.0, delay=10.0, friction=0.55),
    case("light-lowgrip-side", 1114, 0.0, 0.2, 0.0,
         body="light-lowgrip", rail=10.0, delay=10.0, friction=0.55),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--body-dir", type=Path, default=OUT)
    parser.add_argument("--case", action="append", help="Run only named cases")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--action-limit-margin-rad", type=float,
                        help="Clip leg position targets inside Cove joint travel")
    parser.add_argument("--ankle-target-margin-rad", type=float,
                        help="Clip only ankle position targets inside Cove joint travel")
    args = parser.parse_args()
    if args.action_limit_margin_rad is not None and args.ankle_target_margin_rad is not None:
        parser.error("Choose either all-leg or ankle-only target clipping")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    bodies = args.body_dir.resolve()
    cases = CASES
    if args.case:
        unknown = set(args.case) - {spec["name"] for spec in CASES}
        if unknown:
            parser.error(f"Unknown case names: {', '.join(sorted(unknown))}")
        cases = [spec for spec in CASES if spec["name"] in args.case]
    rows = []
    for spec in cases:
        name = spec["name"]
        body = bodies / ("cove50-direct-training.xml" if spec["body"] == "training"
                      else f"cove50-direct-{spec['body']}.xml")
        report = out / f"{args.prefix}-{name}.json"
        log = out / f"{args.prefix}-{name}.log"
        cmd = [sys.executable, str(EVALUATOR), str(checkpoint),
               "--task", "Mjlab-Velocity-Flat-Ape-Elbows-Planar",
               "--body-xml", str(body), "--physics-model", "frame50",
               "--yaw-only-head", "--arms-down", "--worlds", str(args.worlds),
               "--seed", str(spec["seed"] + args.seed_offset),
               "--velocity-x", str(spec["vx"]),
               "--velocity-y", str(spec["vy"]),
               "--angular-z", str(spec["yaw"]),
               "--motor-rail-v", str(spec["rail"]),
               "--command-delay-ms", str(spec["delay"]),
               "--out", str(report)]
        if spec["friction"] is not None:
            cmd += ["--contact-slide-friction", str(spec["friction"])]
        if spec["stop"] is not None:
            cmd += ["--stop-at-s", str(spec["stop"])]
        if args.action_limit_margin_rad is not None:
            cmd += ["--action-limit-margin-rad", str(args.action_limit_margin_rad)]
        if args.ankle_target_margin_rad is not None:
            cmd += ["--ankle-target-margin-rad", str(args.ankle_target_margin_rad)]
        if name in ("forward", "backward"):
            cmd += ["--state-out", str(out / f"{args.prefix}-{name}.npz")]
        with log.open("w") as stream:
            process = subprocess.run(cmd, cwd=ROOT, stdout=stream,
                                     stderr=subprocess.STDOUT,
                                     env=os.environ.copy(),
                                     check=False)
        row = {"case": name, "seed": spec["seed"] + args.seed_offset, "velocity_command":
               [spec["vx"], spec["vy"], spec["yaw"]],
               "body_sha256": hashlib.sha256(body.read_bytes()).hexdigest(),
               "motor_rail_v": spec["rail"], "command_delay_ms": spec["delay"],
               "action_limit_margin_rad": args.action_limit_margin_rad,
               "ankle_target_margin_rad": args.ankle_target_margin_rad,
               "imported_contact_slide_friction": spec["friction"],
               "stop_at_s": spec["stop"], "exit_code": process.returncode,
               "report": str(report.relative_to(ROOT)),
               "log": str(log.relative_to(ROOT))}
        if process.returncode == 0:
            result = json.loads(report.read_text())
            row.update(summarize(result))
            worlds = result["worlds"]
            mean = lambda key: round(sum(w[key] for w in worlds) / len(worlds), 4)
            row["mean_command_progress_m"] = mean("command_progress_m")
            row["mean_cross_track_m"] = mean("cross_track_m")
            row["mean_yaw_error_rad"] = round(sum(
                w["net_yaw_rad"] - w["target_yaw_change_rad"]
                for w in worlds) / len(worlds), 4)
            if spec["stop"] is not None:
                row["mean_late_speed_m_s"] = round(sum(
                    w["stop"]["late_mean_planar_speed_m_s"]
                    for w in worlds) / len(worlds), 4)
                row["mean_yaw_after_stop_rad"] = round(sum(
                    w["stop"]["yaw_after_command_rad"]
                    for w in worlds) / len(worlds), 4)
                row["mean_late_abs_yaw_speed_rad_s"] = round(sum(
                    w["stop"]["late_mean_abs_yaw_speed_rad_s"]
                    for w in worlds) / len(worlds), 4)
                row["mean_late_double_support_fraction"] = round(sum(
                    w["stop"]["late_double_support_fraction"]
                    for w in worlds) / len(worlds), 4)
        else:
            row["error_tail"] = "\n".join(log.read_text(errors="replace").splitlines()[-12:])
        rows.append(row)
        (out / f"{args.prefix}-planar-matrix.json").write_text(json.dumps({
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "cases": rows,
        }, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k not in
                          ("world_results", "report", "log")}), flush=True)
    if any(row["exit_code"] != 0 for row in rows):
        raise SystemExit("One or more native rollouts failed; see case logs")


if __name__ == "__main__":
    main()
