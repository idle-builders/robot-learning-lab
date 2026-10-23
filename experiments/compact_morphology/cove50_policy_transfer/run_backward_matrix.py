"""Fresh nominal and boundary reverse-command rollouts for a Cove checkpoint."""

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
EVAL = ROOT / "experiments/compact_morphology/mjlab_ape/evaluate.py"


def summarize(report: dict) -> dict:
    worlds = report["worlds"]
    mean = lambda key: round(sum(w[key] for w in worlds) / len(worlds), 4)
    result = {
        "worlds": len(worlds),
        "survived_8s": sum(w["ended_by_timeout"] for w in worlds),
        "one_meter_reverse_alternating": sum(
            w["ended_by_timeout"] and w["forward_m"] <= -1.0
            and w["alternating_touchdowns"] >= 2 for w in worlds
        ),
        "mean_backward_m": -mean("forward_m"),
        "mean_lateral_m": mean("lateral_m"),
        "mean_yaw_rad": mean("net_yaw_rad"),
        "mean_leg_saturated_fraction": mean("fraction_frames_with_saturated_leg"),
        "max_body_contact_fraction": max(w["fraction_frames_with_body_contact"] for w in worlds),
        "world_results": [{
            "backward_m": round(-w["forward_m"], 4),
            "lateral_m": w["lateral_m"],
            "net_yaw_rad": w["net_yaw_rad"],
            "ended_by_timeout": w["ended_by_timeout"],
            "alternating_touchdowns": w["alternating_touchdowns"],
            "stop": w.get("stop"),
        } for w in worlds],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    base = OUT / "cove50-direct-training.xml"
    cases = [
        ("backward-seed144", base, 144, 12.0, 0.0, None, None),
        ("backward-seed905", base, 905, 12.0, 0.0, None, None),
        ("backward-stop", base, 906, 12.0, 0.0, 4.0, None),
        ("heavy-lowgrip-weak-delay", OUT / "cove50-direct-heavy-lowgrip.xml",
         907, 10.0, 10.0, None, 0.55),
    ]
    rows = []
    for name, body, seed, rail, delay, stop, grip in cases:
        report = OUT / f"{args.prefix}-{name}.json"
        log = OUT / f"{args.prefix}-{name}.log"
        command = [sys.executable, str(EVAL), str(checkpoint),
                   "--task", "Mjlab-Velocity-Flat-Ape-Elbows-Backward",
                   "--body-xml", str(body), "--physics-model", "frame50",
                   "--yaw-only-head", "--arms-down", "--worlds", "4",
                   "--seed", str(seed), "--velocity-x", "-0.35",
                   "--velocity-y", "0", "--angular-z", "0",
                   "--motor-rail-v", str(rail), "--command-delay-ms", str(delay),
                   "--out", str(report)]
        if stop is not None:
            command += ["--stop-at-s", str(stop)]
        if grip is not None:
            command += ["--contact-slide-friction", str(grip)]
        if name == "backward-seed144":
            command += ["--state-out", str(OUT / f"{args.prefix}-{name}.npz")]
        with log.open("w") as stream:
            process = subprocess.run(command, cwd=ROOT, stdout=stream,
                                     stderr=subprocess.STDOUT,
                                     env={**os.environ, "MUJOCO_GL": "egl"}, check=False)
        row = {
            "case": name, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "body_sha256": hashlib.sha256(body.read_bytes()).hexdigest(),
            "seed": seed, "motor_rail_v": rail, "command_delay_ms": delay,
            "contact_slide_friction": grip, "stop_at_s": stop,
            "exit_code": process.returncode,
            "report": str(report.relative_to(ROOT)), "log": str(log.relative_to(ROOT)),
        }
        if process.returncode == 0:
            row.update(summarize(json.loads(report.read_text())))
        else:
            row["error_tail"] = "\n".join(log.read_text(errors="replace").splitlines()[-12:])
        rows.append(row)
        (OUT / f"{args.prefix}-matrix.json").write_text(json.dumps({
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "cases": rows,
        }, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "world_results"}), flush=True)
    if any(row["exit_code"] != 0 for row in rows):
        raise SystemExit("One or more backward replays failed")


if __name__ == "__main__":
    main()
